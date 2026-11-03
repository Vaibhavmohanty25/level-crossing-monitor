"""Session-specific, time-weighted intrusion occupancy maps."""

import csv
import io
import json
import math
import sqlite3
import uuid
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from crossing_monitor.storage import LEVEL_RANK, utc_now


class GridRecorder:
    def __init__(
        self,
        store,
        source,
        reference,
        points,
        max_interval=2.0,
    ):
        if not math.isfinite(max_interval) or max_interval <= 0:
            raise ValueError(
                "Maximum observation interval must be positive."
            )

        self.store = store
        self.height, self.width = reference.shape[:2]

        if min(self.height, self.width) < 2:
            raise ValueError("Reference image is too small.")

        self.points = np.asarray(points, dtype=np.int32)

        if (
            self.points.shape != (4, 2)
            or not cv2.isContourConvex(self.points)
        ):
            raise ValueError(
                "A convex four-point zone is required."
            )

        self.mask = np.zeros(
            (self.height, self.width),
            np.uint8,
        )
        cv2.fillPoly(self.mask, [self.points], 255)

        self.columns = 48
        self.rows = max(
            1,
            round(self.columns * self.height / self.width),
        )

        self.max_interval = max_interval
        self.previous = {}
        self.last_timestamp = None
        self.session_id = uuid.uuid4().hex

        store.connection.executescript("""
            CREATE TABLE IF NOT EXISTS heatmap_sessions (
                session_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                grid_columns INTEGER NOT NULL,
                grid_rows INTEGER NOT NULL,
                points_json TEXT NOT NULL,
                reference_path TEXT NOT NULL,
                max_interval REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS heatmap_cells (
                session_id TEXT NOT NULL
                    REFERENCES heatmap_sessions(session_id),
                incident_id TEXT NOT NULL
                    REFERENCES incidents(incident_id),
                cell_x INTEGER NOT NULL,
                cell_y INTEGER NOT NULL,
                seconds REAL NOT NULL DEFAULT 0
                    CHECK(seconds >= 0),
                observations INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (
                    session_id,
                    incident_id,
                    cell_x,
                    cell_y
                )
            );
        """)

        path = (
            store.path.parent
            / "heatmaps"
            / self.session_id
            / "reference.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        success, encoded = cv2.imencode(".png", reference)

        if not success:
            raise OSError(
                "Could not encode heatmap reference image."
            )

        path.write_bytes(encoded.tobytes())

        with store.connection:
            store.connection.execute(
                """
                INSERT INTO heatmap_sessions
                VALUES (
                    ?, ?, ?, NULL,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    self.session_id,
                    source,
                    utc_now(),
                    self.width,
                    self.height,
                    self.columns,
                    self.rows,
                    json.dumps(self.points.tolist()),
                    str(path.resolve()),
                    max_interval,
                ),
            )

    def _cell(self, box):
        values = [float(value) for value in box]

        if (
            len(values) != 4
            or not all(math.isfinite(v) for v in values)
        ):
            return None

        x1, y1, x2, y2 = values

        left = max(0, math.floor(x1))
        top = max(0, math.floor(y1))
        right = min(self.width, math.ceil(x2))
        bottom = min(self.height, math.ceil(y2))

        if right <= left or bottom <= top:
            return None

        moments = cv2.moments(
            self.mask[top:bottom, left:right],
            binaryImage=True,
        )

        if moments["m00"] == 0:
            return None

        # Representative position within the part of the
        # vehicle box that intersects the crossing zone.
        x = (
            left + moments["m10"] / moments["m00"]
        ) / (self.width - 1)

        y = (
            top + moments["m01"] / moments["m00"]
        ) / (self.height - 1)

        return (
            min(
                self.columns - 1,
                int(x * self.columns),
            ),
            min(
                self.rows - 1,
                int(y * self.rows),
            ),
        )

    def update(self, timestamp, observations):
        """
        observations:
            {track_id: (active_incident_id, visible_inside_box)}
        """

        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(
                "Timestamp must be finite and nonnegative."
            )

        if (
            self.last_timestamp is not None
            and timestamp < self.last_timestamp
        ):
            raise ValueError(
                "Timestamps must not move backwards."
            )

        delta = (
            0
            if self.last_timestamp is None
            else timestamp - self.last_timestamp
        )

        current = {}
        increments = defaultdict(lambda: [0.0, 0])

        for track_id, (incident_id, box) in observations.items():
            cell = self._cell(box)

            if cell is None:
                continue

            current[track_id] = (incident_id, *cell)
            increments[(incident_id, *cell)][1] += 1

            previous = self.previous.get(track_id)

            if (
                previous
                and previous[0] == incident_id
                and 0 < delta <= self.max_interval
            ):
                # Estimate the interval at the preceding
                # observed position.
                increments[previous][0] += delta

        with self.store.connection:
            self.store.connection.executemany(
                """
                INSERT INTO heatmap_cells
                VALUES (?, ?, ?, ?, ?, ?)

                ON CONFLICT(
                    session_id,
                    incident_id,
                    cell_x,
                    cell_y
                )
                DO UPDATE SET
                    seconds = seconds + excluded.seconds,
                    observations = (
                        observations + excluded.observations
                    )
                """,
                [
                    (
                        self.session_id,
                        *key,
                        seconds,
                        count,
                    )
                    for key, (seconds, count)
                    in increments.items()
                ],
            )

        self.previous = current
        self.last_timestamp = timestamp

    def finish(self):
        with self.store.connection:
            self.store.connection.execute(
                """
                UPDATE heatmap_sessions
                SET ended_at_utc = ?
                WHERE session_id = ?
                """,
                (utc_now(), self.session_id),
            )


def _connect(path):
    return sqlite3.connect(
        Path(path).resolve().as_uri() + "?mode=ro",
        uri=True,
    )


def list_sessions(path):
    if not Path(path).is_file():
        return []

    with closing(_connect(path)) as connection:
        connection.row_factory = sqlite3.Row

        exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE
                type = 'table'
                AND name = 'heatmap_sessions'
            """
        ).fetchone()

        if not exists:
            return []

        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM heatmap_sessions
                ORDER BY rowid DESC
                """
            )
        ]


def read_heatmap(path, session_id, levels=None):
    levels = (
        list(LEVEL_RANK)
        if levels is None
        else list(levels)
    )

    if any(level not in LEVEL_RANK for level in levels):
        raise ValueError("Unknown escalation level.")

    with closing(_connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")

        session = connection.execute(
            """
            SELECT *
            FROM heatmap_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        if session is None:
            raise KeyError("Unknown monitoring session.")

        session = dict(session)
        session["points"] = json.loads(
            session["points_json"]
        )

        if not levels:
            return session, []

        placeholders = ",".join("?" for _ in levels)

        rows = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT
                    c.incident_id,
                    i.track_id,
                    i.max_level,
                    c.cell_x,
                    c.cell_y,
                    c.seconds,
                    c.observations
                FROM heatmap_cells c
                JOIN incidents i USING (incident_id)
                WHERE
                    c.session_id = ?
                    AND i.max_level IN ({placeholders})
                ORDER BY
                    c.incident_id,
                    c.cell_y,
                    c.cell_x
                """,
                (session_id, *levels),
            )
        ]

        for row in rows:
            row["x_normalized"] = (
                row["cell_x"] + 0.5
            ) / session["grid_columns"]

            row["y_normalized"] = (
                row["cell_y"] + 0.5
            ) / session["grid_rows"]

        return session, rows


def render_heatmap(
    session,
    rows,
    opacity=0.7,
    smoothing=1.0,
):
    """Return a BGR image without changing stored durations."""

    if (
        not 0 <= opacity <= 1
        or not math.isfinite(smoothing)
        or smoothing < 0
    ):
        raise ValueError(
            "Invalid heatmap display settings."
        )

    data = np.frombuffer(
        Path(session["reference_path"]).read_bytes(),
        np.uint8,
    )
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    height = session["height"]
    width = session["width"]

    if (
        image is None
        or image.shape[:2] != (height, width)
    ):
        raise ValueError(
            "Reference image is missing, invalid, "
            "or has changed dimensions."
        )

    grid = np.zeros(
        (
            session["grid_rows"],
            session["grid_columns"],
        ),
        np.float32,
    )

    for row in rows:
        grid[row["cell_y"], row["cell_x"]] += (
            row["seconds"]
        )

    heat = cv2.resize(
        grid,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    if smoothing > 0:
        heat = cv2.GaussianBlur(
            heat,
            (0, 0),
            sigmaX=(
                smoothing
                * width
                / session["grid_columns"]
            ),
            sigmaY=(
                smoothing
                * height
                / session["grid_rows"]
            ),
        )

    points = np.asarray(
        session["points"],
        np.int32,
    )

    mask = np.zeros(
        (height, width),
        np.uint8,
    )
    cv2.fillPoly(mask, [points], 255)

    heat[mask == 0] = 0
    peak = float(heat.max())

    if peak > 0:
        intensity = heat / peak

        colours = cv2.applyColorMap(
            (intensity * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )

        alpha = (
            opacity * np.sqrt(intensity)
        )[..., None]

        image = np.clip(
            image * (1 - alpha) + colours * alpha,
            0,
            255,
        ).astype(np.uint8)

    cv2.polylines(
        image,
        [points],
        True,
        (0, 255, 255),
        2,
    )

    return image


def cells_csv(rows):
    output = io.StringIO(newline="")

    fields = [
        "incident_id",
        "track_id",
        "max_level",
        "cell_x",
        "cell_y",
        "x_normalized",
        "y_normalized",
        "seconds",
        "observations",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fields,
    )
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()