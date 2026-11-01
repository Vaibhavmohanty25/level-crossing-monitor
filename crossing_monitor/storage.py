"""SQLite incident records and CSV export."""

import argparse
import csv
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


LEVEL_RANK = {
    "ENTRY": 0,
    "PERSISTENT": 1,
    "PROLONGED": 2,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "runtime" / "incidents.db"


def utc_now():
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


class IncidentStore:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

        self.connection.execute("PRAGMA foreign_keys = ON")

        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                start_seconds REAL NOT NULL,
                end_seconds REAL,
                duration_seconds REAL NOT NULL DEFAULT 0,
                max_level TEXT NOT NULL DEFAULT 'ENTRY',
                status TEXT NOT NULL DEFAULT 'active',
                end_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL
                    REFERENCES incidents(incident_id),
                level TEXT NOT NULL,
                path TEXT NOT NULL,
                recorded_at_utc TEXT NOT NULL,
                UNIQUE(incident_id, level)
            );
        """)

    def start(self, source, track_id, start_seconds):
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError(
                "Start time must be finite and nonnegative."
            )

        incident_id = uuid.uuid4().hex

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO incidents (
                    incident_id,
                    source,
                    track_id,
                    started_at_utc,
                    start_seconds
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    source,
                    track_id,
                    utc_now(),
                    start_seconds,
                ),
            )

        return incident_id

    def get(self, incident_id):
        row = self.connection.execute(
            "SELECT * FROM incidents WHERE incident_id = ?",
            (incident_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Unknown incident: {incident_id}")

        event = dict(row)

        snapshot_rows = self.connection.execute(
            """
            SELECT *
            FROM snapshots
            WHERE incident_id = ?
            ORDER BY snapshot_id
            """,
            (incident_id,),
        ).fetchall()

        event["snapshots"] = [
            dict(item) for item in snapshot_rows
        ]

        return event

    def update(self, incident_id, duration, level):
        if (
            not math.isfinite(duration)
            or duration < 0
            or level not in LEVEL_RANK
        ):
            raise ValueError(
                "Invalid duration or escalation level."
            )

        previous = self.get(incident_id)

        if previous["status"] != "active":
            raise ValueError("Cannot update a closed incident.")

        highest = max(
            (previous["max_level"], level),
            key=LEVEL_RANK.get,
        )

        longest_duration = max(
            previous["duration_seconds"],
            duration,
        )

        with self.connection:
            self.connection.execute(
                """
                UPDATE incidents
                SET duration_seconds = ?, max_level = ?
                WHERE incident_id = ?
                """,
                (
                    longest_duration,
                    highest,
                    incident_id,
                ),
            )

    def add_snapshot(self, incident_id, level, path):
        if level not in LEVEL_RANK:
            raise ValueError("Invalid escalation level.")

        self.get(incident_id)

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO snapshots (
                    incident_id,
                    level,
                    path,
                    recorded_at_utc
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(incident_id, level) DO NOTHING
                """,
                (
                    incident_id,
                    level,
                    str(path),
                    utc_now(),
                ),
            )

    def finish(self, incident_id, end_seconds, reason):
        event = self.get(incident_id)

        if (
            not math.isfinite(end_seconds)
            or end_seconds < event["start_seconds"]
        ):
            raise ValueError(
                "End time must be finite and after the start."
            )

        final_duration = (
            end_seconds - event["start_seconds"]
        )

        if (
            final_duration
            < event["duration_seconds"] - 1e-6
        ):
            raise ValueError(
                "End time cannot precede the latest duration."
            )

        with self.connection:
            self.connection.execute(
                """
                UPDATE incidents
                SET
                    status = 'closed',
                    end_seconds = ?,
                    duration_seconds = ?,
                    ended_at_utc = ?,
                    end_reason = ?
                WHERE incident_id = ? AND status = 'active'
                """,
                (
                    end_seconds,
                    final_duration,
                    utc_now(),
                    reason,
                    incident_id,
                ),
            )

    def list_recent(self, limit=20):
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(
                "Limit must be a positive integer."
            )

        rows = self.connection.execute(
            """
            SELECT *
            FROM incidents
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    def export_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cursor = self.connection.execute(
            "SELECT * FROM incidents ORDER BY rowid"
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                column[0] for column in cursor.description
            ])

            writer.writerows(cursor.fetchall())

    def close(self):
        self.connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect saved incident records."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--export",
        type=Path,
        help="Export all incidents to CSV.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("Limit must be greater than zero.")

    store = IncidentStore()

    try:
        events = store.list_recent(args.limit)

        if not events:
            print("No incidents saved yet.")

        for event in events:
            print(
                f'{event["incident_id"][:8]} | '
                f'Track {event["track_id"]} | '
                f'{event["max_level"]} | '
                f'{event["duration_seconds"]:.1f}s | '
                f'{event["status"]} | '
                f'{event["end_reason"] or "ongoing"}'
            )

        if args.export:
            store.export_csv(args.export)

            print(
                f"CSV saved: {args.export.resolve()}"
            )

    finally:
        store.close()


if __name__ == "__main__":
    main()