"""Monitor a crossing and save incident and spatial evidence."""

import argparse
import math
import sqlite3
import time
from pathlib import Path

import cv2

from crossing_monitor.detector import ObjectDetector
from crossing_monitor.escalation import EscalationMonitor
from crossing_monitor.heatmap import GridRecorder
from crossing_monitor.recorder import IncidentRecorder
from crossing_monitor.storage import IncidentStore
from crossing_monitor.tracker import ObjectTracker
from crossing_monitor.zones import select_zone


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VIDEO = (
    PROJECT_ROOT / "data" / "videos" / "sample.mp4"
)

WINDOW_NAME = "Level Crossing Monitor"

VEHICLE_CLASSES = {1, 2, 3, 5, 7}
MIN_OVERLAP = 0.10

LEVEL_RANK = {
    "ENTRY": 0,
    "PERSISTENT": 1,
    "PROLONGED": 2,
}

LEVEL_COLORS = {
    "ENTRY": (0, 255, 255),
    "PERSISTENT": (0, 165, 255),
    "PROLONGED": (0, 0, 255),
}


def fit_frame(frame):
    height, width = frame.shape[:2]

    if width <= 1280:
        return frame

    scale = 1280 / width

    return cv2.resize(
        frame,
        (1280, max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def draw_status(frame, lines, color):
    cv2.rectangle(
        frame,
        (0, 0),
        (
            frame.shape[1] - 1,
            15 + len(lines) * 28,
        ),
        (25, 25, 25),
        -1,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def preview_video(
    source,
    detect=False,
    track=False,
    monitor=False,
):
    detector = None
    tracker = None
    zone = None
    store = None
    recorder = None
    heatmap = None

    escalation = EscalationMonitor()

    if track or monitor:
        print("Loading YOLO and ByteTrack...")
        tracker = ObjectTracker()

    elif detect:
        detector = ObjectDetector()

    capture = cv2.VideoCapture(source)
    is_camera = isinstance(source, int)

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source}"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0

        delay_ms = max(
            1,
            round(1000 / min(fps, 120)),
        )

        success, raw_frame = capture.read()

        if not success:
            raise RuntimeError(
                "Could not read the first frame."
            )

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        if monitor:
            # Save an unannotated reference for this session.
            reference_frame = fit_frame(
                raw_frame
            ).copy()

            zone = select_zone(
                WINDOW_NAME,
                reference_frame,
            )

            if zone is None:
                return

            if is_camera:
                success, raw_frame = capture.read()

        if monitor:
            runtime = (
                PROJECT_ROOT / "data" / "runtime"
            )

            store = IncidentStore(
                runtime / "incidents.db"
            )

            source_name = (
                f"camera:{source}"
                if is_camera
                else f"video:{source}"
            )

            recorder = IncidentRecorder(
                store,
                runtime / "evidence",
                source_name,
            )

            heatmap = GridRecorder(
                store,
                source_name,
                reference_frame,
                zone.points,
            )

            print(
                "Heatmap session: "
                f"{heatmap.session_id[:8]}"
            )

            print(
                "Incident database: "
                f"{store.path.resolve()}"
            )

        crossing_closed = False
        frame_count = 0

        start_clock = time.monotonic()
        received_at = start_clock

        while success:
            frame = fit_frame(raw_frame)
            frame_count += 1

            timestamp = (
                received_at - start_clock
                if is_camera
                else (frame_count - 1) / fps
            )

            result = None
            visible_tracks = 0

            if tracker is not None:
                result = tracker.track(frame)
                frame = result.plot()

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    visible_tracks = len(
                        result.boxes.id
                    )

            elif detector is not None:
                result = detector.detect(frame)
                frame = result.plot()

            visible_ids = set()
            inside_ids = set()
            vehicle_boxes = {}

            if (
                monitor
                and result is not None
                and result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = (
                    result.boxes.xyxy.cpu().numpy()
                )

                classes = (
                    result.boxes.cls.int().cpu().tolist()
                )

                track_ids = (
                    result.boxes.id.int().cpu().tolist()
                )

                for box, class_id, track_id in zip(
                    boxes,
                    classes,
                    track_ids,
                ):
                    if class_id not in VEHICLE_CLASSES:
                        continue

                    visible_ids.add(track_id)
                    vehicle_boxes[track_id] = box

                    if (
                        zone.overlap_ratio(box)
                        >= MIN_OVERLAP
                    ):
                        inside_ids.add(track_id)

            if monitor:
                states = escalation.update(
                    inside_ids,
                    visible_ids,
                    timestamp,
                    crossing_closed,
                )

                color = (
                    (0, 255, 255)
                    if crossing_closed
                    else (0, 255, 0)
                )

                status = (
                    "CLOSED - CLEAR"
                    if crossing_closed
                    else "OPEN"
                )

                if states:
                    highest = max(
                        states.values(),
                        key=lambda state: (
                            LEVEL_RANK[state.level]
                        ),
                    )

                    status = (
                        f"INTRUSION - {highest.level}"
                    )

                    color = LEVEL_COLORS[
                        highest.level
                    ]

                zone.draw(frame, color)

                for track_id, state in states.items():
                    if track_id not in vehicle_boxes:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        vehicle_boxes[track_id],
                    )

                    box_color = LEVEL_COLORS[
                        state.level
                    ]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        3,
                    )

                    label_y = min(
                        frame.shape[0] - 10,
                        max(20, y2 + 20),
                    )

                    cv2.putText(
                        frame,
                        (
                            f"ID {track_id}: "
                            f"{state.level} "
                            f"{state.duration:.1f}s"
                        ),
                        (max(0, x1), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                    )

                missing_count = sum(
                    state.missing
                    for state in states.values()
                )

                lines = [
                    f"Crossing: {status}",
                    (
                        f"In zone: {len(inside_ids)} | "
                        f"Active: {len(states)} | "
                        f"Unseen: {missing_count}"
                    ),
                    (
                        "Escalation: 3s persistent | "
                        "6s prolonged"
                    ),
                    (
                        "C: toggle Open/Closed | "
                        "Q: quit"
                    ),
                ]

            else:
                mode = (
                    "Tracking"
                    if track
                    else "Detection"
                    if detect
                    else "Preview"
                )

                color = (0, 255, 0)

                lines = [
                    (
                        f"Mode: {mode} | "
                        f"Frame: {frame_count}"
                    ),
                    (
                        f"Tracks visible: "
                        f"{visible_tracks} | Q: quit"
                    ),
                ]

            draw_status(frame, lines, color)

            if recorder is not None:
                recorder.sync(
                    states,
                    frame,
                    crossing_closed,
                    visible_ids,
                    timestamp,
                )

                # Only currently visible intrusion tracks
                # contribute to spatial occupancy.
                heatmap.update(
                    timestamp,
                    {
                        track_id: (
                            recorder.active[
                                track_id
                            ].incident_id,
                            vehicle_boxes[track_id],
                        )
                        for track_id in inside_ids
                        if track_id in recorder.active
                    },
                )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(delay_ms) & 0xFF

            if key == ord("q"):
                break

            if monitor and key == ord("c"):
                crossing_closed = not crossing_closed

            if cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break

            success, raw_frame = capture.read()
            received_at = time.monotonic()

        print(
            f"Stopped. Frames displayed: {frame_count}"
        )

    finally:
        capture.release()
        cv2.destroyAllWindows()

        if store is not None:
            try:
                if recorder is not None:
                    recorder.finish_all()

                if heatmap is not None:
                    heatmap.finish()

                report = (
                    PROJECT_ROOT
                    / "data"
                    / "runtime"
                    / "reports"
                    / "incidents.csv"
                )

                try:
                    store.export_csv(report)
                    print(
                        f"Incident report: {report}"
                    )

                except OSError as error:
                    print(
                        "CSV export failed; "
                        "database records retained: "
                        f"{error}"
                    )

            finally:
                store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Crossing monitoring."
    )

    sources = parser.add_mutually_exclusive_group()

    sources.add_argument(
        "--video",
        type=Path,
        help="Recorded video path.",
    )

    sources.add_argument(
        "--camera",
        type=int,
        help="Camera index.",
    )

    modes = parser.add_mutually_exclusive_group()

    modes.add_argument(
        "--detect",
        action="store_true",
    )

    modes.add_argument(
        "--track",
        action="store_true",
    )

    modes.add_argument(
        "--monitor",
        action="store_true",
    )

    args = parser.parse_args()

    if args.camera is not None:
        if args.camera < 0:
            parser.error(
                "Camera index must be zero or greater."
            )

        source = args.camera

    else:
        path = (
            args.video
            if args.video is not None
            else DEFAULT_VIDEO
        ).resolve()

        if not path.is_file():
            parser.error(
                f"Video file does not exist: {path}"
            )

        source = str(path)

    try:
        preview_video(
            source,
            args.detect,
            args.track,
            args.monitor,
        )

    except (
        RuntimeError,
        ValueError,
        OSError,
        sqlite3.Error,
        cv2.error,
    ) as error:
        parser.exit(
            status=1,
            message=f"Monitoring failed: {error}\n",
        )


if __name__ == "__main__":
    main()