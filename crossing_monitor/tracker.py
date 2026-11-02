"""Track people and vehicles across consecutive video frames."""

from pathlib import Path

import numpy as np
from ultralytics.engine.results import Results

from crossing_monitor.detector import ObjectDetector, TARGET_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKER_CONFIG = PROJECT_ROOT / "config" / "bytetrack.yaml"


class ObjectTracker:
    """Reuse one YOLO model and one persistent ByteTrack session."""

    def __init__(self) -> None:
        if not TRACKER_CONFIG.is_file():
            raise FileNotFoundError(
                f"Missing tracker configuration: {TRACKER_CONFIG}"
            )

        # Keep lower-confidence detections available to ByteTrack.
        self.detector = ObjectDetector(confidence=0.1)

        print(
            f"Tracker: {TRACKER_CONFIG.name} | "
            f"Image size: {self.detector.image_size} | "
            f"Device: {self.detector.device}"
        )

    def track(self, frame: np.ndarray) -> Results:
        """Process the next frame from the same video source."""

        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker=str(TRACKER_CONFIG),
            conf=self.detector.confidence,
            classes=TARGET_CLASSES,
            imgsz=self.detector.image_size,
            device=self.detector.device,
            verbose=False,
        )

        return results[0]