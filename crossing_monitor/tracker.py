"""Track people and vehicles across consecutive video frames."""

import numpy as np
from ultralytics.engine.results import Results

from crossing_monitor.detector import ObjectDetector, TARGET_CLASSES


class ObjectTracker:
    """Combine YOLO detection with ByteTrack object tracking."""

    def __init__(self) -> None:
        # ByteTrack uses lower-confidence detections to help
        # maintain existing tracks when objects become less visible.
        self.detector = ObjectDetector(confidence=0.1)

    def track(self, frame: np.ndarray) -> Results:
        """Detect and track objects in the next frame of one video."""
    
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.detector.confidence,
            classes=TARGET_CLASSES,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        return results[0]