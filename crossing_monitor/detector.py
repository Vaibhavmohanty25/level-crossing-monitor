"""Detect people and vehicles using YOLO."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"

# COCO class IDs: person, bicycle, car, motorcycle, bus, truck.
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]


class ObjectDetector:
    """Load the model once and reuse it across frames."""

    def __init__(self, confidence: float = 0.4) -> None:
        if not 0 < confidence <= 1:
            raise ValueError(
                "Confidence must be greater than 0 and at most 1."
            )

        self.confidence = confidence

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Downloads pretrained weights on the first run.
        self.model = YOLO(str(MODEL_PATH))

    def detect(self, frame: np.ndarray) -> Results:
        """Detect target objects in an OpenCV frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=TARGET_CLASSES,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        return results[0]