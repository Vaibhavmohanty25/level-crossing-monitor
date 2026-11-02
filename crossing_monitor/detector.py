"""Detect people and vehicles using a reusable YOLO model."""

import math
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / os.getenv(
    "CROSSING_MODEL", "yolo11n.pt"
)

INFERENCE_SIZE = int(os.getenv("CROSSING_IMGSZ", "640"))
INFERENCE_DEVICE = os.getenv("CROSSING_DEVICE", "cpu")

# COCO: person, bicycle, car, motorcycle, bus, truck.
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]


class ObjectDetector:
    """Load the model once and reuse it across frames."""

    def __init__(self, confidence: float = 0.4) -> None:
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError(
                "Confidence must be finite and in (0, 1]."
            )

        if INFERENCE_SIZE < 32 or INFERENCE_SIZE % 32 != 0:
            raise ValueError(
                "CROSSING_IMGSZ must be a positive multiple of 32."
            )

        self.confidence = confidence
        self.image_size = INFERENCE_SIZE
        self.device = INFERENCE_DEVICE

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Downloads supported pretrained weights if missing.
        self.model = YOLO(str(MODEL_PATH))

    def detect(self, frame: np.ndarray) -> Results:
        """Detect target objects in an OpenCV frame."""

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=TARGET_CLASSES,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        return results[0]