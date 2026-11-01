"""Select a crossing zone and check vehicle overlap."""

import math

import cv2
import numpy as np


class CrossingZone:
    """A polygon and its pixel mask for one fixed camera view."""

    def __init__(
        self,
        frame_shape: tuple,
        points: list[tuple[int, int]],
    ) -> None:
        self.height, self.width = frame_shape[:2]
        self.points = np.array(points, dtype=np.int32)

        self.mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        cv2.fillPoly(self.mask, [self.points], 255)

    def overlap_ratio(self, box) -> float:
        """Return the fraction of a visible bounding box in the zone."""

        x1, y1, x2, y2 = [float(value) for value in box]

        # Clip coordinates to the image boundaries.
        left = max(0, min(self.width, math.floor(x1)))
        top = max(0, min(self.height, math.floor(y1)))
        right = max(0, min(self.width, math.ceil(x2)))
        bottom = max(0, min(self.height, math.ceil(y2)))

        if right <= left or bottom <= top:
            return 0.0

        box_area = (right - left) * (bottom - top)

        intersection = cv2.countNonZero(
            self.mask[top:bottom, left:right]
        )

        return intersection / box_area

    def draw(self, frame: np.ndarray, color: tuple) -> None:
        """Draw the crossing boundary onto a displayed frame."""

        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                "Video dimensions changed. Restart and select the zone again."
            )

        cv2.polylines(
            frame,
            [self.points],
            isClosed=True,
            color=color,
            thickness=3,
        )


def select_zone(
    window_name: str,
    frame: np.ndarray,
) -> CrossingZone | None:
    """Let the user select four corners on a paused frame."""

    points: list[tuple[int, int]] = []
    message = "Click 4 corners in order around the crossing."

    def on_mouse(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            preview = frame.copy()

            for index, point in enumerate(points):
                cv2.circle(preview, point, 6, (0, 255, 255), -1)

                cv2.putText(
                    preview,
                    str(index + 1),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if len(points) >= 2:
                polygon = np.array(points, dtype=np.int32)

                cv2.polylines(
                    preview,
                    [polygon],
                    isClosed=len(points) == 4,
                    color=(0, 255, 255),
                    thickness=2,
                )

            cv2.putText(
                preview,
                message,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                preview,
                "ENTER: confirm | R: reset | Q: quit",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                return None

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                return None

            if key == ord("r"):
                points.clear()
                message = "Click 4 corners in order around the crossing."

            if key in (10, 13):
                if len(points) != 4:
                    message = "Select exactly 4 corners first."
                    continue

                polygon = np.array(points, dtype=np.int32)

                if (
                    not cv2.isContourConvex(polygon)
                    or cv2.contourArea(polygon) < 25
                ):
                    message = "Invalid shape. Press R and select corners in order."
                    continue

                return CrossingZone(frame.shape, points)

    finally:
        # Remove the selection callback before normal playback.
        cv2.setMouseCallback(window_name, lambda *args: None)