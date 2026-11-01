"""Save annotated incident screenshots."""

from pathlib import Path

import cv2


def save_snapshot(
    frame,
    directory,
    incident_id,
    level,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = (
        directory
        / f"{incident_id}_{level.lower()}.jpg"
    )

    if not cv2.imwrite(str(path), frame):
        raise OSError(
            f"Could not write screenshot: {path}"
        )

    return str(path.resolve())