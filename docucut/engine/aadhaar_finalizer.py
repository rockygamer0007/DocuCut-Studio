from __future__ import annotations

import cv2
import numpy as np

from docucut.engine.finalizer import id1_height_for_width, resize_to_exact


def remove_aadhaar_bottom_border_safely(
    image: np.ndarray,
    max_scan_px: int = 8,
) -> np.ndarray:
    """
    Remove only a thin, very dark bottom border line.

    Important: do not remove the surrounding content area.
    This keeps the Aadhaar VID region intact.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]

    scan = min(max_scan_px, max(0, height - 1))

    for offset in range(scan):
        y = height - 1 - offset
        row = gray[y]

        dark_ratio = float(np.mean(row < 130))

        # A true border line is normally dark across a substantial
        # part of the row. Avoid treating normal printed content as
        # a removable border.
        if dark_ratio >= 0.28:
            return image[:y]

        break

    return image


def finalize_aadhaar(
    image: np.ndarray,
    target_width: int,
) -> np.ndarray:
    """
    Aadhaar-specific finalization.

    Keeps the lower card content intact, removes only a thin
    bottom border when clearly detected, then applies the ID-1
    aspect ratio and exact output dimensions.
    """
    image = remove_aadhaar_bottom_border_safely(image)

    target_height = id1_height_for_width(target_width)

    return resize_to_exact(
        image,
        target_width,
        target_height,
    )
