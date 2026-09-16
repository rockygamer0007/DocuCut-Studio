from __future__ import annotations

import numpy as np


AADHAAR_FRONT = (0.084, 0.726, 0.488, 0.925)
AADHAAR_BACK = (0.509, 0.726, 0.919, 0.925)


def find_aadhaar_boxes(
    page_width: float,
    page_height: float,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] | None:
    """
    Dedicated calibrated detector for standard e-Aadhaar
    letter-size layouts.

    Returns normalized front/back boxes.
    """

    if page_width <= 0 or page_height <= 0:
        return None

    # The reference layout is intended for the lower
    # e-Aadhaar cut-out region on letter-size pages.
    aspect = page_width / page_height

    if not (0.70 <= aspect <= 0.80):
        return None

    return AADHAAR_FRONT, AADHAAR_BACK


def boxes_to_pixel_rects(
    image: np.ndarray,
    boxes: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ],
) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]

    result = []

    for x0, y0, x1, y1 in boxes:
        left = max(0, min(width - 1, round(x0 * width)))
        top = max(0, min(height - 1, round(y0 * height)))
        right = max(left + 1, min(width, round(x1 * width)))
        bottom = max(top + 1, min(height, round(y1 * height)))

        result.append((left, top, right, bottom))

    return result
