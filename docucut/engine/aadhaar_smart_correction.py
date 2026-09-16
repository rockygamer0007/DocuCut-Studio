from __future__ import annotations

from typing import Iterable

AadhaarBox = tuple[float, float, float, float]


def extend_aadhaar_bottom(
    boxes: Iterable[AadhaarBox],
    image_height: int,
    extra_pixels: int = 32,
) -> list[AadhaarBox]:
    """
    Extend only the bottom edge of AI-detected Aadhaar boxes.

    The correction is expressed in source-image pixels so that the
    same physical correction is applied regardless of image size.
    """
    if image_height <= 0:
        return list(boxes)

    extra = extra_pixels / image_height

    result: list[AadhaarBox] = []

    for x0, y0, x1, y1 in boxes:
        result.append(
            (
                x0,
                y0,
                x1,
                min(1.0, y1 + extra),
            )
        )

    return result
