from __future__ import annotations

import cv2
import numpy as np

ID1_WIDTH_MM = 85.6
ID1_HEIGHT_MM = 53.98
MIN_ASPECT = 1.4
MAX_ASPECT = 1.8


def id1_height_for_width(target_width: int) -> int:
    return max(1, round(target_width * ID1_HEIGHT_MM / ID1_WIDTH_MM))


def resize_to_exact(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    *,
    anchor_top: bool = False,
    anchor_bottom: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return image

    target_aspect = target_width / target_height
    current_aspect = width / height

    if current_aspect > target_aspect + 1e-6:
        new_width = max(1, int(round(height * target_aspect)))
        x0 = max(0, (width - new_width) // 2)
        image = image[:, x0:x0 + new_width]

    elif current_aspect < target_aspect - 1e-6:
        new_height = max(1, int(round(width / target_aspect)))

        if anchor_bottom:
            y0 = max(0, height - new_height)
        elif anchor_top:
            y0 = 0
        else:
            y0 = max(0, (height - new_height) // 2)

        image = image[y0:y0 + new_height, :]

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )


def resize_to_target(
    image: np.ndarray,
    target_width: int,
    *,
    exact: bool = False,
) -> np.ndarray:
    height, width = image.shape[:2]

    if width <= 0:
        return image

    if not exact and width >= target_width:
        return image

    scale = target_width / width
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (target_width, new_height),
        interpolation=cv2.INTER_LANCZOS4,
    )


def finalize_smart(
    image: np.ndarray,
    target_width: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    aspect = width / height if height else 0.0
    target_height = id1_height_for_width(target_width)

    if MIN_ASPECT < aspect < MAX_ASPECT:
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return resize_to_target(
        image,
        target_width,
        exact=True,
    )
