from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from docucut.engine.aadhaar_smart_correction import extend_aadhaar_bottom


IDEAL_ASPECT = 1.585185185185185
MIN_ASPECT = 1.4
MAX_ASPECT = 1.8

MIN_PAIR_SIMILARITY = 0.8
MAX_PAIR_OVERLAP = 0.05
ALIGN_TOLERANCE = 0.15

MIN_WIDTH_RATIO = 0.12
MIN_HEIGHT_RATIO = 0.04


def _looks_like_card(x0: int, y0: int, x1: int, y1: int) -> bool:
    width = x1 - x0
    height = y1 - y0

    if height <= 0:
        return False

    aspect = width / height
    return width > height and MIN_ASPECT <= aspect <= MAX_ASPECT


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _aspect_fit(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0

    if height <= 0:
        return 0.0

    aspect = width / height

    return 1.0 - min(
        1.0,
        abs(aspect - IDEAL_ASPECT) / 0.4,
    )


def _pair_score(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    if _iou(a, b) > MAX_PAIR_OVERLAP:
        return float("-inf")

    aw = a[2] - a[0]
    ah = a[3] - a[1]
    bw = b[2] - b[0]
    bh = b[3] - b[1]

    similarity = (
        min(aw, bw) / max(aw, bw)
    ) * (
        min(ah, bh) / max(ah, bh)
    )

    if similarity < MIN_PAIR_SIMILARITY:
        return float("-inf")

    y_diff = abs(a[1] - b[1]) / ALIGN_TOLERANCE
    x_diff = abs(a[0] - b[0]) / ALIGN_TOLERANCE

    aligned_y = max(0.0, 1.0 - y_diff)
    aligned_x = max(0.0, 1.0 - x_diff)
    aligned = min(1.0, aligned_y + aligned_x)

    area = min(
        max(0.0, aw * ah),
        max(0.0, bw * bh),
    )

    return (
        similarity * 3.0
        + aligned * 3.0
        + (_aspect_fit(a) + _aspect_fit(b)) * 2.0
        + area * 2.0
    )


def _reading_order(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    return sorted(
        boxes,
        key=lambda box: (
            round(box[1], 2),
            box[0],
        ),
    )


def card_candidates_from_image(
    image: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]

    if width < 20 or height < 20:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    masks = [
        gray < 245,
        saturation > 40,
        cv2.dilate(
            cv2.Canny(gray, 50, 150),
            np.ones((3, 3), dtype=np.uint8),
            iterations=2,
        ) > 0,
    ]

    boxes: list[tuple[float, float, float, float]] = []

    for mask in masks:
        mask_u8 = np.asarray(mask, dtype=np.uint8) * 255

        for kernel_width in (0, 5, 15, 31):
            if kernel_width:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (kernel_width, kernel_width),
                )
                processed = cv2.morphologyEx(
                    mask_u8,
                    cv2.MORPH_CLOSE,
                    kernel,
                )
            else:
                processed = mask_u8

            for mode in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
                contours, _ = cv2.findContours(
                    processed,
                    mode,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    if w < width * MIN_WIDTH_RATIO:
                        continue

                    if h < height * MIN_HEIGHT_RATIO:
                        continue

                    if not _looks_like_card(x, y, x + w, y + h):
                        continue

                    box = (
                        round(x / width, 4),
                        round(y / height, 4),
                        round((x + w) / width, 4),
                        round((y + h) / height, 4),
                    )

                    if box not in boxes:
                        boxes.append(box)

    return boxes


def select_cards(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    if not boxes:
        return []

    best_pair = None
    best_score = float("-inf")

    for a, b in combinations(boxes, 2):
        score = _pair_score(a, b)

        if score > best_score:
            best_score = score
            best_pair = (a, b)

    if best_pair is not None and best_score != float("-inf"):
        return _reading_order(list(best_pair))

    return [
        max(
            boxes,
            key=lambda box: (
                (box[2] - box[0]) *
                (box[3] - box[1])
            ),
        )
    ]


def find_card_boxes_in_image(
    image: np.ndarray,
    *,
    profile_id: str | None = None,
) -> list[tuple[float, float, float, float]]:
    boxes = select_cards(card_candidates_from_image(image))

    if profile_id and profile_id.upper() == "AADHAAR" and boxes:
        boxes = extend_aadhaar_bottom(
            boxes,
            image.shape[0],
            extra_pixels=32,
        )

    return boxes
