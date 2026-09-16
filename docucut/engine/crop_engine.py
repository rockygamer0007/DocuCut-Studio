from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from docucut.engine.finalizer import finalize_smart
from docucut.engine.smart_detector import find_card_boxes_in_image
from docucut.profiles.store import CropBox, Profile


@dataclass(frozen=True)
class PixelBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class CropResult:
    source: Path
    output: Path
    profile_id: str
    side: str
    source_size: tuple[int, int]
    crop_box: PixelBox
    output_size: tuple[int, int]


class CropEngine:

    def normalized_to_pixels(
        self,
        image_size: tuple[int, int],
        box: CropBox,
    ) -> PixelBox:
        width, height = image_size

        left = round(box.x0 * width)
        top = round(box.y0 * height)
        right = round(box.x1 * width)
        bottom = round(box.y1 * height)

        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))

        return PixelBox(left, top, right, bottom)

    def crop(
        self,
        image: np.ndarray,
        box: CropBox,
    ) -> tuple[np.ndarray, PixelBox]:
        height, width = image.shape[:2]

        pixel_box = self.normalized_to_pixels(
            (width, height),
            box,
        )

        cropped = image[
            pixel_box.top:pixel_box.bottom,
            pixel_box.left:pixel_box.right,
        ]

        return cropped, pixel_box

    def crop_side(
        self,
        image: np.ndarray,
        profile: Profile,
        side: str,
    ) -> tuple[np.ndarray, PixelBox]:
        side = side.lower()

        if side == "front":
            box = profile.front
        elif side == "back":
            box = profile.back
        else:
            raise ValueError(
                f"Unknown side: {side!r}. Expected 'front' or 'back'."
            )

        if box is None:
            raise ValueError(
                f"Profile {profile.id} has no {side} crop box."
            )

        return self.crop(image, box)

    def save(
        self,
        image: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Write through a temporary file first. This makes repeated
        # processing safer and avoids leaving a partially written PNG.
        temp_output = output.with_name(
            output.stem + ".tmp" + output.suffix
        )

        try:
            ok = cv2.imwrite(
                str(temp_output),
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )

            if not ok or not temp_output.is_file():
                raise IOError(
                    f"Failed to write temporary PNG: {temp_output}"
                )

            temp_output.replace(output)

        finally:
            if temp_output.exists():
                try:
                    temp_output.unlink()
                except OSError:
                    pass

        return output

    def process_image(
        self,
        source_path: str | Path,
        profile: Profile,
        side: str,
        output_path: str | Path,
        *,
        smart: bool = False,
        target_width: int | None = None,
    ) -> CropResult:
        source = Path(source_path)

        if not source.is_file():
            raise FileNotFoundError(source)

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Unable to read image: {source}")

        source_height, source_width = image.shape[:2]

        if smart:
            if target_width is None:
                raise ValueError(
                    "target_width is required when smart=True"
                )

            boxes = find_card_boxes_in_image(
                image,
                profile_id=profile.id,
            )

            if not boxes:
                raise ValueError(
                    f"SMART detection found no card boxes for {profile.id}"
                )

            wanted_side = side.lower()

            if wanted_side not in {"front", "back"}:
                raise ValueError(
                    f"Unknown side: {side!r}"
                )

            index = 0 if wanted_side == "front" else 1

            if index >= len(boxes):
                raise ValueError(
                    f"SMART detection found only {len(boxes)} card(s)"
                )

            x0, y0, x1, y1 = boxes[index]

            left = max(0, min(source_width - 1, round(x0 * source_width)))
            top = max(0, min(source_height - 1, round(y0 * source_height)))
            right = max(left + 1, min(source_width, round(x1 * source_width)))
            bottom = max(top + 1, min(source_height, round(y1 * source_height)))

            cropped = image[top:bottom, left:right]

            pixel_box = PixelBox(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )

            cropped = finalize_smart(
                cropped,
                target_width,
            )

        else:
            cropped, pixel_box = self.crop_side(
                image=image,
                profile=profile,
                side=side,
            )

        output = self.save(
            cropped,
            output_path,
        )

        output_height, output_width = cropped.shape[:2]

        return CropResult(
            source=source,
            output=output,
            profile_id=profile.id,
            side=side.lower(),
            source_size=(source_width, source_height),
            crop_box=pixel_box,
            output_size=(output_width, output_height),
        )
