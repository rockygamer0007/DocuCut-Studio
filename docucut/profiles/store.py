from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROFILE_FILE = ROOT / "docucut" / "profiles" / "profiles.json"


class ProfileError(ValueError):
    """Invalid DocuCut profile configuration."""


def _number(value: Any, *, profile: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(
            f"{profile}.{field} must be numeric; got {value!r}"
        ) from exc

    if number != number:  # NaN
        raise ProfileError(
            f"{profile}.{field} cannot be NaN"
        )

    return number


@dataclass(frozen=True)
class CropBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_sequence(
        cls,
        values: list[Any] | tuple[Any, ...],
        *,
        profile: str,
        field: str,
    ) -> "CropBox":
        if len(values) != 4:
            raise ProfileError(
                f"{profile}.{field} must contain exactly "
                f"4 values; got {len(values)}"
            )

        box = cls(
            x0=_number(values[0], profile=profile, field=f"{field}[0]"),
            y0=_number(values[1], profile=profile, field=f"{field}[1]"),
            x1=_number(values[2], profile=profile, field=f"{field}[2]"),
            y1=_number(values[3], profile=profile, field=f"{field}[3]"),
        )

        box.validate(profile=profile, field=field)
        return box

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any],
        *,
        profile: str,
        field: str,
    ) -> "CropBox":
        required = ("x0", "y0", "x1", "y1")

        missing = [
            key for key in required
            if key not in values
        ]

        if missing:
            raise ProfileError(
                f"{profile}.{field} is missing: "
                f"{', '.join(missing)}"
            )

        box = cls(
            x0=_number(
                values["x0"],
                profile=profile,
                field=f"{field}.x0",
            ),
            y0=_number(
                values["y0"],
                profile=profile,
                field=f"{field}.y0",
            ),
            x1=_number(
                values["x1"],
                profile=profile,
                field=f"{field}.x1",
            ),
            y1=_number(
                values["y1"],
                profile=profile,
                field=f"{field}.y1",
            ),
        )

        box.validate(profile=profile, field=field)
        return box

    def validate(self, *, profile: str = "unknown", field: str = "box") -> None:
        if not (0.0 <= self.x0 < self.x1 <= 1.0):
            raise ProfileError(
                f"{profile}.{field} has invalid X coordinates: "
                f"{self.x0}, {self.x1}"
            )

        if not (0.0 <= self.y0 < self.y1 <= 1.0):
            raise ProfileError(
                f"{profile}.{field} has invalid Y coordinates: "
                f"{self.y0}, {self.y1}"
            )


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    enabled: bool
    sides: str
    front: CropBox | None
    back: CropBox | None
    output_pvc_width: int | None
    output_standard_width: int | None
    processing: str
    detector: str | None
    inset: dict[str, float]
    detect_keywords: tuple[str, ...]
    detect_filename: tuple[str, ...]
    detect_min_page_w: float | None
    detect_layout: str | None

    @property
    def aspect_ratio(self) -> float | None:
        if not self.front:
            return None

        width = self.front.x1 - self.front.x0
        height = self.front.y1 - self.front.y0

        if height <= 0:
            return None

        return width / height


class ProfileStore:
    def __init__(self, path: Path = PROFILE_FILE):
        self.path = Path(path)
        self._profiles: dict[str, Profile] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.is_file():
            raise ProfileError(
                f"Profile file not found: {self.path}"
            )

        try:
            raw = json.loads(
                self.path.read_text(
                    encoding="utf-8-sig"
                )
            )
        except json.JSONDecodeError as exc:
            raise ProfileError(
                f"Invalid JSON in {self.path}: {exc}"
            ) from exc

        raw_profiles = raw.get("profiles")

        if not isinstance(raw_profiles, dict):
            raise ProfileError(
                "profiles.json must contain a 'profiles' object."
            )

        profiles: dict[str, Profile] = {}

        for profile_id, data in raw_profiles.items():
            if not isinstance(data, dict):
                raise ProfileError(
                    f"{profile_id} must be an object."
                )

            front = self._parse_box(
                data.get("front"),
                profile=profile_id,
                field="front",
            )

            back = self._parse_box(
                data.get("back"),
                profile=profile_id,
                field="back",
            )

            inset_data = data.get("inset") or {}

            if not isinstance(inset_data, dict):
                raise ProfileError(
                    f"{profile_id}.inset must be an object."
                )

            inset = {
                name: _number(
                    inset_data.get(name, 0.0),
                    profile=profile_id,
                    field=f"inset.{name}",
                )
                for name in ("left", "top", "right", "bottom")
            }

            output_pvc = data.get("output_pvc")
            output_standard = data.get("output_standard")

            pvc_width = (
                int(_number(
                    output_pvc,
                    profile=profile_id,
                    field="output_pvc",
                ))
                if output_pvc is not None
                else None
            )

            standard_width = (
                int(_number(
                    output_standard,
                    profile=profile_id,
                    field="output_standard",
                ))
                if output_standard is not None
                else None
            )

            min_page_w = data.get("detect_min_page_w")

            if min_page_w is not None:
                min_page_w = _number(
                    min_page_w,
                    profile=profile_id,
                    field="detect_min_page_w",
                )

            keywords = data.get("detect_keywords") or []
            filenames = data.get("detect_filename") or []

            if not isinstance(keywords, list):
                raise ProfileError(
                    f"{profile_id}.detect_keywords must be a list."
                )

            if not isinstance(filenames, list):
                raise ProfileError(
                    f"{profile_id}.detect_filename must be a list."
                )

            profiles[profile_id] = Profile(
                id=profile_id,
                label=str(data.get("label", profile_id)),
                enabled=bool(data.get("enabled", True)),
                sides=str(data.get("sides", "single")),
                front=front,
                back=back,
                output_pvc_width=pvc_width,
                output_standard_width=standard_width,
                processing=str(
                    data.get("processing", "none")
                ),
                detector=data.get("detect_layout"),
                inset=inset,
                detect_keywords=tuple(
                    str(x) for x in keywords
                ),
                detect_filename=tuple(
                    str(x) for x in filenames
                ),
                detect_min_page_w=min_page_w,
                detect_layout=data.get("detect_layout"),
            )

        self._profiles = profiles

    @staticmethod
    def _parse_box(
        value: Any,
        *,
        profile: str,
        field: str,
    ) -> CropBox | None:
        if value is None:
            return None

        if isinstance(value, dict):
            return CropBox.from_mapping(
                value,
                profile=profile,
                field=field,
            )

        if isinstance(value, (list, tuple)):
            return CropBox.from_sequence(
                value,
                profile=profile,
                field=field,
            )

        raise ProfileError(
            f"{profile}.{field} must be an array or object."
        )

    def get(self, profile_id: str) -> Profile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            available = ", ".join(self._profiles)
            raise KeyError(
                f"Unknown profile {profile_id!r}. "
                f"Available profiles: {available}"
            ) from exc

    def all(self) -> list[Profile]:
        return list(self._profiles.values())
