"""
toolkit/annotations/yolo.py
────────────────────────────
YOLO annotation reader and validator.

Label file format (one row per object):
    <class_id> <x_center> <y_center> <width> <height>

All values are normalised to [0, 1].
Empty label files are valid (image contains no annotated objects).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class YoloAnnotation:
    """One bounding-box row from a YOLO label file."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class YoloFileResult:
    image_path: str
    label_path: str | None
    annotations: list[YoloAnnotation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True
    label_missing: bool = False


@dataclass
class YoloValidationReport:
    results: list[YoloFileResult] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return sum(1 for r in self.results if r.is_valid)

    @property
    def invalid_count(self) -> int:
        return sum(1 for r in self.results if not r.is_valid)

    @property
    def missing_label_count(self) -> int:
        return sum(1 for r in self.results if r.label_missing)

    def to_dict(self) -> dict:
        return {
            "format": "yolo",
            "total": len(self.results),
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "missing_labels": self.missing_label_count,
            "errors": [
                {"image": r.image_path, "errors": r.errors}
                for r in self.results if r.errors
            ],
            "warnings": [
                {"image": r.image_path, "warnings": r.warnings}
                for r in self.results if r.warnings
            ],
        }


def _parse_label_file(
    label_path: Path,
    num_classes: int | None,
    image_path_str: str,
) -> YoloFileResult:
    result = YoloFileResult(
        image_path=image_path_str,
        label_path=str(label_path),
    )

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        result.errors.append(f"Cannot read label file: {exc}")
        result.is_valid = False
        return result

    if not text:
        # Empty label file is valid — image has no annotated objects
        return result

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            result.errors.append(
                f"Line {line_no}: expected 5 values, got {len(parts)}: {line!r}"
            )
            result.is_valid = False
            continue

        try:
            class_id = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:])
        except ValueError:
            result.errors.append(f"Line {line_no}: non-numeric values: {line!r}")
            result.is_valid = False
            continue

        # Track per-line validity
        line_errors_before = len(result.errors)

        # Class ID
        if class_id < 0:
            result.errors.append(f"Line {line_no}: negative class_id {class_id}")
            result.is_valid = False
        if num_classes is not None and class_id >= num_classes:
            result.errors.append(
                f"Line {line_no}: class_id {class_id} >= num_classes {num_classes}"
            )
            result.is_valid = False

        # Coordinate bounds
        for val, name in ((x_c, "x_center"), (y_c, "y_center"), (bw, "width"), (bh, "height")):
            if not (0.0 <= val <= 1.0):
                result.errors.append(
                    f"Line {line_no}: {name}={val:.6f} is outside [0, 1]"
                )
                result.is_valid = False

        # Box extent
        if bw <= 0 or bh <= 0:
            result.errors.append(
                f"Line {line_no}: zero or negative box dimensions (w={bw}, h={bh})"
            )
            result.is_valid = False

        # Boundary check (box must fit within image)
        x_min = x_c - bw / 2
        x_max = x_c + bw / 2
        y_min = y_c - bh / 2
        y_max = y_c + bh / 2
        for v, name in ((x_min, "x_min"), (x_max, "x_max"), (y_min, "y_min"), (y_max, "y_max")):
            if v < -1e-6 or v > 1.0 + 1e-6:
                result.errors.append(
                    f"Line {line_no}: bounding box edge '{name}'={v:.6f} out of image"
                )
                result.is_valid = False

        # Only store annotation if this specific line had no new errors
        if len(result.errors) == line_errors_before:
            result.annotations.append(
                YoloAnnotation(class_id=class_id, x_center=x_c, y_center=y_c,
                               width=bw, height=bh)
            )

    return result


def validate_yolo(
    image_paths: Sequence[str | Path],
    labels_dir: str | Path,
    num_classes: int | None = None,
) -> YoloValidationReport:
    """
    Validate YOLO label files for each image in *image_paths*.

    Parameters
    ----------
    image_paths:
        Absolute paths to image files.
    labels_dir:
        Directory containing ``.txt`` label files (one per image, same stem).
    num_classes:
        If provided, class IDs >= *num_classes* are flagged as errors.

    Returns
    -------
    YoloValidationReport
    """
    labels_root = Path(labels_dir)
    report = YoloValidationReport()

    for img_path in image_paths:
        img_p = Path(img_path)
        label_p = labels_root / (img_p.stem + ".txt")

        if not label_p.exists():
            result = YoloFileResult(
                image_path=str(img_p),
                label_path=None,
                is_valid=False,
                label_missing=True,
            )
            result.warnings.append(f"Label file not found: {label_p}")
            report.results.append(result)
            logger.debug("Missing label: %s", label_p)
            continue

        result = _parse_label_file(label_p, num_classes, str(img_p))
        report.results.append(result)
        if not result.is_valid:
            logger.warning("Invalid YOLO label: %s — %s", label_p, result.errors)

    logger.info(
        "YOLO validation: %d total, %d valid, %d invalid, %d missing labels",
        len(report.results),
        report.valid_count,
        report.invalid_count,
        report.missing_label_count,
    )
    return report
