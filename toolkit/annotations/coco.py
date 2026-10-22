"""
toolkit/annotations/coco.py
────────────────────────────
COCO JSON annotation reader and validator.

Expected JSON structure::

    {
        "images":      [{"id": int, "file_name": str, "width": int, "height": int}, ...],
        "annotations": [{"id": int, "image_id": int, "category_id": int,
                          "bbox": [x, y, w, h], ...}, ...],
        "categories":  [{"id": int, "name": str}, ...]
    }

Conversion assumptions (documented):
- COCO bounding boxes are in absolute pixel coordinates [x_topleft, y_topleft, w, h].
- YOLO format uses normalised [x_center, y_center, w, h].
- Converting COCO → YOLO requires image dimensions from the "images" array.
- Category ID mapping must be supplied explicitly; IDs are NOT remapped silently.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CocoValidationReport:
    json_path: str = ""
    image_count: int = 0
    annotation_count: int = 0
    category_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "format": "coco",
            "json_path": self.json_path,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "category_count": self.category_count,
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_coco(json_path: str | Path) -> CocoValidationReport:
    """
    Parse and validate a COCO-format JSON annotation file.

    Parameters
    ----------
    json_path:
        Path to the ``annotations.json`` file.

    Returns
    -------
    CocoValidationReport
    """
    p = Path(json_path)
    report = CocoValidationReport(json_path=str(p))

    if not p.exists():
        report.errors.append(f"File not found: {p}")
        report.is_valid = False
        return report

    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        report.errors.append(f"JSON parse error: {exc}")
        report.is_valid = False
        return report
    except OSError as exc:
        report.errors.append(f"Cannot read file: {exc}")
        report.is_valid = False
        return report

    # ── Top-level keys ────────────────────────────────────────────────────────
    for key in ("images", "annotations", "categories"):
        if key not in data:
            report.errors.append(f"Missing top-level key: '{key}'")
            report.is_valid = False

    if not report.is_valid:
        return report

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    report.image_count = len(images)
    report.annotation_count = len(annotations)
    report.category_count = len(categories)

    # ── Build lookup sets ─────────────────────────────────────────────────────
    image_ids: dict[int, dict] = {}
    for img in images:
        if not isinstance(img, dict):
            report.errors.append("'images' array contains a non-object entry")
            report.is_valid = False
            continue
        iid = img.get("id")
        if iid is None:
            report.errors.append(f"Image entry missing 'id': {img}")
            report.is_valid = False
            continue
        if iid in image_ids:
            report.errors.append(f"Duplicate image id: {iid}")
            report.is_valid = False
        image_ids[iid] = img
        for field_name in ("file_name", "width", "height"):
            if field_name not in img:
                report.warnings.append(f"Image id={iid} missing field '{field_name}'")
        w = img.get("width", 0)
        h = img.get("height", 0)
        if not isinstance(w, (int, float)) or w <= 0:
            report.errors.append(f"Image id={iid} has invalid width: {w}")
            report.is_valid = False
        if not isinstance(h, (int, float)) or h <= 0:
            report.errors.append(f"Image id={iid} has invalid height: {h}")
            report.is_valid = False

    category_ids: set[int] = set()
    for cat in categories:
        if not isinstance(cat, dict):
            report.errors.append("'categories' array contains a non-object entry")
            report.is_valid = False
            continue
        cid = cat.get("id")
        if cid is None:
            report.errors.append(f"Category entry missing 'id': {cat}")
            report.is_valid = False
            continue
        category_ids.add(cid)
        if "name" not in cat:
            report.warnings.append(f"Category id={cid} missing 'name'")

    # ── Validate annotations ──────────────────────────────────────────────────
    annotation_ids: set[int] = set()
    for ann in annotations:
        if not isinstance(ann, dict):
            report.errors.append("'annotations' array contains a non-object entry")
            report.is_valid = False
            continue

        aid = ann.get("id")
        if aid in annotation_ids:
            report.errors.append(f"Duplicate annotation id: {aid}")
            report.is_valid = False
        if aid is not None:
            annotation_ids.add(aid)

        img_id = ann.get("image_id")
        if img_id not in image_ids:
            report.errors.append(
                f"Annotation id={aid} references nonexistent image_id={img_id}"
            )
            report.is_valid = False
            continue  # Cannot validate bbox without image dims

        cat_id = ann.get("category_id")
        if cat_id not in category_ids:
            report.errors.append(
                f"Annotation id={aid} references nonexistent category_id={cat_id}"
            )
            report.is_valid = False

        bbox = ann.get("bbox")
        if bbox is None:
            report.warnings.append(f"Annotation id={aid} has no 'bbox'")
            continue

        if not (isinstance(bbox, list) and len(bbox) == 4):
            report.errors.append(
                f"Annotation id={aid} 'bbox' must be a list of 4 numbers, got: {bbox}"
            )
            report.is_valid = False
            continue

        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            report.errors.append(
                f"Annotation id={aid} has non-positive bbox dimensions: w={bw}, h={bh}"
            )
            report.is_valid = False

        img_meta = image_ids.get(img_id, {})
        img_w = img_meta.get("width", 0)
        img_h = img_meta.get("height", 0)
        if img_w > 0 and img_h > 0:
            if x < 0 or y < 0 or x + bw > img_w + 1e-6 or y + bh > img_h + 1e-6:
                report.errors.append(
                    f"Annotation id={aid} bbox [{x},{y},{bw},{bh}] exceeds image "
                    f"dimensions {img_w}x{img_h}"
                )
                report.is_valid = False

    logger.info(
        "COCO validation: %d images, %d annotations, %d categories, valid=%s",
        report.image_count,
        report.annotation_count,
        report.category_count,
        report.is_valid,
    )
    return report
