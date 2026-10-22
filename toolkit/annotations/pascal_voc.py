"""
toolkit/annotations/pascal_voc.py
──────────────────────────────────
Pascal VOC XML annotation reader and validator.

Expected XML structure::

    <annotation>
        <filename>image.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
            <depth>3</depth>
        </size>
        <object>
            <name>cat</name>
            <bndbox>
                <xmin>10</xmin>
                <ymin>20</ymin>
                <xmax>100</xmax>
                <ymax>150</ymax>
            </bndbox>
        </object>
        ...
    </annotation>
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class VocObject:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocFileResult:
    xml_path: str
    image_filename: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    objects: list[VocObject] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class VocValidationReport:
    results: list[VocFileResult] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return sum(1 for r in self.results if r.is_valid)

    @property
    def invalid_count(self) -> int:
        return sum(1 for r in self.results if not r.is_valid)

    def to_dict(self) -> dict:
        return {
            "format": "pascal_voc",
            "total": len(self.results),
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "errors": [
                {"xml": r.xml_path, "errors": r.errors}
                for r in self.results if r.errors
            ],
            "warnings": [
                {"xml": r.xml_path, "warnings": r.warnings}
                for r in self.results if r.warnings
            ],
        }


def _parse_voc_xml(xml_path: Path) -> VocFileResult:
    result = VocFileResult(xml_path=str(xml_path))

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as exc:
        result.errors.append(f"Malformed XML: {exc}")
        result.is_valid = False
        return result
    except OSError as exc:
        result.errors.append(f"Cannot read file: {exc}")
        result.is_valid = False
        return result

    # ── Filename ──────────────────────────────────────────────────────────────
    fn_el = root.find("filename")
    if fn_el is None or not fn_el.text:
        result.warnings.append("Missing <filename> element")
    else:
        result.image_filename = fn_el.text.strip()

    # ── Image dimensions ──────────────────────────────────────────────────────
    size_el = root.find("size")
    if size_el is None:
        result.errors.append("Missing <size> element")
        result.is_valid = False
    else:
        try:
            result.image_width = int(size_el.findtext("width", "0"))
            result.image_height = int(size_el.findtext("height", "0"))
        except ValueError:
            result.errors.append("<size> contains non-integer width/height")
            result.is_valid = False

        if result.image_width is not None and result.image_width <= 0:
            result.errors.append(f"Invalid image width: {result.image_width}")
            result.is_valid = False
        if result.image_height is not None and result.image_height <= 0:
            result.errors.append(f"Invalid image height: {result.image_height}")
            result.is_valid = False

    # ── Objects ───────────────────────────────────────────────────────────────
    for obj_el in root.findall("object"):
        name_el = obj_el.find("name")
        if name_el is None or not (name_el.text or "").strip():
            result.errors.append("Object has empty or missing <name>")
            result.is_valid = False
            continue
        class_name = name_el.text.strip()

        bbox_el = obj_el.find("bndbox")
        if bbox_el is None:
            result.errors.append(f"Object '{class_name}' has no <bndbox>")
            result.is_valid = False
            continue

        try:
            xmin = float(bbox_el.findtext("xmin", "nan"))
            ymin = float(bbox_el.findtext("ymin", "nan"))
            xmax = float(bbox_el.findtext("xmax", "nan"))
            ymax = float(bbox_el.findtext("ymax", "nan"))
        except ValueError:
            result.errors.append(
                f"Object '{class_name}' has non-numeric bndbox values"
            )
            result.is_valid = False
            continue

        import math
        for v, name in ((xmin, "xmin"), (ymin, "ymin"), (xmax, "xmax"), (ymax, "ymax")):
            if math.isnan(v):
                result.errors.append(f"Object '{class_name}' missing bndbox field '{name}'")
                result.is_valid = False

        if xmax <= xmin:
            result.errors.append(
                f"Object '{class_name}': xmax ({xmax}) <= xmin ({xmin})"
            )
            result.is_valid = False

        if ymax <= ymin:
            result.errors.append(
                f"Object '{class_name}': ymax ({ymax}) <= ymin ({ymin})"
            )
            result.is_valid = False

        if result.image_width and result.image_height:
            if xmin < 0 or ymin < 0:
                result.errors.append(
                    f"Object '{class_name}': bbox has negative coordinates"
                )
                result.is_valid = False
            if xmax > result.image_width + 1e-6:
                result.errors.append(
                    f"Object '{class_name}': xmax {xmax} > image width {result.image_width}"
                )
                result.is_valid = False
            if ymax > result.image_height + 1e-6:
                result.errors.append(
                    f"Object '{class_name}': ymax {ymax} > image height {result.image_height}"
                )
                result.is_valid = False

        result.objects.append(
            VocObject(name=class_name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        )

    return result


def validate_pascal_voc(
    xml_paths: Sequence[str | Path],
) -> VocValidationReport:
    """
    Validate Pascal VOC XML annotation files.

    Parameters
    ----------
    xml_paths:
        Paths to ``.xml`` annotation files.

    Returns
    -------
    VocValidationReport
    """
    report = VocValidationReport()
    for p in xml_paths:
        result = _parse_voc_xml(Path(p))
        report.results.append(result)
        if not result.is_valid:
            logger.warning("Invalid VOC annotation: %s — %s", p, result.errors)

    logger.info(
        "Pascal VOC validation: %d total, %d valid, %d invalid",
        len(report.results),
        report.valid_count,
        report.invalid_count,
    )
    return report
