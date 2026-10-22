"""tests/test_annotations.py – Tests for YOLO, COCO, and Pascal VOC validators."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from toolkit.annotations.yolo import validate_yolo
from toolkit.annotations.coco import validate_coco
from toolkit.annotations.pascal_voc import validate_pascal_voc
from tests.conftest import make_rgb_image


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO
# ═══════════════════════════════════════════════════════════════════════════════

class TestYoloValidation:
    def _setup(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        lbl = tmp_path / "labels"
        lbl.mkdir()
        make_rgb_image(raw / "img.jpg")
        return raw, lbl

    def test_valid_label(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.valid_count == 1
        assert report.invalid_count == 0

    def test_empty_label_file_is_valid(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.valid_count == 1

    def test_missing_label_file(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        # No label file written
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.missing_label_count == 1

    def test_out_of_range_coordinate(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("0 1.5 0.5 0.4 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.invalid_count == 1
        assert any("x_center" in e for e in report.results[0].errors)

    def test_negative_class_id(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("-1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.invalid_count == 1

    def test_class_id_exceeds_num_classes(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("5 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl, num_classes=3)
        assert report.invalid_count == 1

    def test_zero_width_box(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("0 0.5 0.5 0.0 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.invalid_count == 1

    def test_wrong_column_count(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("0 0.5 0.5 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.invalid_count == 1

    def test_non_numeric_values(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text("cat 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.invalid_count == 1

    def test_multiple_objects_per_image(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        (lbl / "img.txt").write_text(
            "0 0.25 0.25 0.2 0.2\n1 0.75 0.75 0.2 0.2\n", encoding="utf-8"
        )
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.valid_count == 1
        assert len(report.results[0].annotations) == 2

    def test_box_exceeds_image_boundary(self, tmp_path):
        raw, lbl = self._setup(tmp_path)
        # cx=0.9, bw=0.4 → x_max = 0.9 + 0.2 = 1.1 > 1.0
        (lbl / "img.txt").write_text("0 0.9 0.5 0.4 0.4\n", encoding="utf-8")
        report = validate_yolo([raw / "img.jpg"], lbl)
        assert report.invalid_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# COCO
# ═══════════════════════════════════════════════════════════════════════════════

class TestCocoValidation:
    def _write_coco(self, path: Path, data: dict) -> Path:
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_coco(self, tmp_path):
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                              "bbox": [10, 10, 100, 80]}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        p = self._write_coco(tmp_path / "ann.json", data)
        report = validate_coco(p)
        assert report.is_valid
        assert report.image_count == 1
        assert report.annotation_count == 1

    def test_missing_images_key(self, tmp_path):
        data = {"annotations": [], "categories": []}
        p = self._write_coco(tmp_path / "ann.json", data)
        report = validate_coco(p)
        assert not report.is_valid
        assert any("images" in e for e in report.errors)

    def test_annotation_references_nonexistent_image(self, tmp_path):
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 999, "category_id": 0,
                              "bbox": [10, 10, 100, 80]}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        p = self._write_coco(tmp_path / "ann.json", data)
        report = validate_coco(p)
        assert not report.is_valid
        assert any("999" in e for e in report.errors)

    def test_invalid_bbox_exceeds_image(self, tmp_path):
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                              "bbox": [80, 80, 50, 50]}],  # goes outside
            "categories": [{"id": 0, "name": "cat"}],
        }
        p = self._write_coco(tmp_path / "ann.json", data)
        report = validate_coco(p)
        assert not report.is_valid

    def test_zero_dimensions_image(self, tmp_path):
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 0, "height": 0}],
            "annotations": [],
            "categories": [],
        }
        p = self._write_coco(tmp_path / "ann.json", data)
        report = validate_coco(p)
        assert not report.is_valid

    def test_missing_file(self, tmp_path):
        report = validate_coco(tmp_path / "nonexistent.json")
        assert not report.is_valid

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json", encoding="utf-8")
        report = validate_coco(p)
        assert not report.is_valid

    def test_duplicate_image_ids(self, tmp_path):
        data = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
                {"id": 1, "file_name": "b.jpg", "width": 100, "height": 100},
            ],
            "annotations": [],
            "categories": [],
        }
        p = self._write_coco(tmp_path / "ann.json", data)
        report = validate_coco(p)
        assert not report.is_valid


# ═══════════════════════════════════════════════════════════════════════════════
# Pascal VOC
# ═══════════════════════════════════════════════════════════════════════════════

class TestPascalVocValidation:
    def _write_xml(self, path: Path, xml_str: str) -> Path:
        path.write_text(xml_str, encoding="utf-8")
        return path

    def _valid_xml(self, name="cat", xmin=10, ymin=10, xmax=100, ymax=90):
        return f"""<annotation>
  <filename>img.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>{name}</name>
    <bndbox>
      <xmin>{xmin}</xmin><ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax><ymax>{ymax}</ymax>
    </bndbox>
  </object>
</annotation>"""

    def test_valid_annotation(self, tmp_path):
        p = self._write_xml(tmp_path / "img.xml", self._valid_xml())
        report = validate_pascal_voc([p])
        assert report.valid_count == 1

    def test_malformed_xml(self, tmp_path):
        p = tmp_path / "bad.xml"
        p.write_text("<annotation><unclosed>", encoding="utf-8")
        report = validate_pascal_voc([p])
        assert report.invalid_count == 1
        assert any("Malformed" in e for r in report.results for e in r.errors)

    def test_xmax_less_than_xmin(self, tmp_path):
        p = self._write_xml(tmp_path / "img.xml", self._valid_xml(xmin=200, xmax=50))
        report = validate_pascal_voc([p])
        assert report.invalid_count == 1

    def test_box_exceeds_image(self, tmp_path):
        p = self._write_xml(tmp_path / "img.xml", self._valid_xml(xmax=700))  # > 640
        report = validate_pascal_voc([p])
        assert report.invalid_count == 1

    def test_missing_size_element(self, tmp_path):
        xml = "<annotation><filename>img.jpg</filename><object><name>cat</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>"
        p = self._write_xml(tmp_path / "img.xml", xml)
        report = validate_pascal_voc([p])
        assert report.invalid_count == 1

    def test_empty_class_name(self, tmp_path):
        xml = "<annotation><filename>img.jpg</filename><size><width>640</width><height>480</height><depth>3</depth></size><object><name></name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>100</xmax><ymax>90</ymax></bndbox></object></annotation>"
        p = self._write_xml(tmp_path / "img.xml", xml)
        report = validate_pascal_voc([p])
        assert report.invalid_count == 1

    def test_missing_file(self, tmp_path):
        report = validate_pascal_voc([tmp_path / "nonexistent.xml"])
        assert report.invalid_count == 1

    def test_multiple_objects(self, tmp_path):
        xml = """<annotation>
  <filename>img.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object><name>cat</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>100</xmax><ymax>90</ymax></bndbox></object>
  <object><name>dog</name><bndbox><xmin>200</xmin><ymin>200</ymin><xmax>400</xmax><ymax>400</ymax></bndbox></object>
</annotation>"""
        p = self._write_xml(tmp_path / "img.xml", xml)
        report = validate_pascal_voc([p])
        assert report.valid_count == 1
        assert len(report.results[0].objects) == 2
