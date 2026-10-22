"""tests/test_reporter.py – Tests for toolkit.reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from toolkit.scanner import scan_dataset
from toolkit.duplicates import detect_duplicates
from toolkit.dimensions import analyze_dimensions
from toolkit.splitter import split_dataset
from toolkit.reporter import generate_reports
from tests.conftest import make_rgb_image


def _make_scan(tmp_path: Path, n: int = 5):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for i in range(n):
        make_rgb_image(raw / f"img_{i}.jpg", 100 + i * 20, 100 + i * 20)
    return scan_dataset(raw, compute_hash=True)


class TestGenerateReports:
    def test_all_reports_written(self, tmp_path):
        scan = _make_scan(tmp_path)
        dup = detect_duplicates(scan.records, enable_perceptual=False)
        dims = analyze_dimensions(scan.records)
        split = split_dataset(scan.valid_records, seed=42)
        out = tmp_path / "reports"
        written = generate_reports(
            out,
            scan_result=scan,
            duplicate_report=dup,
            dimension_stats=dims,
            split_result=split,
        )
        expected = [
            "dataset_summary", "image_inventory", "invalid_images",
            "duplicate_groups", "dimension_statistics", "split_manifest",
            "dataset_report",
        ]
        for name in expected:
            assert name in written, f"Missing report: {name}"
            assert written[name].exists(), f"Report file not created: {written[name]}"

    def test_dataset_summary_json_valid(self, tmp_path):
        scan = _make_scan(tmp_path)
        out = tmp_path / "reports"
        written = generate_reports(out, scan_result=scan)
        data = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
        assert "scan" in data
        assert data["scan"]["valid_images"] == scan.valid_count

    def test_image_inventory_csv_columns(self, tmp_path):
        scan = _make_scan(tmp_path)
        out = tmp_path / "reports"
        generate_reports(out, scan_result=scan)
        df = pd.read_csv(out / "image_inventory.csv")
        for col in ("path", "relative_path", "width", "height", "is_valid"):
            assert col in df.columns

    def test_invalid_images_csv(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "good.jpg", 100, 100)
        (raw / "bad.jpg").write_bytes(b"")
        scan = scan_dataset(raw)
        out = tmp_path / "reports"
        generate_reports(out, scan_result=scan)
        df = pd.read_csv(out / "invalid_images.csv")
        assert len(df) == 1

    def test_duplicate_groups_csv(self, tmp_path):
        import shutil
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "a.jpg")
        shutil.copy(raw / "a.jpg", raw / "b.jpg")
        scan = scan_dataset(raw)
        dup = detect_duplicates(scan.records, enable_perceptual=False)
        out = tmp_path / "reports"
        generate_reports(out, duplicate_report=dup)
        df = pd.read_csv(out / "duplicate_groups.csv")
        assert len(df) == 2

    def test_split_manifest_csv(self, tmp_path):
        scan = _make_scan(tmp_path, 10)
        split = split_dataset(scan.valid_records, seed=42)
        out = tmp_path / "reports"
        generate_reports(out, split_result=split)
        df = pd.read_csv(out / "split_manifest.csv")
        assert "split" in df.columns
        assert len(df) == 10

    def test_markdown_report_written(self, tmp_path):
        scan = _make_scan(tmp_path)
        out = tmp_path / "reports"
        generate_reports(out, scan_result=scan)
        md = (out / "dataset_report.md").read_text(encoding="utf-8")
        assert "# Dataset Quality Report" in md
        assert "Scan Summary" in md

    def test_no_data_still_writes_summary(self, tmp_path):
        out = tmp_path / "reports"
        written = generate_reports(out)
        assert (out / "dataset_summary.json").exists()

    def test_annotation_report_included(self, tmp_path):
        from toolkit.annotations.yolo import YoloValidationReport
        scan = _make_scan(tmp_path, 3)
        ann_report = YoloValidationReport()
        out = tmp_path / "reports"
        generate_reports(out, scan_result=scan, annotation_reports={"yolo": ann_report})
        assert (out / "annotation_validation.json").exists()
        data = json.loads((out / "annotation_validation.json").read_text())
        assert "yolo" in data

    def test_output_dir_created_automatically(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "reports"
        assert not out.exists()
        generate_reports(out)
        assert out.exists()
