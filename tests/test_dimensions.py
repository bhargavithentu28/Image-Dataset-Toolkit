"""tests/test_dimensions.py – Tests for toolkit.dimensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolkit.scanner import scan_dataset
from toolkit.dimensions import analyze_dimensions
from tests.conftest import make_rgb_image


class TestAnalyzeDimensions:
    def test_basic_stats(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "small.jpg", 100, 100)
        make_rgb_image(raw / "large.jpg", 640, 480)
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records)
        assert stats.count == 2
        assert stats.min_width == 100
        assert stats.max_width == 640
        assert stats.min_height == 100
        assert stats.max_height == 480

    def test_mean_median_computed(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for w, h in [(100, 100), (200, 200), (300, 300)]:
            make_rgb_image(raw / f"{w}x{h}.jpg", w, h)
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records)
        assert stats.mean_width == pytest.approx(200.0)
        assert stats.median_width == 200.0

    def test_small_images_flagged(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "tiny.jpg", 32, 32)
        make_rgb_image(raw / "normal.jpg", 640, 480)
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records, small_threshold=64)
        assert len(stats.small_images) == 1
        assert "tiny" in stats.small_images[0]

    def test_large_images_flagged(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "huge.jpg", 9000, 9000)
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records, large_threshold=8192)
        assert len(stats.large_images) == 1

    def test_empty_dataset(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records)
        assert stats.count == 0
        assert stats.min_width is None

    def test_invalid_images_excluded(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "good.jpg", 200, 200)
        (raw / "bad.jpg").write_bytes(b"")
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records)
        assert stats.count == 1

    def test_aspect_ratios_computed(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "wide.jpg", 200, 100)   # ratio = 2.0
        make_rgb_image(raw / "tall.jpg", 100, 200)   # ratio = 0.5
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records)
        assert len(stats.aspect_ratios) == 2
        assert 2.0 in stats.aspect_ratios
        assert 0.5 in stats.aspect_ratios

    def test_to_dict_keys(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg", 100, 100)
        scan = scan_dataset(raw, compute_hash=False)
        stats = analyze_dimensions(scan.records)
        d = stats.to_dict()
        for key in ("count", "min_width", "max_width", "mean_width", "median_width",
                    "min_height", "max_height", "mean_height", "median_height"):
            assert key in d
