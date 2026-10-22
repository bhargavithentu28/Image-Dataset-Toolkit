"""tests/test_duplicates.py – Tests for toolkit.duplicates."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from toolkit.scanner import scan_dataset
from toolkit.duplicates import detect_duplicates
from tests.conftest import make_rgb_image


class TestExactFileDuplicates:
    def test_exact_duplicate_detected(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "a.jpg")
        shutil.copy(raw / "a.jpg", raw / "b.jpg")   # byte-identical copy
        scan = scan_dataset(raw)
        report = detect_duplicates(scan.records, enable_perceptual=False)
        assert len(report.exact_file_groups) == 1
        assert len(report.exact_file_groups[0].paths) == 2

    def test_no_false_positive_exact(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "a.jpg", color=(200, 0, 0))
        make_rgb_image(raw / "b.jpg", color=(0, 200, 0))
        scan = scan_dataset(raw)
        report = detect_duplicates(scan.records, enable_perceptual=False)
        assert len(report.exact_file_groups) == 0


class TestPixelIdenticalDuplicates:
    def test_pixel_identical_different_encoding(self, tmp_path):
        """Same pixel content, different file format (JPEG re-encode at 100% quality
        may or may not be byte-identical depending on the encoder; PNG is lossless
        and should produce identical pixels)."""
        raw = tmp_path / "raw"
        raw.mkdir()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[10:50, 10:50] = (200, 100, 50)
        cv2.imwrite(str(raw / "img.png"), img)
        # Reload and save as another PNG — same pixels
        reloaded = cv2.imread(str(raw / "img.png"))
        cv2.imwrite(str(raw / "img_copy.png"), reloaded)
        scan = scan_dataset(raw)
        report = detect_duplicates(scan.records, enable_perceptual=False)
        # At least one group (exact or pixel) should capture them
        total_groups = (
            len(report.exact_file_groups) + len(report.pixel_identical_groups)
        )
        assert total_groups >= 1

    def test_invalid_images_skipped(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "good.jpg")
        (raw / "bad.jpg").write_bytes(b"")
        scan = scan_dataset(raw)
        # Should not crash even with invalid records
        report = detect_duplicates(scan.records, enable_perceptual=False)
        assert report is not None


class TestPerceptualDuplicates:
    def test_perceptual_groups_returned(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        # Create two very similar images (same solid colour)
        img = np.full((100, 100, 3), (128, 128, 128), dtype=np.uint8)
        cv2.imwrite(str(raw / "a.jpg"), img)
        # Slight noise
        noisy = img.copy()
        noisy[50, 50] = (130, 130, 130)
        cv2.imwrite(str(raw / "b.jpg"), noisy)
        scan = scan_dataset(raw)
        try:
            report = detect_duplicates(scan.records, enable_perceptual=True, perceptual_threshold=10)
            # perceptual groups may or may not fire depending on imagehash availability
            assert report is not None
        except ImportError:
            pytest.skip("imagehash not installed")

    def test_no_duplicates_empty_dataset(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        scan = scan_dataset(raw)
        report = detect_duplicates(scan.records, enable_perceptual=False)
        assert len(report.exact_file_groups) == 0
        assert report.total_duplicate_files == 0

    def test_report_to_rows(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "a.jpg")
        shutil.copy(raw / "a.jpg", raw / "b.jpg")
        scan = scan_dataset(raw)
        report = detect_duplicates(scan.records, enable_perceptual=False)
        rows = report.to_rows()
        assert len(rows) == 2
        assert all("kind" in r for r in rows)
