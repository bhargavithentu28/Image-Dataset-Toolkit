"""tests/test_scanner.py – Tests for toolkit.scanner."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from toolkit.scanner import scan_dataset, SUPPORTED_EXTENSIONS
from tests.conftest import make_rgb_image, make_zero_byte, make_corrupt_jpeg, make_non_image


class TestScanDataset:
    """Core scanning behaviour."""

    def test_valid_images_detected(self, tmp_data):
        result = scan_dataset(tmp_data / "raw")
        # img_a.jpg, img_b.jpg, img_c.png, img_a_dup.jpg = 4 valid
        assert result.valid_count == 4

    def test_invalid_images_detected(self, tmp_data):
        result = scan_dataset(tmp_data / "raw")
        # zero.jpg + corrupt.jpg = 2 invalid
        assert result.invalid_count == 2

    def test_non_image_skipped(self, tmp_data):
        result = scan_dataset(tmp_data / "raw")
        # readme.txt has unsupported extension → skipped
        assert result.skipped_extensions >= 1

    def test_zero_byte_invalid(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_zero_byte(raw / "empty.jpg")
        result = scan_dataset(raw)
        assert result.invalid_count == 1
        assert "Zero-byte" in result.invalid_records[0].error_message

    def test_corrupt_jpeg_invalid(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_corrupt_jpeg(raw / "bad.jpg")
        result = scan_dataset(raw)
        assert result.invalid_count == 1

    def test_valid_image_metadata(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "test.jpg", width=320, height=240)
        result = scan_dataset(raw)
        assert result.valid_count == 1
        rec = result.valid_records[0]
        assert rec.width == 320
        assert rec.height == 240
        assert rec.channels == 3
        assert rec.extension == ".jpg"
        assert rec.is_valid

    def test_empty_directory(self, tmp_path):
        raw = tmp_path / "empty"
        raw.mkdir()
        result = scan_dataset(raw)
        assert result.total_files == 0
        assert result.valid_count == 0

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_dataset(tmp_path / "does_not_exist")

    def test_path_is_file_raises(self, tmp_path):
        f = tmp_path / "file.jpg"
        make_rgb_image(f)
        with pytest.raises(NotADirectoryError):
            scan_dataset(f)

    def test_sha256_computed(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg")
        result = scan_dataset(raw, compute_hash=True)
        assert result.valid_records[0].sha256 is not None
        assert len(result.valid_records[0].sha256) == 64

    def test_no_hash_when_disabled(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg")
        result = scan_dataset(raw, compute_hash=False)
        assert result.valid_records[0].sha256 is None

    def test_recursive_scan(self, tmp_path):
        raw = tmp_path / "raw"
        sub = raw / "subdir"
        sub.mkdir(parents=True)
        make_rgb_image(raw / "top.jpg")
        make_rgb_image(sub / "nested.jpg")
        result = scan_dataset(raw)
        assert result.valid_count == 2

    def test_mixed_extensions(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for ext in (".jpg", ".png", ".bmp"):
            make_rgb_image(raw / f"img{ext}")
        result = scan_dataset(raw)
        assert result.valid_count == 3

    def test_rgba_png(self, tmp_path):
        from tests.conftest import make_rgba_image
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgba_image(raw / "rgba.png")
        result = scan_dataset(raw)
        assert result.valid_count == 1
        assert result.valid_records[0].channels == 4

    def test_source_files_not_modified(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        p = raw / "img.jpg"
        make_rgb_image(p)
        original_bytes = p.read_bytes()
        scan_dataset(raw)
        assert p.read_bytes() == original_bytes
