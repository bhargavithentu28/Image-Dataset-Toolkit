"""tests/test_preprocessor.py – Tests for toolkit.preprocessor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from toolkit.scanner import scan_dataset
from toolkit.preprocessor import (
    preprocess_images,
    _letterbox,
    _stretch,
    _crop_center,
    LetterboxTransform,
)
from tests.conftest import make_rgb_image, make_rgba_image


class TestLetterbox:
    def test_output_size(self, tmp_path):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out, _ = _letterbox(img, 416, 416, (114, 114, 114),
                             cv2.INTER_CUBIC, cv2.INTER_AREA)
        assert out.shape == (416, 416, 3)

    def test_aspect_ratio_preserved(self, tmp_path):
        # 640x320 → letterboxed to 416x416; content should be 416x208
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        out, t = _letterbox(img, 416, 416, (114, 114, 114),
                             cv2.INTER_CUBIC, cv2.INTER_AREA)
        assert out.shape[:2] == (416, 416)
        # The unpadded region should be ~208 pixels tall
        unpadded_h = round(320 * t.scale)
        assert abs(unpadded_h - 208) <= 2

    def test_transform_bbox_valid(self):
        t = LetterboxTransform(
            scale=0.5, pad_top=10, pad_left=0,
            target_w=320, target_h=320, orig_w=640, orig_h=600
        )
        # Box that fits within the image
        ncx, ncy, nbw, nbh = t.transform_yolo_box(0.5, 0.5, 0.4, 0.4)
        assert 0.0 <= ncx <= 1.0
        assert 0.0 <= ncy <= 1.0

    def test_transform_bbox_out_of_bounds_raises(self):
        t = LetterboxTransform(
            scale=2.0, pad_top=0, pad_left=0,
            target_w=100, target_h=100, orig_w=100, orig_h=100
        )
        with pytest.raises(ValueError):
            t.transform_yolo_box(0.9, 0.9, 0.5, 0.5)


class TestStretch:
    def test_exact_size(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _stretch(img, 224, 224, cv2.INTER_CUBIC, cv2.INTER_AREA)
        assert out.shape == (224, 224, 3)


class TestCropCenter:
    def test_output_size(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _crop_center(img, 224, 224, cv2.INTER_CUBIC, cv2.INTER_AREA)
        assert out.shape == (224, 224, 3)


class TestPreprocessImages:
    def test_letterbox_produces_output_files(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg", 320, 240)
        scan = scan_dataset(raw, compute_hash=False)
        out = tmp_path / "processed"
        summary = preprocess_images(scan.valid_records, out, 416, 416, mode="letterbox")
        assert summary.success_count == 1
        assert summary.failure_count == 0
        assert (out / "img.jpg").exists()

    def test_stretch_mode(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg", 320, 240)
        scan = scan_dataset(raw, compute_hash=False)
        out = tmp_path / "processed"
        preprocess_images(scan.valid_records, out, 224, 224, mode="stretch")
        processed = cv2.imread(str(out / "img.jpg"))
        assert processed.shape == (224, 224, 3)

    def test_crop_center_mode(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg", 320, 240)
        scan = scan_dataset(raw, compute_hash=False)
        out = tmp_path / "processed"
        preprocess_images(scan.valid_records, out, 224, 224, mode="crop_center")
        processed = cv2.imread(str(out / "img.jpg"))
        assert processed.shape == (224, 224, 3)

    def test_invalid_mode_raises(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg")
        scan = scan_dataset(raw, compute_hash=False)
        with pytest.raises(ValueError, match="Unknown mode"):
            preprocess_images(scan.valid_records, tmp_path / "out", 224, 224, mode="invalid_mode")

    def test_source_files_unchanged(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        p = raw / "img.jpg"
        make_rgb_image(p)
        original = p.read_bytes()
        scan = scan_dataset(raw, compute_hash=False)
        preprocess_images(scan.valid_records, tmp_path / "out", 224, 224)
        assert p.read_bytes() == original

    def test_alpha_channel_preserved(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgba_image(raw / "rgba.png")
        scan = scan_dataset(raw, compute_hash=False)
        out = tmp_path / "processed"
        summary = preprocess_images(
            scan.valid_records, out, 64, 64, mode="letterbox", preserve_alpha=True
        )
        assert summary.success_count == 1

    def test_result_dimensions_recorded(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        make_rgb_image(raw / "img.jpg", 320, 240)
        scan = scan_dataset(raw, compute_hash=False)
        out = tmp_path / "processed"
        summary = preprocess_images(scan.valid_records, out, 416, 416)
        res = summary.results[0]
        assert res.orig_width == 320
        assert res.orig_height == 240
        assert res.out_width == 416
        assert res.out_height == 416
