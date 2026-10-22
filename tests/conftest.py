"""tests/conftest.py – Shared pytest fixtures for the Image Dataset Toolkit."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_rgb_image(path: Path, width: int = 100, height: int = 100,
                   color: tuple = (128, 64, 32)) -> Path:
    """Write a solid-colour JPEG and return the path."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def make_rgba_image(path: Path, width: int = 80, height: int = 80) -> Path:
    """Write a transparent PNG with an alpha channel and return the path."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, 0] = 200   # R
    img[:, :, 3] = 128   # alpha
    cv2.imwrite(str(path), img)
    return path


def make_zero_byte(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def make_corrupt_jpeg(path: Path) -> Path:
    """Write a file with a JPEG magic number but garbage content."""
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00GARBAGE_DATA")
    return path


def make_non_image(path: Path) -> Path:
    path.write_text("Not an image", encoding="utf-8")
    return path


# ── Base fixture ──────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data(tmp_path: Path) -> Path:
    """Return a temporary directory pre-populated with a standard test dataset."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # Three valid images of different sizes
    make_rgb_image(raw / "img_a.jpg", 320, 240, (200, 100, 50))
    make_rgb_image(raw / "img_b.jpg", 640, 480, (50, 200, 100))
    make_rgb_image(raw / "img_c.png", 128, 128, (100, 50, 200))

    # Corrupt files
    make_zero_byte(raw / "zero.jpg")
    make_corrupt_jpeg(raw / "corrupt.jpg")

    # Exact duplicate (byte copy)
    shutil.copy(raw / "img_a.jpg", raw / "img_a_dup.jpg")

    # Non-image file (should be skipped)
    make_non_image(raw / "readme.txt")

    return tmp_path


@pytest.fixture()
def labels_dir(tmp_data: Path) -> Path:
    """Create YOLO label files matching the valid images in tmp_data/raw."""
    raw = tmp_data / "raw"
    lbl = tmp_data / "labels"
    lbl.mkdir()

    # Valid labels
    (lbl / "img_a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (lbl / "img_b.txt").write_text("1 0.3 0.3 0.2 0.2\n2 0.7 0.7 0.1 0.1\n", encoding="utf-8")
    (lbl / "img_c.txt").write_text("", encoding="utf-8")   # empty = valid (no objects)
    (lbl / "img_a_dup.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    return lbl
