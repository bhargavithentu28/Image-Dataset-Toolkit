"""
toolkit/duplicates.py
─────────────────────
Three-tier duplicate detection:

1. SHA-256 file hash  → byte-identical files
2. Decoded-pixel MD5  → files with identical decoded pixels but different bytes
3. Perceptual hash    → visually similar images (optional, requires *imagehash*)

No files are deleted automatically.  The caller receives a report and must
take explicit action.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from toolkit.scanner import ImageRecord

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DuplicateGroup:
    """A cluster of files that are duplicates of each other."""

    kind: str          # "exact_file" | "pixel_identical" | "visually_similar"
    paths: list[Path]
    hash_value: str    # The shared hash (or representative hash)
    distance: int = 0  # Hamming distance for perceptual duplicates


@dataclass
class DuplicateReport:
    exact_file_groups: list[DuplicateGroup] = field(default_factory=list)
    pixel_identical_groups: list[DuplicateGroup] = field(default_factory=list)
    perceptual_groups: list[DuplicateGroup] = field(default_factory=list)

    @property
    def total_duplicate_files(self) -> int:
        count = 0
        for groups in (
            self.exact_file_groups,
            self.pixel_identical_groups,
            self.perceptual_groups,
        ):
            for g in groups:
                count += len(g.paths) - 1  # one is the "original"
        return count

    def to_rows(self) -> list[dict]:
        """Flatten all groups into a list of dicts for CSV export."""
        rows = []
        for group_list, kind in (
            (self.exact_file_groups, "exact_file"),
            (self.pixel_identical_groups, "pixel_identical"),
            (self.perceptual_groups, "visually_similar"),
        ):
            for g in group_list:
                for path in g.paths:
                    rows.append(
                        {
                            "kind": kind,
                            "group_hash": g.hash_value,
                            "hamming_distance": g.distance,
                            "path": str(path),
                        }
                    )
        return rows


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pixel_hash(path: Path) -> str | None:
    """
    Decode the image and return an MD5 of its raw pixel bytes.
    Returns *None* if the image cannot be decoded.
    """
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()  # noqa: S324


def _perceptual_hash(path: Path):
    """
    Compute an *imagehash* perceptual hash.  Returns *None* if the library is
    unavailable or the image cannot be opened.
    """
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Perceptual hash failed for %s: %s", path, exc)
        return None


def _group_by_hash(hash_map: dict[Path, str], kind: str) -> list[DuplicateGroup]:
    """Invert *hash_map* and return groups with more than one member."""
    buckets: dict[str, list[Path]] = {}
    for path, h in hash_map.items():
        buckets.setdefault(h, []).append(path)
    return [
        DuplicateGroup(kind=kind, paths=sorted(paths), hash_value=h)
        for h, paths in buckets.items()
        if len(paths) > 1
    ]


def _group_perceptual(
    phash_map: dict[Path, object],
    threshold: int,
) -> list[DuplicateGroup]:
    """Build groups of perceptually similar images using a greedy approach."""
    paths = list(phash_map.keys())
    visited: set[int] = set()
    groups: list[DuplicateGroup] = []

    for i in range(len(paths)):
        if i in visited:
            continue
        group_paths: list[Path] = [paths[i]]
        max_dist = 0
        for j in range(i + 1, len(paths)):
            if j in visited:
                continue
            try:
                dist = phash_map[paths[i]] - phash_map[paths[j]]
            except Exception:  # noqa: BLE001
                continue
            if dist <= threshold:
                group_paths.append(paths[j])
                visited.add(j)
                max_dist = max(max_dist, dist)
        if len(group_paths) > 1:
            visited.add(i)
            groups.append(
                DuplicateGroup(
                    kind="visually_similar",
                    paths=sorted(group_paths),
                    hash_value=str(phash_map[paths[i]]),
                    distance=max_dist,
                )
            )
    return groups


# ── Public API ────────────────────────────────────────────────────────────────

def detect_duplicates(
    records: Sequence[ImageRecord],
    enable_perceptual: bool = True,
    perceptual_threshold: int = 10,
) -> DuplicateReport:
    """
    Run all three duplicate-detection tiers on *records*.

    Parameters
    ----------
    records:
        Valid :class:`~toolkit.scanner.ImageRecord` objects (invalid images are
        skipped automatically).
    enable_perceptual:
        Whether to run perceptual hashing (slower; requires *imagehash*).
    perceptual_threshold:
        Maximum Hamming distance to consider two images visually similar.

    Returns
    -------
    DuplicateReport
    """
    valid = [r for r in records if r.is_valid]
    logger.info("Running duplicate detection on %d valid images", len(valid))

    report = DuplicateReport()

    # ── Tier 1: exact file hash ───────────────────────────────────────────────
    file_hash_map: dict[Path, str] = {}
    pixel_hash_needed: list[ImageRecord] = []

    for rec in valid:
        if rec.sha256:
            file_hash_map[rec.path] = rec.sha256
            pixel_hash_needed.append(rec)

    report.exact_file_groups = _group_by_hash(file_hash_map, "exact_file")
    exact_paths: set[Path] = {
        p for g in report.exact_file_groups for p in g.paths
    }
    logger.info("Exact-file duplicate groups: %d", len(report.exact_file_groups))

    # ── Tier 2: decoded-pixel hash ────────────────────────────────────────────
    pixel_hash_map: dict[Path, str] = {}
    for rec in pixel_hash_needed:
        if rec.path in exact_paths:
            continue
        h = _pixel_hash(rec.path)
        if h:
            pixel_hash_map[rec.path] = h

    report.pixel_identical_groups = _group_by_hash(pixel_hash_map, "pixel_identical")
    logger.info(
        "Pixel-identical duplicate groups: %d", len(report.pixel_identical_groups)
    )

    # ── Tier 3: perceptual hash ───────────────────────────────────────────────
    if enable_perceptual:
        phash_map: dict[Path, object] = {}
        for rec in valid:
            h = _perceptual_hash(rec.path)
            if h is not None:
                phash_map[rec.path] = h

        if phash_map:
            report.perceptual_groups = _group_perceptual(phash_map, perceptual_threshold)
            logger.info(
                "Perceptual duplicate groups (threshold=%d): %d",
                perceptual_threshold,
                len(report.perceptual_groups),
            )
        else:
            logger.warning(
                "Perceptual hashing skipped — imagehash not available or no valid images"
            )

    return report
