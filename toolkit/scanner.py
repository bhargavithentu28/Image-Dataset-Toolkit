"""
toolkit/scanner.py
──────────────────
Recursively scan an image directory, validate each file, and return a
structured inventory without modifying any source file.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ImageRecord:
    """All metadata captured for one file during a scan."""

    path: Path
    relative_path: str
    extension: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    format: str | None = None
    is_valid: bool = False
    error_message: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "sha256": self.sha256,
        }


@dataclass
class ScanResult:
    """Aggregated result of scanning an entire directory."""

    records: list[ImageRecord] = field(default_factory=list)
    total_files: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    skipped_extensions: int = 0

    @property
    def valid_records(self) -> list[ImageRecord]:
        return [r for r in self.records if r.is_valid]

    @property
    def invalid_records(self) -> list[ImageRecord]:
        return [r for r in self.records if not r.is_valid]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sha256_file(path: Path, chunk_size: int = 65_536) -> str:
    """Return the hex SHA-256 digest of *path* without loading it entirely."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_image(path: Path) -> tuple[bool, dict, str | None]:
    """
    Try to open and decode *path* with both Pillow and OpenCV.

    Returns
    -------
    (is_valid, metadata_dict, error_message)
    """
    meta: dict = {}
    # ── Zero-byte check ───────────────────────────────────────────────────────
    size = path.stat().st_size
    meta["size_bytes"] = size
    if size == 0:
        return False, meta, "Zero-byte file"

    # ── Pillow round-trip ─────────────────────────────────────────────────────
    try:
        with Image.open(path) as img:
            img.verify()          # catches truncated / corrupt headers
        with Image.open(path) as img:
            img.load()            # fully decode pixel data
            meta["format"] = img.format or path.suffix.lstrip(".").upper()
            meta["width"], meta["height"] = img.size
            meta["channels"] = len(img.getbands())
    except UnidentifiedImageError:
        return False, meta, "Unidentified image format (Pillow)"
    except Exception as exc:  # noqa: BLE001
        return False, meta, f"Pillow error: {exc}"

    # ── OpenCV cross-check ────────────────────────────────────────────────────
    try:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return False, meta, "OpenCV could not decode image"
        h, w = arr.shape[:2]
        ch = arr.shape[2] if arr.ndim == 3 else 1
        # Prefer OpenCV channel count (handles alpha correctly)
        meta["channels"] = ch
        meta["width"] = w
        meta["height"] = h
    except Exception as exc:  # noqa: BLE001
        return False, meta, f"OpenCV error: {exc}"

    return True, meta, None


# ── Public API ────────────────────────────────────────────────────────────────

def scan_dataset(
    input_dir: str | Path,
    supported_extensions: Sequence[str] | None = None,
    compute_hash: bool = True,
    exclude_dirs: Sequence[str] | None = None,
    progress_callback=None,
) -> ScanResult:
    """
    Scan *input_dir* recursively, validate each image, and return a
    :class:`ScanResult`.

    Parameters
    ----------
    input_dir:
        Root directory to scan.
    supported_extensions:
        Iterable of lower-case extensions including the dot (e.g. ``".jpg"``).
        Defaults to :data:`SUPPORTED_EXTENSIONS`.
    compute_hash:
        Whether to compute SHA-256 for each file.
    exclude_dirs:
        Directory names to skip (e.g. ``["processed", ".venv"]``).
    progress_callback:
        Optional callable ``(current_index, total)`` for UI progress reporting.

    Returns
    -------
    ScanResult
    """
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    exts: frozenset[str] = (
        frozenset(e.lower() for e in supported_extensions)
        if supported_extensions is not None
        else SUPPORTED_EXTENSIONS
    )
    exclude: frozenset[str] = frozenset(exclude_dirs or [])

    result = ScanResult()

    all_files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            if exclude and any(part in exclude for part in p.parts):
                continue
            all_files.append(p)

    total = len(all_files)
    logger.info("Scanning %s — %d candidate files found", root, total)

    for idx, path in enumerate(all_files):
        if progress_callback:
            progress_callback(idx, total)

        suffix = path.suffix.lower()
        size_bytes = 0
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            logger.warning("Cannot stat %s: %s", path, exc)
            rec = ImageRecord(
                path=path,
                relative_path=str(path.relative_to(root)),
                extension=suffix,
                size_bytes=0,
                is_valid=False,
                error_message=f"OS error reading file metadata: {exc}",
            )
            result.records.append(rec)
            result.total_files += 1
            result.invalid_count += 1
            continue

        if suffix not in exts:
            result.skipped_extensions += 1
            continue

        result.total_files += 1
        is_valid, meta, error = _validate_image(path)

        sha = None
        if compute_hash and size_bytes > 0:
            try:
                sha = _sha256_file(path)
            except OSError as exc:
                logger.warning("Cannot hash %s: %s", path, exc)

        rec = ImageRecord(
            path=path,
            relative_path=str(path.relative_to(root)),
            extension=suffix,
            size_bytes=meta.get("size_bytes", size_bytes),
            width=meta.get("width"),
            height=meta.get("height"),
            channels=meta.get("channels"),
            format=meta.get("format"),
            is_valid=is_valid,
            error_message=error,
            sha256=sha,
        )
        result.records.append(rec)

        if is_valid:
            result.valid_count += 1
            logger.debug("OK  %s (%dx%d)", rec.relative_path, rec.width, rec.height)
        else:
            result.invalid_count += 1
            logger.warning("BAD %s — %s", rec.relative_path, error)

    if progress_callback:
        progress_callback(total, total)

    logger.info(
        "Scan complete: %d total, %d valid, %d invalid, %d skipped (unsupported ext)",
        result.total_files,
        result.valid_count,
        result.invalid_count,
        result.skipped_extensions,
    )
    return result
