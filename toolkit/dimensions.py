"""
toolkit/dimensions.py
─────────────────────
Compute dimension statistics and identify outlier images.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from toolkit.scanner import ImageRecord

logger = logging.getLogger(__name__)


@dataclass
class DimensionStats:
    count: int = 0
    min_width: int | None = None
    max_width: int | None = None
    mean_width: float | None = None
    median_width: float | None = None
    min_height: int | None = None
    max_height: int | None = None
    mean_height: float | None = None
    median_height: float | None = None
    aspect_ratios: list[float] = field(default_factory=list)
    small_images: list[str] = field(default_factory=list)
    large_images: list[str] = field(default_factory=list)
    inconsistent_images: list[str] = field(default_factory=list)
    unique_sizes: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "mean_width": round(self.mean_width, 2) if self.mean_width else None,
            "median_width": self.median_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "mean_height": round(self.mean_height, 2) if self.mean_height else None,
            "median_height": self.median_height,
            "mean_aspect_ratio": (
                round(statistics.mean(self.aspect_ratios), 4)
                if self.aspect_ratios else None
            ),
            "unique_size_count": len(self.unique_sizes),
            "small_image_count": len(self.small_images),
            "large_image_count": len(self.large_images),
            "inconsistent_image_count": len(self.inconsistent_images),
            "small_images": self.small_images,
            "large_images": self.large_images,
            "inconsistent_images": self.inconsistent_images,
        }


def analyze_dimensions(
    records: Sequence[ImageRecord],
    small_threshold: int = 64,
    large_threshold: int = 8192,
    inconsistency_ratio: float = 0.1,
) -> DimensionStats:
    """
    Compute dimension statistics for *records*.

    Parameters
    ----------
    records:
        Valid image records produced by the scanner.
    small_threshold:
        Images with width **or** height below this value are flagged as small.
    large_threshold:
        Images with width **or** height above this value are flagged as large.
    inconsistency_ratio:
        Fraction deviation from median to flag as inconsistent.

    Returns
    -------
    DimensionStats
    """
    valid = [r for r in records if r.is_valid and r.width and r.height]

    stats = DimensionStats()
    if not valid:
        logger.warning("No valid images with dimension data — returning empty stats")
        return stats

    widths = [r.width for r in valid]  # type: ignore[arg-type]
    heights = [r.height for r in valid]  # type: ignore[arg-type]

    stats.count = len(valid)
    stats.min_width = min(widths)
    stats.max_width = max(widths)
    stats.mean_width = statistics.mean(widths)
    stats.median_width = statistics.median(widths)
    stats.min_height = min(heights)
    stats.max_height = max(heights)
    stats.mean_height = statistics.mean(heights)
    stats.median_height = statistics.median(heights)
    stats.aspect_ratios = [w / h for w, h in zip(widths, heights) if h > 0]
    stats.unique_sizes = sorted(set(zip(widths, heights)))

    med_w = stats.median_width
    med_h = stats.median_height

    for rec in valid:
        path_str = rec.relative_path
        w, h = rec.width, rec.height  # type: ignore[assignment]

        if w < small_threshold or h < small_threshold:
            stats.small_images.append(path_str)
        if w > large_threshold or h > large_threshold:
            stats.large_images.append(path_str)
        if (
            abs(w - med_w) / (med_w + 1e-9) > inconsistency_ratio
            or abs(h - med_h) / (med_h + 1e-9) > inconsistency_ratio
        ):
            stats.inconsistent_images.append(path_str)

    logger.info(
        "Dimension analysis: %d images, width %d–%d, height %d–%d",
        stats.count,
        stats.min_width,
        stats.max_width,
        stats.min_height,
        stats.max_height,
    )
    return stats
