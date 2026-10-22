"""
toolkit/splitter.py
───────────────────
Reproducible dataset splitting with optional class-aware stratification.

Every eligible image appears in exactly one split.
Duplicate groups are kept together to prevent label leakage.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from toolkit.scanner import ImageRecord

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def to_rows(self) -> list[dict]:
        rows = []
        for split, paths in (("train", self.train), ("val", self.val), ("test", self.test)):
            for p in paths:
                rows.append({"split": split, "relative_path": p})
        return rows


# ── Validation ────────────────────────────────────────────────────────────────

def validate_ratios(train: float, val: float, test: float) -> None:
    """Raise *ValueError* if ratios don't sum to ~1 or any is negative."""
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train}+{val}+{test}={total:.4f}"
        )
    for name, ratio in (("train", train), ("val", val), ("test", test)):
        # Use math.copysign to catch -0.0, which compares equal to 0.0
        if ratio < 0 or math.copysign(1.0, ratio) < 0:
            raise ValueError(f"Ratio '{name}' must be >= 0, got {ratio}")


# ── Core splitting logic ──────────────────────────────────────────────────────

def _shuffle_and_split(
    items: list,
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list, list, list]:
    rng.shuffle(items)
    n = len(items)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)
    return items[:n_train], items[n_train: n_train + n_val], items[n_train + n_val:]


def split_dataset(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    labels: dict[str, str] | None = None,
    duplicate_groups: list[list[str]] | None = None,
) -> SplitResult:
    """
    Split *records* into train / val / test sets.

    Parameters
    ----------
    records:
        Valid image records.
    train_ratio, val_ratio, test_ratio:
        Must sum to 1.0.
    seed:
        Random seed for reproducibility.
    labels:
        Optional mapping of ``relative_path → class_name`` for stratification.
    duplicate_groups:
        List of groups that must stay together to prevent leakage.

    Returns
    -------
    SplitResult
    """
    validate_ratios(train_ratio, val_ratio, test_ratio)

    valid = [r for r in records if r.is_valid]
    if not valid:
        return SplitResult(warnings=["No valid images to split"])

    rng = random.Random(seed)
    result = SplitResult()

    path_to_rec: dict[str, ImageRecord] = {r.relative_path: r for r in valid}
    all_paths = list(path_to_rec.keys())

    assigned_group: dict[str, int] = {}
    groups: list[list[str]] = []

    if duplicate_groups:
        for cluster in duplicate_groups:
            gid = len(groups)
            groups.append([p for p in cluster if p in path_to_rec])
            for p in groups[-1]:
                assigned_group[p] = gid

    for p in all_paths:
        if p not in assigned_group:
            gid = len(groups)
            groups.append([p])
            assigned_group[p] = gid

    if labels:
        group_class: dict[int, str] = {}
        for gid, members in enumerate(groups):
            class_votes: dict[str, int] = defaultdict(int)
            for p in members:
                cls = labels.get(p, "unknown")
                class_votes[cls] += 1
            group_class[gid] = max(class_votes, key=class_votes.__getitem__)

        class_to_groups: dict[str, list[int]] = defaultdict(list)
        for gid, cls in group_class.items():
            class_to_groups[cls].append(gid)

        train_gids: list[int] = []
        val_gids: list[int] = []
        test_gids: list[int] = []

        for cls, gids in class_to_groups.items():
            if len(gids) < 3:
                result.warnings.append(
                    f"Class '{cls}' has only {len(gids)} group(s); "
                    "stratification may not be possible — images placed in train."
                )
                train_gids.extend(gids)
                continue
            tr, va, te = _shuffle_and_split(gids, train_ratio, val_ratio, rng)
            train_gids.extend(tr)
            val_gids.extend(va)
            test_gids.extend(te)
    else:
        all_gids = list(range(len(groups)))
        train_gids, val_gids, test_gids = _shuffle_and_split(
            all_gids, train_ratio, val_ratio, rng
        )

    def _expand(gids: list[int]) -> list[str]:
        paths: list[str] = []
        for gid in gids:
            paths.extend(groups[gid])
        return sorted(paths)

    result.train = _expand(train_gids)
    result.val = _expand(val_gids)
    result.test = _expand(test_gids)

    combined = set(result.train) | set(result.val) | set(result.test)
    if len(combined) != len(all_paths):
        result.warnings.append(
            "WARNING: some images were lost during splitting. "
            "Check duplicate_groups entries."
        )

    logger.info(
        "Split: train=%d  val=%d  test=%d  (seed=%d)",
        len(result.train), len(result.val), len(result.test), seed,
    )
    return result


def save_split_manifest(result: SplitResult, output_dir: str | Path) -> Path:
    """Write split_manifest.csv to *output_dir* and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "split_manifest.csv"
    df = pd.DataFrame(result.to_rows())
    df.to_csv(csv_path, index=False)
    logger.info("Split manifest saved to %s", csv_path)
    return csv_path
