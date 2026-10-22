"""tests/test_splitter.py – Tests for toolkit.splitter."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolkit.scanner import scan_dataset
from toolkit.splitter import split_dataset, validate_ratios, save_split_manifest
from tests.conftest import make_rgb_image


def _make_records(tmp_path: Path, n: int):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for i in range(n):
        make_rgb_image(raw / f"img_{i:04d}.jpg", 100, 100)
    return scan_dataset(raw, compute_hash=False).valid_records


class TestValidateRatios:
    def test_valid_ratios(self):
        validate_ratios(0.7, 0.15, 0.15)   # should not raise

    def test_ratios_not_summing_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_ratios(0.6, 0.2, 0.1)

    def test_negative_ratio(self):
        with pytest.raises(ValueError, match=">= 0"):
            validate_ratios(0.8, 0.2, -0.0)   # -0.0 == 0.0 but still checks sign

    def test_zero_test_ratio_allowed(self):
        validate_ratios(0.8, 0.2, 0.0)   # should not raise


class TestSplitDataset:
    def test_basic_split_counts(self, tmp_path):
        records = _make_records(tmp_path, 20)
        result = split_dataset(records, 0.7, 0.15, 0.15, seed=42)
        total = sum(result.counts().values())
        assert total == 20

    def test_no_image_in_two_splits(self, tmp_path):
        records = _make_records(tmp_path, 30)
        result = split_dataset(records, 0.7, 0.15, 0.15, seed=42)
        train_set = set(result.train)
        val_set = set(result.val)
        test_set = set(result.test)
        assert train_set.isdisjoint(val_set), "Train and val overlap!"
        assert train_set.isdisjoint(test_set), "Train and test overlap!"
        assert val_set.isdisjoint(test_set), "Val and test overlap!"

    def test_reproducible_with_same_seed(self, tmp_path):
        records = _make_records(tmp_path, 20)
        r1 = split_dataset(records, seed=42)
        r2 = split_dataset(records, seed=42)
        assert r1.train == r2.train
        assert r1.val == r2.val
        assert r1.test == r2.test

    def test_different_seeds_give_different_splits(self, tmp_path):
        records = _make_records(tmp_path, 20)
        r1 = split_dataset(records, seed=0)
        r2 = split_dataset(records, seed=99)
        # Very unlikely to be identical
        assert r1.train != r2.train

    def test_empty_dataset(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        scan = scan_dataset(raw, compute_hash=False)
        result = split_dataset(scan.valid_records)
        assert result.counts() == {"train": 0, "val": 0, "test": 0}
        assert result.warnings

    def test_small_dataset_warning(self, tmp_path):
        records = _make_records(tmp_path, 2)
        result = split_dataset(records, 0.7, 0.15, 0.15, seed=42)
        # Should not crash; all images should still appear exactly once
        total = sum(result.counts().values())
        assert total == 2

    def test_duplicate_groups_kept_together(self, tmp_path):
        records = _make_records(tmp_path, 10)
        # Force images 0 and 1 into the same group
        paths = [r.relative_path for r in records]
        dup_groups = [[paths[0], paths[1]]]
        r1 = split_dataset(records, seed=42, duplicate_groups=dup_groups)
        # Both paths should be in the same split
        for split_paths in (r1.train, r1.val, r1.test):
            if paths[0] in split_paths:
                assert paths[1] in split_paths
                break
        else:
            pytest.fail("Neither path found in any split")

    def test_stratified_split_with_labels(self, tmp_path):
        records = _make_records(tmp_path, 15)
        # Assign 3 classes
        labels = {r.relative_path: f"class_{i % 3}" for i, r in enumerate(records)}
        result = split_dataset(records, 0.7, 0.15, 0.15, seed=42, labels=labels)
        total = sum(result.counts().values())
        assert total == 15

    def test_invalid_ratios_raise(self, tmp_path):
        records = _make_records(tmp_path, 10)
        with pytest.raises(ValueError):
            split_dataset(records, 0.5, 0.5, 0.5)   # sum = 1.5

    def test_save_manifest(self, tmp_path):
        records = _make_records(tmp_path, 10)
        result = split_dataset(records, seed=42)
        csv_path = save_split_manifest(result, tmp_path / "reports")
        assert csv_path.exists()
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "split" in df.columns
        assert "relative_path" in df.columns
        assert len(df) == 10
