"""
toolkit/reporter.py
───────────────────
Generate all dataset quality reports from scan / analysis results.

Outputs:
  dataset_summary.json
  image_inventory.csv
  duplicate_groups.csv
  invalid_images.csv
  dimension_statistics.json
  split_manifest.csv
  annotation_validation.json
  dataset_report.md    (human-readable Markdown)
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to *path* atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=path.parent, suffix=".tmp"))
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame CSV to *path* atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=path.parent, suffix=".tmp"))
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def generate_reports(
    output_dir: str | Path,
    scan_result=None,
    duplicate_report=None,
    dimension_stats=None,
    split_result=None,
    annotation_reports: dict | None = None,
) -> dict[str, Path]:
    """
    Write all report files to *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory where reports are written.
    scan_result:
        :class:`~toolkit.scanner.ScanResult` (optional).
    duplicate_report:
        :class:`~toolkit.duplicates.DuplicateReport` (optional).
    dimension_stats:
        :class:`~toolkit.dimensions.DimensionStats` (optional).
    split_result:
        :class:`~toolkit.splitter.SplitResult` (optional).
    annotation_reports:
        Dict of ``{format_name: report_object}`` (optional).

    Returns
    -------
    dict mapping report name → Path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    written: dict[str, Path] = {}

    # ── dataset_summary.json ──────────────────────────────────────────────────
    summary: dict[str, Any] = {"generated_at": now}
    if scan_result is not None:
        summary["scan"] = {
            "total_files": scan_result.total_files,
            "valid_images": scan_result.valid_count,
            "invalid_images": scan_result.invalid_count,
            "skipped_extensions": scan_result.skipped_extensions,
        }
    if duplicate_report is not None:
        summary["duplicates"] = {
            "exact_file_groups": len(duplicate_report.exact_file_groups),
            "pixel_identical_groups": len(duplicate_report.pixel_identical_groups),
            "perceptual_groups": len(duplicate_report.perceptual_groups),
            "total_duplicate_files": duplicate_report.total_duplicate_files,
        }
    if dimension_stats is not None:
        summary["dimensions"] = dimension_stats.to_dict()
    if split_result is not None:
        summary["split"] = split_result.counts()
        if split_result.warnings:
            summary["split"]["warnings"] = split_result.warnings
    if annotation_reports:
        summary["annotations"] = {
            fmt: (rep.to_dict() if hasattr(rep, "to_dict") else str(rep))
            for fmt, rep in annotation_reports.items()
        }

    p = out / "dataset_summary.json"
    _atomic_write_json(p, summary)
    written["dataset_summary"] = p
    logger.info("Wrote %s", p)

    # ── image_inventory.csv ───────────────────────────────────────────────────
    if scan_result is not None:
        rows = [r.to_dict() for r in scan_result.records]
        p = out / "image_inventory.csv"
        _atomic_write_csv(p, pd.DataFrame(rows))
        written["image_inventory"] = p
        logger.info("Wrote %s", p)

        # ── invalid_images.csv ────────────────────────────────────────────────
        invalid_rows = [r.to_dict() for r in scan_result.invalid_records]
        p = out / "invalid_images.csv"
        _atomic_write_csv(p, pd.DataFrame(invalid_rows))
        written["invalid_images"] = p
        logger.info("Wrote %s", p)

    # ── duplicate_groups.csv ──────────────────────────────────────────────────
    if duplicate_report is not None:
        rows = duplicate_report.to_rows()
        p = out / "duplicate_groups.csv"
        _atomic_write_csv(p, pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["kind", "group_hash", "hamming_distance", "path"]
        ))
        written["duplicate_groups"] = p
        logger.info("Wrote %s", p)

    # ── dimension_statistics.json ─────────────────────────────────────────────
    if dimension_stats is not None:
        p = out / "dimension_statistics.json"
        _atomic_write_json(p, dimension_stats.to_dict())
        written["dimension_statistics"] = p
        logger.info("Wrote %s", p)

    # ── split_manifest.csv ────────────────────────────────────────────────────
    if split_result is not None:
        rows = split_result.to_rows()
        p = out / "split_manifest.csv"
        _atomic_write_csv(p, pd.DataFrame(rows))
        written["split_manifest"] = p
        logger.info("Wrote %s", p)

    # ── annotation_validation.json ────────────────────────────────────────────
    if annotation_reports:
        p = out / "annotation_validation.json"
        ann_data = {
            fmt: (rep.to_dict() if hasattr(rep, "to_dict") else str(rep))
            for fmt, rep in annotation_reports.items()
        }
        _atomic_write_json(p, ann_data)
        written["annotation_validation"] = p
        logger.info("Wrote %s", p)

    # ── dataset_report.md ─────────────────────────────────────────────────────
    md = _build_markdown(summary, written)
    p = out / "dataset_report.md"
    p.write_text(md, encoding="utf-8")
    written["dataset_report"] = p
    logger.info("Wrote %s", p)

    return written


def _build_markdown(summary: dict, written: dict[str, Path]) -> str:
    lines = [
        "# Dataset Quality Report",
        "",
        f"_Generated: {summary.get('generated_at', 'n/a')}_",
        "",
        "---",
        "",
    ]

    scan = summary.get("scan")
    if scan:
        lines += [
            "## Scan Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total files scanned | {scan['total_files']} |",
            f"| Valid images | {scan['valid_images']} |",
            f"| Invalid images | {scan['invalid_images']} |",
            f"| Skipped (unsupported ext) | {scan['skipped_extensions']} |",
            "",
        ]

    dup = summary.get("duplicates")
    if dup:
        lines += [
            "## Duplicate Detection",
            "",
            f"| Type | Groups |",
            f"|------|--------|",
            f"| Exact-file duplicates | {dup['exact_file_groups']} |",
            f"| Pixel-identical duplicates | {dup['pixel_identical_groups']} |",
            f"| Visually similar (perceptual) | {dup['perceptual_groups']} |",
            f"| Total duplicate files | {dup['total_duplicate_files']} |",
            "",
            "> **No files have been deleted.** Review `duplicate_groups.csv` and take",
            "> explicit action if removal is desired.",
            "",
        ]

    dims = summary.get("dimensions")
    if dims:
        lines += [
            "## Dimension Statistics",
            "",
            f"| Metric | Width | Height |",
            f"|--------|-------|--------|",
            f"| Min | {dims.get('min_width')} | {dims.get('min_height')} |",
            f"| Max | {dims.get('max_width')} | {dims.get('max_height')} |",
            f"| Mean | {dims.get('mean_width')} | {dims.get('mean_height')} |",
            f"| Median | {dims.get('median_width')} | {dims.get('median_height')} |",
            "",
            f"- Unique sizes: {dims.get('unique_size_count')}",
            f"- Small images (flagged): {dims.get('small_image_count')}",
            f"- Large images (flagged): {dims.get('large_image_count')}",
            f"- Inconsistent images (flagged): {dims.get('inconsistent_image_count')}",
            "",
        ]

    split = summary.get("split")
    if split:
        lines += [
            "## Dataset Split",
            "",
            f"| Split | Count |",
            f"|-------|-------|",
            f"| Train | {split.get('train', 0)} |",
            f"| Val   | {split.get('val', 0)} |",
            f"| Test  | {split.get('test', 0)} |",
            "",
        ]
        for w in split.get("warnings", []):
            lines.append(f"> ⚠️ {w}")
        lines.append("")

    ann = summary.get("annotations")
    if ann:
        lines += ["## Annotation Validation", ""]
        for fmt, rep_data in ann.items():
            if isinstance(rep_data, dict):
                lines.append(f"### {fmt.upper()}")
                lines.append("")
                for k, v in rep_data.items():
                    if not isinstance(v, (list, dict)):
                        lines.append(f"- **{k}**: {v}")
                lines.append("")

    lines += [
        "## Generated Files",
        "",
    ]
    for name, path in written.items():
        lines.append(f"- `{path.name}`")
    lines.append("")

    return "\n".join(lines)
