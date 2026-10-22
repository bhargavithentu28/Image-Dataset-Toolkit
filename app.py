"""
app.py – CLI entry point for the Image Dataset Preparation and Annotation Toolkit.

Usage:
    python app.py --help
    python app.py scan        --input data/raw --output reports
    python app.py duplicates  --input data/raw --output reports
    python app.py dimensions  --input data/raw --output reports
    python app.py preprocess  --input data/raw --output data/processed
    python app.py split       --input data/raw --output reports
    python app.py validate-annotations --input data/raw --labels data/labels --format yolo
    python app.py report      --input data/raw --output reports
    python app.py demo        --output demo_output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from toolkit.config import load_config, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Image Dataset Preparation and Annotation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to a custom config.yaml (default: config.yaml alongside app.py)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── scan ──────────────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Scan and validate an image directory")
    p_scan.add_argument("--input", required=True, metavar="DIR", help="Input image directory")
    p_scan.add_argument("--output", required=True, metavar="DIR", help="Reports output directory")
    p_scan.add_argument("--no-hash", action="store_true", help="Skip SHA-256 hashing")

    # ── duplicates ────────────────────────────────────────────────────────────
    p_dup = sub.add_parser("duplicates", help="Detect duplicate images")
    p_dup.add_argument("--input", required=True, metavar="DIR")
    p_dup.add_argument("--output", required=True, metavar="DIR")
    p_dup.add_argument("--no-perceptual", action="store_true",
                       help="Disable perceptual hashing")
    p_dup.add_argument("--threshold", type=int, default=10,
                       help="Perceptual hash Hamming distance threshold (default: 10)")

    # ── dimensions ────────────────────────────────────────────────────────────
    p_dim = sub.add_parser("dimensions", help="Analyse image dimensions")
    p_dim.add_argument("--input", required=True, metavar="DIR")
    p_dim.add_argument("--output", required=True, metavar="DIR")

    # ── preprocess ────────────────────────────────────────────────────────────
    p_pre = sub.add_parser("preprocess", help="Resize and preprocess images")
    p_pre.add_argument("--input", required=True, metavar="DIR")
    p_pre.add_argument("--output", required=True, metavar="DIR")
    p_pre.add_argument("--width", type=int, default=640)
    p_pre.add_argument("--height", type=int, default=640)
    p_pre.add_argument("--mode", choices=["letterbox", "stretch", "crop_center"],
                       default="letterbox")

    # ── split ─────────────────────────────────────────────────────────────────
    p_split = sub.add_parser("split", help="Create train/val/test splits")
    p_split.add_argument("--input", required=True, metavar="DIR")
    p_split.add_argument("--output", required=True, metavar="DIR")
    p_split.add_argument("--train", type=float, default=0.70)
    p_split.add_argument("--val", type=float, default=0.15)
    p_split.add_argument("--test", type=float, default=0.15)
    p_split.add_argument("--seed", type=int, default=42)

    # ── validate-annotations ──────────────────────────────────────────────────
    p_ann = sub.add_parser("validate-annotations", help="Validate annotation files")
    p_ann.add_argument("--input", required=True, metavar="DIR",
                       help="Image directory")
    p_ann.add_argument("--output", required=True, metavar="DIR")
    p_ann.add_argument("--format", dest="ann_format",
                       choices=["yolo", "coco", "pascal_voc"], default="yolo")
    p_ann.add_argument("--labels", metavar="DIR",
                       help="YOLO/VOC label directory")
    p_ann.add_argument("--coco-json", metavar="FILE",
                       help="COCO annotations JSON file")
    p_ann.add_argument("--num-classes", type=int, default=None,
                       help="Expected number of YOLO classes for validation")

    # ── report (full pipeline) ────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Run full pipeline and generate all reports")
    p_rep.add_argument("--input", required=True, metavar="DIR")
    p_rep.add_argument("--output", required=True, metavar="DIR")
    p_rep.add_argument("--labels", metavar="DIR", default=None)
    p_rep.add_argument("--ann-format", choices=["yolo", "coco", "pascal_voc"],
                       default=None)

    # ── demo ──────────────────────────────────────────────────────────────────
    p_demo = sub.add_parser("demo", help="Generate sample data and run the full pipeline")
    p_demo.add_argument("--output", required=True, metavar="DIR")

    return parser


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_scan(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.reporter import generate_reports

    result = scan_dataset(
        input_dir=args.input,
        compute_hash=not args.no_hash,
    )
    generate_reports(output_dir=args.output, scan_result=result)
    print(
        f"Scan complete: {result.valid_count} valid, "
        f"{result.invalid_count} invalid, "
        f"{result.total_files} total."
    )


def cmd_duplicates(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.duplicates import detect_duplicates
    from toolkit.reporter import generate_reports

    scan = scan_dataset(args.input)
    dup_report = detect_duplicates(
        scan.records,
        enable_perceptual=not args.no_perceptual,
        perceptual_threshold=args.threshold,
    )
    generate_reports(output_dir=args.output, duplicate_report=dup_report)
    print(
        f"Duplicates: {len(dup_report.exact_file_groups)} exact-file groups, "
        f"{len(dup_report.pixel_identical_groups)} pixel-identical groups, "
        f"{len(dup_report.perceptual_groups)} perceptual groups."
    )


def cmd_dimensions(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.dimensions import analyze_dimensions
    from toolkit.reporter import generate_reports

    scan = scan_dataset(args.input)
    stats = analyze_dimensions(scan.records)
    generate_reports(output_dir=args.output, dimension_stats=stats)
    print(
        f"Dimensions: {stats.count} images, "
        f"width {stats.min_width}–{stats.max_width}, "
        f"height {stats.min_height}–{stats.max_height}"
    )


def cmd_preprocess(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.preprocessor import preprocess_images

    scan = scan_dataset(args.input, compute_hash=False)
    summary = preprocess_images(
        records=scan.valid_records,
        output_dir=args.output,
        target_width=args.width,
        target_height=args.height,
        mode=args.mode,
    )
    print(
        f"Preprocessing: {summary.success_count} success, "
        f"{summary.failure_count} failed → {args.output}"
    )


def cmd_split(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.splitter import split_dataset, save_split_manifest

    scan = scan_dataset(args.input, compute_hash=False)
    result = split_dataset(
        records=scan.valid_records,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    save_split_manifest(result, args.output)
    counts = result.counts()
    print(
        f"Split: train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )
    for w in result.warnings:
        print(f"  Warning: {w}")


def cmd_validate_annotations(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.annotations import validate_yolo, validate_coco, validate_pascal_voc
    from toolkit.reporter import generate_reports

    scan = scan_dataset(args.input, compute_hash=False)
    ann_format = args.ann_format
    ann_reports = {}

    if ann_format == "yolo":
        if not args.labels:
            print("ERROR: --labels is required for YOLO validation", file=sys.stderr)
            sys.exit(1)
        report = validate_yolo(
            image_paths=[r.path for r in scan.valid_records],
            labels_dir=args.labels,
            num_classes=args.num_classes,
        )
        ann_reports["yolo"] = report
        print(
            f"YOLO: {report.valid_count} valid, {report.invalid_count} invalid, "
            f"{report.missing_label_count} missing labels"
        )

    elif ann_format == "coco":
        if not args.coco_json:
            print("ERROR: --coco-json is required for COCO validation", file=sys.stderr)
            sys.exit(1)
        report = validate_coco(args.coco_json)
        ann_reports["coco"] = report
        print(
            f"COCO: {report.image_count} images, {report.annotation_count} annotations, "
            f"valid={report.is_valid}"
        )

    elif ann_format == "pascal_voc":
        if not args.labels:
            print("ERROR: --labels is required for Pascal VOC validation", file=sys.stderr)
            sys.exit(1)
        xml_paths = list(Path(args.labels).glob("*.xml"))
        report = validate_pascal_voc(xml_paths)
        ann_reports["pascal_voc"] = report
        print(f"Pascal VOC: {report.valid_count} valid, {report.invalid_count} invalid")

    generate_reports(output_dir=args.output, annotation_reports=ann_reports)


def cmd_report(args, cfg):
    from toolkit.scanner import scan_dataset
    from toolkit.duplicates import detect_duplicates
    from toolkit.dimensions import analyze_dimensions
    from toolkit.splitter import split_dataset
    from toolkit.reporter import generate_reports

    print("Running full pipeline…")
    scan = scan_dataset(args.input)
    dup = detect_duplicates(scan.records)
    dims = analyze_dimensions(scan.records)
    split = split_dataset(scan.valid_records)

    ann_reports = {}
    if args.labels and args.ann_format == "yolo":
        from toolkit.annotations import validate_yolo
        ann_reports["yolo"] = validate_yolo(
            [r.path for r in scan.valid_records], args.labels
        )

    written = generate_reports(
        output_dir=args.output,
        scan_result=scan,
        duplicate_report=dup,
        dimension_stats=dims,
        split_result=split,
        annotation_reports=ann_reports or None,
    )
    print(f"Reports written to {args.output}:")
    for name, path in written.items():
        print(f"  {path.name}")


def cmd_demo(args):
    """Generate a synthetic sample dataset and run the full pipeline."""
    import shutil
    import numpy as np
    import cv2
    import json

    out = Path(args.output)
    raw = out / "raw"
    labels_dir = out / "labels"
    reports_dir = out / "reports"
    processed_dir = out / "processed"

    # Clean slate
    if out.exists():
        shutil.rmtree(out)
    raw.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    print("Generating sample dataset…")

    # 1. Valid colour images (various sizes)
    sizes = [(640, 480), (320, 240), (1280, 720), (100, 100), (224, 224)]
    class_names = ["cat", "dog", "bird"]
    for i, (w, h) in enumerate(sizes):
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        # Put a coloured rectangle as a "fake object"
        cv2.rectangle(img, (10, 10), (w // 2, h // 2), (0, 255, 0), -1)
        path = raw / f"image_{i:03d}.jpg"
        cv2.imwrite(str(path), img)
        # YOLO label
        cx = (10 + w // 2) / (2 * w)
        cy = (10 + h // 2) / (2 * h)
        bw = (w // 2 - 10) / w
        bh = (h // 2 - 10) / h
        class_id = i % len(class_names)
        (labels_dir / f"image_{i:03d}.txt").write_text(
            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
        )

    # 2. PNG image (different extension, same content as image_000)
    img0 = cv2.imread(str(raw / "image_000.jpg"))
    cv2.imwrite(str(raw / "image_000_copy.png"), img0)
    (labels_dir / "image_000_copy.txt").write_text("0 0.25 0.25 0.3 0.3\n", encoding="utf-8")

    # 3. Exact duplicate (byte copy)
    shutil.copy(raw / "image_001.jpg", raw / "image_001_dup.jpg")
    shutil.copy(labels_dir / "image_001.txt", labels_dir / "image_001_dup.txt")

    # 4. Zero-byte file (corrupt)
    (raw / "corrupt_zero.jpg").write_bytes(b"")

    # 5. Truncated/corrupt JPEG
    (raw / "corrupt_trunc.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00GARBAGE")

    # 6. Non-image file (should be skipped)
    (raw / "readme.txt").write_text("This is not an image.", encoding="utf-8")

    # 7. COCO annotations
    coco = {
        "images": [
            {"id": i, "file_name": f"image_{i:03d}.jpg",
             "width": sizes[i][0], "height": sizes[i][1]}
            for i in range(len(sizes))
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": i % len(class_names),
                "bbox": [10, 10, sizes[i][0] // 2 - 10, sizes[i][1] // 2 - 10],
            }
            for i in range(len(sizes))
        ],
        "categories": [
            {"id": j, "name": name} for j, name in enumerate(class_names)
        ],
    }
    (out / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")

    # 8. Pascal VOC XML
    voc_dir = out / "voc_labels"
    voc_dir.mkdir()
    for i, (w, h) in enumerate(sizes):
        xml = f"""<annotation>
  <filename>image_{i:03d}.jpg</filename>
  <size>
    <width>{w}</width>
    <height>{h}</height>
    <depth>3</depth>
  </size>
  <object>
    <name>{class_names[i % len(class_names)]}</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>10</ymin>
      <xmax>{w // 2}</xmax>
      <ymax>{h // 2}</ymax>
    </bndbox>
  </object>
</annotation>"""
        (voc_dir / f"image_{i:03d}.xml").write_text(xml, encoding="utf-8")

    print(f"Sample dataset created in {raw}")
    print(f"  {len(sizes)} valid images, 1 PNG duplicate, 1 byte-identical duplicate,")
    print(f"  1 zero-byte corrupt, 1 truncated corrupt, 1 non-image file\n")

    # ── Run full pipeline ─────────────────────────────────────────────────────
    from toolkit.scanner import scan_dataset
    from toolkit.duplicates import detect_duplicates
    from toolkit.dimensions import analyze_dimensions
    from toolkit.preprocessor import preprocess_images
    from toolkit.splitter import split_dataset
    from toolkit.annotations import validate_yolo, validate_coco, validate_pascal_voc
    from toolkit.reporter import generate_reports

    print("Step 1/6 — Scanning…")
    scan = scan_dataset(raw)
    print(f"  {scan.valid_count} valid, {scan.invalid_count} invalid\n")

    print("Step 2/6 — Detecting duplicates…")
    dup = detect_duplicates(scan.records)
    print(
        f"  Exact-file groups: {len(dup.exact_file_groups)}, "
        f"Pixel-identical: {len(dup.pixel_identical_groups)}, "
        f"Perceptual: {len(dup.perceptual_groups)}\n"
    )

    print("Step 3/6 — Analysing dimensions…")
    dims = analyze_dimensions(scan.records)
    print(
        f"  Width {dims.min_width}–{dims.max_width}, "
        f"Height {dims.min_height}–{dims.max_height}\n"
    )

    print("Step 4/6 — Preprocessing (letterbox 640×640)…")
    prep = preprocess_images(scan.valid_records, processed_dir)
    print(f"  {prep.success_count} success, {prep.failure_count} failed\n")

    print("Step 5/6 — Splitting dataset…")
    split = split_dataset(scan.valid_records, seed=42)
    counts = split.counts()
    print(f"  train={counts['train']}, val={counts['val']}, test={counts['test']}\n")

    print("Step 6/6 — Validating annotations & generating reports…")
    yolo_report = validate_yolo(
        [r.path for r in scan.valid_records], labels_dir
    )
    coco_report = validate_coco(out / "annotations.json")
    voc_report = validate_pascal_voc(list(voc_dir.glob("*.xml")))

    ann_reports = {
        "yolo": yolo_report,
        "coco": coco_report,
        "pascal_voc": voc_report,
    }
    written = generate_reports(
        output_dir=reports_dir,
        scan_result=scan,
        duplicate_report=dup,
        dimension_stats=dims,
        split_result=split,
        annotation_reports=ann_reports,
    )
    print(f"\nAll reports written to {reports_dir}:")
    for name, p in written.items():
        print(f"  {p.name}")
    print("\nDemo complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config if hasattr(args, "config") else None)
    setup_logging(cfg)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "scan": cmd_scan,
        "duplicates": cmd_duplicates,
        "dimensions": cmd_dimensions,
        "preprocess": cmd_preprocess,
        "split": cmd_split,
        "validate-annotations": cmd_validate_annotations,
        "report": cmd_report,
    }

    if args.command == "demo":
        cmd_demo(args)
    elif args.command in dispatch:
        dispatch[args.command](args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
