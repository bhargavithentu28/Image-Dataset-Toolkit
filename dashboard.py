"""
dashboard.py – Streamlit interface for the Image Dataset Preparation and Annotation Toolkit.

Launch:
    streamlit run dashboard.py

All business logic is delegated to the toolkit modules.
The UI never reimplements core functionality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Dataset Toolkit",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoid importing heavy libs at module level) ─────────────────
@st.cache_resource(show_spinner=False)
def _import_toolkit():
    from toolkit.config import load_config
    from toolkit.scanner import scan_dataset
    from toolkit.duplicates import detect_duplicates
    from toolkit.dimensions import analyze_dimensions
    from toolkit.preprocessor import preprocess_images
    from toolkit.splitter import split_dataset, validate_ratios
    from toolkit.annotations import validate_yolo, validate_coco, validate_pascal_voc
    from toolkit.reporter import generate_reports
    return {
        "load_config": load_config,
        "scan_dataset": scan_dataset,
        "detect_duplicates": detect_duplicates,
        "analyze_dimensions": analyze_dimensions,
        "preprocess_images": preprocess_images,
        "split_dataset": split_dataset,
        "validate_ratios": validate_ratios,
        "validate_yolo": validate_yolo,
        "validate_coco": validate_coco,
        "validate_pascal_voc": validate_pascal_voc,
        "generate_reports": generate_reports,
    }


tk = _import_toolkit()

# ── Session state helpers ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "scan_result": None,
        "dup_report": None,
        "dim_stats": None,
        "split_result": None,
        "ann_reports": {},
        "prep_summary": None,
        "reports_dir": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🖼️ Dataset Toolkit")
    st.markdown("---")

    st.subheader("📁 Input Directory")
    input_dir = st.text_input(
        "Image directory path",
        value="data/raw",
        help="Enter the **absolute or relative** path to your image folder. "
             "Browser file upload is not used for large datasets.",
    )

    st.subheader("📂 Output Directory")
    output_dir = st.text_input("Reports / output directory", value="reports")

    st.markdown("---")
    st.subheader("⚙️ Preprocessing")
    target_w = st.number_input("Target width", value=640, min_value=32, step=32)
    target_h = st.number_input("Target height", value=640, min_value=32, step=32)
    mode = st.selectbox("Resize mode", ["letterbox", "stretch", "crop_center"])

    st.markdown("---")
    st.subheader("✂️ Dataset Split")
    train_r = st.slider("Train ratio", 0.0, 1.0, 0.70, 0.05)
    val_r   = st.slider("Val ratio",   0.0, 1.0, 0.15, 0.05)
    test_r  = st.slider("Test ratio",  0.0, 1.0, 0.15, 0.05)
    seed    = st.number_input("Random seed", value=42, step=1)

    st.markdown("---")
    st.subheader("🏷️ Annotation Validation")
    ann_fmt  = st.selectbox("Format", ["yolo", "coco", "pascal_voc"])
    labels_d = st.text_input("Labels directory (YOLO/VOC)", value="data/labels")
    coco_j   = st.text_input("COCO JSON path", value="data/annotations.json")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Image Dataset Preparation and Annotation Toolkit")

tab_scan, tab_dup, tab_dim, tab_pre, tab_split, tab_ann, tab_report = st.tabs([
    "Scan", "Duplicates", "Dimensions", "Preprocess", "Split", "Annotations", "Reports"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB: SCAN
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:
    st.header("Dataset Scan & Validation")
    if st.button("▶ Scan Directory", key="btn_scan"):
        if not Path(input_dir).is_dir():
            st.error(f"Directory not found: `{input_dir}`")
        else:
            with st.spinner("Scanning…"):
                result = tk["scan_dataset"](input_dir)
                st.session_state["scan_result"] = result

    r = st.session_state["scan_result"]
    if r is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total files", r.total_files)
        col2.metric("Valid images", r.valid_count)
        col3.metric("Invalid images", r.invalid_count)
        col4.metric("Skipped (ext)", r.skipped_extensions)

        st.subheader("Image Inventory")
        rows = [rec.to_dict() for rec in r.records]
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇ Download image_inventory.csv",
                df.to_csv(index=False).encode(),
                "image_inventory.csv",
                "text/csv",
            )

        if r.invalid_records:
            st.subheader("⚠️ Invalid Images")
            inv_df = pd.DataFrame([rec.to_dict() for rec in r.invalid_records])
            st.dataframe(inv_df[["relative_path", "size_bytes", "error_message"]],
                         use_container_width=True)

        # Image preview
        st.subheader("Image Preview")
        valid_paths = [rec.path for rec in r.valid_records]
        if valid_paths:
            selected = st.selectbox(
                "Select an image to preview",
                [str(p) for p in valid_paths[:50]],
            )
            if selected:
                st.image(selected, caption=Path(selected).name, use_column_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────
with tab_dup:
    st.header("Duplicate Detection")
    st.info("No files are deleted automatically. Review and take explicit action.")

    use_perceptual = st.checkbox("Enable perceptual hashing", value=True)
    perc_threshold = st.slider("Perceptual hash threshold (Hamming)", 1, 30, 10)

    if st.button("▶ Detect Duplicates", key="btn_dup"):
        scan = st.session_state["scan_result"]
        if scan is None:
            st.warning("Run **Scan** first.")
        else:
            with st.spinner("Detecting duplicates…"):
                rep = tk["detect_duplicates"](
                    scan.records,
                    enable_perceptual=use_perceptual,
                    perceptual_threshold=perc_threshold,
                )
                st.session_state["dup_report"] = rep

    dr = st.session_state["dup_report"]
    if dr is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Exact-file groups", len(dr.exact_file_groups))
        col2.metric("Pixel-identical groups", len(dr.pixel_identical_groups))
        col3.metric("Perceptual groups", len(dr.perceptual_groups))

        rows = dr.to_rows()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇ Download duplicate_groups.csv",
                df.to_csv(index=False).encode(),
                "duplicate_groups.csv",
                "text/csv",
            )
        else:
            st.success("No duplicates detected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB: DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_dim:
    st.header("Image Dimension Analysis")
    if st.button("▶ Analyse Dimensions", key="btn_dim"):
        scan = st.session_state["scan_result"]
        if scan is None:
            st.warning("Run **Scan** first.")
        else:
            with st.spinner("Analysing…"):
                stats = tk["analyze_dimensions"](scan.records)
                st.session_state["dim_stats"] = stats

    ds = st.session_state["dim_stats"]
    if ds is not None:
        d = ds.to_dict()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Width")
            st.write({k: d[k] for k in ["min_width", "max_width", "mean_width", "median_width"]})
        with col2:
            st.subheader("Height")
            st.write({k: d[k] for k in ["min_height", "max_height", "mean_height", "median_height"]})

        col3, col4, col5 = st.columns(3)
        col3.metric("Unique sizes", d["unique_size_count"])
        col4.metric("Small images", d["small_image_count"])
        col5.metric("Large images", d["large_image_count"])

        # Distribution chart (widths vs heights)
        scan = st.session_state["scan_result"]
        if scan:
            chart_data = pd.DataFrame([
                {"width": r.width, "height": r.height}
                for r in scan.valid_records
                if r.width and r.height
            ])
            if not chart_data.empty:
                st.subheader("Width Distribution")
                st.bar_chart(chart_data["width"].value_counts().sort_index())
                st.subheader("Height Distribution")
                st.bar_chart(chart_data["height"].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# TAB: PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
with tab_pre:
    st.header("Image Preprocessing")
    processed_dir = st.text_input("Processed output directory", value="data/processed")

    if st.button("▶ Preprocess Images", key="btn_pre"):
        scan = st.session_state["scan_result"]
        if scan is None:
            st.warning("Run **Scan** first.")
        else:
            with st.spinner(f"Preprocessing → {processed_dir}…"):
                summary = tk["preprocess_images"](
                    records=scan.valid_records,
                    output_dir=processed_dir,
                    target_width=int(target_w),
                    target_height=int(target_h),
                    mode=mode,
                )
                st.session_state["prep_summary"] = summary
            st.success(
                f"Done: {summary.success_count} success, {summary.failure_count} failed"
            )

    ps = st.session_state["prep_summary"]
    if ps is not None:
        rows = [
            {
                "source": str(r.source_path),
                "output": str(r.output_path),
                "orig": f"{r.orig_width}×{r.orig_height}",
                "out": f"{r.out_width}×{r.out_height}" if r.out_width else "—",
                "success": r.success,
                "error": r.error or "",
            }
            for r in ps.results
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: SPLIT
# ─────────────────────────────────────────────────────────────────────────────
with tab_split:
    st.header("Dataset Splitting")
    total_ratio = round(train_r + val_r + test_r, 4)
    if abs(total_ratio - 1.0) > 1e-3:
        st.warning(f"Ratios sum to {total_ratio:.4f} — must equal 1.0")

    if st.button("▶ Split Dataset", key="btn_split"):
        scan = st.session_state["scan_result"]
        if scan is None:
            st.warning("Run **Scan** first.")
        elif abs(total_ratio - 1.0) > 1e-3:
            st.error("Fix ratios before splitting.")
        else:
            with st.spinner("Splitting…"):
                result = tk["split_dataset"](
                    records=scan.valid_records,
                    train_ratio=train_r,
                    val_ratio=val_r,
                    test_ratio=test_r,
                    seed=int(seed),
                )
                st.session_state["split_result"] = result

    sr = st.session_state["split_result"]
    if sr is not None:
        counts = sr.counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Train", counts["train"])
        col2.metric("Val", counts["val"])
        col3.metric("Test", counts["test"])

        for w in sr.warnings:
            st.warning(w)

        rows = sr.to_rows()
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇ Download split_manifest.csv",
            df.to_csv(index=False).encode(),
            "split_manifest.csv",
            "text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB: ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_ann:
    st.header("Annotation Validation")
    if st.button("▶ Validate Annotations", key="btn_ann"):
        scan = st.session_state["scan_result"]
        if scan is None:
            st.warning("Run **Scan** first.")
        else:
            ann_reports = {}
            with st.spinner(f"Validating {ann_fmt} annotations…"):
                if ann_fmt == "yolo":
                    rep = tk["validate_yolo"](
                        [r.path for r in scan.valid_records], labels_d
                    )
                    ann_reports["yolo"] = rep
                elif ann_fmt == "coco":
                    rep = tk["validate_coco"](coco_j)
                    ann_reports["coco"] = rep
                elif ann_fmt == "pascal_voc":
                    from pathlib import Path as _P
                    xmls = list(_P(labels_d).glob("*.xml"))
                    rep = tk["validate_pascal_voc"](xmls)
                    ann_reports["pascal_voc"] = rep
            st.session_state["ann_reports"] = ann_reports

    ann_reports = st.session_state["ann_reports"]
    if ann_reports:
        for fmt, rep in ann_reports.items():
            st.subheader(f"{fmt.upper()} Validation")
            d = rep.to_dict()
            st.json(d)

# ─────────────────────────────────────────────────────────────────────────────
# TAB: REPORTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    st.header("Generate Full Quality Report")
    if st.button("▶ Generate Reports", key="btn_report"):
        scan = st.session_state["scan_result"]
        if scan is None:
            st.warning("Run **Scan** first.")
        else:
            with st.spinner("Generating reports…"):
                written = tk["generate_reports"](
                    output_dir=output_dir,
                    scan_result=scan,
                    duplicate_report=st.session_state["dup_report"],
                    dimension_stats=st.session_state["dim_stats"],
                    split_result=st.session_state["split_result"],
                    annotation_reports=st.session_state["ann_reports"] or None,
                )
                st.session_state["reports_dir"] = output_dir

            st.success(f"Reports written to `{output_dir}`")
            for name, p in written.items():
                st.write(f"• `{p.name}`")

    rd = st.session_state["reports_dir"]
    if rd and Path(rd).is_dir():
        md_path = Path(rd) / "dataset_report.md"
        if md_path.exists():
            st.markdown("---")
            st.subheader("Markdown Summary")
            st.markdown(md_path.read_text(encoding="utf-8"))

        st.markdown("---")
        st.subheader("Download Reports")
        for csv_file in Path(rd).glob("*.csv"):
            data = csv_file.read_bytes()
            st.download_button(
                f"⬇ {csv_file.name}",
                data,
                csv_file.name,
                "text/csv",
                key=f"dl_{csv_file.stem}",
            )
        for json_file in Path(rd).glob("*.json"):
            data = json_file.read_bytes()
            st.download_button(
                f"⬇ {json_file.name}",
                data,
                json_file.name,
                "application/json",
                key=f"dl_{json_file.stem}",
            )
