# 🖼️ Image Dataset Preparation and Annotation Toolkit

A local Python toolkit for preparing image datasets for computer vision and
object detection workflows. Handles validation, deduplication, resizing,
dataset splitting, annotation validation, and quality report generation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

| Module | Capability |
|--------|-----------|
| **Scanner** | Recursive scan, format validation, corrupt/zero-byte detection, SHA-256 hashing |
| **Duplicates** | Exact-file, decoded-pixel, and perceptual-hash duplicate detection |
| **Dimensions** | Min/max/mean/median stats, aspect-ratio analysis, flagging small/large images |
| **Preprocessor** | Letterbox / stretch / crop-center resize with bounding-box transform |
| **Splitter** | Reproducible train/val/test splits with optional class stratification |
| **Annotations** | YOLO, COCO JSON, Pascal VOC XML readers and validators |
| **Reporter** | JSON, CSV, and Markdown quality-report generation |
| **Dashboard** | Streamlit UI wiring the same core functions — no duplicated logic |

---

## 📋 Requirements

- Python 3.10, 3.11, or 3.12
- Windows 10/11 (PowerShell or CMD), Linux, or macOS

---

## ⚡ Quick Start — One-Shot Bootstrap

```powershell
cd e:\Image-Dataset-Toolkit
python run_all.py
```

This single command:
1. Creates a virtual environment (`.venv/`)
2. Installs all dependencies
3. Runs the full test suite (94 tests)
4. Runs the end-to-end demo pipeline

---

## 🛠️ Manual Setup (Windows PowerShell)

```powershell
# 1. Navigate to the project folder
cd e:\Image-Dataset-Toolkit

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it (PowerShell)
.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Tip:** If `Activate.ps1` is blocked by execution policy, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

> **Important:** If the venv is not active, use `.venv\Scripts\python.exe` and
> `.venv\Scripts\streamlit.exe` instead of bare `python` / `streamlit`.

---

## 🧪 Running Tests

```powershell
# Quick (summary only)
.venv\Scripts\python.exe -m pytest -q

# Verbose with short tracebacks
.venv\Scripts\python.exe -m pytest -v --tb=short

# With coverage report
.venv\Scripts\python.exe -m pytest --cov=toolkit --cov-report=term-missing -q
```

Expected: **94 passed**

---

## 💻 CLI Usage

> **Note:** Always use `.venv\Scripts\python.exe` (or activate the venv first).

```powershell
# Show all available commands
.venv\Scripts\python.exe app.py --help

# Scan a dataset directory and produce reports
.venv\Scripts\python.exe app.py scan `
    --input "D:\my_images" --output reports

# Detect duplicate images
.venv\Scripts\python.exe app.py duplicates `
    --input "D:\my_images" --output reports

# Analyse image dimensions
.venv\Scripts\python.exe app.py dimensions `
    --input "D:\my_images" --output reports

# Preprocess images (letterbox resize to 640×640)
.venv\Scripts\python.exe app.py preprocess `
    --input "D:\my_images" --output "D:\processed" `
    --width 640 --height 640 --mode letterbox

# Create reproducible train/val/test splits
.venv\Scripts\python.exe app.py split `
    --input "D:\my_images" --output reports `
    --train 0.7 --val 0.15 --test 0.15 --seed 42

# Validate YOLO annotations
.venv\Scripts\python.exe app.py validate-annotations `
    --input "D:\my_images" --labels "D:\labels" `
    --format yolo --output reports

# Validate COCO annotations
.venv\Scripts\python.exe app.py validate-annotations `
    --input "D:\my_images" --coco-json "D:\annotations.json" `
    --format coco --output reports

# Validate Pascal VOC annotations
.venv\Scripts\python.exe app.py validate-annotations `
    --input "D:\my_images" --labels "D:\voc_labels" `
    --format pascal_voc --output reports

# Full pipeline — scan, deduplicate, dimension stats, split, reports
.venv\Scripts\python.exe app.py report `
    --input "D:\my_images" --output reports

# End-to-end demo (auto-generates a synthetic dataset)
.venv\Scripts\python.exe app.py demo --output demo_output
```

---

## 🌐 Streamlit Dashboard

```powershell
.venv\Scripts\streamlit.exe run dashboard.py
```

Open **http://localhost:8501** in your browser.

The sidebar lets you set an image directory path (absolute or relative).
Tabs provide:
- **Scan** — validate and preview images
- **Duplicates** — review duplicate groups (no auto-deletion)
- **Dimensions** — distribution charts, flagged images
- **Preprocess** — resize with configurable mode
- **Split** — interactive ratio sliders, manifest download
- **Annotations** — YOLO / COCO / Pascal VOC validation
- **Reports** — generate and download all report files

---

## 📁 Project Structure

```
Image-Dataset-Toolkit/
├── toolkit/
│   ├── __init__.py
│   ├── config.py            # Config loading & defaults
│   ├── scanner.py           # Dataset scanning & validation
│   ├── duplicates.py        # Duplicate detection (SHA-256, pixel, perceptual)
│   ├── dimensions.py        # Dimension analysis & statistics
│   ├── preprocessor.py      # Image resizing & padding
│   ├── splitter.py          # Reproducible dataset splitting
│   ├── reporter.py          # Report generation (JSON, CSV, Markdown)
│   └── annotations/
│       ├── __init__.py
│       ├── yolo.py          # YOLO reader & validator
│       ├── coco.py          # COCO JSON reader & validator
│       └── pascal_voc.py    # Pascal VOC XML reader & validator
├── tests/
│   ├── conftest.py          # Shared pytest fixtures
│   ├── test_scanner.py      # 15 tests
│   ├── test_duplicates.py   # 7 tests
│   ├── test_dimensions.py   # 8 tests
│   ├── test_preprocessor.py # 8 tests
│   ├── test_splitter.py     # 11 tests
│   ├── test_annotations.py  # 27 tests
│   └── test_reporter.py     # 10 tests
├── app.py                   # CLI entry point
├── dashboard.py             # Streamlit dashboard
├── run_all.py               # One-shot bootstrap script
├── config.yaml              # Default configuration
├── requirements.txt         # Pinned dependencies
├── pytest.ini
├── setup.cfg
└── .gitignore
```

---

## 📊 Generated Reports

After running a scan or full pipeline, the `reports/` folder will contain:

| File | Contents |
|------|----------|
| `dataset_summary.json` | High-level stats: scan, duplicates, dimensions, splits |
| `image_inventory.csv` | Per-image record: path, size, dimensions, hash, validity |
| `invalid_images.csv` | Details on corrupt/zero-byte/unreadable files |
| `duplicate_groups.csv` | All duplicate groups with kind and file paths |
| `dimension_statistics.json` | Min/max/mean/median width & height, flagged images |
| `split_manifest.csv` | Per-image split assignment (train/val/test) |
| `annotation_validation.json` | Per-format validation results with errors/warnings |
| `dataset_report.md` | Human-readable Markdown summary of all above |

---

## ⚙️ Configuration

Edit [`config.yaml`](config.yaml) to change defaults:

```yaml
preprocessing:
  target_width: 640
  target_height: 640
  mode: "letterbox"        # letterbox | stretch | crop_center

splitting:
  train_ratio: 0.70
  val_ratio:   0.15
  test_ratio:  0.15
  random_seed: 42

duplicates:
  enable_perceptual_hash: true
  perceptual_hash_threshold: 10

logging:
  level: "INFO"            # DEBUG | INFO | WARNING | ERROR
```

All CLI flags override corresponding config values.

---

## ⚠️ Known Limitations

- **Perceptual hashing** requires images to be decodable; corrupt files are skipped.
- **Stratified splitting** requires at least one annotation per image; unannotated images fall into an `unknown` class bucket.
- **COCO → YOLO conversion** requires an explicit class-name mapping (not auto-remapped).
- **Very large datasets** (>100k images) may be slow on pixel-hash computation; SHA-256 is always fast.
- **Streamlit file upload** is browser-memory limited; use the directory-path input for large datasets.

---

## 👤 Author

**Bhargavi Thentu**
GitHub: [@bhargavithentu28](https://github.com/bhargavithentu28)
