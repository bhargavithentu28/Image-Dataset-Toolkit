"""
run_all.py
──────────
Self-contained installer + test runner + demo.
Run this script with *any* Python 3.10+ to:
  1. Create a virtual environment (.venv/)
  2. Install all requirements
  3. Run the full test suite
  4. Run the end-to-end demo

Usage (no venv needed):
    python run_all.py

Or, after creating a venv manually:
    .venv\\Scripts\\python.exe run_all.py
"""
from __future__ import annotations
import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WIN = sys.platform == "win32"

def venv_python():
    return str(VENV / ("Scripts/python.exe" if WIN else "bin/python"))

def run(args: list[str], *, check: bool = True, cwd=None) -> int:
    print(f"\n$ {' '.join(str(a) for a in args)}")
    r = subprocess.run(args, cwd=cwd or ROOT)
    if check and r.returncode != 0:
        print(f"  [exit {r.returncode}]")
    return r.returncode

print("=" * 60)
print("  Image Dataset Preparation and Annotation Toolkit")
print("  Bootstrap + Test Runner")
print("=" * 60)

# ── Step 1: virtual environment ───────────────────────────────────────────────
if not VENV.exists():
    print(f"\n[1] Creating virtual environment at {VENV}")
    run([sys.executable, "-m", "venv", str(VENV)])
else:
    print(f"\n[1] Virtual environment already exists at {VENV}")

py = venv_python()
if not Path(py).exists():
    print(f"ERROR: expected venv python not found at {py}")
    sys.exit(1)

# ── Step 2: install deps ──────────────────────────────────────────────────────
print("\n[2] Installing/updating dependencies…")
run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
rc = run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "--quiet"])
if rc != 0:
    print("WARNING: some dependencies may not have installed correctly.")

# ── Step 3: run tests ─────────────────────────────────────────────────────────
print("\n[3] Running test suite…")
test_rc = run([py, "-m", "pytest", "-v", "--tb=short", str(ROOT / "tests")], check=False)

# ── Step 4: run demo ──────────────────────────────────────────────────────────
print("\n[4] Running end-to-end demo…")
demo_dir = ROOT / "demo_output"
if demo_dir.exists():
    shutil.rmtree(demo_dir)
demo_rc = run([py, str(ROOT / "app.py"), "demo", "--output", str(demo_dir)], check=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  Tests  : {'PASSED' if test_rc == 0 else f'FAILED (exit {test_rc})'}")
print(f"  Demo   : {'OK' if demo_rc == 0 else f'FAILED (exit {demo_rc})'}")
print("=" * 60)

if test_rc == 0:
    print("\nAll tests passed!")
    print(f"\nReports written to: {demo_dir / 'reports'}")
    print("\nTo launch the dashboard:")
    venv_stream = str(VENV / ("Scripts/streamlit.exe" if WIN else "bin/streamlit"))
    print(f"  {venv_stream} run dashboard.py")
else:
    print("\nSome tests failed — check output above.")

sys.exit(test_rc)
