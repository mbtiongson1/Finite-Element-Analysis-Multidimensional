from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    # This file lives at <repo>/paths.py
    return Path(__file__).resolve().parent


REPO_ROOT = _repo_root()
OUT_DIR = REPO_ROOT / "out"
CSV_DIR = OUT_DIR / "csv"
FIGURES_DIR = OUT_DIR / "figures"
REPORTS_DIR = OUT_DIR / "reports"


def csv_path(filename: str) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    return CSV_DIR / filename
