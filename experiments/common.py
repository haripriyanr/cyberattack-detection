"""Shared helpers for the experiment scripts."""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics(df, name: str):
    ensure_dirs()
    path = METRICS_DIR / name
    df.to_csv(path, index=False)
    print(f"[save] {path}")
    return path


def save_fig(fig, name: str):
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150)
    print(f"[save] {path}")
    return path