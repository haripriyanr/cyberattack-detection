"""Persist / reload trained artifacts with joblib."""

from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def save_bundle(bundle: dict, name: str = "best_model.joblib") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    joblib.dump(bundle, path)
    print(f"[persist] saved {path}")
    return path


def load_bundle(name: str = "best_model.joblib") -> dict:
    return joblib.load(MODEL_DIR / name)