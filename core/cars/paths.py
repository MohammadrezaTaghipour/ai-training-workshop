"""Local paths for the car-price studio dataset and model files."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "projects" / "car-price-studio"
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

RAW_CSV = DATA_DIR / "listings_raw.csv"
CLEAN_CSV = DATA_DIR / "listings_clean.csv"
ARTIFACTS_JOBLIB = MODEL_DIR / "car_price_model.joblib"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
