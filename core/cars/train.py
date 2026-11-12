"""Train / save / load a car-price linear regression (session 13)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from core.cars.paths import ARTIFACTS_JOBLIB, ensure_dirs
from core.cars.prepare import FEATURE_COLUMNS, PrepareError, clean_for_training


def train_price_model(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit LinearRegression and return model, metrics, and encoders."""
    clean, meta = clean_for_training(frame)
    feature_columns = meta["feature_columns"]
    X = clean[feature_columns]
    y = clean["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    probe = LinearRegression()
    probe.fit(X_train.values, y_train)
    y_pred = probe.predict(X_test.values)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))

    final_model = LinearRegression()
    final_model.fit(X.values, y)

    coef = pd.DataFrame(
        final_model.coef_, index=feature_columns, columns=["coefficient"]
    )

    return {
        "model": final_model,
        "encoders": meta["encoders"],
        "feature_columns": feature_columns,
        "clean_frame": clean,
        "metrics": {
            "n_raw": meta["n_raw"],
            "n_clean": meta["n_clean"],
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "price_mean": float(y.mean()),
        },
        "coefficients": coef,
        "choices": {
            col: sorted(encoder.classes_.tolist())
            for col, encoder in meta["encoders"].items()
        },
    }


def save_artifacts(result: dict[str, Any], path: Path | None = None) -> Path:
    """Persist model + encoders with joblib (session 13 deployment)."""
    ensure_dirs()
    target = path or ARTIFACTS_JOBLIB
    payload = {
        "model": result["model"],
        "encoders": result["encoders"],
        "feature_columns": result["feature_columns"],
        "metrics": result["metrics"],
        "choices": result["choices"],
    }
    dump(payload, target)
    return target


def load_artifacts(path: Path | None = None) -> dict[str, Any]:
    """Load a previously saved car-price model bundle."""
    target = path or ARTIFACTS_JOBLIB
    if not target.exists():
        raise PrepareError(f"No saved model at {target}")
    artifacts = load(target)
    if "model" not in artifacts or "encoders" not in artifacts:
        raise PrepareError("Saved file is not a car-price model bundle.")
    artifacts.setdefault("feature_columns", list(FEATURE_COLUMNS))
    return artifacts
