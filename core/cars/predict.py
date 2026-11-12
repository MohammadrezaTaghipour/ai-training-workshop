"""Singleton car-price prediction from a saved joblib bundle."""

from __future__ import annotations

from typing import Any

from core.cars.prepare import encode_prediction_row


def predict_price(
    artifacts: dict[str, Any],
    *,
    year: int,
    mileage: int,
    color: str,
    transmission: str,
    body_status: str,
    chassis_status: str,
    engine_status: str,
) -> float:
    """Predict price (tomans) for one car specification."""
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    vector = encode_prediction_row(
        year=year,
        mileage=mileage,
        color=color,
        transmission=transmission,
        body_status=body_status,
        chassis_status=chassis_status,
        engine_status=engine_status,
        encoders=encoders,
    )
    prediction = model.predict([vector])[0]
    return float(prediction)
