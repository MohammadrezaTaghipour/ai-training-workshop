"""Turn crawled Divar rows into a regression-ready table (session 12 prep)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

_PERSIAN_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)

FEATURE_COLUMNS = [
    "year",
    "mileage",
    "color_code",
    "transmission_code",
    "body_status_code",
    "chassis_status_code",
    "engine_status_code",
]
CATEGORICAL_COLUMNS = [
    "color",
    "transmission",
    "body_status",
    "chassis_status",
    "engine_status",
]


class PrepareError(ValueError):
    """Dataset could not be prepared for training."""


def listings_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a raw DataFrame and parse numeric year / mileage / price."""
    if not rows:
        raise PrepareError("No crawled rows to convert.")
    frame = pd.DataFrame(rows)
    frame["year"] = frame.apply(_pick_year, axis=1)
    frame["mileage"] = frame.apply(_pick_mileage, axis=1)
    frame["price"] = frame.apply(_pick_price, axis=1)
    for col in CATEGORICAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
        frame[col] = (
            frame[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": "نامشخص", "nan": "نامشخص"})
        )
    return frame


def clean_for_training(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop bad rows, remove price outliers (IQR), encode categoricals."""
    needed = ["year", "mileage", "price", *CATEGORICAL_COLUMNS]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise PrepareError(f"Missing columns: {', '.join(missing)}")

    clean = frame.dropna(subset=["year", "mileage", "price"]).copy()
    clean = clean[(clean["year"] > 1370) & (clean["year"] < 1410)]
    clean = clean[(clean["mileage"] >= 0) & (clean["mileage"] < 1_000_000)]
    clean = clean[(clean["price"] > 10_000_000) & (clean["price"] < 20_000_000_000)]
    # IQR needs a bit of volume; skip on tiny crawls so training still works.
    if len(clean) >= 20:
        clean = _drop_iqr_outliers(clean, "price")
    clean = clean.reset_index(drop=True)

    if len(clean) < 10:
        raise PrepareError(
            f"Only {len(clean)} clean rows left; need at least 10 to train."
        )

    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        clean[f"{col}_code"] = encoder.fit_transform(clean[col].astype(str))
        encoders[col] = encoder

    meta = {
        "encoders": encoders,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_raw": int(len(frame)),
        "n_clean": int(len(clean)),
    }
    return clean, meta


def encode_prediction_row(
    *,
    year: int,
    mileage: int,
    color: str,
    transmission: str,
    body_status: str,
    chassis_status: str,
    engine_status: str,
    encoders: dict[str, LabelEncoder],
) -> list[float]:
    """Map a form submission into the model feature vector."""
    values = {
        "year": float(year),
        "mileage": float(mileage),
        "color": _known_or_default(color, encoders["color"]),
        "transmission": _known_or_default(transmission, encoders["transmission"]),
        "body_status": _known_or_default(body_status, encoders["body_status"]),
        "chassis_status": _known_or_default(chassis_status, encoders["chassis_status"]),
        "engine_status": _known_or_default(engine_status, encoders["engine_status"]),
    }
    return [
        values["year"],
        values["mileage"],
        float(encoders["color"].transform([values["color"]])[0]),
        float(encoders["transmission"].transform([values["transmission"]])[0]),
        float(encoders["body_status"].transform([values["body_status"]])[0]),
        float(encoders["chassis_status"].transform([values["chassis_status"]])[0]),
        float(encoders["engine_status"].transform([values["engine_status"]])[0]),
    ]


def _known_or_default(value: str, encoder: LabelEncoder) -> str:
    classes = set(encoder.classes_)
    text = (value or "").strip() or "نامشخص"
    if text in classes:
        return text
    if "نامشخص" in classes:
        return "نامشخص"
    return str(encoder.classes_[0])


def _pick_year(row: pd.Series) -> float | None:
    text = str(row.get("year_text") or "")
    year = _first_int(text)
    if year is not None and year > 1900:
        # Detail sometimes shows "1404 - 2025"; keep Jalali year.
        parts = [_first_int(p) for p in re.split(r"[-–]", text)]
        parts = [p for p in parts if p is not None]
        jalali = [p for p in parts if 1300 <= p <= 1500]
        if jalali:
            return float(jalali[0])
    if year is not None and 1300 <= year <= 1500:
        return float(year)
    # Fallback: look for a year-like token in the title.
    # People write 1404, ۴۰۴, 04, or 99.
    title = str(row.get("title") or "")
    for token in re.findall(r"\d{2,4}", _to_ascii_digits(title)):
        value = int(token)
        if 1390 <= value <= 1410:
            return float(value)
        if 400 <= value <= 410:  # ۴۰۴ → 1404
            return float(1000 + value)
        if 90 <= value <= 99:
            return float(1300 + value)
        if 0 <= value <= 15:
            return float(1400 + value)
    return None


def _pick_mileage(row: pd.Series) -> float | None:
    for key in ("mileage_text", "list_mileage_text"):
        text = str(row.get(key) or "")
        if "صفر" in text:
            return 0.0
        value = _first_int(text.replace(",", "").replace("٬", "").replace("،", ""))
        if value is not None:
            return float(value)
    return None


def _pick_price(row: pd.Series) -> float | None:
    price = row.get("price")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)
    for key in ("price_text", "list_price_text"):
        text = str(row.get(key) or "")
        value = _first_int(text.replace(",", "").replace("٬", "").replace("،", ""))
        if value is not None and value > 0:
            return float(value)
    return None


def _drop_iqr_outliers(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = frame[column].quantile(0.25)
    q3 = frame[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return frame[(frame[column] >= lower) & (frame[column] <= upper)]


def _to_ascii_digits(text: str) -> str:
    return str(text).translate(_PERSIAN_DIGITS)


def _first_int(text: str) -> int | None:
    match = re.search(r"\d+", _to_ascii_digits(text).replace(",", ""))
    if not match:
        return None
    return int(match.group(0))
