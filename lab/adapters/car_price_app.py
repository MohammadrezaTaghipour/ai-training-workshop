"""Streamlit app: Divar crawl → dataset → train → predict car price.

Run from the repo root:

    uv run streamlit run lab/adapters/car_price_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core.cars.crawl import DEFAULT_URL, MAX_LISTINGS, CrawlError, crawl_listings
from core.cars.paths import ARTIFACTS_JOBLIB, CLEAN_CSV, RAW_CSV, ensure_dirs
from core.cars.predict import predict_price
from core.cars.prepare import PrepareError, listings_to_frame
from core.cars.train import load_artifacts, save_artifacts, train_price_model

_PREVIEW_COLS = [
    "title",
    "year",
    "mileage",
    "color",
    "transmission",
    "body_status",
    "chassis_status",
    "engine_status",
    "price",
    "url",
]


def _init_state() -> None:
    st.session_state.setdefault("raw_frame", None)
    st.session_state.setdefault("train_result", None)
    st.session_state.setdefault("artifacts", None)
    st.session_state.setdefault("crawl_log", [])
    st.session_state.setdefault("status", "")
    if st.session_state.artifacts is None and ARTIFACTS_JOBLIB.exists():
        try:
            st.session_state.artifacts = load_artifacts()
        except PrepareError:
            st.session_state.artifacts = None


def _log(message: str) -> None:
    st.session_state.crawl_log.append(message)


def _format_toman(value: float) -> str:
    return f"{value:,.0f} تومان"


def _save_raw(frame: pd.DataFrame) -> Path:
    ensure_dirs()
    frame.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    return RAW_CSV


def main() -> None:
    st.set_page_config(page_title="Car Price Studio", layout="wide")
    _init_state()

    st.title("Car Price Studio")
    st.caption(
        "Divar crawl → local CSV → clean → Linear Regression → joblib → predict. "
        "Sessions 11–13 in one Streamlit app."
    )

    tab_crawl, tab_train, tab_predict = st.tabs(
        ["1. Crawl dataset", "2. Train model", "3. Predict price"]
    )

    with tab_crawl:
        _crawl_tab()
    with tab_train:
        _train_tab()
    with tab_predict:
        _predict_tab()


def _crawl_tab() -> None:
    st.subheader("Crawl Divar listings")
    url = st.text_input("Divar search URL", value=DEFAULT_URL)
    max_listings = st.number_input(
        "Max listings",
        min_value=10,
        max_value=200,
        value=MAX_LISTINGS,
        step=10,
    )

    if st.button("Crawl & save CSV", type="primary"):
        st.session_state.crawl_log = []
        st.session_state.status = ""
        progress = st.progress(0, text="Starting…")
        log_box = st.empty()

        def on_progress(message: str) -> None:
            _log(message)
            # Rough progress: pages then details.
            n = len(st.session_state.crawl_log)
            progress.progress(min(n / (int(max_listings) + 5), 0.99), text=message)
            log_box.code("\n".join(st.session_state.crawl_log[-12:]))

        try:
            rows = crawl_listings(
                url,
                max_listings=int(max_listings),
                fetch_details=True,
                progress=on_progress,
            )
            frame = listings_to_frame(rows)
            path = _save_raw(frame)
            st.session_state.raw_frame = frame
            st.session_state.train_result = None
            progress.progress(1.0, text="Done")
            st.session_state.status = (
                f"Saved {len(frame)} rows to {path.relative_to(path.parents[2])}"
            )
            st.success(st.session_state.status)
        except (CrawlError, PrepareError) as exc:
            st.error(str(exc))
        except Exception as exc:  # network / unexpected parse failures
            st.error(f"Crawl failed: {exc}")

    if st.session_state.status:
        st.info(st.session_state.status)

    frame = st.session_state.raw_frame
    if frame is None and RAW_CSV.exists():
        frame = pd.read_csv(RAW_CSV)
        st.session_state.raw_frame = frame

    if frame is not None:
        st.write(f"Dataset preview ({len(frame)} rows)")
        cols = [c for c in _PREVIEW_COLS if c in frame.columns]
        st.dataframe(frame[cols], width="stretch")
        st.download_button(
            "Download raw CSV",
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="listings_raw.csv",
            mime="text/csv",
        )


def _train_tab() -> None:
    st.subheader("Train linear regression")
    frame = st.session_state.raw_frame
    if frame is None and RAW_CSV.exists():
        frame = pd.read_csv(RAW_CSV)
        st.session_state.raw_frame = frame

    if frame is None:
        st.warning("Crawl a dataset first (tab 1).")
        return

    st.write(f"Using {len(frame)} crawled rows.")
    if st.button("Train model", type="primary"):
        try:
            result = train_price_model(frame)
            path = save_artifacts(result)
            ensure_dirs()
            result["clean_frame"].to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
            st.session_state.train_result = result
            st.session_state.artifacts = {
                "model": result["model"],
                "encoders": result["encoders"],
                "feature_columns": result["feature_columns"],
                "metrics": result["metrics"],
                "choices": result["choices"],
            }
            st.success(
                f"Training complete. Model saved to "
                f"{path.relative_to(path.parents[2])}"
            )
        except PrepareError as exc:
            st.error(str(exc))

    result = st.session_state.train_result
    artifacts = st.session_state.artifacts
    metrics = None
    if result is not None:
        metrics = result["metrics"]
    elif artifacts is not None:
        metrics = artifacts.get("metrics")

    if metrics:
        st.markdown("#### Evaluation (20% test split)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", _format_toman(metrics["mae"]))
        c2.metric("RMSE", _format_toman(metrics["rmse"]))
        c3.metric("Mean price", _format_toman(metrics["price_mean"]))
        c4.metric("Clean rows", f"{metrics['n_clean']} / {metrics['n_raw']}")
        st.caption(
            f"Train size {metrics['n_train']}, test size {metrics['n_test']}. "
            f"MSE={metrics['mse']:,.0f}"
        )

    if result is not None:
        st.markdown("#### Coefficients")
        st.dataframe(result["coefficients"], width="stretch")
        st.markdown("#### Clean dataset preview")
        cols = [c for c in _PREVIEW_COLS if c in result["clean_frame"].columns]
        st.dataframe(result["clean_frame"][cols].head(20), width="stretch")


def _predict_tab() -> None:
    st.subheader("Predict car price")
    artifacts = st.session_state.artifacts
    if artifacts is None:
        st.warning("Train a model first (tab 2), or place a saved joblib under models/.")
        return

    choices = artifacts.get("choices") or {}
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Model year (Jalali)", min_value=1390, max_value=1405, value=1402)
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500_000, value=50_000, step=1000)
        color = st.selectbox("Color", choices.get("color") or ["نامشخص"])
        transmission = st.selectbox(
            "Transmission", choices.get("transmission") or ["نامشخص"]
        )
    with c2:
        body_status = st.selectbox(
            "Body status", choices.get("body_status") or ["نامشخص"]
        )
        chassis_status = st.selectbox(
            "Chassis status", choices.get("chassis_status") or ["نامشخص"]
        )
        engine_status = st.selectbox(
            "Engine status", choices.get("engine_status") or ["نامشخص"]
        )

    if st.button("Predict price", type="primary"):
        price = predict_price(
            artifacts,
            year=int(year),
            mileage=int(mileage),
            color=str(color),
            transmission=str(transmission),
            body_status=str(body_status),
            chassis_status=str(chassis_status),
            engine_status=str(engine_status),
        )
        st.markdown("### Estimated price")
        st.markdown(f"## {_format_toman(price)}")


if __name__ == "__main__":
    main()
