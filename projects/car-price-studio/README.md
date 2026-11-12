# Car Price Studio

End-to-end workshop project (sessions 11–13):

**Divar URL → crawl API → local CSV → clean → Linear Regression → joblib → predict**

Logic lives in UI-free `core/cars/`. The Streamlit UI is the web adapter.

## Run

From the repository root:

```bash
uv run streamlit run lab/adapters/car_price_app.py
```

Default search URL:

`https://divar.ir/s/tehran/car/peugeot/207i`

## Flow in the app

1. **Crawl dataset** — load page 1 from Divar HTML preloaded state, then paginate with `POST /v8/postlist/w/search` (lazy load). Fetch each listing detail from `/v8/posts-v2/web/{token}` (up to 50). Save `data/listings_raw.csv`.
2. **Train model** — drop bad rows, IQR price outliers, `LabelEncoder` for categoricals, `LinearRegression`, show MAE / MSE / RMSE, save `models/car_price_model.joblib`.
3. **Predict price** — enter year, mileage, color, transmission, body / chassis / engine status.

## Stack (already used in the course)

`requests` · `pandas` · `scikit-learn` · `joblib` · Streamlit
