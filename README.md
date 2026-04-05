# Energy Demand Forecasting API

Hourly building energy demand prediction using LightGBM, deployed as a REST API on Render.

**Live API:** https://YOUR-APP-NAME.onrender.com

---

## Model Performance

| Metric | Value |
|--------|-------|
| RMSE | 5.5169 |
| MAE | 2.7230 |
| MAPE | 9.23% |
| R² | 0.9528 |

### Cross-Validation (TimeSeriesSplit — 5 Fold)

| Metric | Mean ± Std |
|--------|------------|
| RMSE | 5.375 ± 0.961 |
| MAE | 3.426 ± 0.784 |
| R² | 0.9478 ± 0.0223 |

---

## Model Details

- **Algorithm:** LightGBM Regressor
- **Tuning:** Optuna (50 trials, TimeSeriesSplit inner loop)
- **Features:** 13 (calendar, cyclic hour encoding, lag, rolling mean)
- **Training period:** 2018-07 → 2019-06
- **Test period:** 2019-07 onwards

### Features Used

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `weekday` | Day of week (0=Monday, 6=Sunday) |
| `month` | Month of year (1–12) |
| `is_weekend` | 1 if Saturday/Sunday, else 0 |
| `hour_sin` | sin(2π × hour / 24) — cyclic encoding |
| `hour_cos` | cos(2π × hour / 24) — cyclic encoding |
| `lag_1` | Total demand 1 hour ago (kW) |
| `lag_24` | Total demand 24 hours ago (kW) |
| `lag_48` | Total demand 48 hours ago (kW) |
| `lag_168` | Total demand same hour last week (kW) |
| `rolling_24` | 24-hour rolling mean demand (kW) |
| `rolling_48` | 48-hour rolling mean demand (kW) |
| `rolling_168` | 168-hour rolling mean demand (kW) |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + model metrics |
| GET | `/features` | Lists all required input features |
| POST | `/predict` | Predict demand for a single record |
| POST | `/predict_batch` | Predict demand for multiple records |

---

## Usage Examples

### Health Check

```bash
curl https://YOUR-APP-NAME.onrender.com/
```

**Response:**
```json
{
  "status": "ok",
  "model": "LightGBM Energy Demand Forecaster",
  "metrics": {
    "rmse": 5.5169,
    "mae": 2.723,
    "mape": 9.23,
    "r2": 0.9528
  }
}
```

---

### Single Prediction

```bash
curl -X POST https://YOUR-APP-NAME.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 9,
    "weekday": 0,
    "month": 7,
    "is_weekend": 0,
    "hour_sin": 0.7071,
    "hour_cos": 0.7071,
    "lag_1": 45.2,
    "lag_24": 44.8,
    "lag_48": 43.1,
    "lag_168": 46.3,
    "rolling_24": 35.6,
    "rolling_48": 34.2,
    "rolling_168": 33.8
  }'
```

**Response:**
```json
{
  "predicted_total_demand_kW": 52.3141,
  "input": { "hour": 9, "weekday": 0, "..." : "..." }
}
```

---

### Batch Prediction

```bash
curl -X POST https://YOUR-APP-NAME.onrender.com/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "hour": 9, "weekday": 0, "month": 7, "is_weekend": 0,
        "hour_sin": 0.7071, "hour_cos": 0.7071,
        "lag_1": 45.2, "lag_24": 44.8, "lag_48": 43.1, "lag_168": 46.3,
        "rolling_24": 35.6, "rolling_48": 34.2, "rolling_168": 33.8
      },
      {
        "hour": 10, "weekday": 0, "month": 7, "is_weekend": 0,
        "hour_sin": 1.0, "hour_cos": 0.0,
        "lag_1": 52.3, "lag_24": 55.1, "lag_48": 54.0, "lag_168": 53.2,
        "rolling_24": 36.1, "rolling_48": 34.8, "rolling_168": 34.0
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    { "index": 0, "predicted_total_demand_kW": 52.3141 },
    { "index": 1, "predicted_total_demand_kW": 61.0823 }
  ]
}
```

---

## Repo Structure

```
├── app.py              # Flask REST API
├── model.pkl           # Trained LightGBM model artifact
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container config for Render
└── README.md
```

---

## Deployment

Deployed on [Render](https://render.com) using Docker.

> **Note:** The free tier sleeps after 15 minutes of inactivity. The first request may take ~30 seconds to wake up.
