# ⚡ Energy Demand Forecasting

Hourly building energy demand prediction using LightGBM, served with a web UI and REST API, deployed on Render.

**🌐 Live Demo:** https://hourly-building-energy-demand-prediction.onrender.com

> **Note:** Free tier sleeps after 15 min of inactivity — first request may take ~30s to wake up.

---

## 🖥️ Web Interface

Visit the live URL to access the interactive prediction UI. No coding required.

- Select **hour, day, and month** from dropdowns
- Enter **lag and rolling mean values** (past demand in kW)
- Click **Predict** to get the forecasted demand instantly
- Weekend flag and cyclic hour encoding are calculated automatically

![UI Preview](https://hourly-building-energy-demand-prediction.onrender.com)

---

## 📊 Model Performance

| Metric | Test Set |
|--------|----------|
| R²     | 0.9528   |
| RMSE   | 5.5169   |
| MAE    | 2.7230   |
| MAPE   | 9.23%    |

### Cross-Validation (TimeSeriesSplit — 5 Fold)

| Metric | Mean ± Std          |
|--------|---------------------|
| RMSE   | 5.375 ± 0.961       |
| MAE    | 3.426 ± 0.784       |
| R²     | 0.9478 ± 0.0223     |

---

## 🤖 Model Details

- **Algorithm:** LightGBM Regressor
- **Tuning:** Optuna (50 trials, TimeSeriesSplit inner loop)
- **Training period:** Jul 2018 → Jun 2019
- **Test period:** Jul 2019 onwards

### Features (13 total)

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

## 🔗 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api` | Health check + model metrics (JSON) |
| GET | `/features` | Lists all required input features |
| POST | `/predict` | Single prediction |
| POST | `/predict_batch` | Batch predictions |

### Single Prediction

```bash
curl -X POST https://hourly-building-energy-demand-prediction.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 9, "weekday": 0, "month": 7, "is_weekend": 0,
    "hour_sin": 0.7071, "hour_cos": 0.7071,
    "lag_1": 45.2, "lag_24": 44.8, "lag_48": 43.1, "lag_168": 46.3,
    "rolling_24": 35.6, "rolling_48": 34.2, "rolling_168": 33.8
  }'
```

**Response:**
```json
{
  "predicted_total_demand_kW": 52.3141,
  "input": { "hour": 9, "weekday": 0, "..." : "..." }
}
```

### Batch Prediction

```bash
curl -X POST https://hourly-building-energy-demand-prediction.onrender.com/predict_batch \
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

## 📁 Repo Structure

```
├── app.py                  # Flask app (UI + REST API)
├── model.pkl               # Trained LightGBM model artifact
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config for Render
├── templates/
│   └── index.html          # Web UI
└── README.md
```

---

## 🚀 Deployment

Deployed on [Render](https://render.com) using Docker (Singapore region).
