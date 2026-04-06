# ⚡ Energy Demand Forecasting API

Hourly building energy demand prediction using LightGBM, served with a web UI and REST API, deployed on Render.

**🌐 Live Demo:** https://hourly-building-energy-demand-prediction.onrender.com/

> **Note:** Free tier sleeps after 15 min of inactivity — first request may take ~30s to wake up.

---

## 🖥️ Web Interface

Visit the live URL to access the interactive prediction UI. No coding required.

- Pick any **date and time** from the calendar
- All 13 features are **calculated automatically** from historical averages
- Lag and rolling values are **editable** if you have actual meter readings
- Click **Predict** to get the forecasted demand instantly

---

## 📊 Dataset

The dataset used for training this model is publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/claytonmiller/cubems-smart-building-energy-and-iaq-data

Due to size constraints, the dataset is not included in this repository.  
Please download it manually and place it in the `data/` directory before running the notebook.

---

## 📊 Model Comparison

Three models were evaluated on the same test set (Jul 2019 onwards) to justify the final model choice.

| Model | R² | RMSE (kW) | MAE (kW) | MAPE (%) | Selected |
|---|---|---|---|---|---|
| Linear Regression | 0.8240 | 10.653 | 6.535 | 24.57% | ✗ |
| LightGBM (default) | 0.9387 | 6.286 | 2.857 | 9.46% | ✗ |
| **LightGBM (Optuna tuned)** | **0.9528** | **5.517** | **2.723** | **9.23%** | **✓** |

LightGBM tuned outperforms Linear Regression by **48% on RMSE** and **35% on R²**, confirming that energy demand patterns are strongly non-linear and that hyperparameter tuning provides meaningful additional gains.

---

## ✅ Final Model Performance

| Metric | Value |
|--------|-------|
| R²     | 0.9528 |
| RMSE   | 5.517 kW |
| MAE    | 2.723 kW |
| MAPE   | 9.23% |

### Cross-Validation (TimeSeriesSplit — 5 Fold)

| Metric | Mean ± Std |
|--------|------------|
| RMSE   | 5.375 ± 0.961 kW |
| MAE    | 3.426 ± 0.784 kW |
| R²     | 0.9478 ± 0.0223 |

---

## 🤖 Model Details

- **Algorithm:** LightGBM Regressor
- **Tuning:** Optuna (50 trials, TPE sampler, TimeSeriesSplit inner loop)
- **Training period:** Jul 2018 → Jun 2019
- **Test period:** Jul 2019 onwards

### Best Hyperparameters (Optuna)

| Parameter | Value |
|-----------|-------|
| n_estimators | 498 |
| learning_rate | 0.0615 |
| num_leaves | 39 |
| max_depth | 7 |
| min_child_samples | 44 |
| subsample | 0.710 |
| colsample_bytree | 0.764 |
| reg_alpha | 1.81e-06 |
| reg_lambda | 6.44e-05 |

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
  "input": { "hour": 9, "weekday": 0, "...": "..." }
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
├── app.py                          # Flask app (UI + REST API)
├── model.pkl                       # Trained LightGBM model artifact
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container config for Render
├── templates/
│   └── index.html                  # Web UI
├── notebooks/
│   └── energy_demand_forecasting.ipynb   # ML workflow (EDA, feature engineering, training)
└── README.md
```

---

## 🚀 Deployment

Deployed on [Render](https://render.com) using Docker (Singapore region).
