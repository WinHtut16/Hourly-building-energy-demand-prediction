"""
train_model.py
--------------
Run this script ONCE locally to train and save the model.
It produces:  model.pkl   (the trained LightGBM regressor)
              scaler.pkl  (not used here but kept for extensibility)

Usage:
    python train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── 1. Load & merge CSVs ──────────────────────────────────────────────────────
FOLDER = "data"

df_list = []
for fname in os.listdir(FOLDER):
    if fname.endswith(".csv"):
        tmp = pd.read_csv(os.path.join(FOLDER, fname))
        tmp["source_file"] = fname
        df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
print(f"Merged shape: {df.shape}")

# ── 2. Clean ──────────────────────────────────────────────────────────────────
missing_pct  = df.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 70].index
df = df.drop(columns=cols_to_drop)
df = df.ffill().bfill()

# ── 3. Parse date & aggregate ─────────────────────────────────────────────────
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.set_index("Date").sort_index()

df_power = df.loc[:, df.columns.str.contains("kW")].copy()
df_power["total_demand"] = df_power.sum(axis=1)
df_hourly = df_power["total_demand"].resample("h").mean()

# ── 4. Feature engineering ───────────────────────────────────────────────────
df_feat = df_hourly.reset_index().copy()
df_feat.columns = ["Date", "total_demand"]
df_feat = df_feat.dropna()

df_feat["hour"]       = df_feat["Date"].dt.hour
df_feat["weekday"]    = df_feat["Date"].dt.weekday
df_feat["month"]      = df_feat["Date"].dt.month
df_feat["is_weekend"] = (df_feat["weekday"] >= 5).astype(int)
df_feat["hour_sin"]   = np.sin(2 * np.pi * df_feat["hour"] / 24)
df_feat["hour_cos"]   = np.cos(2 * np.pi * df_feat["hour"] / 24)
df_feat["lag_1"]      = df_feat["total_demand"].shift(1)
df_feat["lag_24"]     = df_feat["total_demand"].shift(24)
df_feat["lag_48"]     = df_feat["total_demand"].shift(48)
df_feat["lag_168"]    = df_feat["total_demand"].shift(168)
df_feat["rolling_24"] = df_feat["total_demand"].rolling(24).mean()
df_feat["rolling_48"] = df_feat["total_demand"].rolling(48).mean()
df_feat["rolling_168"]= df_feat["total_demand"].rolling(168).mean()

df_feat = df_feat.set_index("Date").dropna()

FEATURES = [
    "hour", "weekday", "month", "is_weekend",
    "hour_sin", "hour_cos",
    "lag_1", "lag_24", "lag_48", "lag_168",
    "rolling_24", "rolling_48", "rolling_168",
]
TARGET = "total_demand"

train = df_feat.loc[: "2019-06-30"]
test  = df_feat.loc["2019-07-01":]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ── 5. Train with best Optuna params (paste yours here) ───────────────────────
# Replace the dict below with your actual study.best_params if you ran tuning.
best_params = {
    "n_estimators"     : 700,
    "learning_rate"    : 0.05,
    "num_leaves"       : 80,
    "max_depth"        : 8,
    "min_child_samples": 30,
    "subsample"        : 0.85,
    "colsample_bytree" : 0.85,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "random_state"     : 42,
    "verbose"          : -1,
}

model = lgb.LGBMRegressor(**best_params)
model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
preds = model.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
mape  = 100 * np.mean(np.abs(preds - y_test) / np.where(y_test == 0, 1, y_test))

print(f"\nTest RMSE : {rmse:.3f}")
print(f"Test MAE  : {mae:.3f}")
print(f"Test MAPE : {mape:.2f}%")
print(f"Test R²   : {r2:.4f}")

# ── 7. Save model & metadata ──────────────────────────────────────────────────
artifact = {
    "model"   : model,
    "features": FEATURES,
    "metrics" : {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2},
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved → model.pkl")
