"""
app.py
------
Flask app — serves a UI at / and a REST API at /api, /predict, /predict_batch
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# ── Load model artifact ────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model    = artifact["model"]
FEATURES = artifact["features"]
metrics  = artifact["metrics"]

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)


def validate_input(data: dict):
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return None, f"Missing fields: {missing}"
    try:
        row = {f: float(data[f]) for f in FEATURES}
    except (ValueError, TypeError) as e:
        return None, f"All fields must be numeric. Error: {e}"
    return row, None


# ── UI ────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", metrics=metrics)


# ── API routes ────────────────────────────────────────────────────────────────
@app.route("/api", methods=["GET"])
def health():
    return jsonify({
        "status" : "ok",
        "model"  : "LightGBM Energy Demand Forecaster",
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "usage"  : {
            "predict_single": "POST /predict  with JSON body containing feature values",
            "predict_batch" : "POST /predict_batch  with JSON body {\"records\": [...]}",
            "list_features" : "GET  /features",
        }
    })


@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "required_features": FEATURES,
        "descriptions": {
            "hour"       : "Hour of day (0-23)",
            "weekday"    : "Day of week (0=Monday, 6=Sunday)",
            "month"      : "Month of year (1-12)",
            "is_weekend" : "1 if Saturday/Sunday, else 0",
            "hour_sin"   : "sin(2pi x hour / 24) - cyclic hour encoding",
            "hour_cos"   : "cos(2pi x hour / 24) - cyclic hour encoding",
            "lag_1"      : "Total demand 1 hour ago (kW)",
            "lag_24"     : "Total demand 24 hours ago (kW)",
            "lag_48"     : "Total demand 48 hours ago (kW)",
            "lag_168"    : "Total demand 168 hours ago / same hour last week (kW)",
            "rolling_24" : "24-hour rolling mean demand (kW)",
            "rolling_48" : "48-hour rolling mean demand (kW)",
            "rolling_168": "168-hour rolling mean demand (kW)",
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    row, err = validate_input(data)
    if err:
        return jsonify({"error": err}), 422

    X    = np.array([[row[f] for f in FEATURES]])
    pred = float(model.predict(X)[0])

    return jsonify({
        "predicted_total_demand_kW": round(pred, 4),
        "input": row,
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True, silent=True)
    if data is None or "records" not in data:
        return jsonify({"error": "Body must be {\"records\": [...]}"}), 400

    records = data["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "\"records\" must be a non-empty list"}), 422

    cleaned, errors = [], []
    for i, rec in enumerate(records):
        row, err = validate_input(rec)
        if err:
            errors.append({"index": i, "error": err})
        else:
            cleaned.append(row)

    if errors:
        return jsonify({"errors": errors}), 422

    X     = np.array([[r[f] for f in FEATURES] for r in cleaned])
    preds = model.predict(X).tolist()

    return jsonify({
        "predictions": [
            {"index": i, "predicted_total_demand_kW": round(p, 4)}
            for i, p in enumerate(preds)
        ]
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)