# src/predict_context.py v1
"""
Predict next-day trend and strength using contextual LightGBM models.

Workflow:
- Reads latest contextual features (features_context.parquet)
- Loads contextual models from models_context/
- Predicts Trend (Up/Down) and Strength (Weak/Medium/Strong)
- Outputs final signal with probabilities and interpretation
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models_context")

# -------------------------------------------------------------------
# Load latest contextual features
# -------------------------------------------------------------------
context_path = os.path.join(DATA_DIR, "features_context.parquet")
df = pd.read_parquet(context_path)
df = df.sort_values("timestamp").reset_index(drop=True)

latest_row = df.iloc[-1:]
latest_date = latest_row["timestamp"].iloc[0]
print(f"Using latest available data: {latest_date}")

# -------------------------------------------------------------------
# Load models and feature columns
# -------------------------------------------------------------------
model_trend = joblib.load(os.path.join(MODEL_DIR, "model_trend_context.pkl"))
model_strength = joblib.load(os.path.join(MODEL_DIR, "model_strength_context.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns_context.pkl"))

# Some older features may be missing — align automatically
available_cols = [c for c in feature_cols if c in df.columns]
if len(available_cols) < len(feature_cols):
    missing = set(feature_cols) - set(available_cols)
    print(f"⚠️ Missing {len(missing)} feature columns — skipped: {list(missing)[:5]}...")

X_latest = latest_row[available_cols].fillna(0)

# -------------------------------------------------------------------
# Predict Trend (binary) and Strength (3-class)
# -------------------------------------------------------------------
prob_up = model_trend.predict_proba(X_latest)[0, 1]
pred_trend = int(prob_up > 0.5)

prob_strength = model_strength.predict_proba(X_latest)[0]
pred_strength = int(np.argmax(prob_strength))

# -------------------------------------------------------------------
# Interpret strength category
# -------------------------------------------------------------------
strength_map = {0: "Weak", 1: "Medium", 2: "Strong"}
trend_map = {0: "Down", 1: "Up"}

signal = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "latest_data_date": str(latest_date),
    "predicted_trend": trend_map[pred_trend],
    "prob_up": round(prob_up, 3),
    "predicted_strength": strength_map[pred_strength],
    "strength_probabilities": {
        "Weak": round(prob_strength[0], 3),
        "Medium": round(prob_strength[1], 3),
        "Strong": round(prob_strength[2], 3),
    }
}

# -------------------------------------------------------------------
# Print clean summary
# -------------------------------------------------------------------
print("\n=== Next-Day Market Prediction ===")
print(f"Data up to: {signal['latest_data_date']}")
print(f"Prediction Date: {signal['date']}")
print(f"-> Trend: {signal['predicted_trend']}  (Prob Up: {signal['prob_up']})")
print(f"-> Strength: {signal['predicted_strength']}")
print(f"Probabilities: {signal['strength_probabilities']}")

# -------------------------------------------------------------------
# Optional: Save to artifacts for recordkeeping
# -------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
out_path = os.path.join(ARTIFACTS_DIR, f"prediction_context_{signal['date']}.json")

import json
with open(out_path, "w") as f:
    json.dump(signal, f, indent=2)

print(f"\n[OK] Prediction saved to: {out_path}")
