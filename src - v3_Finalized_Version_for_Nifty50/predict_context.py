# src/predict_context.py v2.1
"""
Predict next-day trend and strength using contextual LightGBM models
with confidence-based classification logic.

Changelog:
- v2.1: Replace non-ASCII comparison symbols with ASCII for Windows console compatibility.
- v2.0: Added configurable probability thresholds (default: 0.45 / 0.55),
        Sideways output when model confidence is low, confidence score reporting.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

__version__ = "2.1"

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CONF_THRESH_LOW = 0.45
CONF_THRESH_HIGH = 0.55

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models_context")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

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

# Align columns (in case of minor feature drift)
available_cols = [c for c in feature_cols if c in df.columns]
if len(available_cols) < len(feature_cols):
    missing = set(feature_cols) - set(available_cols)
    print(f"[WARN] Missing {len(missing)} feature columns — skipped: {list(missing)[:5]}...")
X_latest = latest_row[available_cols].fillna(0)

# -------------------------------------------------------------------
# Predict Trend (binary) and Strength (3-class)
# -------------------------------------------------------------------
# If model_trend is binary with classes [0,1] and index 1 is 'Up'
# predict_proba returns [prob_down, prob_up] or [prob_class0, prob_class1]
proba = model_trend.predict_proba(X_latest)
# handle shape robustness
if proba.shape[1] == 2:
    prob_up = float(proba[0, 1])
else:
    # If model has classes ordering, try to find index of 'Up' (2) or label 1
    # fallback: assume last column is Up
    prob_up = float(proba[0, -1])

prob_down = 1.0 - prob_up

# Confidence-based classification
if prob_up >= CONF_THRESH_HIGH:
    trend_label = "Up"
elif prob_up <= CONF_THRESH_LOW:
    trend_label = "Down"
else:
    trend_label = "Sideways"

# Confidence score (0–1)
confidence_score = abs(prob_up - 0.5) * 2  # how far from neutral

# Strength prediction
prob_strength = model_strength.predict_proba(X_latest)[0]
pred_strength = int(np.argmax(prob_strength))
strength_map = {0: "Weak", 1: "Medium", 2: "Strong"}
strength_label = strength_map.get(pred_strength, "Unknown")

# -------------------------------------------------------------------
# Build final signal dictionary
# -------------------------------------------------------------------
signal = {
    "version": __version__,
    "date": datetime.now().strftime("%Y-%m-%d"),
    "latest_data_date": str(latest_date),
    "predicted_trend": trend_label,
    "prob_up": round(float(prob_up), 3),
    "prob_down": round(float(prob_down), 3),
    "confidence_score": round(float(confidence_score), 3),
    "predicted_strength": strength_label,
    "strength_probabilities": {
        "Weak": round(float(prob_strength[0]), 3),
        "Medium": round(float(prob_strength[1]), 3),
        "Strong": round(float(prob_strength[2]), 3),
    }
}

# -------------------------------------------------------------------
# Console Output
# -------------------------------------------------------------------
print("\n=== Next-Day Market Prediction ===")
print(f"Data up to: {signal['latest_data_date']}")
print(f"Prediction Date: {signal['date']}")
print(f"Model Version: {__version__}")
print(f"-> Trend: {signal['predicted_trend']}  (Prob Up: {signal['prob_up']}, Prob Down: {signal['prob_down']})")
print(f"-> Confidence: {signal['confidence_score']}")
print(f"-> Strength: {signal['predicted_strength']}")
print(f"Strength Probabilities: {signal['strength_probabilities']}")

# -------------------------------------------------------------------
# Save JSON output
# -------------------------------------------------------------------
out_path = os.path.join(ARTIFACTS_DIR, f"prediction_context_{signal['date']}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(signal, f, indent=2)

print(f"\n[OK] Prediction saved to: {out_path}")
print(f"[INFO] Confidence-based classification thresholds: Down <= {CONF_THRESH_LOW}, Up >= {CONF_THRESH_HIGH}")

# -------------------------------------------------------------------
# End of file
# -------------------------------------------------------------------
