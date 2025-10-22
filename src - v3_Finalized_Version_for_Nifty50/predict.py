# src/predict.py v3
"""
Predict next-day trend & strength with thresholds.

- Loads latest binary trend and strength models.
- Supports asymmetric thresholds (--thresh_up, --thresh_down).
- Outputs tomorrow's prediction (blended label).
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_safe.parquet")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NON_FEATURES = [
    "label_raw", "label_class", "label_name", "label_class_bin",
    "strength_01", "strength_bin",
    "date", "timestamp", "adj_close",
    "open", "high", "low", "close", "volume"
]

def blended_label(prob_up, strength, thresh_up, thresh_down, true_label_class=None):
    """Return final signal label given prob_up and thresholds."""
    if true_label_class == 1:  # original sideways
        return "Sideways"

    if prob_up >= thresh_up:
        t = "Up"
    elif prob_up <= thresh_down:
        t = "Down"
    else:
        return "No-trade"

    if strength == 0:
        s = "Weak"
    elif strength == 1:
        s = "Medium"
    else:
        s = "Strong"
    return f"{s} {t}"

def main(thresh_up, thresh_down):
    # Load data
    df = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)
    df_labels = pd.read_parquet(LABELS_PATH).sort_values("date").reset_index(drop=True)

    # Load models
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    clf_trend = joblib.load(os.path.join(MODELS_DIR, "lgbm_trend_binary.joblib"))
    clf_strength = joblib.load(os.path.join(MODELS_DIR, "lgbm_strength.joblib"))

    # Prepare features
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=feature_cols, index=X.index)

    # Predict last available day
    last_idx = X_imp.index[-1]
    prob_up = clf_trend.predict_proba(X_imp.iloc[[last_idx]])[0, 1]
    pred_strength = clf_strength.predict(X_imp.iloc[[last_idx]])[0]

    # True label_class (for Sideways filtering)
    true_label_class = None
    if "label_class" in df_labels.columns:
        true_label_class = df_labels.iloc[last_idx]["label_class"]

    # Blended signal
    signal = blended_label(prob_up, pred_strength, thresh_up, thresh_down, true_label_class)

    print(f"\n=== Prediction for next day ({df.iloc[last_idx]['date'].date()}) ===")
    print(f"Prob Up: {prob_up:.3f}, Strength: {pred_strength}")
    print(f"Signal: {signal}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh_up", type=float, default=0.55,
                        help="Probability threshold for Up (default 0.55)")
    parser.add_argument("--thresh_down", type=float, default=0.45,
                        help="Probability threshold for Down (default 0.45)")
    args = parser.parse_args()
    main(args.thresh_up, args.thresh_down)
