# src/inspect_preds.py v3
"""
Inspect predictions for binary trend (Up vs Down) + Strength.

- Loads saved models from train.py v6.
- Computes probabilities for Up/Down.
- Applies thresholding (default 0.5, configurable via --threshold).
- Blends binary trend + strength + Sideways into human-readable categories:
    * Weak Up, Medium Up, Strong Up
    * Weak Down, Medium Down, Strong Down
    * Sideways (if original label_class == 1, or prob in "uncertain zone")
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

NON_FEATURES = [
    "label_raw", "label_class", "label_name", "label_class_bin",
    "strength_01", "strength_bin",
    "date", "timestamp", "adj_close",
    "open", "high", "low", "close", "volume"
]

def blended_label(trend, strength, prob_up, threshold, true_label_class):
    """
    Return human-readable blended category.
    - If original label_class==1 (Sideways), always "Sideways"
    - Else if prob_up between (0.5-thresh_margin, 0.5+thresh_margin) -> "No-trade"
    - Else blend trend (Up/Down) with strength.
    """
    if pd.notna(true_label_class) and int(true_label_class) == 1:
        return "Sideways"

    if prob_up >= threshold:
        t = "Up"
    elif prob_up <= (1 - threshold):
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

def main(threshold):
    # Load data + models
    df = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    clf_trend = joblib.load(os.path.join(MODELS_DIR, "lgbm_trend_binary.joblib"))
    clf_strength = joblib.load(os.path.join(MODELS_DIR, "lgbm_strength.joblib"))

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=feature_cols, index=X.index)

    # Predictions
    prob_up = clf_trend.predict_proba(X_imp)[:, 1]
    pred_trend = (prob_up >= threshold).astype(int)  # 0=Down, 1=Up
    pred_strength = clf_strength.predict(X_imp)

    # Build result DataFrame
    out = pd.DataFrame({
        "date": df["date"],
        "true_trend_bin": df["label_class_bin"],  # 0=Down,1=Up,NaN=Sideways
        "pred_trend_bin": pred_trend,
        "true_strength_bin": df["strength_bin"],
        "pred_strength_bin": pred_strength,
        "prob_up": prob_up,
    })

    out["blended"] = [
        blended_label(pt, ps, p, threshold, tlc)
        for pt, ps, p, tlc in zip(
            out["pred_trend_bin"], out["pred_strength_bin"], out["prob_up"], df["label_class"]
        )
    ]

    # Save outputs
    out_csv = os.path.join(ARTIFACTS, f"inspect_preds_binary_{threshold:.2f}.csv")
    out.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to {out_csv}")

    # Print quick diagnostics
    print("\n=== HEAD (first 20 rows) ===")
    print(out.head(20))

    print("\n=== Blended Value Counts ===")
    print(out["blended"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for Up classification (default=0.5). "
                             "E.g., 0.55 means only predict Up if prob_up >= 0.55.")
    args = parser.parse_args()
    main(args.threshold)
