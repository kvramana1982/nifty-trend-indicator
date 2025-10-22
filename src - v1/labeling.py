# src/labeling.py
import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features_daily.parquet")
LABELED_PATH = os.path.join(PROCESSED_DIR, "labeled_daily.parquet")


def main():
    print(f"Reading features: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    # Ensure we have standardized "date" column
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in features dataset. Please re-run features.py.")

    # Sort chronologically to prevent leakage
    df = df.sort_values("date").reset_index(drop=True)

    # --- Labeling: open-to-close return ---
    if "open" not in df.columns or "close" not in df.columns:
        raise ValueError("Input features must contain 'open' and 'close' columns.")
    ret = (df["close"] - df["open"]) / df["open"]

    # --- Ensure ATR available ---
    if "atr_14" not in df.columns:
        raise ValueError("ATR feature missing. Please ensure features.py was run correctly or compute atr_14.")

    # --- Dynamic Sideways threshold: fraction of ATR ---
    atr_frac = 0.25
    atr_pct = df["atr_14"] / df["open"]
    sideways_mask = ret.abs() < (atr_frac * atr_pct)

    # Assign trend labels
    df["label_class"] = 0  # default sideways
    df.loc[(~sideways_mask) & (ret > 0), "label_class"] = 1   # Up
    df.loc[(~sideways_mask) & (ret < 0), "label_class"] = -1  # Down

    # --- Strength: volatility-adjusted & normalized ---
    raw_strength = (ret.abs() / df["atr_14"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Normalize 0..1 robustly
    min_val = raw_strength.min()
    max_val = raw_strength.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        norm_strength = raw_strength.clip(0, 1)
    else:
        norm_strength = ((raw_strength - min_val) / (max_val - min_val)).clip(0, 1)

    df["strength_01"] = norm_strength

    # --- Strength bins: tertiles (Weak / Medium / Strong) ---
    q1 = df["strength_01"].quantile(0.33)
    q2 = df["strength_01"].quantile(0.66)

    def strength_bin(val):
        if val < q1:
            return 0  # Weak
        elif val < q2:
            return 1  # Medium
        else:
            return 2  # Strong

    df["strength_bin"] = df["strength_01"].apply(strength_bin)

    # --- Diagnostics ---
    print("Label distribution (trend):")
    print(df["label_class"].value_counts())
    print("Strength (continuous) stats:")
    print(df["strength_01"].describe())
    print("Strength bin counts (0=Weak, 1=Medium, 2=Strong):")
    print(df["strength_bin"].value_counts())

    # Save
    df.to_parquet(LABELED_PATH, index=False)
    print(f"Labeled dataset saved to {LABELED_PATH}")


if __name__ == "__main__":
    main()
