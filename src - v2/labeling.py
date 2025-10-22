# src/labeling.py v5
"""
Labeling script for Nifty features.

Changelog v5:
- Fixed true data leakage:
  • Keeps OHLCV columns (open, high, low, close, adj_close, volume) UNLAGGED
    for label calculation.
  • Lags only derived / indicator / return features by 1 day.
  • Prevents model from seeing same-day close/returns when predicting labels.
- Prints exact number of lagged columns for transparency.

Purpose:
- Reads processed features parquet.
- Creates trend labels (multi-class or binary).
- Creates strength (continuous + binned).
- Saves labeled parquet.
"""

import os
import pandas as pd
import numpy as np
import argparse


def add_labels(df: pd.DataFrame, atr_frac: float = 0.25, binary_trend: bool = False) -> pd.DataFrame:
    """
    Add trend and strength labels to dataframe.
    Uses today's close and yesterday's ATR for label logic.
    """
    df = df.copy()

    # --- Trend labeling ---
    df["label_raw"] = np.nan
    # Up: today's close sufficiently above yesterday's close
    df.loc[
        df["close"] > df["close"].shift(1) * (1 + atr_frac * df["atr_14"].shift(1) / df["close"].shift(1)),
        "label_raw",
    ] = 1
    # Down: today's close sufficiently below yesterday's close
    df.loc[
        df["close"] < df["close"].shift(1) * (1 - atr_frac * df["atr_14"].shift(1) / df["close"].shift(1)),
        "label_raw",
    ] = -1
    # Remaining = sideways
    df.loc[df["label_raw"].isna(), "label_raw"] = 0
    df["label_raw"] = df["label_raw"].astype(int)

    if binary_trend:
        # 0=Down, 1=Up, NaN=Sideways
        df["label_class_bin"] = np.where(
            df["label_raw"] == -1, 0,
            np.where(df["label_raw"] == 1, 1, np.nan)
        )

    # Map to classes 0/1/2
    mapping = {-1: 0, 0: 1, 1: 2}
    df["label_class"] = df["label_raw"].map(mapping)
    df["label_name"] = df["label_class"].map({0: "Down", 1: "Sideways", 2: "Up"})

    # --- Strength labeling ---
    df["strength_01"] = df["close"].pct_change().abs().fillna(0)
    if df["strength_01"].max() > 0:
        df["strength_01"] = df["strength_01"] / df["strength_01"].max()
    else:
        df["strength_01"] = 0
    df["strength_bin"] = pd.qcut(df["strength_01"], q=3, labels=[0, 1, 2]).astype(int)

    return df


def main(input_path, output_path, atr_frac=0.25, binary_trend=False):
    # --- Load features ---
    df = pd.read_parquet(input_path)
    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Reading features: {input_path}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Using atr_frac = {atr_frac}, binary_trend = {binary_trend}")

    # --- Separate raw OHLCV columns from derived features ---
    ohlc_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    ignore_cols = ["date", "timestamp"] + ohlc_cols
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in ignore_cols]

    # --- Lag only derived features ---
    df_lagged = df.copy()
    df_lagged[feature_cols] = df_lagged[feature_cols].shift(1)
    print(f"Lagged {len(feature_cols)} derived feature columns by 1 day "
          f"(examples: {feature_cols[:6] if len(feature_cols) > 0 else 'None'})")

    # --- Compute labels using unlagged OHLC data ---
    df_out = add_labels(df_lagged, atr_frac=atr_frac, binary_trend=binary_trend)

    # --- Diagnostics ---
    if not binary_trend:
        print("\n--- Label diagnostics ---")
        print("label_raw counts (-1=Down,0=Sideways,1=Up):")
        print(df_out["label_raw"].value_counts())
        print("\nlabel_class counts (0=Down,1=Sideways,2=Up):")
        print(df_out["label_class"].value_counts())
    else:
        print("\nBinary Trend Labels (0=Down,1=Up, NaN=Sideways) counts:")
        print(df_out["label_class_bin"].value_counts(dropna=False))

    print("\nStrength bin counts (0=Weak,1=Medium,2=Strong):")
    print(df_out["strength_bin"].value_counts())

    # --- Save labeled dataset ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"\nLabeled dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_daily.parquet"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_daily.parquet"))
    parser.add_argument("--atr_frac", type=float, default=0.25)
    parser.add_argument("--binary_trend", action="store_true")
    args = parser.parse_args()

    main(args.input, args.output, args.atr_frac, args.binary_trend)
