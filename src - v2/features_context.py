# src/features_context.py v2
"""
Enhanced contextual feature generator for Nifty Trend Indicator.

Changelog v2:
- Adds rolling linear regression slopes (via np.polyfit) for close, EMA10/20/50
- Adds rolling gradients for RSI, MACD, and ATR
- Drops columns with >50% NaN values
- Ensures all derived features are lagged by 1 day to prevent data leakage
- Saves contextual features to data/processed/features_context.parquet
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "processed", "features_daily.parquet")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed", "features_context.parquet")

# --------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------

def rolling_slope(series, window=10):
    """Compute rolling slope using linear regression (np.polyfit)."""
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
            continue
        y = series.iloc[i - window + 1 : i + 1].values
        x = np.arange(window)
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def safe_diff(series, periods=1):
    """Return difference while preserving index and avoiding lookahead."""
    return series.diff(periods=periods)

def safe_pct_change(series, periods=1):
    """Safe percentage change computation."""
    return series.pct_change(periods=periods)

# --------------------------------------------------------------
# Main feature generator
# --------------------------------------------------------------

def add_contextual_features(df):
    print("Adding contextual features...")

    # Rolling linear slopes
    for col in ["close", "ema_10", "ema_20", "ema_50"]:
        if col in df.columns:
            for w in [5, 10, 20]:
                df[f"slope_{col}_{w}"] = rolling_slope(df[col], window=w)

    # RSI, MACD, ATR gradients
    for col in ["rsi_14", "macd_diff", "atr_14"]:
        if col in df.columns:
            df[f"{col}_grad"] = safe_diff(df[col], periods=1)
            for w in [5, 10]:
                df[f"{col}_grad_mean_{w}"] = df[f"{col}_grad"].rolling(w).mean()

    # Volatility compression — rolling % change in ATR
    if "atr_14" in df.columns:
        df["atr_compression_5"] = safe_pct_change(df["atr_14"].rolling(5).mean())
        df["atr_compression_10"] = safe_pct_change(df["atr_14"].rolling(10).mean())

    # Lag everything by 1 day to avoid future leakage
    derived_cols = [c for c in df.columns if c not in ["date", "timestamp"]]
    df[derived_cols] = df[derived_cols].shift(1)

    # Drop columns with >50% NaN
    nan_ratio = df.isna().mean()
    drop_cols = nan_ratio[nan_ratio > 0.5].index.tolist()
    if drop_cols:
        print(f"Dropping {len(drop_cols)} columns (>50% NaN): {drop_cols[:5]}...")
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Drop first few NaN rows
    df = df.dropna().reset_index(drop=True)
    print(f"Contextual features generated. Shape: {df.shape}")

    return df

# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------

def main():
    print(f"Reading base features: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Rows: {len(df)}, Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    df_context = add_contextual_features(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_context.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved contextual features to {OUTPUT_PATH} — rows: {len(df_context)}")

if __name__ == "__main__":
    main()
