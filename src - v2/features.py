# src/features.py v5
"""
Generate technical features for NIFTY index data.

Reads raw CSV from data/raw/nifty_daily.csv, computes features, and saves:
- features_daily.parquet (all features, including those used for training/analysis)
- features_safe.parquet (subset safe for prediction, no label leakage)
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nifty_daily.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def add_features(df):
    """Compute technical features for the dataset."""
    # returns
    df["ret_1"] = df["close"].pct_change(fill_method=None)
    df["ret_2"] = df["close"].pct_change(2, fill_method=None)
    df["ret_3"] = df["close"].pct_change(3, fill_method=None)
    df["ret_5"] = df["close"].pct_change(5, fill_method=None)

    df["logret_1"] = np.log(df["close"] / df["close"].shift(1))

    # ranges and volatilities
    df["range"] = df["high"] - df["low"]
    df["range_pct"] = df["range"] / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
    df["hl_pct"] = (df["high"] - df["low"]) / df["close"]

    # rolling stats
    df["range_sma_5"] = df["range"].rolling(5).mean()
    df["range_sma_10"] = df["range"].rolling(10).mean()
    df["range_std_10"] = df["range"].rolling(10).std()
    df["atr_14"] = (
        np.maximum(df["high"] - df["low"],
                   np.maximum(abs(df["high"] - df["close"].shift(1)),
                              abs(df["low"] - df["close"].shift(1))))
        .rolling(14).mean()
    )

    # moving averages
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # distance from MAs
    df["ema_cross_diff"] = df["ema_10"] - df["ema_50"]
    df["dist_ema10"] = df["close"] / df["ema_10"] - 1
    df["dist_ema50"] = df["close"] / df["ema_50"] - 1

    # volume-based
    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_10"]
    df["volume_ema_10"] = df["volume"].ewm(span=10, adjust=False).mean()
    df["volume_zscore_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()

    # slopes
    df["slope_10"] = df["close"].diff(10) / 10
    df["slope_20"] = df["close"].diff(20) / 20
    df["slope10_distema10"] = df["slope_10"] - df["dist_ema10"]
    df["slope20_distema50"] = df["slope_20"] - df["dist_ema50"]

    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_diff"] = ema_12 - ema_26

    # Bollinger Bands
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    df["bollinger_band_width"] = (2 * std_20) / sma_20
    df["bb_pct"] = (df["close"] - (sma_20 - 2 * std_20)) / (4 * std_20)
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20

    return df

def main():
    print(f"Reading raw CSV: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])

    print(f"Total rows read: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ✅ Drop unwanted Yahoo columns if present
    df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

    # ✅ Force numeric dtypes for OHLCV to avoid string issues
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Generate features
    df = add_features(df)

    # Save full feature set
    out_path = os.path.join(OUT_DIR, "features_daily.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Features saved to {out_path} — rows: {len(df)}")

    # Save safe subset (no raw OHLC/labels)
    non_features = ["timestamp", "date", "open", "high", "low", "close", "adj_close", "volume"]
    safe_df = df.drop(columns=non_features, errors="ignore")
    out_path_safe = os.path.join(OUT_DIR, "features_safe.parquet")
    safe_df.to_parquet(out_path_safe, index=False)
    print(f"Prediction-safe features saved to {out_path_safe} — rows: {len(safe_df)}")

if __name__ == "__main__":
    main()
