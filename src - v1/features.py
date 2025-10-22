# src/features.py
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_PATH = os.path.join(RAW_DIR, "nifty_daily.csv")
OUT_PATH = os.path.join(PROCESSED_DIR, "features_daily.parquet")


def compute_features(df):
    """
    Compute features *safe for prediction*:
    - All features are computed using information available BEFORE the day being predicted.
    - That is achieved by shifting the raw price/volume series by 1 (t-1) and computing
      rolling / ratio / EMA / ATR based on those lagged series.
    - We preserve original raw columns (open/high/low/close/volume) in the output so
      labeling.py can still compute labels from the raw day's open/close.
    """

    # Ensure numeric for raw price columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # PREP: create t-1 series (previous day's values) and t-2 where needed
    prev_close = df['close'].shift(1) if 'close' in df.columns else pd.Series(np.nan, index=df.index)
    prev_open = df['open'].shift(1) if 'open' in df.columns else pd.Series(np.nan, index=df.index)
    prev_high = df['high'].shift(1) if 'high' in df.columns else pd.Series(np.nan, index=df.index)
    prev_low = df['low'].shift(1) if 'low' in df.columns else pd.Series(np.nan, index=df.index)
    prev_volume = df['volume'].shift(1) if 'volume' in df.columns else pd.Series(np.nan, index=df.index)
    prev_close_prev = df['close'].shift(2) if 'close' in df.columns else pd.Series(np.nan, index=df.index)

    # -------------------------
    # Basic returns (lagged)
    # ret_1 = previous day's close pct change (i.e., return realized on t-1).
    df['ret_1'] = df['close'].pct_change(1, fill_method=None).shift(1)
    df['logret_1'] = np.log1p(df['ret_1']).fillna(0.0)

    # -------------------------
    # Ranges and ratios computed on previous day's prices
    prev_range = (prev_high - prev_low)
    df['range'] = prev_range  # previous day's high - low
    df['range_pct'] = prev_range / prev_open.replace(0, np.nan)

    # close_open_pct: previous day's close vs previous day's open
    df['close_open_pct'] = (prev_close - prev_open) / prev_open.replace(0, np.nan)

    # hl_pct: previous day's (high-low) relative to previous day's close
    df['hl_pct'] = prev_range / prev_close.replace(0, np.nan)

    # Rolling stats for previous-day range
    df['range_sma_5'] = prev_range.rolling(5).mean()
    df['range_sma_10'] = prev_range.rolling(10).mean()
    df['range_std_10'] = prev_range.rolling(10).std()

    # -------------------------
    # ATR (14) computed up to t-1 (so ATR_t uses data through t-1)
    # TR for the day t-1 should use high_{t-1}, low_{t-1}, close_{t-2}
    hl = prev_high - prev_low
    hc = (prev_high - prev_close_prev).abs()
    lc = (prev_low - prev_close_prev).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # Small safeguard
    df['atr_14'] = df['atr_14'].replace([np.inf, -np.inf], np.nan)

    # -------------------------
    # Moving averages computed on prev_close (so they don't see today's close)
    df['sma_50'] = prev_close.rolling(50).mean()
    df['ema_10'] = prev_close.ewm(span=10, adjust=False).mean()
    df['ema_20'] = prev_close.ewm(span=20, adjust=False).mean()
    df['ema_50'] = prev_close.ewm(span=50, adjust=False).mean()
    df['ema_cross_diff'] = df['ema_10'] - df['ema_50']

    # Distances to EMAs (relative to EMA)
    df['dist_ema10'] = (prev_close - df['ema_10']) / df['ema_10'].replace(0, np.nan)
    df['dist_ema50'] = (prev_close - df['ema_50']) / df['ema_50'].replace(0, np.nan)

    # -------------------------
    # Volume features (lagged)
    df['volume_sma_10'] = prev_volume.rolling(10).mean()
    df['volume_ratio'] = prev_volume / df['volume_sma_10'].replace(0, np.nan)

    # -------------------------
    # Momentum-ish / slope features computed on prev_close
    # slope_10: percent change of prev_close vs prev_close 10 periods before (i.e. ends at t-1)
    df['slope_10'] = (prev_close - prev_close.shift(10)) / prev_close.shift(10).replace(0, np.nan)
    df['slope_20'] = (prev_close - prev_close.shift(20)) / prev_close.shift(20).replace(0, np.nan)

    # -------------------------
    # Distance/ratios using lagged moving averages or indicators
    # (already done above: dist_ema10, dist_ema50)

    # -------------------------
    # Ensure technical columns exist and are safe (shift them by 1 if present,
    # otherwise create NaN placeholders). This prevents using a same-day RSI, etc.
    tech_cols = ['rsi_14', 'adx_14', 'macd_diff', 'bollinger_band_width', 'bb_pct', 'stoch_oscillator']
    for col in tech_cols:
        if col in df.columns:
            # shift in-place so col now represents its value available at prediction time
            df[col] = df[col].shift(1)
        else:
            df[col] = np.nan

    # If bollinger components were not present, we could compute them from prev_close (optional)
    # but for now we keep placeholders to match earlier pipeline.

    # -------------------------
    # Extra features (keep original naming but ensure they are lagged)
    # If these columns exist already (e.g. computed in raw), shift them.
    for col in ['macd', 'bb_pct', 'bb_upper', 'bb_lower']:
        if col in df.columns:
            df[col] = df[col].shift(1)

    # -------------------------
    # Safety: replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def main():
    print(f"Reading raw CSV: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Detect datetime column
    datetime_col = None
    for candidate in ['date', 'timestamp', 'Date', 'Datetime']:
        if candidate in df.columns:
            datetime_col = candidate
            break
    if datetime_col is None:
        raise ValueError(f"No date/timestamp column found in {RAW_PATH}. Columns available: {df.columns.tolist()}")

    # Normalize to "date" column and sort
    df['date'] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    print("Total rows read:", len(df))

    df = compute_features(df)

    # Save parquet (we keep original raw columns plus the new lagged features)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Features saved to {OUT_PATH} — rows: {len(df)}")


if __name__ == "__main__":
    main()
