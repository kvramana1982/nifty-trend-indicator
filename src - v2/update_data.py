"""
Update raw Nifty data (nifty_daily.csv) with the latest prices.

- Cleans existing CSV timestamps (drops floats/NaN).
- Normalizes to date-only (no time, no tz).
- Drops Dividends / Stock Splits.
- Forces OHLCV to numeric.
- src/update_data.py v8
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nifty_daily.csv")

def clean_timestamp(series):
    """Force column to pure datetime.date, drop invalids."""
    s = pd.to_datetime(series, errors="coerce")   # convert all
    s = s.dropna()                                # drop invalids
    return s.dt.date

def main():
    # Load existing
    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)

        # Normalize column name
        if "timestamp" not in df.columns:
            if "Date" in df.columns:
                df.rename(columns={"Date": "timestamp"}, inplace=True)
            elif "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            else:
                raise RuntimeError("Raw CSV missing timestamp column")

        # Clean timestamp
        df["timestamp"] = clean_timestamp(df["timestamp"])

        # Drop any rows where timestamp is still missing
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            last_date = datetime(2010, 1, 1).date()
        else:
            last_date = max(df["timestamp"])

        print(f"Loaded existing data. Last available date: {last_date}")
    else:
        df = pd.DataFrame()
        last_date = datetime(2010, 1, 1).date()
        print("No existing file found. Starting fresh from 2010-01-01")

    # Fetch new
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching new data from {start} to {end} ...")

    ticker = yf.Ticker("^NSEI")
    new_df = ticker.history(interval="1d", start=start, end=end)

    if new_df is None or new_df.empty:
        print("No new data available.")
        return

    new_df = new_df.reset_index()
    if "Date" in new_df.columns:
        new_df.rename(columns={"Date": "timestamp"}, inplace=True)

    new_df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    }, inplace=True)

    new_df["timestamp"] = clean_timestamp(new_df["timestamp"])

    # Drop Yahoo junk
    new_df = new_df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

    # Enforce numeric
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    # Merge
    df_all = pd.concat([df, new_df], ignore_index=True) if not df.empty else new_df
    df_all = df_all.dropna(subset=["timestamp"])   # final hard-clean
    df_all = df_all.drop_duplicates(subset=["timestamp"])
    df_all = df_all.sort_values("timestamp").reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df_all.to_csv(RAW_PATH, index=False)
    print(f"Updated dataset saved to {RAW_PATH}")
    print(f"Now covering: {min(df_all['timestamp'])} -> {max(df_all['timestamp'])}")

if __name__ == "__main__":
    main()
