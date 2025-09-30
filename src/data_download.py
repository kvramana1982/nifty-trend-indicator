"""
download_daily_nifty: downloads historical NIFTY (daily) using yfinance.
Saves CSV to data/raw/nifty_daily.csv
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

def download_daily_nifty(start="1996-01-01", end=None, ticker="^NSEI"):
    """
    Downloads daily historical data for the given ticker (default NIFTY '^NSEI').
    start: yyyy-mm-dd
    end: yyyy-mm-dd or None (till today)
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    print(f"Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check network or ticker symbol.")
    df = df.reset_index()
    # standardize column names (lowercase)
    df.rename(columns={
        "Date": "timestamp", "Open":"open", "High":"high", "Low":"low",
        "Close":"close", "Adj Close":"adj_close", "Volume":"volume"
    }, inplace=True)
    # save CSV
    out_path = os.path.join(OUT_DIR, "nifty_daily.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return out_path

if __name__ == "__main__":
    download_daily_nifty()
