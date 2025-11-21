"""
Fetch OHLCV data using AlphaVantage (NO rate limit errors like Yahoo).
Saves a single CSV with columns: time,ticker,open,high,low,close,volume
"""

import time
import requests
import pandas as pd
from pathlib import Path
import argparse
import sys
from src.config import TICKERS, START_DATE, END_DATE, DATA_RAW, ALPHA_VANTAGE_KEY


BASE_URL = "https://www.alphavantage.co/query"


def fetch_one_symbol(symbol, api_key):
    """Fetch full daily OHLCV for one ticker from AlphaVantage."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",  # full history
        "apikey": api_key,
    }

    print(f"Fetching {symbol}...")

    try:
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] HTTP error for {symbol}: {e}")
        return None

    if "Time Series (Daily)" not in data:
        print(f"[ERROR] No time series returned for {symbol}: {data}")
        return None

    ts = data["Time Series (Daily)"]
    rows = []

    for date, values in ts.items():
        rows.append({
            "time": date,
            "ticker": symbol,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": float(values["5. volume"]),
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    return df


def fetch_all(tickers=TICKERS, out=DATA_RAW):
    all_data = []

    for i, t in enumerate(tickers):

        df = fetch_one_symbol(t, ALPHA_VANTAGE_KEY)

        if df is None or df.empty:
            print(f"[WARN] Skipping {t}, no data.")
        else:
            all_data.append(df)

        # AlphaVantage free API: 5 requests per minute → sleep 15 sec to be safe
        if i < len(tickers) - 1:
            time.sleep(15)

    if not all_data:
        raise RuntimeError("No data downloaded.")

    final_df = pd.concat(all_data)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out, index=False)

    print(f"Saved all data to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_RAW))
    args = parser.parse_args()
    fetch_all(out=args.out)
