"""
Data preprocessing:
- read raw csv
- compute log_price and multi-horizon log-return targets
- drop rows with missing targets
- save processed csv
"""
import argparse
import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_PROCESSED, HORIZONS
from pathlib import Path

def prepare(infile=DATA_RAW, outfile=DATA_PROCESSED):
    df = pd.read_csv(infile, parse_dates=['time'])
    df = df.sort_values(['ticker','time']).drop_duplicates(['ticker','time'])
    df = df.dropna(subset=['close'])
    df['log_price'] = np.log(df['close'].replace(0, np.nan)).fillna(method='ffill')
    for h in HORIZONS:
        df[f'log_price_t_plus_{h}'] = df.groupby('ticker')['log_price'].shift(-h)
        df[f'log_return_{h}'] = df[f'log_price_t_plus_{h}'] - df['log_price']
    df = df.dropna(subset=[f'log_return_{h}' for h in HORIZONS])
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print("Saved processed to", outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', default=str(DATA_RAW))
    parser.add_argument('--out', dest='outfile', default=str(DATA_PROCESSED))
    args = parser.parse_args()
    prepare(args.infile, args.outfile)
