"""
Add rolling features (mean/std) and calendar features.
"""
import argparse
import pandas as pd
from pathlib import Path
from src.config import DATA_PROCESSED, DATA_FEATURES

def add_features(infile=DATA_PROCESSED, outfile=DATA_FEATURES, windows=[5,10,21,63]):
    df = pd.read_csv(infile, parse_dates=['time'])
    df = df.sort_values(['ticker','time'])
    for w in windows:
        df[f'ret_mean_{w}'] = (
            df.groupby("ticker")["log_price"]
            .transform(lambda s: s.diff().rolling(w, min_periods=1).mean())
            .fillna(0)
        )
        # df[f'ret_mean_{w}'] = df.groupby('ticker')['log_price'].apply(lambda s: s.diff().rolling(w, min_periods=1).mean()).fillna(0)
        
        # df[f'ret_std_{w}'] = df.groupby('ticker')['log_price'].apply(lambda s: s.diff().rolling(w, min_periods=1).std()).fillna(0)
        df[f"ret_std_{w}"] = (
            df.groupby("ticker")["log_price"]
              .transform(lambda s: s.diff().rolling(w, min_periods=1).std())
              .fillna(0)
        )

    df['dow'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    print("Saved features to", outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', default=str(DATA_PROCESSED))
    parser.add_argument('--out', dest='outfile', default=str(DATA_FEATURES))
    args = parser.parse_args()
    add_features(args.infile, args.outfile)
