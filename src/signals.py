"""
Convert predictions into portfolio signals (top-k, threshold, score-weighted).
Output CSV: time,ticker,signal
"""
import argparse
import pandas as pd
from pathlib import Path
from src.config import PRED_DIR, SIGNAL_DIR, TOP_K

def generate(preds_csv, strategy="topk", topk=TOP_K, threshold=0.02):
    df = pd.read_csv(preds_csv, parse_dates=['time'])
    df = df.sort_values('time')
    outs=[]
    for date, g in df.groupby('time'):
        g = g.copy()
        if strategy == "topk":
            g = g.sort_values('pred_logret_30', ascending=False).reset_index(drop=True)
            g['signal'] = 0
            if len(g) > 0:
                g.loc[:topk-1,'signal'] = 1
                g.loc[len(g)-topk:,'signal'] = -1
        elif strategy == "threshold":
            g['signal'] = 0
            g.loc[g['pred_logret_30']>threshold,'signal'] = 1
        elif strategy == "score_weight":
            s = g['pred_logret_30'].clip(lower=0)
            if s.sum()>0:
                w = s / s.sum()
                g['signal'] = w
            else:
                g['signal'] = 0
        outs.append(g[['time','ticker','signal']])
    out_df = pd.concat(outs).reset_index(drop=True)
    Path(SIGNAL_DIR).mkdir(parents=True, exist_ok=True)
    out_file = SIGNAL_DIR / f"signals_{strategy}.csv"
    out_df.to_csv(out_file, index=False)
    print("Saved signals to", out_file)
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default=str(PRED_DIR/'preds.csv'))
    parser.add_argument('--strategy', default='topk')
    args = parser.parse_args()
    generate(args.preds, args.strategy)
