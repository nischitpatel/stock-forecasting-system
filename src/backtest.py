"""
Simple vectorized backtester:
- loads prices and signals
- aligns by time/ticker (pivot)
- computes daily returns based on positions
- considers transaction costs by turnover
- returns stats dictionary plus equity series persisted
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import TRANSACTION_COST, SLIPPAGE, RESULTS

def backtest(prices_csv, signals_csv, out_pickle=None):
    prices = pd.read_csv(prices_csv, parse_dates=['time'])
    signals = pd.read_csv(signals_csv, parse_dates=['time'])
    prices = prices[['time','ticker','close']].sort_values(['ticker','time'])
    # pivot
    price_pivot = prices.pivot(index='time', columns='ticker', values='close').sort_index()
    sig_pivot = signals.pivot(index='time', columns='ticker', values='signal').reindex(price_pivot.index).fillna(0)
    # next-day returns
    returns = price_pivot.pct_change().fillna(0)
    # assume position at close day t yields returns on t+1
    pos = sig_pivot.shift(0).fillna(0)
    # daily strategy return normalized by notional: sum(pos*returns)/sum(abs(pos)) to simulate equal-dollar/normalized
    denom = pos.abs().sum(axis=1).replace(0, np.nan)
    strat_daily = (pos * returns).sum(axis=1) / denom
    strat_daily = strat_daily.fillna(0)
    # turnover cost (absolute change in positions)
    turnover = pos.diff().abs().sum(axis=1).fillna(0)
    strat_daily = strat_daily - turnover * TRANSACTION_COST
    equity = (1 + strat_daily).cumprod().fillna(method='ffill').fillna(1)
    # metrics
    stats = performance_stats(strat_daily)
    out = {"daily_returns": strat_daily, "equity": equity, "stats": stats}
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    out_file = out_pickle or (RESULTS / "backtest.pkl")
    pd.to_pickle(out, out_file)
    print("Backtest saved to", out_file)
    print("Stats:", stats)
    return out

def performance_stats(daily_ret):
    days = daily_ret.shape[0]
    compounded = (1 + daily_ret).prod()
    cagr = compounded ** (252/days) - 1 if days>0 else np.nan
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.02)/vol if vol>0 else np.nan
    eq = (1 + daily_ret).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return {"cagr": cagr, "annual_vol": vol, "sharpe": sharpe, "max_drawdown": mdd, "total_return": eq.iloc[-1]-1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', required=True)
    parser.add_argument('--signals', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    backtest(args.prices, args.signals, args.out)
