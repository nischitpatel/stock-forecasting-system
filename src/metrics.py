# helper metrics used by backtest etc.
import numpy as np
import pandas as pd

def sharpe_ratio(daily_returns, rf=0.02):
    days = len(daily_returns)
    if days == 0:
        return np.nan
    ar = (1 + daily_returns).prod() ** (252/days) - 1
    vol = daily_returns.std() * np.sqrt(252)
    return (ar - rf) / vol if vol > 0 else np.nan
