"""
Build sliding windows across all tickers and return PyTorch Dataset wrapper.
Also provides a scaler that saves per-feature scaler fit on train windows.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from src.config import DATA_FEATURES, ENCODER_LENGTH
import joblib
from pathlib import Path

class TimeSeriesWindows(Dataset):
    def __init__(self, df, encoder_length, feature_cols, target_cols, scaler=None, fit_scaler=False, scaler_path=None):
        self.encoder_length = encoder_length
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        X_list, y_list = [], []
        meta = []
        for ticker, g in df.groupby('ticker'):
            g = g.sort_values('time').reset_index(drop=True)
            n = len(g)
            for end_idx in range(encoder_length-1, n):
                start = end_idx - (encoder_length - 1)
                x = g.loc[start:end_idx, feature_cols].values
                y = g.loc[end_idx, target_cols].values.astype('float32')
                if x.shape[0] == encoder_length:
                    X_list.append(x)
                    y_list.append(y)
                    meta.append({'ticker': ticker, 'time': g.loc[end_idx,'time']})
        if not X_list:
            raise ValueError("No windows created. Check data and encoder_length.")
        self.X = np.stack(X_list).astype('float32')
        self.y = np.stack(y_list).astype('float32')
        self.meta = pd.DataFrame(meta)
        # scaler
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
        if fit_scaler:
            flat = self.X.reshape(-1, self.X.shape[-1])
            self.scaler.fit(flat)
            if scaler_path:
                Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.scaler, scaler_path)
        # transform
        flat = self.X.reshape(-1, self.X.shape[-1])
        flat_s = self.scaler.transform(flat)
        self.X = flat_s.reshape(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def info(self):
        return {"n_samples": len(self), "encoder_length": self.encoder_length, "n_features": len(self.feature_cols)}
