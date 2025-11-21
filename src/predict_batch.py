"""
Load a trained model and run batch prediction for all windows -> produce per-date,ticker predictions
"""
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_FEATURES, PRED_DIR, ENCODER_LENGTH, HORIZONS, MODEL_DIR
from src.dataset import TimeSeriesWindows
from src.models.nbeats import NBeats
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster
import joblib

def predict(model_path, model_name, out_path):
    df = pd.read_csv(DATA_FEATURES, parse_dates=['time'])
    feature_cols = ['log_price','volume','ret_mean_5','ret_std_5','ret_mean_21','ret_std_21','dow','month']
    target_cols = [f'log_return_{h}' for h in HORIZONS]
    scaler = joblib.load(MODEL_DIR/'scaler.joblib')
    dataset = TimeSeriesWindows(df, ENCODER_LENGTH, feature_cols, target_cols, scaler=scaler)
    X = dataset.X  # already scaled
    meta = dataset.meta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "nbeats":
        model = NBeats(ENCODER_LENGTH, len(HORIZONS), n_blocks=4, hidden_size=256, n_layers=3)
    elif model_name == "lstm":
        model = LSTMForecaster(len(feature_cols), hidden_size=128, num_layers=2, output_size=len(HORIZONS))
    elif model_name == "transformer":
        model = TransformerForecaster(len(feature_cols), d_model=128, nhead=4, num_layers=2, output_size=len(HORIZONS))
    else:
        raise ValueError("invalid model")
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()
    batch=256
    preds=[]
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch]).to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
    preds = np.vstack(preds)
    out_df = meta.copy()
    for idx,h in enumerate(HORIZONS):
        out_df[f'pred_logret_{h}'] = preds[:, idx]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Saved predictions to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', default='nbeats', choices=['nbeats','lstm','transformer'])
    parser.add_argument('--out', default=str(PRED_DIR/'preds.csv'))
    args = parser.parse_args()
    predict(args.model, args.name, args.out)
