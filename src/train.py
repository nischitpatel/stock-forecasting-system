"""
Train script for universal model. Use --model [nbeats|lstm|transformer]
Saves model to models/<model>.pth and scaler to models/scaler.joblib
"""
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import DATA_FEATURES, MODEL_DIR, ENCODER_LENGTH, HORIZONS, BATCH_SIZE, EPOCHS, LR
from src.dataset import TimeSeriesWindows
from src.models.nbeats import NBeats
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster
from src.utils import set_seed
import pandas as pd

def train(model_name="nbeats"):
    set_seed(42)
    df = pd.read_csv(DATA_FEATURES, parse_dates=['time'])
    feature_cols = ['log_price','volume','ret_mean_5','ret_std_5','ret_mean_21','ret_std_21','dow','month']
    target_cols = [f'log_return_{h}' for h in HORIZONS]
    dataset = TimeSeriesWindows(df, ENCODER_LENGTH, feature_cols, target_cols, fit_scaler=True, scaler_path=str(MODEL_DIR/'scaler.joblib'))
    # splits
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.12, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    # instantiate model
    input_size = len(feature_cols)
    if model_name == "nbeats":
        model = NBeats(ENCODER_LENGTH, len(HORIZONS), n_blocks=4, hidden_size=256, n_layers=3)
    elif model_name == "lstm":
        model = LSTMForecaster(input_size, hidden_size=128, num_layers=2, output_size=len(HORIZONS))
    elif model_name == "transformer":
        model = TransformerForecaster(input_size, d_model=128, nhead=4, num_layers=2, output_size=len(HORIZONS))
    else:
        raise ValueError("Invalid model name")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = torch.nn.MSELoss()
    best_val = float('inf'); best_epoch = 0; best_state=None
    for epoch in range(1, EPOCHS+1):
        model.train()
        total=0; train_loss=0
        for xb, yb in train_loader:
            xb = torch.tensor(xb).to(device)
            yb = torch.tensor(yb).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()*xb.size(0)
            total += xb.size(0)
        train_loss /= total
        # val
        model.eval()
        total=0; val_loss=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = torch.tensor(xb).to(device)
                yb = torch.tensor(yb).to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()*xb.size(0)
                total += xb.size(0)
        val_loss /= total
        print(f"[{model_name}] Epoch {epoch} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state, MODEL_DIR/f"{model_name}.pth")
        if epoch - best_epoch > 6:
            print("Early stopping at epoch", epoch)
            break
    # save scaler (already saved)
    print("Done. Best val:", best_val, "epoch", best_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nbeats', choices=['nbeats','lstm','transformer'])
    args = parser.parse_args()
    train(args.model)
