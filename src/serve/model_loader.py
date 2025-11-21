"""
Load model architecture and weights, and scaler. Return ready-to-use models for inference.
"""
import torch
from pathlib import Path
from src.models.nbeats import NBeats
from src.models.lstm import LSTMForecaster
from src.models.transformer import TransformerForecaster
import joblib
from src.config import MODEL_DIR
from src.config import ENCODER_LENGTH, HORIZONS

MODEL_DIR = Path(MODEL_DIR)

def load_scaler():
    scaler_path = MODEL_DIR / "scaler.joblib"
    return joblib.load(scaler_path)

def load_nbeats(path=MODEL_DIR/"nbeats.pth"):
    model = NBeats(ENCODER_LENGTH, len(HORIZONS), n_blocks=4, hidden_size=256, n_layers=3)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

def load_lstm(path=MODEL_DIR/"lstm.pth"):
    # infer input_size from scaler? assume same features as training
    from src.dataset import TimeSeriesWindows
    model = LSTMForecaster(input_size=8, hidden_size=128, num_layers=2, output_size=len(HORIZONS))
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

def load_transformer(path=MODEL_DIR/"transformer.pth"):
    model = TransformerForecaster(input_size=8, d_model=128, nhead=4, num_layers=2, output_size=len(HORIZONS))
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model
