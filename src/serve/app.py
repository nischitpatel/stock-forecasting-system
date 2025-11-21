"""
FastAPI service that exposes /predict?ticker=XXX&horizon=30
It uses saved models & scaler and reconstructs last ENCODER_LENGTH days from processed features csv.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.serve import model_loader
from src.config import DATA_FEATURES, ENCODER_LENGTH, HORIZONS
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import torch

app = FastAPI(title="Stock Forecasting API")
# load models/scaler
scaler = model_loader.load_scaler()
nbeats = model_loader.load_nbeats()
lstm = model_loader.load_lstm()
trans = model_loader.load_transformer()

class PredResp(BaseModel):
    ticker: str
    horizon: int
    preds: dict

@app.get("/predict", response_model=PredResp)
def predict(ticker: str, horizon: int = 30):
    if horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail="horizon must be one of " + ",".join(map(str,HORIZONS)))
    df = pd.read_csv(DATA_FEATURES, parse_dates=['time'])
    df = df[df['ticker']==ticker].sort_values('time').tail(ENCODER_LENGTH)
    if len(df) < ENCODER_LENGTH:
        raise HTTPException(status_code=400, detail=f"not enough history for {ticker}, need {ENCODER_LENGTH} rows")
    feature_cols = ['log_price','volume','ret_mean_5','ret_std_5','ret_mean_21','ret_std_21','dow','month']
    X = df[feature_cols].values.astype('float32')[None,:,:]  # (1,T,F)
    flat = X.reshape(-1, X.shape[-1])
    Xs = scaler.transform(flat).reshape(X.shape)
    # predict
    with torch.no_grad():
        p_n = nbeats(torch.tensor(Xs)).numpy()[0]
        p_l = lstm(torch.tensor(Xs)).numpy()[0]
        p_t = trans(torch.tensor(Xs)).numpy()[0]
    ensemble = (p_n + p_l + p_t)/3.0
    ix = HORIZONS.index(horizon)
    resp = {
        "nbeats": float(p_n[ix]),
        "lstm": float(p_l[ix]),
        "transformer": float(p_t[ix]),
        "ensemble": float(ensemble[ix])
    }
    return PredResp(ticker=ticker, horizon=horizon, preds=resp)
