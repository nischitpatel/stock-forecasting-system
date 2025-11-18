from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE

# load yaml
cfg = yaml.safe_load(open(BASE / "src" / "config.yaml" if BASE.joinpath("src","config.yaml").exists() else BASE / "src" / "config.yaml"))

# paths
DATA_RAW = BASE / "data" / "raw.csv"
DATA_PROCESSED = BASE / "data" / "processed.csv"
DATA_FEATURES = BASE / "data" / "processed_features.csv"
MODEL_DIR = BASE / "models"
PRED_DIR = BASE / "predictions"
SIGNAL_DIR = BASE / "signals"
RESULTS = BASE / "results"

# create dirs
for p in [MODEL_DIR, PRED_DIR, SIGNAL_DIR, RESULTS, BASE / "data"]:
    p.mkdir(parents=True, exist_ok=True)

# load parameters
TICKERS = cfg.get("tickers", ["AAPL","MSFT","AMZN"])
START_DATE = cfg.get("start_date", "2018-01-01")
END_DATE = cfg.get("end_date", "2024-12-31")
ENCODER_LENGTH = cfg.get("encoder_length", 180)
HORIZONS = cfg.get("horizons", [30,60,90])
BATCH_SIZE = cfg.get("batch_size", 64)
EPOCHS = cfg.get("epochs", 40)
LR = cfg.get("lr", 1e-3)
SEED = cfg.get("seed", 42)
TRANSACTION_COST = cfg.get("transaction_cost", 0.0005)
SLIPPAGE = cfg.get("slippage", 0.0005)
TOP_K = cfg.get("top_k", 10)
