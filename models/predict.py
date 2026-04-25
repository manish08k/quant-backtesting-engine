"""
models/predict.py
Load saved model + feature list → predict on latest bars.
Run: python3 models/predict.py --symbol RELIANCE.NS
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from data.data_loader             import YFinanceLoader
from data.data_cleaning           import DataCleaner
from features.feature_engineering import FeatureEngineer
from models.ensemble              import EnsembleModel
from strategy.signal_filter       import SignalGenerator
from utils.logger                 import get_logger

log = get_logger(__name__)


def predict(symbol: str, n_bars: int = 5):
    feat_path = os.path.join("models", "saved", "features.json")
    if not os.path.exists(feat_path):
        raise FileNotFoundError("Run train.py first to generate features.json")

    with open(feat_path) as f:
        selected = json.load(f)

    model   = EnsembleModel.load()
    loader  = YFinanceLoader()
    cleaner = DataCleaner()
    eng     = FeatureEngineer()

    raw = loader.fetch(symbol)
    df  = cleaner.clean(raw, symbol)

    bench_raw = loader.fetch_benchmark()
    bench = pd.DataFrame()
    if not bench_raw.empty:
        bench = cleaner.clean(bench_raw, "NIFTY50")
        bi = bench.reindex(df.index, method="ffill").copy()
        if "volume" not in bi.columns:
            bi["volume"] = 1_000_000
        bench = bi

    X = eng.build(df, benchmark=bench)

    # Align to saved feature set
    missing = [c for c in selected if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X_sel = X[selected].tail(n_bars)

    prob    = model.predict_proba(X_sel)
    signals = SignalGenerator().generate(
        prob, features=X_sel, threshold=model.threshold
    )

    out = pd.DataFrame({
        "prob":   prob,
        "signal": signals,
        "close":  df["close"].reindex(X_sel.index),
    })
    print(out.to_string())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="RELIANCE.NS")
    p.add_argument("--bars",   type=int, default=5)
    args = p.parse_args()
    predict(args.symbol, args.bars)
