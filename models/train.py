"""
models/train.py
Standalone training script. Saves model + selected feature list.
Run: python3 models/train.py --symbol RELIANCE.NS
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from data.data_loader        import YFinanceLoader
from data.data_cleaning      import DataCleaner
from features.feature_engineering import FeatureEngineer
from features.feature_selection   import FeatureSelector
from labeling.barriers       import select_label
from models.ensemble         import EnsembleModel
from models.evaluation       import Evaluator
from utils.config            import CFG
from utils.logger            import get_logger

log = get_logger(__name__)


def train(symbol: str):
    log.info(f"Training on {symbol}")
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

    X   = eng.build(df, benchmark=bench)
    atr = X["atr"].copy() if "atr" in X.columns else None
    c   = df["close"].reindex(X.index)
    atr_a = atr.reindex(X.index) if atr is not None else c * 0.005

    y = select_label(c, atr_a)
    y = y.dropna().astype(int)
    X = X.loc[y.index]

    sel   = FeatureSelector(top_k=30)
    X_sel = sel.fit_transform(X, y)

    model = EnsembleModel()
    folds = model.walk_forward(X_sel, y)
    Evaluator().summarise_folds(folds)
    model.fit(X_sel, y)
    model.save()

    # Save selected feature names for inference
    feat_path = os.path.join("models", "saved", "features.json")
    with open(feat_path, "w") as f:
        json.dump(sel.selected, f)
    log.info(f"Feature list saved → {feat_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="RELIANCE.NS")
    args = p.parse_args()
    train(args.symbol)
