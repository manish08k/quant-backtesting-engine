"""
feature_engineering.py — Event-driven alpha features. ≤12 features. Zero leakage.

CHANGES vs prior version
-------------------------
- Removed all sym_* dummy columns (cause of feature mismatch bug)
- symbol_id passed through as single LabelEncoded int column
- Kept exactly 12 event-based features with no redundancy
- regime composite feature retained
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("pipeline")


def add_alpha_features(df: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df.get("volume", pd.Series(1.0, index=df.index))

    bar_range = (h - l).replace(0, np.nan)
    hl = (bar_range / (c.abs() + 1e-9)).clip(0, 0.2)

    # 1. 1-bar return
    out["ret_1"] = c.pct_change(1)

    # 2. Breakout event — close > 20-bar prior high (shift prevents lookahead)
    hh_20 = h.rolling(20, min_periods=10).max().shift(1)
    out["breakout_event"] = (c > hh_20).astype(float)

    # 3. Volatility regime — compressed vs expanding
    vol_20  = hl.rolling(20, min_periods=10).std()
    vol_q30 = vol_20.rolling(50, min_periods=20).quantile(0.30)
    out["vol_shift"] = (vol_20 < vol_q30).astype(float)

    # 4. Momentum flip — short vs medium term direction disagree
    out["mom_flip"] = (np.sign(c.diff(3)) != np.sign(c.diff(10))).astype(float)

    # 5. Trend alignment — fast MA > slow MA
    ma_fast = c.rolling(10, min_periods=5).mean()
    ma_slow = c.rolling(30, min_periods=15).mean()
    out["trend_align"] = (ma_fast > ma_slow).astype(float)

    # 6. Volume shock — spike vs 20-bar mean
    vol_ma = v.rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["vol_shock"] = (v > 1.5 * vol_ma).astype(float)

    # 7. Expansion — current bar range vs compressed baseline
    compression = vol_20.replace(0, np.nan)
    out["expansion"] = (hl / (compression + 1e-6)).clip(0, 10)

    # 8. Range z-score
    hl_mean = hl.rolling(50, min_periods=20).mean()
    hl_std  = hl.rolling(50, min_periods=20).std().replace(0, np.nan)
    out["range_z"] = ((hl - hl_mean) / (hl_std + 1e-9)).clip(-5, 5)

    # 9. Relative strength vs benchmark (5-bar)
    try:
        bm_c = bm["close"].reindex(df.index, method="ffill")
        out["rel_bm"] = c.pct_change(5) - bm_c.pct_change(5)
    except Exception:
        out["rel_bm"] = 0.0

    # 10. Candle body
    out["body"] = ((c - o) / bar_range.replace(0, 1e-6)).clip(-1, 1)

    # 11. Raw hl (used downstream for ATR proxy)
    out["hl_range"] = hl

    # 12. Regime: trending AND NOT compressed
    out["regime"] = ((ma_fast > ma_slow) & (vol_20 >= vol_q30)).astype(float)

    # FIX: pass symbol_id through as single int column (NOT dummies)
    # symbol_id must already be LabelEncoded (int) before calling this fn
    if "symbol_id" in df.columns:
        out["symbol_id"] = df["symbol_id"].values

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    log.info(f"add_alpha_features: {len(out)} rows, {len(out.columns)} features")
    return out


class FeatureEngineer:
    def fit_transform(self, df: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
        return add_alpha_features(df, bm)

    def transform(self, df: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
        return add_alpha_features(df, bm)