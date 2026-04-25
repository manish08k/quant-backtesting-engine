"""
regime.py  (NEW — fixes "regime column not found" warning)
===========================================================
Provides two things:

  1. add_regime_col(df) — causal, look-ahead-free function that adds a
     'regime' column to any OHLCV-style DataFrame.
     Call this inside feature_engineering / add_alpha_features BEFORE
     the data reaches the model.

  2. RegimeModel — wrapper used by ensemble.py / main.py.
     Fits a per-regime sub-model when the 'regime' column is present;
     falls back gracefully to the global model when it is not.

Regime classification (3 states)
----------------------------------
  0 = CHOPPY   — low directional energy, avoid trading
  1 = TRENDING — sustained move, full model output
  2 = VOLATILE — high energy but mean-reverting, reduce size

Detection method: rolling ADX proxy + volatility percentile.
All computations are strictly causal (only past bars used).

How to wire it in
-----------------
In feature_engineering.py / add_alpha_features():

    from models.regime import add_regime_col
    df = add_regime_col(df)   # adds 'regime' int column

In main.py (already exists, just needs the column present):

    regime_model = RegimeModel(base_model=ensemble)
    regime_model.fit(X_train, y_train, regime=train_df["regime"])
    probs = regime_model.predict_proba(X_holdout, regime=holdout_df["regime"])
"""

from __future__ import annotations
import logging
import numpy  as np
import pandas as pd
from typing import Optional

log = logging.getLogger("pipeline")

# ── Regime constants ──────────────────────────────────────────────────────────
CHOPPY   = 0
TRENDING = 1
VOLATILE = 2

_REGIME_NAMES = {CHOPPY: "CHOPPY", TRENDING: "TRENDING", VOLATILE: "VOLATILE"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Causal regime column builder
# ─────────────────────────────────────────────────────────────────────────────

def add_regime_col(
    df: pd.DataFrame,
    adx_window:      int   = 14,
    vol_window:      int   = 20,
    adx_trend_thr:   float = 25.0,   # ADX proxy > this → trending
    vol_spike_pct:   float = 80.0,   # vol percentile > this → volatile
) -> pd.DataFrame:
    """
    Add an integer 'regime' column to df.

    Requires columns: High, Low, Close  (case-insensitive).
    Safe to call with any column naming — falls back to flat regime=1 if
    OHLC not available (avoids breaking existing pipelines).

    Returns df with 'regime' column added in-place (copy-safe).
    """
    df = df.copy()

    # Normalise column names
    col_map = {c.lower(): c for c in df.columns}
    high_col  = col_map.get("high")
    low_col   = col_map.get("low")
    close_col = col_map.get("close")

    if not all([high_col, low_col, close_col]):
        log.warning("regime.add_regime_col: High/Low/Close not found — defaulting regime=1 (TRENDING)")
        df["regime"] = TRENDING
        return df

    H = df[high_col]
    L = df[low_col]
    C = df[close_col]

    # ── ADX proxy (Wilder smoothed DM / TR) ──────────────────────────────────
    # True Range
    tr = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    dm_plus  = (H - H.shift(1)).clip(lower=0)
    dm_minus = (L.shift(1) - L).clip(lower=0)
    # Zero out where the opposite DM is larger (standard Wilder rule)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0.0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0.0)

    # Wilder smoothing (EWM approximation)
    alpha = 1.0 / adx_window
    atr   = tr.ewm(alpha=alpha, adjust=False).mean()
    di_p  = 100 * dm_plus.ewm(alpha=alpha,  adjust=False).mean() / atr.replace(0, np.nan)
    di_m  = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    dx    = (100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)).fillna(0)
    adx   = dx.ewm(alpha=alpha, adjust=False).mean()

    # ── Volatility percentile (causal rolling) ────────────────────────────────
    log_ret  = np.log(C / C.shift(1)).abs()
    vol_pct  = log_ret.rolling(vol_window, min_periods=5).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]) * 100,
        raw=False,
    )

    # ── Classify ──────────────────────────────────────────────────────────────
    regime = pd.Series(CHOPPY, index=df.index, dtype=int)
    is_trending  = adx >= adx_trend_thr
    is_volatile  = vol_pct >= vol_spike_pct

    # Priority: volatile > trending > choppy
    regime[is_trending]             = TRENDING
    regime[is_trending & is_volatile] = VOLATILE  # high ADX + vol spike = breakout/reversal
    regime[~is_trending & is_volatile] = VOLATILE # low ADX + vol spike = news/shock

    df["regime"] = regime.values

    counts = regime.value_counts().to_dict()
    log.info(
        f"add_regime_col: CHOPPY={counts.get(CHOPPY,0)}  "
        f"TRENDING={counts.get(TRENDING,0)}  "
        f"VOLATILE={counts.get(VOLATILE,0)}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RegimeModel wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RegimeModel:
    """
    Wraps any sklearn-compatible base model and fits per-regime sub-models.

    If the 'regime' Series is missing or has < min_regime_bars in a regime,
    the global model is used — no silent failure, but no crash either.

    Usage
    -----
    rm = RegimeModel(base_model=EnsembleModel(...))
    rm.fit(X_train, y_train, regime=train_df["regime"])
    probs = rm.predict_proba(X_holdout, regime=holdout_df["regime"])
    """

    def __init__(
        self,
        base_model,
        min_regime_bars: int = 200,   # sub-model only fit if >= this many bars
    ):
        self.base_model      = base_model
        self.min_regime_bars = min_regime_bars
        self._regime_models: dict[int, object] = {}
        self._fitted_global  = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _clone_base(self):
        """Return a fresh copy of the base model class with same params."""
        try:
            from sklearn.base import clone
            return clone(self.base_model)
        except Exception:
            return type(self.base_model)()

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime: Optional[pd.Series] = None,
    ) -> "RegimeModel":
        """
        Fit global model always.
        Fit per-regime sub-models when regime Series is provided and
        each regime has enough bars.
        """
        # Global model
        self.base_model.fit(X, y)
        self._fitted_global = True
        log.info(f"RegimeModel: global model fitted on {len(X)} bars")

        if regime is None:
            log.warning("RegimeModel: no regime Series provided — global model only.")
            return self

        regime = regime.reindex(X.index)
        missing = regime.isna().sum()
        if missing > 0:
            log.warning(f"RegimeModel: {missing} NaN regime values — filling with TRENDING(1)")
            regime = regime.fillna(TRENDING)

        for r in [CHOPPY, TRENDING, VOLATILE]:
            mask = regime == r
            n    = int(mask.sum())
            if n < self.min_regime_bars:
                log.info(
                    f"RegimeModel: regime={_REGIME_NAMES[r]} has only {n} bars "
                    f"(< {self.min_regime_bars}) — using global model"
                )
                continue
            try:
                sub = self._clone_base()
                sub.fit(X[mask], y[mask])
                self._regime_models[r] = sub
                log.info(f"RegimeModel: sub-model fitted for {_REGIME_NAMES[r]}  n={n}")
            except Exception as e:
                log.warning(f"RegimeModel: sub-model fit failed for {_REGIME_NAMES[r]}: {e}")

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        X: pd.DataFrame,
        regime: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Route each bar to its regime sub-model if available; else global.
        Returns probability of class 1 as a 1-D numpy array.
        """
        if not self._fitted_global:
            raise RuntimeError("RegimeModel.predict_proba() called before fit().")

        probs = np.full(len(X), np.nan)

        if regime is None or not self._regime_models:
            # No regime info or no sub-models — use global everywhere
            probs = self.base_model.predict_proba(X)[:, 1]
            return probs

        regime = regime.reindex(X.index).fillna(TRENDING).astype(int)

        for r in [CHOPPY, TRENDING, VOLATILE]:
            mask = (regime == r).values
            if not mask.any():
                continue
            model = self._regime_models.get(r, self.base_model)
            probs[mask] = model.predict_proba(X.iloc[mask])[:, 1]

        # Safety: fill any remaining NaN with global
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            probs[nan_mask] = self.base_model.predict_proba(X.iloc[nan_mask])[:, 1]

        return probs

    # ── choppy gate ───────────────────────────────────────────────────────────

    @staticmethod
    def choppy_mask(regime: pd.Series) -> pd.Series:
        """
        Returns a boolean Series: True where regime == CHOPPY.
        Use this in signal_filter to suppress all signals in choppy periods.

        Example in main.py:
            choppy = RegimeModel.choppy_mask(holdout_df["regime"])
            prob[choppy] = 0.0   # zero out choppy-bar probs before filtering
        """
        return regime == CHOPPY
