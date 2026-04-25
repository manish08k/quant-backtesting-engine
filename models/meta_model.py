"""
models/meta_model.py
====================
Stacked meta-learner that re-scores base-model probabilities using
regime context features.  Fits ONLY on train-split data; never sees
holdout or future information.

Design
------
  Input features  : [prob, expansion, vol_spike, rel_bm, range_z]
  Target          : same y labels used for base model
  Estimator       : LGBMClassifier (shallow — max_depth=3, n_estimators=100)
                    Falls back to LogisticRegression if LightGBM unavailable.

No-leakage contract
-------------------
  • fit()     must be called only on the *train* split.
  • predict() is called on *test/holdout* split.
  • The meta-model is never re-fitted inside the backtest loop.

Usage
-----
    meta = MetaModel()
    meta.fit(X_train, y_train, probs_train)
    meta_probs = meta.predict(X_holdout, probs_holdout)
"""

from __future__ import annotations
import logging
import numpy  as np
import pandas as pd

log = logging.getLogger("pipeline")

_FEATURE_COLS = ["prob", "expansion", "vol_spike", "rel_bm", "range_z"]


def _get_col(
    source: pd.DataFrame | dict,
    col:    str,
    index=None,
) -> pd.Series:
    """
    Safely pull a column from a DataFrame or a dict-like.
    Returns zeros aligned to `index` if not found.
    """
    if isinstance(source, pd.DataFrame):
        if col in source.columns:
            return source[col]
        # Return zeros with same index
        idx = source.index if index is None else index
        return pd.Series(0.0, index=idx)
    # dict / SimpleNamespace path
    val = source.get(col, 0) if isinstance(source, dict) else 0
    idx = index if index is not None else pd.RangeIndex(1)
    return pd.Series(float(val), index=idx)


def _build_meta_features(
    X:    pd.DataFrame | dict,
    prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Construct the 5-column meta feature matrix.
    Works whether X is a DataFrame (live path) or a dict/SimpleNamespace
    (testing path).  Missing columns are filled with 0 — model degrades
    gracefully rather than crashing.
    """
    if isinstance(prob, np.ndarray):
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(prob))
        prob_s = pd.Series(prob, index=idx)
    else:
        prob_s = prob

    idx = prob_s.index

    feats = pd.DataFrame(index=idx)
    feats["prob"]      = prob_s.values
    feats["expansion"] = _get_col(X, "expansion",  idx).reindex(idx).fillna(0).values
    feats["vol_spike"] = _get_col(X, "vol_spike",  idx).reindex(idx).fillna(0).values
    feats["rel_bm"]    = _get_col(X, "rel_bm",     idx).reindex(idx).fillna(0).values
    feats["range_z"]   = _get_col(X, "range_z",    idx).reindex(idx).fillna(0).values

    return feats


class MetaModel:
    """
    Lightweight stacked meta-learner for probability re-calibration.

    Parameters
    ----------
    n_estimators : int
        Trees in LightGBM (default 100 — shallow to avoid overfitting meta layer)
    max_depth    : int
        Max tree depth (default 3)
    random_state : int

    Methods
    -------
    fit(X, y, prob)      → trains meta-model on train split only
    predict(X, prob)     → returns recalibrated probability array [0, 1]
    is_fitted            → bool property
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth:    int = 3,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self._model       = None
        self._fitted      = False

    # ── private: build estimator ──────────────────────────────────────────────

    def _make_estimator(self):
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators     = self.n_estimators,
                max_depth        = self.max_depth,
                learning_rate    = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                min_child_samples= 10,
                class_weight     = "balanced",
                verbose          = -1,
                random_state     = self.random_state,
                n_jobs           = -1,
            )
        except ImportError:
            log.warning(
                "MetaModel: LightGBM not installed — falling back to "
                "LogisticRegression."
            )
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C            = 0.5,
                max_iter     = 500,
                class_weight = "balanced",
                random_state = self.random_state,
            )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X:    pd.DataFrame,
        y:    pd.Series,
        prob: np.ndarray | pd.Series,
    ) -> "MetaModel":
        """
        Train the meta-model.

        Parameters
        ----------
        X    : feature DataFrame from the **train split only**
        y    : binary label Series (same rows as X)
        prob : base-model probability predictions on the train split
               (out-of-fold probabilities preferred to avoid overfitting)
        """
        meta_X = _build_meta_features(X, prob)
        n, pos = len(y), int((y == 1).sum())

        if n < 50 or pos < 10 or y.nunique() < 2:
            log.warning(
                f"MetaModel.fit: insufficient data (n={n}, pos={pos}). "
                "Skipping — meta-model will pass through base probs unchanged."
            )
            return self

        self._model = self._make_estimator()
        try:
            self._model.fit(meta_X.values, y.values)
            self._fitted = True
            log.info(
                f"MetaModel fitted on {n} samples  pos_rate={pos/n:.3f}  "
                f"features={list(meta_X.columns)}"
            )
        except Exception as exc:
            log.warning(f"MetaModel.fit failed: {exc} — will pass through base probs.")
            self._fitted = False

        return self

    def predict(
        self,
        X:    pd.DataFrame,
        prob: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Return recalibrated probabilities.

        If the meta-model was not successfully fitted, returns the base
        `prob` unchanged — the pipeline degrades gracefully.

        Parameters
        ----------
        X    : feature DataFrame (test/holdout split)
        prob : base-model probability predictions for the same rows
        """
        if isinstance(prob, pd.Series):
            prob_arr = prob.values
        else:
            prob_arr = np.asarray(prob, dtype=float)

        if not self._fitted or self._model is None:
            log.debug("MetaModel.predict: not fitted — returning base probs unchanged.")
            return prob_arr

        meta_X = _build_meta_features(X, prob)
        try:
            meta_prob = self._model.predict_proba(meta_X.values)[:, 1]
        except Exception as exc:
            log.warning(f"MetaModel.predict failed: {exc} — returning base probs.")
            return prob_arr

        # Blend: 0.40 meta + 0.60 base — meta adds information, base anchors it
        blended = 0.60 * prob_arr + 0.40 * meta_prob
        blended = np.clip(blended, 0.0, 1.0)

        log.info(
            f"MetaModel.predict: base_mean={prob_arr.mean():.4f}  "
            f"meta_mean={meta_prob.mean():.4f}  "
            f"blended_mean={blended.mean():.4f}  "
            f"blended_std={blended.std():.4f}"
        )
        return blended
