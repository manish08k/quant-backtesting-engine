"""
feature_selection.py — Fit once globally. Transform everywhere. No refit in folds.

CHANGES vs prior version
-------------------------
- selected_columns stored as sorted list for deterministic ordering
- transform() strictly reindexes to selected_columns (fill_value=0)
- assertion added before transform returns to catch shape mismatches early
- sym_* passthrough removed — symbol_id handled as single numeric column
- _PASSTHROUGH updated: symbol_id kept, hl_range kept, regime kept
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("pipeline")

_DEFAULT_TOP_K    = 10
_DEFAULT_MIN_FEAT = 6
_CORR_THRESHOLD   = 0.90

# Columns that must never be scaled — passed through raw
# FIX: removed sym_* pattern; symbol_id is now a single int passthrough
_PASSTHROUGH = {"symbol_id", "regime", "hl_range"}


def _make_importance_model(pos_weight: float):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            scale_pos_weight=pos_weight, verbose=-1, random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )


class FeatureSelector:
    def __init__(self, top_k=_DEFAULT_TOP_K, corr_threshold=_CORR_THRESHOLD,
                 min_features=_DEFAULT_MIN_FEAT):
        self.top_k          = top_k
        self.corr_threshold = corr_threshold
        self.min_features   = min_features
        self.selected_columns: list = []   # SINGLE SOURCE OF TRUTH — set at fit, never changed
        self.selected:         list = []   # alias kept for back-compat
        self._passthrough_cols: list = []
        self._alpha_cols:       list = []
        self.importance_df = pd.DataFrame()
        self._scaler       = StandardScaler()
        self._is_fitted    = False

    @staticmethod
    def _remove_dead(X: pd.DataFrame) -> pd.DataFrame:
        alive   = X.std() > 1e-6
        dropped = alive[~alive].index.tolist()
        if dropped:
            log.info(f"FeatureSelector: dropped dead: {dropped}")
        return X.loc[:, alive]

    @staticmethod
    def _corr_filter(X: pd.DataFrame, importances: pd.Series) -> list:
        if X.shape[1] < 2:
            return X.columns.tolist()
        corr  = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            for partner in upper.index[upper[col] > _CORR_THRESHOLD].tolist():
                worse = col if importances.get(col, 0) < importances.get(partner, 0) else partner
                to_drop.add(worse)
        kept = [c for c in X.columns if c not in to_drop]
        if to_drop:
            log.info(f"FeatureSelector: corr-dropped: {sorted(to_drop)}")
        return kept

    def _rank(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        model = _make_importance_model(n_neg / max(n_pos, 1.0))
        try:
            model.fit(X, y)
            return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        except Exception as e:
            log.warning(f"FeatureSelector: importance model failed: {e}")
            return pd.Series(1.0, index=X.columns)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        log.info(f"FeatureSelector.fit_transform: {X.shape}  pos_rate={y.mean():.3f}")

        # FIX: no sym_* columns — only exact _PASSTHROUGH matches
        pt_cols    = [c for c in X.columns if c in _PASSTHROUGH]
        alpha_cols = [c for c in X.columns if c not in pt_cols]

        X_pt    = X[pt_cols].copy()    if pt_cols    else pd.DataFrame(index=X.index)
        X_alpha = X[alpha_cols].copy()

        X_alpha = self._remove_dead(X_alpha)
        if X_alpha.empty:
            raise ValueError("No alpha features remain after dead-feature removal.")

        X_alpha = X_alpha[self._corr_filter(X_alpha, pd.Series(1.0, index=X_alpha.columns))]
        imp     = self._rank(X_alpha, y)
        X_alpha = X_alpha[self._corr_filter(X_alpha, imp)]
        imp     = imp.reindex(X_alpha.columns).fillna(0)

        k        = max(min(self.top_k, len(X_alpha.columns)), self.min_features)
        top_cols = imp.sort_values(ascending=False).head(k).index.tolist()
        if len(top_cols) < self.min_features:
            extra    = [c for c in X_alpha.columns if c not in top_cols]
            top_cols += extra[:self.min_features - len(top_cols)]

        self._alpha_cols       = top_cols
        self._passthrough_cols = [c for c in pt_cols if c in X.columns]

        # FIX: store as a fixed ordered list — this is the contract for transform()
        self.selected_columns = top_cols + self._passthrough_cols
        self.selected         = self.selected_columns   # back-compat alias
        self.importance_df    = imp.reindex(top_cols).fillna(0).to_frame("importance")

        log.info(f"FeatureSelector: alpha={top_cols}  passthrough={self._passthrough_cols}")

        scaled = self._scaler.fit_transform(X_alpha[top_cols].values)
        self._is_fitted = True

        X_out = pd.DataFrame(scaled, index=X_alpha.index, columns=top_cols)
        if not X_pt.empty:
            X_out = pd.concat([X_out, X_pt.reindex(X_out.index).fillna(0)], axis=1)

        # Enforce exact column order
        X_out = X_out.reindex(columns=self.selected_columns, fill_value=0.0)

        # Assertion: output must exactly match stored columns
        assert list(X_out.columns) == self.selected_columns, \
            f"fit_transform column mismatch: {list(X_out.columns)} != {self.selected_columns}"

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform before transform.")

        missing = [c for c in self._alpha_cols if c not in X.columns]
        if missing:
            log.warning(f"FeatureSelector.transform: filling missing alpha cols: {missing}")

        X_alpha = X.reindex(columns=self._alpha_cols, fill_value=0.0)
        scaled  = self._scaler.transform(X_alpha.values)
        X_out   = pd.DataFrame(scaled, index=X.index, columns=self._alpha_cols)

        if self._passthrough_cols:
            missing_pt = [c for c in self._passthrough_cols if c not in X.columns]
            if missing_pt:
                log.warning(f"FeatureSelector.transform: filling missing passthrough: {missing_pt}")
            X_pt  = X.reindex(columns=self._passthrough_cols, fill_value=0.0)
            X_out = pd.concat([X_out, X_pt], axis=1)

        # Enforce exact column order from fit — THIS IS THE FIX for feature mismatch
        X_out = X_out.reindex(columns=self.selected_columns, fill_value=0.0)

        # Assertion: catches any shape/order bug before it reaches the model
        assert list(X_out.columns) == self.selected_columns, (
            f"transform column mismatch!\n"
            f"  expected ({len(self.selected_columns)}): {self.selected_columns}\n"
            f"  got      ({len(X_out.columns)}): {list(X_out.columns)}"
        )

        return X_out