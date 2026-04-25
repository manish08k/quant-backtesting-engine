"""
ensemble.py — LightGBM-primary ensemble. Stable predict_proba. No crashes.

CHANGES vs prior version
-------------------------
- Blend weights: LGB=0.60, XGB=0.40 (simpler, documented)
- predict_proba: always returns pd.Series; 0.5 fallback if all models fail
- Feature shape validation uses _n_features stored at fit time
- walk_forward: TimeSeriesSplit(n_splits=2) default — more stable for small datasets
- No stacking, no meta-model complexity
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import roc_auc_score, f1_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger("pipeline")

# FIX: simplified, documented weights
_LGB_W      = 0.60
_XGB_W      = 0.40
_TEMPERATURE = 0.85


def calibrate(prob: np.ndarray, temp: float = _TEMPERATURE) -> np.ndarray:
    prob  = np.clip(prob, 1e-6, 1 - 1e-6)
    logit = np.log(prob / (1 - prob))
    return 1.0 / (1.0 + np.exp(-logit / temp))


def _calibration_params(n_train: int) -> dict:
    return {"cv": 3, "method": "sigmoid"} if n_train < 1500 else {"cv": 5, "method": "isotonic"}


def _pos_weight(y: pd.Series) -> float:
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return n_neg / max(n_pos, 1.0)


def _sample_weights(y: pd.Series, base_sw: np.ndarray | None = None) -> np.ndarray:
    pw      = _pos_weight(y)
    class_w = np.where(y.values == 1, pw, 1.0)
    return class_w * base_sw if base_sw is not None else class_w


def _make_estimators(pw: float) -> list:
    estimators = []
    try:
        from lightgbm import LGBMClassifier
        estimators.append(("lgb", LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=pw,
            verbose=-1, random_state=42, n_jobs=-1,
        )))
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        estimators.append(("xgb", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pw, use_label_encoder=False,
            eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1,
        )))
    except ImportError:
        pass

    if not estimators:
        from sklearn.ensemble import RandomForestClassifier
        estimators.append(("rf", RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )))

    return estimators


_BLEND_W = {"lgb": _LGB_W, "xgb": _XGB_W, "rf": 1.0}


def _blend_probs(preds: list) -> np.ndarray:
    total_w = sum(_BLEND_W.get(n, 1.0) for n, _ in preds)
    blend   = sum(_BLEND_W.get(n, 1.0) / total_w * p for n, p in preds)
    return calibrate(blend)


class EnsembleModel:
    def __init__(self, threshold: float = 0.55):
        self.threshold   = threshold
        self._calibrated = []
        self._pw: float  = 1.0
        self._n_features: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: pd.Series | None = None) -> "EnsembleModel":
        self._pw         = _pos_weight(y)
        self._n_features = X.shape[1]
        sw        = _sample_weights(y, sample_weight.values if sample_weight is not None else None)
        cal_kw    = _calibration_params(len(X))
        estimators = _make_estimators(self._pw)

        self._calibrated = []
        for name, base in estimators:
            try:
                cal = CalibratedClassifierCV(base, **cal_kw)
                cal.fit(X.values, y, sample_weight=sw)
                self._calibrated.append((name, cal))
                log.info(f"EnsembleModel: fitted '{name}'  n={len(X)}  pos_rate={y.mean():.3f}")
            except Exception as e:
                log.warning(f"EnsembleModel: '{name}' fit failed: {e}")

        if not self._calibrated:
            raise RuntimeError("All estimators failed to fit.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if not self._calibrated:
            raise RuntimeError("Call fit() before predict_proba().")

        if len(X) == 0:
            return pd.Series(dtype=float, name="prob")

        # FIX: validate feature count before calling any model
        if X.shape[1] != self._n_features:
            raise RuntimeError(
                f"Feature mismatch: model trained on {self._n_features} features, "
                f"got {X.shape[1]}. Columns: {X.columns.tolist()}"
            )

        preds = []
        for name, cal in self._calibrated:
            try:
                p = cal.predict_proba(X.values)[:, 1]
                preds.append((name, p))
            except Exception as e:
                log.warning(f"predict_proba: '{name}' failed: {e}")

        if not preds:
            # FIX: safe fallback — never crash
            log.error("All models failed to predict — returning 0.5 fallback.")
            return pd.Series(np.full(len(X), 0.5), index=X.index, name="prob")

        prob     = _blend_probs(preds)
        prob_std = float(prob.std())
        log.info(
            f"predict_proba: mean={prob.mean():.4f}  std={prob_std:.4f}  "
            f"max={prob.max():.4f}  n={len(prob)}"
        )
        if prob_std < 0.02:
            log.warning(f"Low prob_std={prob_std:.4f} — check feature variance and label balance.")

        return pd.Series(prob, index=X.index, name="prob")

    def walk_forward(self, X: pd.DataFrame, y: pd.Series,
                     n_splits: int = 2,   # FIX: default 2 for intraday stability
                     sample_weight: pd.Series | None = None) -> list:
        tscv    = TimeSeriesSplit(n_splits=n_splits)
        results = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            if len(y_tr) < 50 or y_tr.nunique() < 2:
                log.warning(f"Fold {fold}: insufficient data — skipping.")
                continue

            sw_tr = (sample_weight.iloc[tr_idx].values
                     if sample_weight is not None else None)

            fold_model = EnsembleModel(threshold=self.threshold)
            try:
                sw_series = pd.Series(sw_tr, index=y_tr.index) if sw_tr is not None else None
                fold_model.fit(X_tr, y_tr, sample_weight=sw_series)
            except Exception as e:
                log.warning(f"Fold {fold} fit failed: {e}")
                continue

            blend    = fold_model.predict_proba(X_te).values
            prob_std = float(blend.std())
            pred     = (blend >= self.threshold).astype(int)

            try:
                auc   = roc_auc_score(y_te, blend)
                f1    = f1_score(y_te, pred, zero_division=0)
                brier = brier_score_loss(y_te, blend)
                wr    = float(pred[y_te.values == 1].mean()) if y_te.sum() > 0 else 0.0
                n_sig = int(pred.sum())
                results.append({
                    "fold": fold, "auc": auc, "f1": f1, "brier": brier,
                    "win_rate": wr, "n_signals": n_sig,
                    "mean_prob": float(blend.mean()), "max_prob": float(blend.max()),
                    "std_prob": prob_std, "train_size": len(y_tr), "test_size": len(y_te),
                    "pos_rate_train": float(y_tr.mean()), "pos_rate_test": float(y_te.mean()),
                })
                log.info(f"Fold {fold}: AUC={auc:.4f}  F1={f1:.4f}  "
                         f"prob_std={prob_std:.4f}  n_sig={n_sig}")
            except Exception as e:
                log.warning(f"Fold {fold}: metrics failed: {e}")

        return results

    def feature_importance(self, feature_names: list | None = None) -> pd.DataFrame:
        records = []
        for name, cal in self._calibrated:
            base = getattr(cal, "estimator", None) or getattr(cal, "base_estimator", None)
            if base is not None and hasattr(base, "feature_importances_"):
                fi = pd.Series(base.feature_importances_, name=name)
                if feature_names is not None and len(feature_names) == len(fi):
                    fi.index = feature_names
                records.append(fi)
        if not records:
            return pd.DataFrame()
        return (pd.concat(records, axis=1).mean(axis=1)
                .sort_values(ascending=False).to_frame("importance"))

    def meta_coefficients(self):
        return None

    def save(self, path: str = "models/ensemble.pkl") -> None:
        import pickle, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "models/ensemble.pkl") -> "EnsembleModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Module-level helpers used by main.py ──────────────────────────────────────

def _make_base_estimators() -> list:
    return _make_estimators(pw=1.0)


def _class_sample_weights(y_tr: pd.Series, base_sw: np.ndarray | None) -> np.ndarray:
    return _sample_weights(y_tr, base_sw)


def _weighted_blend(base_probs: list) -> np.ndarray:
    return _blend_probs(base_probs)