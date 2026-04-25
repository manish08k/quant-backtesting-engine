"""
barriers.py
===========
Triple-barrier labeling with strong noise filtering and balance enforcement.

Key design choices
------------------
  1. Hard minimum return gate (0.005):
       |realised_return| < 0.005 → row dropped unconditionally.

  2. label = 1 ONLY when:
       (a) profit target hit BEFORE stop loss, AND
       (b) |realised_return| >= min_return
       All other outcomes → label = 0 or dropped.

  3. Timeout handling (strict):
       Timeout with |ret| >= min_return → label = 0.
       Timeout with |ret| < min_return  → dropped.

  4. Label balance enforcement via downsampling:
       pos_rate target: [0.30, 0.50]

  5. Volatility-scaled confidence gate (0.25):
       confidence = |realised_return| / (ATR_pct at entry)
       Labels with confidence < 0.25 are dropped.
"""

from __future__ import annotations
import logging
import numpy  as np
import pandas as pd
from types import SimpleNamespace

log = logging.getLogger("pipeline")


# ── default config ────────────────────────────────────────────────────────────

_DEFAULT = SimpleNamespace(
    pt_mult           = 1.5,
    sl_mult           = 1.0,
    max_hold_bars     = 12,
    min_ret_threshold = 0.005,
    min_confidence    = 0.25,
)

_TARGET_LOW  = 0.30
_TARGET_HIGH = 0.50   # FIX: tightened from 0.60 to avoid noisy positives


# ── internal ATR ─────────────────────────────────────────────────────────────

def _atr_for_labeling(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


# ── label balance diagnostic ──────────────────────────────────────────────────

def label_balance_check(y: pd.Series, context: str = "") -> None:
    n_total = len(y)
    if n_total == 0:
        log.warning(f"label_balance_check [{context}]: empty label series")
        return

    n_pos    = int((y == 1).sum())
    n_neg    = int((y == 0).sum())
    pos_rate = n_pos / n_total

    msg = (
        f"label_balance_check [{context}]: "
        f"n={n_total}  n_pos={n_pos}  n_neg={n_neg}  pos_rate={pos_rate:.3f}"
    )

    if pos_rate < 0.20 or pos_rate > 0.80:
        log.warning(msg + "  *** SEVERE IMBALANCE — model collapse risk ***")
    elif pos_rate < _TARGET_LOW or pos_rate > _TARGET_HIGH:
        log.warning(msg + f"  *** IMBALANCED — target [{_TARGET_LOW:.2f}, {_TARGET_HIGH:.2f}] ***")
    else:
        log.info(msg + "  ✓ balanced")


# ── downsampling balance enforcement ─────────────────────────────────────────

def _enforce_balance(y: pd.Series, rng: np.random.Generator) -> pd.Series:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0:
        log.warning("_enforce_balance: one class is empty — cannot balance.")
        return y

    pos_rate = n_pos / len(y)

    if _TARGET_LOW <= pos_rate <= _TARGET_HIGH:
        return y

    target_midpoint = (_TARGET_LOW + _TARGET_HIGH) / 2  # 0.40

    if pos_rate < _TARGET_LOW:
        target_neg = int(n_pos / target_midpoint) - n_pos
        target_neg = max(target_neg, n_pos)
        if target_neg >= n_neg:
            return y
        neg_idx      = y[y == 0].index
        keep_neg_idx = rng.choice(neg_idx, size=target_neg, replace=False)
        keep_idx     = y[y == 1].index.append(pd.Index(keep_neg_idx))
        y_bal        = y.loc[keep_idx].sort_index()
        log.info(
            f"_enforce_balance: downsampled negatives {n_neg}→{target_neg}  "
            f"new_pos_rate={float(y_bal.mean()):.3f}"
        )
        return y_bal
    else:
        target_pos = int(n_neg * target_midpoint / (1 - target_midpoint))
        target_pos = max(target_pos, 1)
        if target_pos >= n_pos:
            return y
        pos_idx      = y[y == 1].index
        keep_pos_idx = rng.choice(pos_idx, size=target_pos, replace=False)
        keep_idx     = y[y == 0].index.append(pd.Index(keep_pos_idx))
        y_bal        = y.loc[keep_idx].sort_index()
        log.info(
            f"_enforce_balance: downsampled positives {n_pos}→{target_pos}  "
            f"new_pos_rate={float(y_bal.mean()):.3f}"
        )
        return y_bal


# ── core labeling ─────────────────────────────────────────────────────────────

def select_label(
    df:      pd.DataFrame,
    cfg:     SimpleNamespace | None = None,
    balance: bool = True,
    seed:    int  = 42,
) -> pd.Series:
    c   = cfg or _DEFAULT
    atr = _atr_for_labeling(df)
    close = df["close"]
    labels: dict = {}

    max_i = len(close) - c.max_hold_bars - 1

    for i in range(max_i):
        idx     = close.index[i]
        entry   = float(close.iloc[i])
        atr_val = float(atr.iloc[i])

        if atr_val <= 0 or entry <= 0:
            continue

        atr_pct = atr_val / entry
        pt      = entry + c.pt_mult * atr_val
        sl      = entry - c.sl_mult * atr_val

        label:        int | float = np.nan
        exit_price:   float       = entry
        hit_barrier:  bool        = False
        pt_hit_first: bool        = False

        for j in range(1, c.max_hold_bars + 1):
            fp = float(close.iloc[i + j])
            if fp >= pt:
                label        = 1
                exit_price   = fp
                hit_barrier  = True
                pt_hit_first = True
                break
            if fp <= sl:
                label      = 0
                exit_price = fp
                hit_barrier = True
                break

        if not hit_barrier:
            exit_price = float(close.iloc[i + c.max_hold_bars])
            realised   = (exit_price - entry) / entry
            if abs(realised) < c.min_ret_threshold:
                continue
            label = 0

        realised = (exit_price - entry) / entry
        if abs(realised) < c.min_ret_threshold:
            continue

        confidence = abs(realised) / (atr_pct + 1e-9)
        if confidence < c.min_confidence:
            continue

        if label == 1 and not pt_hit_first:
            label = 0

        labels[idx] = label

    s = pd.Series(labels, dtype=float).dropna().astype(int)

    if balance and len(s) > 0:
        rng = np.random.default_rng(seed)
        s = _enforce_balance(s, rng)

    label_balance_check(s, context="select_label")
    return s


# ── public entry point ────────────────────────────────────────────────────────

def label_bars(
    df:             pd.DataFrame,
    atr:            pd.Series,
    pt_mult:        float = 1.5,
    sl_mult:        float = 1.0,
    max_hold_bars:  int   = 12,
    min_ret:        float = 0.005,
    min_confidence: float = 0.25,
) -> pd.Series:
    cfg = SimpleNamespace(
        pt_mult           = pt_mult,
        sl_mult           = sl_mult,
        max_hold_bars     = max_hold_bars,
        min_ret_threshold = min_ret,
        min_confidence    = min_confidence,
    )
    return select_label(df, cfg)


# ── sample weights ────────────────────────────────────────────────────────────

def compute_sample_weights(
    y:             pd.Series,
    max_hold_bars: int   = 12,
    decay:         float = 0.95,
    recency_boost: float = 1.5,
) -> pd.Series:
    n       = len(y)
    idx     = np.arange(n)
    overlap = np.minimum(idx + 1, max_hold_bars)
    raw_w   = 1.0 / overlap
    time_w  = decay ** (n - 1 - idx)

    boost  = np.ones(n)
    cutoff = int(n * 0.80)
    if cutoff < n:
        boost[cutoff:] = np.linspace(1.0, recency_boost, n - cutoff)

    weights = raw_w * time_w * boost
    weights = weights / weights.mean()
    return pd.Series(weights, index=y.index)