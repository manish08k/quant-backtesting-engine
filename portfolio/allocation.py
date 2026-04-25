"""
portfolio/allocation.py
=======================
Volatility-aware portfolio allocation for filtered signal DataFrames.

Design
------
  Weight formula  : score = prob² / vol
                    weight = score / sum(scores)
                    allocation = weight × capital, clipped at 40% per position

  Cluster penalty : if ≥2 signals share the same date and are within
                    cluster_window bars of each other, reduce their
                    individual weight by cluster_penalty_factor.

  Volatility guard: if any position's volatility > vol_high_threshold,
                    the *entire* session allocation is scaled down by
                    vol_scale_down_factor (conservative regime).

  Empty-signal guard: returns the input DataFrame unchanged (with an
                      "allocation" column of zeros) if no signals present.

Usage
-----
    from portfolio.allocation import allocate_portfolio

    signals = allocate_portfolio(signals_df, capital=1_000_000)
    # signals now has: weight, score, allocation columns added

No leakage: all inputs (prob, volatility) are from the same test/holdout
row — nothing from future bars is referenced.
"""

from __future__ import annotations
import logging
import numpy  as np
import pandas as pd

log = logging.getLogger("pipeline")

# ── constants ─────────────────────────────────────────────────────────────────

_MAX_SINGLE_WEIGHT     = 0.40   # no position > 40% of capital
_MIN_ALLOCATION        = 0.0    # floor: never negative allocation
_CLUSTER_WINDOW_BARS   = 3      # signals within 3 index positions = clustered
_CLUSTER_PENALTY       = 0.70   # reduce clustered signal weights to 70%
_VOL_HIGH_THRESHOLD    = 2.0    # normalised vol: above this = high-vol regime
_VOL_SCALE_DOWN        = 0.60   # scale total session allocation to 60% in high vol
_DEFAULT_VOLATILITY    = 1.0    # used when "volatility" column is absent / zero


def _cluster_penalty_mask(df: pd.DataFrame, window: int = _CLUSTER_WINDOW_BARS) -> pd.Series:
    """
    Return a Series of penalty multipliers (1.0 = no penalty, <1.0 = penalised).
    Signals are considered clustered if their positional index is within `window`
    steps of another signal — same date only.
    """
    penalty = pd.Series(1.0, index=df.index)
    if len(df) < 2:
        return penalty

    # Use positional integer index for gap calculation (works with DatetimeIndex)
    positions = np.arange(len(df))

    has_date = hasattr(df.index, "date")
    if has_date:
        dates = np.array([i.date() if hasattr(i, "date") else i for i in df.index])
    else:
        dates = np.zeros(len(df))    # treat as single group

    for grp in np.unique(dates):
        mask = dates == grp
        pos_in_group = positions[mask]
        idx_in_group = df.index[mask]

        if len(pos_in_group) < 2:
            continue

        # For each pair, flag as clustered if gap ≤ window
        clustered_flags = np.zeros(len(pos_in_group), dtype=bool)
        for i in range(len(pos_in_group)):
            for j in range(i + 1, len(pos_in_group)):
                if abs(pos_in_group[i] - pos_in_group[j]) <= window:
                    clustered_flags[i] = True
                    clustered_flags[j] = True

        penalty.loc[idx_in_group[clustered_flags]] = _CLUSTER_PENALTY

    n_penalised = int((penalty < 1.0).sum())
    if n_penalised > 0:
        log.debug(f"allocate_portfolio: {n_penalised} clustered signals penalised "
                  f"(×{_CLUSTER_PENALTY})")
    return penalty


def allocate_portfolio(
    df:                   pd.DataFrame,
    capital:              float = 100_000,
    vol_col:              str   = "volatility",
    prob_col:             str   = "prob",
    max_single_weight:    float = _MAX_SINGLE_WEIGHT,
    cluster_window:       int   = _CLUSTER_WINDOW_BARS,
    cluster_penalty:      float = _CLUSTER_PENALTY,
    vol_high_threshold:   float = _VOL_HIGH_THRESHOLD,
    vol_scale_down:       float = _VOL_SCALE_DOWN,
) -> pd.DataFrame:
    """
    Add allocation columns to a signals DataFrame.

    Parameters
    ----------
    df                 : DataFrame with at least a ``prob`` column.
                         Optional: ``volatility`` column (normalised).
    capital            : Total capital to allocate across all signals.
    vol_col            : Column name for normalised volatility.
    prob_col           : Column name for signal probability.
    max_single_weight  : Hard cap per position (fraction of capital).
    cluster_window     : Positional bar gap for cluster detection.
    cluster_penalty    : Weight multiplier for clustered signals.
    vol_high_threshold : Normalised vol level that triggers regime scale-down.
    vol_scale_down     : Fraction of capital to deploy in high-vol regime.

    Returns
    -------
    df copy with added columns: ``score``, ``weight``, ``allocation``.
    """
    df = df.copy()

    # ── Guard: no signals ─────────────────────────────────────────────────────
    if df.empty or prob_col not in df.columns or len(df) == 0:
        log.warning("allocate_portfolio: empty DataFrame — returning zeros.")
        df["score"]      = 0.0
        df["weight"]     = 0.0
        df["allocation"] = 0.0
        return df

    # ── Probability and volatility ────────────────────────────────────────────
    prob = df[prob_col].clip(lower=0.0, upper=1.0)

    if vol_col in df.columns:
        vol = df[vol_col].replace(0, _DEFAULT_VOLATILITY).fillna(_DEFAULT_VOLATILITY)
    else:
        vol = pd.Series(_DEFAULT_VOLATILITY, index=df.index)

    # ── Score: prob² / vol ────────────────────────────────────────────────────
    score = (prob ** 2) / vol.clip(lower=1e-6)
    df["score"] = score

    # ── Cluster penalty ───────────────────────────────────────────────────────
    penalty          = _cluster_penalty_mask(df, window=cluster_window)
    penalised_score  = score * penalty
    score_sum        = penalised_score.sum()

    if score_sum <= 0:
        log.warning("allocate_portfolio: all scores are zero — equal-weight fallback.")
        df["weight"]     = 1.0 / len(df)
        df["allocation"] = (df["weight"] * capital).clip(
            lower=_MIN_ALLOCATION,
            upper=max_single_weight * capital,
        )
        return df

    df["weight"] = penalised_score / score_sum

    # ── Volatility regime guard ───────────────────────────────────────────────
    # Scale down total session deployment when any signal's volatility is elevated
    max_vol = float(vol.max())
    if max_vol > vol_high_threshold:
        effective_capital = capital * vol_scale_down
        log.info(
            f"allocate_portfolio: high-vol regime detected "
            f"(max_vol={max_vol:.2f} > {vol_high_threshold}). "
            f"Scaling capital: {capital:,.0f} → {effective_capital:,.0f} "
            f"(×{vol_scale_down})"
        )
    else:
        effective_capital = capital

    # ── Allocation with per-position cap ─────────────────────────────────────
    raw_alloc       = df["weight"] * effective_capital
    capped_alloc    = raw_alloc.clip(
        lower = _MIN_ALLOCATION,
        upper = max_single_weight * effective_capital,
    )
    df["allocation"] = capped_alloc

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n_sigs     = len(df)
    total_dep  = float(capped_alloc.sum())
    avg_alloc  = float(capped_alloc.mean())
    max_alloc  = float(capped_alloc.max())
    util       = total_dep / max(capital, 1e-9)

    log.info(
        f"allocate_portfolio: {n_sigs} signals | "
        f"capital={capital:,.0f} | deployed={total_dep:,.0f} ({util:.1%}) | "
        f"avg={avg_alloc:,.0f} | max={max_alloc:,.0f} | "
        f"vol_regime={'HIGH' if max_vol > vol_high_threshold else 'normal'}"
    )

    return df
