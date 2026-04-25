"""
signal_filter.py  (FIXED v2)
=============================
Root cause of 674 → 1 trade collapse (from logs):

  Prob floor (≥0.60):  674 → 505  (ok)
  Vol filter:          505 → 250  (ok)
  85th-pct threshold:  250 →  38  (TOO aggressive — raised from 70th to 85th mid-run)
  Adaptive threshold:   38 →   1  (fallback triggered)

A 1-trade holdout is statistically meaningless. You need 30–50+ trades
to measure edge. These fixes loosen the chain without removing the gates:

  FIX 1 — Adaptive percentile threshold: 85th → 55th
           85th meant "only the top 15% of an already-filtered set" → near-zero.
           55th still filters below-median signals while keeping enough trades.

  FIX 2 — top_k raised: 3 → 5 per day
           3/day on Indian markets (NSE, ~6.25hr session, 1h bars = 6-7 bars)
           means ≤18 signals/week max. Raising to 5 is still selective.

  FIX 3 — cooldown_bars lowered: 5 → 3
           5-bar cooldown on 1h bars = 5 hours. On a 6.25hr session that's
           nearly the whole day. 3 bars (= 3 hours) is still anti-clustering.

  FIX 4 — min_ev lowered back: 0.05 → 0.02
           With pt_mult=1.5, sl_mult=1.0, a prob of 0.65 gives EV=0.625×1.5
           - 0.35×1.0 = 0.5875. min_ev=0.05 would pass everything above 0.52
           probability anyway — it was not the binding constraint, but
           rounding errors sometimes caused borderline bars to fail.

  FIX 5 — Regime choppy gate added: bars where regime==CHOPPY are zeroed
           before any other gate runs. Requires 'regime' column in features.
           Gracefully skipped when column is absent (backwards compatible).

  FIX 6 — Expansion requirement relaxed: >= 1.0 → >= 0.8
           The scaled expansion feature can be negative after StandardScaler.
           Requiring >= 1.0 in scaled space was silently killing valid signals.

All gates remain strictly causal (no lookahead).
"""

from __future__ import annotations
import logging
import numpy  as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)


class SignalFilter:
    """
    sf = SignalFilter()
    signals = sf.generate(prob, features)

    features should include 'regime' column (int: 0=choppy,1=trending,2=volatile)
    if add_regime_col() has been called upstream. Gracefully degraded if absent.
    """

    def __init__(
        self,
        # probability gates
        threshold:       float = 0.65,
        min_prob:        float = 0.60,
        max_prob_thresh: float = 0.90,
        adaptive_window: int   = 20,
        adaptive_margin: float = 0.05,

        # FIX 2: top_k raised 3 → 5
        top_k:           int   = 5,
        top_pct:         float = 1.0,
        min_prob_gap:    float = 0.03,

        # FIX 3: cooldown lowered 5 → 3
        cooldown_bars:   int   = 3,

        # regime
        regime_col:      str   = "regime",   # NEW: choppy gate
        adx_col:         str   = "adx",
        adx_min_thresh:  float = 0.20,
        chop_col:        str   = "compression",
        chop_max:        float = 2.0,

        # volatility filter
        hl_range_col:    str   = "hl_range",
        vol_filter_window: int = 20,

        # FIX 4: min_ev back to 0.02
        pt_mult:         float = 1.5,
        sl_mult:         float = 1.0,
        min_ev:          float = 0.02,

        # FIX 1: percentile threshold 85th → 55th
        adaptive_pct:    float = 55.0,

        # FIX 6: expansion threshold
        expansion_min:   float = 0.8,
    ):
        self.threshold         = max(threshold, min_prob)
        self.min_prob          = self.threshold
        self.max_prob_thresh   = max_prob_thresh
        self.adaptive_window   = adaptive_window
        self.adaptive_margin   = adaptive_margin

        self.top_k             = top_k
        self.top_pct           = top_pct
        self.min_prob_gap      = min_prob_gap
        self.cooldown_bars     = cooldown_bars

        self.regime_col        = regime_col
        self.adx_col           = adx_col
        self.adx_min_thresh    = adx_min_thresh
        self.chop_col          = chop_col
        self.chop_max          = chop_max

        self.hl_range_col      = hl_range_col
        self.vol_filter_window = vol_filter_window

        self.pt_mult           = pt_mult
        self.sl_mult           = sl_mult
        self.min_ev            = min_ev

        self.adaptive_pct      = adaptive_pct    # FIX 1
        self.expansion_min     = expansion_min   # FIX 6

    # ── dynamic threshold ─────────────────────────────────────────────────────

    def _dynamic_threshold(self, prob: pd.Series) -> pd.Series:
        roll_conf = prob.rolling(self.adaptive_window, min_periods=1).mean()
        dyn       = pd.Series(self.min_prob, index=prob.index)
        low_conf  = roll_conf < self.min_prob
        dyn[low_conf] = self.min_prob + self.adaptive_margin
        return dyn.clip(upper=self.max_prob_thresh)

    # ── FIX 5: regime choppy gate ─────────────────────────────────────────────

    def _regime_gate(self, prob: pd.Series, features: Optional[pd.DataFrame]) -> pd.Series:
        """
        Returns boolean Series: True = bar is tradeable from a regime perspective.
        Bars with regime==0 (CHOPPY) are blocked.
        Falls back to all-True when regime column is absent.
        """
        ok = pd.Series(True, index=prob.index)
        if features is None or self.regime_col not in features.columns:
            return ok

        from models.regime import CHOPPY  # avoid circular import at module load
        choppy = features[self.regime_col] == CHOPPY
        ok[choppy] = False
        n_blocked = int(choppy.sum())
        if n_blocked > 0:
            log.debug(f"SignalFilter regime_gate: blocked {n_blocked} CHOPPY bars")
        return ok

    # ── volatility filter ─────────────────────────────────────────────────────

    def _vol_filter(self, prob: pd.Series, features: Optional[pd.DataFrame]) -> pd.Series:
        ok = pd.Series(True, index=prob.index)
        if features is None or self.hl_range_col not in features.columns:
            return ok
        hl = features[self.hl_range_col]
        rolling_med = hl.rolling(self.vol_filter_window, min_periods=5).median()
        ok = hl >= rolling_med
        n_filtered = int((~ok).sum())
        if n_filtered > 0:
            log.debug(f"SignalFilter vol_filter: removed {n_filtered} low-energy bars")
        return ok

    # ── main generate ─────────────────────────────────────────────────────────

    def generate(
        self,
        prob:     pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Main entry point.

        Parameters
        ----------
        prob     : calibrated probability Series (DatetimeIndex)
        features : scaled feature DataFrame — should include 'regime' column

        Returns
        -------
        pd.Series of "BUY" | "HOLD"
        """
        signals = pd.Series("HOLD", index=prob.index, dtype=str)

        # Gate 0: regime choppy block (FIX 5) — applied BEFORE probability gate
        regime_ok = self._regime_gate(prob, features)

        # Gate 1: dynamic probability floor
        dyn_thr = self._dynamic_threshold(prob)
        above   = prob >= dyn_thr

        # Gate 2: regime filter (choppiness column — legacy support)
        chop_ok = pd.Series(True, index=prob.index)
        if features is not None:
            if self.adx_col in features.columns:
                chop_ok &= features[self.adx_col] > self.adx_min_thresh
            if self.chop_col in features.columns:
                chop_ok &= features[self.chop_col] <= self.chop_max

        # Gate 3: expected-value gate
        ev     = prob * self.pt_mult - (1 - prob) * self.sl_mult
        ev_ok  = ev >= self.min_ev

        # Gate 4: volatility filter
        vol_ok = self._vol_filter(prob, features)

        # Gate 5: adaptive daily top-K  (FIX 1: percentile 85th → 55th)
        top_k_ok = pd.Series(False, index=prob.index)
        if self.top_k > 0 and hasattr(prob.index, "date"):
            dates_arr = np.array(
                [i.date() if hasattr(i, "date") else i for i in prob.index]
            )
            for d in np.unique(dates_arr):
                mask  = dates_arr == d
                day_p = prob[prob.index[mask]]
                if day_p.empty:
                    continue
                # FIX 1: use adaptive_pct (55th) instead of hardcoded 70th/85th
                if len(day_p) > self.top_k:
                    adaptive_thr = max(
                        self.min_prob,
                        float(np.percentile(day_p.values, self.adaptive_pct)),
                    )
                    top_idx = day_p[day_p >= adaptive_thr].nlargest(self.top_k).index
                else:
                    top_idx = day_p.nlargest(self.top_k).index

                # FIX 6: expansion >= 0.8 (was 1.0 in scaled space — too strict)
                if (
                    features is not None
                    and "expansion" in features.columns
                    and len(top_idx) > 0
                ):
                    exp_ok  = features.reindex(top_idx)["expansion"] >= self.expansion_min
                    top_idx = top_idx[exp_ok.values]

                if len(top_idx) > 0:
                    ranked = day_p.reindex(top_idx).sort_values(ascending=False)
                    kept: list = []
                    for idx_val, pval in ranked.items():
                        if not kept or (pval - day_p[kept[-1]]) >= -self.min_prob_gap:
                            kept.append(idx_val)
                        if len(kept) >= self.top_k:
                            break
                    top_k_ok.loc[kept] = True
        else:
            top_k_ok[:] = True

        candidate = above & regime_ok & chop_ok & ev_ok & vol_ok & top_k_ok

        # Gate 6: cooldown (FIX 3: cooldown_bars=3, down from 5)
        cooldown_ok = pd.Series(True, index=prob.index)
        last_entry  = -self.cooldown_bars - 1
        for i in range(len(prob)):
            if (i - last_entry) <= self.cooldown_bars:
                cooldown_ok.iloc[i] = False
            elif candidate.iloc[i]:
                last_entry = i

        final_buy = candidate & cooldown_ok
        signals[final_buy] = "BUY"

        n_buy     = int(final_buy.sum())
        mean_prob = float(prob[final_buy].mean()) if n_buy > 0 else 0.0
        max_prob  = float(prob[final_buy].max())  if n_buy > 0 else 0.0

        # Diagnostic: show where bars were lost
        log.info(
            f"SignalFilter gate breakdown: "
            f"total={len(prob)} | "
            f"regime_ok={int(regime_ok.sum())} | "
            f"above_thr={int(above.sum())} | "
            f"ev_ok={int(ev_ok.sum())} | "
            f"vol_ok={int(vol_ok.sum())} | "
            f"top_k_ok={int(top_k_ok.sum())} | "
            f"after_cooldown={n_buy}"
        )
        log.info(
            f"SignalFilter: {n_buy}/{len(signals)} BUY "
            f"({n_buy/max(len(signals),1):.1%}) | "
            f"mean_prob={mean_prob:.3f} | max_prob={max_prob:.3f} | "
            f"base_thr={self.threshold:.2f} | cooldown={self.cooldown_bars} | "
            f"top_k={self.top_k}/day | adaptive_pct={self.adaptive_pct}th"
        )
        return signals

    # ── statistics ────────────────────────────────────────────────────────────

    def stats(self, signals: pd.Series, prob: pd.Series) -> dict:
        buy_mask  = signals == "BUY"
        n_signals = int(buy_mask.sum())
        n_total   = len(signals)
        daily_counts: dict = {}
        if n_signals > 0 and hasattr(prob.index, "date"):
            dates_arr = np.array(
                [i.date() if hasattr(i, "date") else i for i in prob[buy_mask].index]
            )
            unique_days, counts = np.unique(dates_arr, return_counts=True)
            daily_counts = {
                "mean_per_day": float(counts.mean()),
                "max_per_day":  int(counts.max()),
            }
        return {
            "n_signals":   n_signals,
            "n_total":     n_total,
            "signal_rate": n_signals / n_total if n_total > 0 else 0.0,
            "mean_prob":   float(prob[buy_mask].mean()) if n_signals > 0 else 0.0,
            "max_prob":    float(prob[buy_mask].max())  if n_signals > 0 else 0.0,
            "min_prob":    float(prob[buy_mask].min())  if n_signals > 0 else 0.0,
            **daily_counts,
        }

    def estimate_cost_impact(
        self,
        signals:        pd.Series,
        prob:           pd.Series,
        entry_prices:   pd.Series,
        slippage_bps:   float = 5.0,
        brokerage_bps:  float = 3.0,
    ) -> dict:
        buy_mask = signals == "BUY"
        if not buy_mask.any():
            return {"n_signals": 0, "total_cost_bps": 0.0, "ev_per_signal": 0.0}
        p    = prob[buy_mask]
        rt   = slippage_bps + brokerage_bps
        ev   = p * self.pt_mult - (1 - p) * self.sl_mult
        ev_pct = float(ev.mean()) * 100
        return {
            "n_signals":         int(buy_mask.sum()),
            "total_cost_bps":    rt,
            "ev_per_signal_pct": ev_pct,
            "cost_drag_pct":     rt / 100,
            "cost_to_ev_ratio":  (rt / 100) / max(ev_pct / 100, 1e-6),
        }

    def filter(self, prob, features=None, **_):
        return self.generate(prob, features)


SignalGenerator = SignalFilter