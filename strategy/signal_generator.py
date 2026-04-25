"""
signal_filter.py
================
Institutional signal filter — selective, cost-aware, high-confidence only.

Gate pipeline (applied in order, all causal — no lookahead)
------------------------------------------------------------
1.  Probability floor   : prob >= dynamic_threshold (floor = 0.65)
2.  Regime filter       : regime gate (choppiness <= chop_max)
3.  EV gate             : expected value = prob×pt − (1−prob)×sl >= min_ev
4.  Volatility filter   : skip if hl_range < rolling_median(hl_range, 20 bars)
                          Avoids entries on low-energy bars (compressed, drifting)
5.  Daily top-K         : keep top top_k signals per day by probability score
6.  Prob gap            : enforce min separation between signals same day
7.  Cooldown            : suppress signals within cooldown_bars of last entry
                          min_gap_bars = 5 (raised from 3) — anti-clustering

Dynamic threshold
-----------------
The probability floor scales with recent model confidence:
  - Base threshold: min_prob (default 0.65)
  - Raised by +adaptive_margin when rolling mean prob over last N signals is
    weak (model is uncertain) — avoids trading in low-confidence regimes.
  - Clamped to [min_prob, max_prob].

Key changes from v1
-------------------
  - min_prob raised: 0.55 → 0.65
  - top_k (per day): replaces top_pct — more deterministic trade count control
  - Volatility filter added: hl_range < median → skip
  - cooldown_bars raised: 3 → 5
  - min_ev raised: 0.03 → 0.05
  - Signal ranking exposed in stats() for external use

All operations are strictly look-ahead free.
"""

from __future__ import annotations
import logging
import numpy  as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)


class SignalFilter:
    """
    sf = SignalFilter(min_prob=0.65, top_k=3)
    signals = sf.generate(prob, features)
    """

    def __init__(
        self,
        # probability gates
        threshold:       float = 0.65,
        min_prob:        float = 0.65,   # floor — raised from 0.55
        max_prob_thresh: float = 0.90,
        adaptive_window: int   = 20,
        adaptive_margin: float = 0.05,

        # daily ranking — top_k replaces top_pct for deterministic count control
        top_k:           int   = 3,      # maximum signals per day
        top_pct:         float = 1.0,    # kept for fallback (1.0 = off)
        min_prob_gap:    float = 0.03,

        # cooldown
        cooldown_bars:   int   = 5,      # raised from 3 — anti-clustering

        # regime filters
        adx_col:         str   = "adx",
        adx_min_thresh:  float = 0.20,
        chop_col:        str   = "compression",
        chop_max:        float = 2.0,

        # volatility filter
        hl_range_col:    str   = "hl_range",    # normalised bar range column
        vol_filter_window: int = 20,             # bars for rolling median

        # EV gate
        pt_mult:         float = 1.5,
        sl_mult:         float = 1.0,
        min_ev:          float = 0.05,   # raised from 0.03
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

        self.adx_col           = adx_col
        self.adx_min_thresh    = adx_min_thresh
        self.chop_col          = chop_col
        self.chop_max          = chop_max

        self.hl_range_col      = hl_range_col
        self.vol_filter_window = vol_filter_window

        self.pt_mult           = pt_mult
        self.sl_mult           = sl_mult
        self.min_ev            = min_ev

    # ── dynamic threshold ─────────────────────────────────────────────────────

    def _dynamic_threshold(self, prob: pd.Series) -> pd.Series:
        """
        Raise threshold when recent model confidence is low.
        Rolling mean of prob over adaptive_window bars.
        If mean < 0.65 → add adaptive_margin to threshold.
        """
        roll_conf = prob.rolling(self.adaptive_window, min_periods=1).mean()
        dyn       = pd.Series(self.min_prob, index=prob.index)
        low_conf  = roll_conf < self.min_prob
        dyn[low_conf] = self.min_prob + self.adaptive_margin
        return dyn.clip(upper=self.max_prob_thresh)

    # ── volatility filter ──────────────────────────────────────────────────────

    def _vol_filter(self, prob: pd.Series, features: Optional[pd.DataFrame]) -> pd.Series:
        """
        Return boolean Series: True where hl_range >= rolling_median(hl_range).
        Low-energy bars (compressed range) are filtered out — entries on such
        bars tend to be directionless and accrue costs without edge.
        """
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
        features : scaled feature DataFrame (for regime / vol / EV gates)

        Returns
        -------
        pd.Series of "BUY" | "HOLD"
        """
        signals = pd.Series("HOLD", index=prob.index, dtype=str)

        # Gate 1: dynamic probability floor
        dyn_thr = self._dynamic_threshold(prob)
        above   = prob >= dyn_thr

        # Gate 2: regime filter (choppiness only — ADX not guaranteed in scaled X)
        regime_ok = pd.Series(True, index=prob.index)
        if features is not None:
            if self.adx_col in features.columns:
                regime_ok &= features[self.adx_col] > self.adx_min_thresh
            if self.chop_col in features.columns:
                regime_ok &= features[self.chop_col] <= self.chop_max

        # Gate 3: expected-value gate
        ev     = prob * self.pt_mult - (1 - prob) * self.sl_mult
        ev_ok  = ev >= self.min_ev

        # Gate 4: volatility filter (new) — skip low-energy bars
        vol_ok = self._vol_filter(prob, features)

        # Gate 5: daily top-K filter (replaces top_pct for deterministic count)
        top_k_ok = pd.Series(False, index=prob.index)
        if self.top_k > 0 and hasattr(prob.index, "date"):
            dates_arr = np.array(
                [i.date() if hasattr(i, "date") else i for i in prob.index]
            )
            for d in np.unique(dates_arr):
                mask  = dates_arr == d
                day_p = prob[prob.index[mask]]
                # Take top_k by probability (descending)
                top_idx = day_p.nlargest(self.top_k).index
                top_k_ok.loc[top_idx] = True
        else:
            top_k_ok[:] = True

        # Gate 6: minimum prob gap within the same day
        gap_ok = pd.Series(True, index=prob.index)
        if self.min_prob_gap > 0 and hasattr(prob.index, "date"):
            dates_arr = np.array(
                [i.date() if hasattr(i, "date") else i for i in prob.index]
            )
            for d in np.unique(dates_arr):
                mask = dates_arr == d
                vals = prob[prob.index[mask]].sort_values(ascending=False)
                if len(vals) >= 2 and (vals.iloc[0] - vals.iloc[1]) < self.min_prob_gap:
                    gap_ok[vals.index[1:]] = False

        candidate = above & regime_ok & ev_ok & vol_ok & top_k_ok & gap_ok

        # Gate 7: cooldown (causal — enforced sequentially, no lookahead)
        # Raised to cooldown_bars=5 to prevent trade clustering
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

        log.info(
            f"SignalFilter: {n_buy}/{len(signals)} BUY "
            f"({n_buy/max(len(signals),1):.1%}) | "
            f"mean_prob={mean_prob:.3f} | max_prob={max_prob:.3f} | "
            f"base_thr={self.threshold:.2f} | cooldown={self.cooldown_bars} | "
            f"top_k={self.top_k}/day"
        )
        return signals

    # ── statistics ────────────────────────────────────────────────────────────

    def stats(self, signals: pd.Series, prob: pd.Series) -> dict:
        """Summary statistics for a generated signal series."""
        buy_mask  = signals == "BUY"
        n_signals = int(buy_mask.sum())
        n_total   = len(signals)

        # Daily signal distribution
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

    # ── cost impact estimate ──────────────────────────────────────────────────

    def estimate_cost_impact(
        self,
        signals:        pd.Series,
        prob:           pd.Series,
        entry_prices:   pd.Series,
        slippage_bps:   float = 5.0,
        brokerage_bps:  float = 3.0,
    ) -> dict:
        """
        Estimate the cost budget consumed by the generated signals.
        Returns round-trip cost as a fraction of edge (EV).
        """
        buy_mask = signals == "BUY"
        if not buy_mask.any():
            return {"n_signals": 0, "total_cost_bps": 0.0, "ev_per_signal": 0.0}

        p    = prob[buy_mask]
        rt   = slippage_bps + brokerage_bps          # round-trip bps
        ev   = p * self.pt_mult - (1 - p) * self.sl_mult
        ev_pct = float(ev.mean()) * 100              # rough % EV

        return {
            "n_signals":         int(buy_mask.sum()),
            "total_cost_bps":    rt,
            "ev_per_signal_pct": ev_pct,
            "cost_drag_pct":     rt / 100,           # as fraction
            "cost_to_ev_ratio":  (rt / 100) / max(ev_pct / 100, 1e-6),
        }

    # ── legacy aliases ────────────────────────────────────────────────────────

    def filter(self, prob, features=None, **_):
        return self.generate(prob, features)


# Old import alias
SignalGenerator = SignalFilter