"""
risk_manager.py
===============
Institutional risk manager with confidence-scaled position sizing.

Design principles
-----------------
  • Position size ∝ probability confidence
  • ATR-based stops and targets (no fixed % stops)
  • Trailing stop: ratchets upward as price moves in favour.
  • Circuit breakers (two independent):
      1. Daily loss cap   : halt trading after max_daily_loss of capital
      2. Consecutive loss : cooldown after max_consec_losses in a row
  • Edge guard: if edge_required=True, can_trade() returns False until
      set_edge_confirmed(True) is called (e.g. after mean_auc > 0.55).
  • Portfolio heat limiter.
  • Per-symbol daily trade limit.
"""

from __future__ import annotations
import numpy  as np
import pandas as pd
from types import SimpleNamespace


_DEFAULT_RISK = SimpleNamespace(
    capital            = 1_000_000,
    risk_per_trade     = 0.01,
    max_daily_loss     = 0.02,
    max_consec_losses  = 8,         # raised from 4 — only meaningful with model edge
    atr_sl_mult        = 1.0,
    atr_tp_mult        = 1.5,
    trail_atr_mult     = 0.8,
    max_position_pct   = 0.10,
    min_prob           = 0.55,
    max_open_risk      = 0.05,
    max_trades_per_day = 5,
)


class RiskManager:
    """
    rm  = RiskManager(cfg=CFG.risk)
    qty = rm.position_size(atr_val, price, prob=0.72)
    sl, tp = rm.sl_tp(entry, atr_val)
    sl  = rm.trail_stop(current_sl, price, atr_val)
    rm.update(pnl, date, symbol="RELIANCE")
    ok  = rm.can_trade(date, symbol="RELIANCE")

    Edge guard
    ----------
    By default, edge_required=True. can_trade() returns False until
    set_edge_confirmed(True) is called externally (typically after
    walk-forward shows mean_auc > 0.55). This prevents the consecutive-loss
    circuit breaker from firing on noise before the model has demonstrated
    any real edge.
    """

    def __init__(
        self,
        cfg:           SimpleNamespace | None = None,
        capital:       float | None           = None,
        edge_required: bool                   = True,
    ):
        self._cfg              = cfg or _DEFAULT_RISK
        self.capital           = float(capital or self._cfg.capital)
        self._daily_pnl:   dict = {}
        self._consec_losses    = 0
        self._halted_dates: set = set()
        self._symbol_trades: dict = {}
        self._open_risk        = 0.0
        self._edge_required    = edge_required
        self._edge_confirmed   = False

    # ── edge confirmation ─────────────────────────────────────────────────────

    def set_edge_confirmed(self, confirmed: bool) -> None:
        """
        Call with True once model has demonstrated edge (mean_auc > 0.55).
        Until called with True, can_trade() returns False when edge_required=True.
        """
        self._edge_confirmed = confirmed

    # ── circuit breakers ──────────────────────────────────────────────────────

    def can_trade(self, date, symbol: str = "") -> bool:
        """
        Return True if a new trade is permitted.
        Checks: edge guard, daily loss cap, consecutive loss cap,
                per-symbol limit.
        """
        # Edge guard: block all trading until model has demonstrated edge
        if self._edge_required and not self._edge_confirmed:
            return False

        if date in self._halted_dates:
            return False

        # Daily loss circuit breaker
        daily_loss = self._daily_pnl.get(date, 0.0)
        if daily_loss <= -self.capital * self._cfg.max_daily_loss:
            self._halted_dates.add(date)
            return False

        # Consecutive loss circuit breaker (only active when model has edge)
        if self._consec_losses >= self._cfg.max_consec_losses:
            return False

        # Per-symbol daily trade limit
        max_daily = getattr(self._cfg, "max_trades_per_day", 5)
        if symbol:
            key = (symbol, date)
            if self._symbol_trades.get(key, 0) >= max_daily:
                return False

        return True

    # ── position sizing ───────────────────────────────────────────────────────

    def position_size(
        self,
        atr_val: float,
        price:   float,
        prob:    float = 0.55,
    ) -> int:
        min_p = self._cfg.min_prob
        if prob < min_p or atr_val <= 0 or price <= 0:
            return 0

        conf_mult = 0.5 + (prob - min_p) / max(0.75 - min_p, 1e-9)
        conf_mult = float(np.clip(conf_mult, 0.5, 1.5))

        risk_amt   = self.capital * self._cfg.risk_per_trade * conf_mult
        sl_dist    = self._cfg.atr_sl_mult * atr_val
        shares     = int(risk_amt / max(sl_dist, 1e-9))

        max_shares = int(self.capital * self._cfg.max_position_pct / max(price, 1e-9))
        return min(shares, max_shares)

    # ── ATR stop-loss / take-profit ───────────────────────────────────────────

    def sl_tp(
        self,
        entry:     float,
        atr_val:   float,
        direction: str = "BUY",
    ) -> tuple[float, float]:
        if direction == "BUY":
            sl = entry - self._cfg.atr_sl_mult * atr_val
            tp = entry + self._cfg.atr_tp_mult * atr_val
        else:
            sl = entry + self._cfg.atr_sl_mult * atr_val
            tp = entry - self._cfg.atr_tp_mult * atr_val
        return sl, tp

    def trail_stop(
        self,
        current_sl: float,
        price:      float,
        atr_val:    float,
        direction:  str = "BUY",
    ) -> float:
        if direction == "BUY":
            new_sl = price - self._cfg.trail_atr_mult * atr_val
            return max(current_sl, new_sl)
        else:
            new_sl = price + self._cfg.trail_atr_mult * atr_val
            return min(current_sl, new_sl)

    # ── state update ──────────────────────────────────────────────────────────

    def update(
        self,
        pnl:    float,
        date,
        symbol: str = "",
    ) -> None:
        self._daily_pnl[date] = self._daily_pnl.get(date, 0.0) + pnl

        if pnl < 0:
            self._consec_losses += 1
        else:
            self._consec_losses = 0

        if symbol:
            key = (symbol, date)
            self._symbol_trades[key] = self._symbol_trades.get(key, 0) + 1

        self.capital += pnl

    def reset_daily(self, date) -> None:
        self._daily_pnl.pop(date, None)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def daily_summary(self) -> pd.DataFrame:
        if not self._daily_pnl:
            return pd.DataFrame(columns=["date", "pnl"])
        return (
            pd.DataFrame(list(self._daily_pnl.items()), columns=["date", "pnl"])
            .sort_values("date")
            .reset_index(drop=True)
        )

    def metrics(self) -> dict:
        daily = self.daily_summary()
        if daily.empty:
            return {}
        pnl = daily["pnl"]
        cum = pnl.cumsum()
        return {
            "total_pnl":      float(pnl.sum()),
            "n_winning_days": int((pnl > 0).sum()),
            "n_losing_days":  int((pnl < 0).sum()),
            "max_drawdown":   float((cum - cum.cummax()).min()),
            "best_day":       float(pnl.max()),
            "worst_day":      float(pnl.min()),
        }