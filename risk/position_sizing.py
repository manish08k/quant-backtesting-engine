"""
risk/position_sizing.py
Three sizing methods: fixed-fraction, volatility-targeting, Kelly.
RiskManager delegates to this module.
"""
import numpy as np
from utils.config import CFG

R = CFG.risk


class PositionSizer:

    @staticmethod
    def fixed_fraction(
        capital:    float,
        price:      float,
        atr:        float,
        risk_pct:   float = None,
    ) -> int:
        """
        Risk `risk_pct` of capital per trade.
        Stop-loss distance = atr_sl_mult × ATR.
        shares = (capital × risk_pct) / (sl_mult × ATR)
        """
        risk_pct  = risk_pct or R.risk_per_trade
        risk_amt  = capital * risk_pct
        stop_dist = R.atr_sl_mult * atr
        if stop_dist <= 0 or price <= 0:
            return 0
        shares    = risk_amt / stop_dist
        max_shares = (capital * R.max_position_pct) / price
        return int(min(shares, max_shares))

    @staticmethod
    def volatility_target(
        capital:     float,
        price:       float,
        daily_vol:   float,   # annualised vol (e.g. 0.20 = 20%)
        target_vol:  float = 0.15,
    ) -> int:
        """
        Size position so portfolio vol contribution = target_vol.
        shares = (capital × target_vol) / (daily_vol × price × sqrt(252))
        """
        if daily_vol <= 0 or price <= 0:
            return 0
        notional   = capital * (target_vol / daily_vol)
        shares     = notional / price
        max_shares = (capital * R.max_position_pct) / price
        return int(min(shares, max_shares))

    @staticmethod
    def half_kelly(
        capital:  float,
        price:    float,
        win_prob: float,
        win_mult: float,    # avg win / avg loss ratio
    ) -> int:
        """
        Half-Kelly fraction (conservative, avoids overbetting).
        f* = (p × b - q) / b   where b = win/loss ratio, q = 1-p
        Use f*/2 for half-Kelly.
        """
        if win_mult <= 0 or price <= 0:
            return 0
        q = 1 - win_prob
        kelly_f = (win_prob * win_mult - q) / win_mult
        half_f  = max(0, kelly_f / 2)
        notional = capital * half_f
        shares   = notional / price
        max_shares = (capital * R.max_position_pct) / price
        return int(min(shares, max_shares))
