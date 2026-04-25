"""
risk/position_sizing.py
Three sizing methods: fixed-fraction, volatility-targeting, Kelly.
All methods cap at max_position_pct of capital.
"""
import numpy as np
from utils.config import CFG

R = CFG.risk


class PositionSizer:

    @staticmethod
    def fixed_fraction(
        capital:   float,
        price:     float,
        atr:       float,
        risk_pct:  float = None,
    ) -> int:
        """
        Risk `risk_pct` of capital per trade.
        Stop = atr_sl_mult × ATR from entry.
        shares = (capital × risk_pct) / (sl_mult × ATR)
        """
        risk_pct   = risk_pct or R.risk_per_trade
        risk_amt   = capital * risk_pct
        stop_dist  = R.atr_sl_mult * atr
        if stop_dist <= 0 or price <= 0:
            return 0
        shares     = risk_amt / stop_dist
        max_shares = (capital * R.max_position_pct) / price
        return int(min(shares, max_shares))

    @staticmethod
    def volatility_target(
        capital:    float,
        price:      float,
        daily_vol:  float,       # annualised vol, e.g. 0.20
        target_vol: float = 0.15,
    ) -> int:
        """
        Size so portfolio contribution = target_vol.
        notional = capital × (target_vol / instrument_vol)
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
        win_mult: float,    # avg_win / avg_loss ratio
    ) -> int:
        """
        Half-Kelly: f* = (p·b - q) / b; use f*/2.
        b = win/loss ratio, q = 1-p.
        """
        if win_mult <= 0 or price <= 0 or win_prob <= 0:
            return 0
        q       = 1.0 - win_prob
        kelly_f = (win_prob * win_mult - q) / win_mult
        half_f  = max(0.0, kelly_f / 2.0)
        # Cap at 10% per trade regardless of Kelly output
        half_f  = min(half_f, 0.10)
        notional = capital * half_f
        shares   = notional / price
        max_shares = (capital * R.max_position_pct) / price
        return int(min(shares, max_shares))

    @staticmethod
    def confidence_scaled(
        capital:        float,
        price:          float,
        atr:            float,
        prob:           float,
        base_threshold: float = 0.55,
        min_scale:      float = 0.50,
    ) -> int:
        """
        Fixed-fraction base, scaled by how far prob exceeds threshold.
        scale = min_scale + (prob - base_thr)/(1 - base_thr) × (1 - min_scale)
        """
        excess = max(0.0, prob - base_threshold)
        denom  = max(1.0 - base_threshold, 1e-6)
        scale  = min_scale + (excess / denom) * (1.0 - min_scale)
        scale  = float(np.clip(scale, min_scale, 1.0))
        base   = PositionSizer.fixed_fraction(capital, price, atr)
        return max(0, int(base * scale))