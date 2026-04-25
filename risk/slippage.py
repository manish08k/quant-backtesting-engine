"""
risk/slippage.py
Realistic slippage model: fixed bps + volume-impact component.
Used by backtester and live execution layer.
"""
import numpy as np
import pandas as pd
from utils.config import CFG

BC = CFG.backtest


class SlippageModel:
    """
    slippage = base_bps + volume_impact
    volume_impact: larger order relative to avg volume → more slippage.
    Impact formula: k × sqrt(order_size / avg_volume) × price
    k ≈ 0.1 for liquid NSE stocks (empirically estimated).
    """

    def __init__(self, base_bps: float = None, impact_k: float = 0.10):
        self.base_bps = base_bps if base_bps is not None else BC.slippage_bps
        self.impact_k = impact_k

    def cost(
        self,
        price:      float,
        shares:     int,
        avg_volume: float,   # 20-bar average volume in shares
        direction:  str = "BUY",
    ) -> float:
        """
        Returns total slippage cost in ₹ for one side (entry OR exit).
        """
        base     = price * shares * (self.base_bps / 10_000)
        notional = price * shares
        avg_notional = avg_volume * price if avg_volume > 0 else notional * 10
        pct_adv  = notional / avg_notional
        impact   = self.impact_k * np.sqrt(pct_adv) * notional
        return float(base + impact)

    def adjusted_price(
        self,
        price:      float,
        shares:     int,
        avg_volume: float,
        direction:  str = "BUY",
    ) -> float:
        """Returns price after slippage (worse for trader)."""
        slip_per_share = self.cost(price, shares, avg_volume, direction) / max(shares, 1)
        if direction == "BUY":
            return price + slip_per_share
        return price - slip_per_share
