"""
strategy/position_sizing.py
Strategy-layer wrapper: picks the right sizing method based on regime.
Trending regime → volatility-target sizing (ride the trend larger).
Ranging regime  → fixed-fraction sizing (tighter risk control).
"""
import pandas as pd
from risk.position_sizing import PositionSizer
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)


class StrategyPositionSizer:

    def __init__(self, capital: float = None):
        self.capital = capital or CFG.risk.capital
        self.sizer   = PositionSizer()

    def size(
        self,
        price:      float,
        atr:        float,
        daily_vol:  float,
        adx:        float,
        win_prob:   float = 0.55,
        win_mult:   float = 1.5,
    ) -> int:
        """
        Route to sizing method based on ADX regime:
          ADX >= 25 → volatility-target (trending)
          ADX <  25 → fixed-fraction (ranging / choppy)
        """
        if adx >= CFG.strategy.adx_trend_thresh:
            shares = self.sizer.volatility_target(
                self.capital, price, daily_vol, target_vol=0.15
            )
            method = "vol_target"
        else:
            shares = self.sizer.fixed_fraction(
                self.capital, price, atr
            )
            method = "fixed_frac"

        log.debug(f"Size [{method}]: price={price:.1f} atr={atr:.2f} → {shares} shares")
        return shares
