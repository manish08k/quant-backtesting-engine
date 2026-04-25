"""data/data_cleaning.py — OHLCV cleaning with leak-safe timestamp alignment."""
import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger(__name__)

_REQUIRED = ["open", "high", "low", "close", "volume"]


class DataCleaner:
    def clean(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        # Ensure required columns
        for c in _REQUIRED:
            if c not in df.columns:
                if c == "volume":
                    df[c] = 1_000_000
                    log.warning(f"{symbol}: volume missing → set to 1M")
                else:
                    raise ValueError(f"{symbol}: required column {c!r} missing")

        df = df[_REQUIRED].copy()
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        # Remove pre/post-market (keep 09:15–15:30 IST)
        if isinstance(df.index, pd.DatetimeIndex):
            t = df.index.time
            mkt_open  = pd.Timestamp("09:15").time()
            mkt_close = pd.Timestamp("15:30").time()
            df = df[(t >= mkt_open) & (t <= mkt_close)]

        # Drop rows with any zero prices
        price_cols = ["open","high","low","close"]
        df = df[(df[price_cols] > 0).all(axis=1)]

        # Fix OHLC ordering violations
        df["high"] = df[["open","high","close"]].max(axis=1)
        df["low"]  = df[["open","low","close"]].min(axis=1)

        # Clip extreme volume spikes (> 50× rolling median)
        vol_med = df["volume"].rolling(20, min_periods=1).median()
        df["volume"] = df["volume"].clip(upper=vol_med * 50).fillna(0).astype(np.int64)

        # Forward-fill any remaining NaN (max 2 bars)
        df.ffill(limit=2, inplace=True)
        df.dropna(inplace=True)

        log.info(f"{symbol}: cleaned → {len(df)} bars")
        return df
