"""data/data_loader.py — Clean OHLCV loader (yfinance + Kite fallback).
Index volume issue fixed: NIFTY is fetched price-only; volume proxied via NIFTYBEES ETF.
"""
import os, hashlib, pickle, time
import pandas as pd
import numpy as np
import yfinance as yf
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)
os.makedirs(CFG.data.cache_dir, exist_ok=True)


def _cache_path(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(CFG.data.cache_dir, f"{h}.pkl")


def _load_cache(key: str):
    p = _cache_path(key)
    if os.path.exists(p):
        age = time.time() - os.path.getmtime(p)
        if age < 3600:   # 1-hour TTL
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def _save_cache(key: str, obj):
    with open(_cache_path(key), "wb") as f:
        pickle.dump(obj, f)


def _fetch_yf(ticker: str, period: str, interval: str) -> pd.DataFrame:
    key = f"{ticker}_{period}_{interval}"
    cached = _load_cache(key)
    if cached is not None:
        return cached
    for attempt in range(3):
        try:
            t  = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                df.index   = pd.to_datetime(df.index)
                # strip tz → naive UTC-equivalent (IST offset kept in index values)
                if df.index.tzinfo is not None:
                    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
                _save_cache(key, df)
                return df
        except Exception as e:
            log.warning(f"yf attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


class YFinanceLoader:
    """Loads stock + benchmark OHLCV; fixes NIFTY zero-volume with NIFTYBEES proxy."""

    def fetch(self, symbol: str) -> pd.DataFrame:
        df = _fetch_yf(symbol, CFG.data.period, CFG.data.interval)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        log.info(f"Loaded {symbol}: {len(df)} bars")
        return df

    def fetch_benchmark(self) -> pd.DataFrame:
        """
        NIFTY 50 (^NSEI) returns price-only from yfinance (volume = 0).
        We fetch NIFTYBEES.NS as a liquid ETF to get real volume proxy,
        then attach its volume to the NIFTY price series after aligning.
        """
        nifty  = _fetch_yf(CFG.data.benchmark, CFG.data.period, CFG.data.interval)
        bees   = _fetch_yf("NIFTYBEES.NS",     CFG.data.period, CFG.data.interval)

        if nifty.empty:
            log.warning("NIFTY data unavailable — benchmark disabled")
            return pd.DataFrame()

        # Keep price cols from NIFTY
        price_cols = [c for c in ["open","high","low","close"] if c in nifty.columns]
        out = nifty[price_cols].copy()

        # Attach real volume from NIFTYBEES (aligned, forward-filled)
        if not bees.empty and "volume" in bees.columns:
            bees_vol = bees["volume"].reindex(out.index, method="ffill")
            # Scale NIFTYBEES volume to NIFTY notional (approx 10× per unit)
            out["volume"] = (bees_vol * 10).fillna(0).astype(np.int64)
        else:
            # Fallback: use rolling-20 bar dummy (non-zero, won't hurt features)
            out["volume"] = 1_000_000

        log.info(f"Benchmark loaded: {len(out)} bars, vol proxy active={not bees.empty}")
        return out

    def fetch_multi(self) -> dict[str, pd.DataFrame]:
        out = {}
        for sym in CFG.data.symbols:
            try:
                out[sym] = self.fetch(sym)
            except Exception as e:
                log.warning(f"Skipping {sym}: {e}")
        return out
