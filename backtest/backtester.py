"""
backtest/backtester.py
Event-driven bar-by-bar backtester.
Includes: slippage, brokerage, latency (execute on next bar open),
          dynamic SL/TP from ATR, trailing stops, daily halt.
"""
import numpy as np
import pandas as pd
from risk.risk_manager import RiskManager
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)

BC = CFG.backtest
R  = CFG.risk


class Backtester:
    def run(
        self,
        df:      pd.DataFrame,   # OHLCV aligned to signal index
        signals: pd.Series,       # BUY / SELL / HOLD
        probas:  pd.Series = None,
        atr:     pd.Series = None,
    ) -> pd.DataFrame:
        """
        Returns trades DataFrame with columns:
        entry_time, exit_time, direction, entry, exit, shares,
        gross_pnl, cost, net_pnl, return_pct, hold_bars
        """
        rm     = RiskManager(BC.capital)
        trades = []
        position = None   # {direction, entry, sl, tp, shares, entry_time, stop}

        bars = df.index.tolist()
        n    = len(bars)

        for i, bar_time in enumerate(bars):
            date  = pd.Timestamp(bar_time).date()
            o, h, l, c = (df.at[bar_time, col] for col in ["open","high","low","close"])
            a     = float(atr.at[bar_time]) if (atr is not None and bar_time in atr.index) else c * 0.005

            # ── Manage open position ──────────────────────────────────────────
            if position is not None:
                d    = position["direction"]
                sl   = position["sl"]
                tp   = position["tp"]
                stop = position["stop"]   # trailing stop

                # Update trailing stop
                stop = rm.trail_stop(stop, c, a, d)
                position["stop"] = stop
                effective_sl = max(sl, stop) if d == "BUY" else min(sl, stop)

                exit_price = None
                if d == "BUY":
                    if l <= effective_sl:
                        exit_price = effective_sl   # stopped out
                    elif h >= tp:
                        exit_price = tp             # TP hit
                else:  # SELL
                    if h >= effective_sl:
                        exit_price = effective_sl
                    elif l <= tp:
                        exit_price = tp

                # Intraday close-out: force exit at last bar of day
                if exit_price is None and i < n - 1:
                    next_bar   = bars[i + 1]
                    next_date  = pd.Timestamp(next_bar).date()
                    if next_date != date:
                        exit_price = c   # EOD exit at close

                if exit_price is not None:
                    slip   = exit_price * (BC.slippage_bps / 10_000)
                    brok   = exit_price * (BC.brokerage_bps / 10_000)
                    if d == "BUY":
                        gross = (exit_price - position["entry"]) * position["shares"]
                        adj_exit = exit_price - slip
                    else:
                        gross = (position["entry"] - exit_price) * position["shares"]
                        adj_exit = exit_price + slip

                    cost    = (brok + slip) * position["shares"]
                    net_pnl = gross - cost * 2   # entry + exit costs

                    trades.append({
                        "entry_time":  position["entry_time"],
                        "exit_time":   bar_time,
                        "direction":   d,
                        "entry":       position["entry"],
                        "exit":        adj_exit,
                        "shares":      position["shares"],
                        "gross_pnl":   gross,
                        "cost":        cost * 2,
                        "net_pnl":     net_pnl,
                        "return_pct":  net_pnl / (position["entry"] * position["shares"] + 1e-9),
                        "hold_bars":   i - position["entry_bar"],
                    })
                    rm.update(net_pnl, date)
                    position = None
                    continue

            # ── Check for new signal ──────────────────────────────────────────
            if position is not None:
                continue   # already in trade

            sig = signals.get(bar_time, "HOLD")
            if sig == "HOLD" or not rm.can_trade(date):
                continue

            # Execute on next bar open (latency = 1 bar)
            exec_idx = i + BC.latency_bars
            if exec_idx >= n:
                continue
            exec_time  = bars[exec_idx]
            exec_open  = df.at[exec_time, "open"]

            slip       = exec_open * (BC.slippage_bps / 10_000)
            entry_px   = exec_open + slip if sig == "BUY" else exec_open - slip
            shares     = rm.position_size(a, entry_px)
            if shares == 0:
                continue

            sl_price, tp_price = rm.sl_tp(entry_px, a, sig)
            position = {
                "direction":  sig,
                "entry":      entry_px,
                "sl":         sl_price,
                "tp":         tp_price,
                "stop":       sl_price,   # trailing starts at SL
                "shares":     shares,
                "entry_time": exec_time,
                "entry_bar":  exec_idx,
            }

        df_trades = pd.DataFrame(trades)
        log.info(f"Backtest complete: {len(df_trades)} trades")
        return df_trades
