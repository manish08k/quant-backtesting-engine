"""backtest/performance.py — Comprehensive performance analytics."""
import numpy as np
import pandas as pd
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)


class PerformanceAnalyser:
    def analyse(self, trades: pd.DataFrame, capital: float) -> dict:
        if trades.empty:
            log.warning("No trades to analyse")
            return {}

        t = trades.copy()
        t["entry_time"] = pd.to_datetime(t["entry_time"])
        t["exit_time"]  = pd.to_datetime(t["exit_time"])
        t.sort_values("exit_time", inplace=True)

        pnl     = t["net_pnl"].values
        eq      = capital + np.cumsum(pnl)
        peak    = np.maximum.accumulate(eq)
        dd      = (eq - peak) / peak
        max_dd  = float(dd.min())

        wins    = pnl[pnl > 0]
        losses  = pnl[pnl < 0]
        n       = len(pnl)
        win_rate= len(wins) / n if n else 0
        avg_win = float(wins.mean()) if len(wins) else 0
        avg_loss= float(losses.mean()) if len(losses) else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Daily P&L for Sharpe
        daily = t.groupby(t["exit_time"].dt.date)["net_pnl"].sum()
        sharpe = (
            float(daily.mean() / daily.std() * np.sqrt(252))
            if daily.std() > 0 else 0.0
        )
        total_return = float((eq[-1] - capital) / capital)
        profit_factor= (
            float(wins.sum() / (-losses.sum()))
            if losses.sum() != 0 else np.inf
        )

        metrics = {
            "n_trades":      n,
            "win_rate":      round(win_rate, 4),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_loss, 2),
            "expectancy":    round(expectancy, 2),
            "profit_factor": round(profit_factor, 3),
            "sharpe":        round(sharpe, 3),
            "max_drawdown":  round(max_dd, 4),
            "total_return":  round(total_return, 4),
            "final_capital": round(float(eq[-1]), 2),
            "equity_curve":  pd.Series(eq, index=t["exit_time"].values),
        }

        log.info(
            f"\n{'='*55}\nPERFORMANCE REPORT\n{'='*55}\n"
            f"Trades:       {n}\n"
            f"Win Rate:     {win_rate:.1%}\n"
            f"Expectancy:   ₹{expectancy:.2f}\n"
            f"Profit Factor:{profit_factor:.2f}\n"
            f"Sharpe:       {sharpe:.2f}\n"
            f"Max Drawdown: {max_dd:.1%}\n"
            f"Total Return: {total_return:.1%}\n"
            f"Final Capital:₹{eq[-1]:,.0f}"
        )
        return metrics
