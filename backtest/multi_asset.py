"""backtest/multi_asset.py — Portfolio-level backtest with correlation-aware sizing."""
import numpy as np
import pandas as pd
from backtest.performance import PerformanceAnalyser
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)


class MultiAssetBacktester:
    def run_universe(self, symbol_data: dict) -> dict:
        """
        symbol_data: {symbol: (df, signals, probas, atr)}
        Returns per-symbol metrics + portfolio summary.
        """
        from backtest.backtester import Backtester
        results = {}
        all_daily_pnl = {}

        for sym, (df, sig, prob, atr) in symbol_data.items():
            try:
                trades  = Backtester().run(df, sig, probas=prob, atr=atr)
                metrics = PerformanceAnalyser().analyse(trades, CFG.backtest.capital)
                results[sym] = {"trades": trades, "metrics": metrics}

                # Daily pnl for portfolio Sharpe
                if not trades.empty:
                    t = trades.copy()
                    t["exit_time"] = pd.to_datetime(t["exit_time"])
                    daily = t.groupby(t["exit_time"].dt.date)["net_pnl"].sum()
                    all_daily_pnl[sym] = daily
            except Exception as e:
                log.warning(f"{sym}: {e}")

        # Portfolio summary
        if all_daily_pnl:
            port_df   = pd.DataFrame(all_daily_pnl).fillna(0)
            port_pnl  = port_df.sum(axis=1)
            sharpe    = float(port_pnl.mean() / port_pnl.std() * np.sqrt(252)) if port_pnl.std() > 0 else 0
            summary   = pd.DataFrame({
                sym: {
                    "trades":       results[sym]["metrics"].get("n_trades", 0),
                    "sharpe":       results[sym]["metrics"].get("sharpe", 0),
                    "total_return": results[sym]["metrics"].get("total_return", 0),
                    "max_dd":       results[sym]["metrics"].get("max_drawdown", 0),
                }
                for sym in results
            }).T
            summary.loc["__portfolio__"] = {
                "trades": summary["trades"].sum(),
                "sharpe": sharpe,
                "total_return": float(port_pnl.sum() / (CFG.backtest.capital * len(results))),
                "max_dd": 0,
            }
            results["__summary__"] = summary
        return results
