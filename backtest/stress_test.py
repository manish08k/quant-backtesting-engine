"""backtest/stress_test.py — Monte Carlo + scenario stress tests."""
import numpy as np
import pandas as pd
from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyser
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)


class StressTester:
    def run(self, df, signals, probas, atr, n_mc=200) -> dict:
        base_trades  = Backtester().run(df, signals, probas=probas, atr=atr)
        base_metrics = PerformanceAnalyser().analyse(base_trades, CFG.backtest.capital)

        if base_trades.empty:
            return {"base": base_metrics}

        # Monte Carlo: shuffle trade order 200× → distribution of Sharpe / drawdown
        pnl = base_trades["net_pnl"].values
        mc_sharpes = []
        mc_dds     = []
        for _ in range(n_mc):
            shuffled = np.random.permutation(pnl)
            eq       = CFG.backtest.capital + np.cumsum(shuffled)
            peak     = np.maximum.accumulate(eq)
            dd       = (eq - peak) / peak
            daily_g  = pd.Series(shuffled).groupby(np.arange(len(shuffled)) // 75).sum()
            if daily_g.std() > 0:
                mc_sharpes.append(float(daily_g.mean() / daily_g.std() * np.sqrt(252)))
            mc_dds.append(float(dd.min()))

        results = {
            "base":          base_metrics,
            "mc_sharpe_p5":  float(np.percentile(mc_sharpes, 5))  if mc_sharpes else 0,
            "mc_sharpe_p50": float(np.percentile(mc_sharpes, 50)) if mc_sharpes else 0,
            "mc_dd_p95":     float(np.percentile(mc_dds, 95))     if mc_dds else 0,
        }
        log.info(f"Stress: Sharpe p5={results['mc_sharpe_p5']:.2f}  "
                 f"DD p95={results['mc_dd_p95']:.1%}")
        return results
