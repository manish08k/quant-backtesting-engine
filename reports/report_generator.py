"""reports/report_generator.py — Equity curve + metrics report."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.logger import get_logger

log = get_logger(__name__)
os.makedirs("reports", exist_ok=True)


class ReportGenerator:
    def generate(self, metrics: dict, trades: pd.DataFrame, symbol: str):
        if not metrics or trades.empty:
            log.warning("No metrics to report")
            return

        eq = metrics.get("equity_curve")
        if eq is None or len(eq) == 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"Backtest Report — {symbol}", fontsize=14, fontweight="bold")

        # ── Equity curve ──────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(eq.values, color="steelblue", linewidth=1.2)
        ax.set_title(f"Equity Curve  |  Sharpe={metrics.get('sharpe',0):.2f}  "
                     f"Return={metrics.get('total_return',0):.1%}")
        ax.set_ylabel("Capital (₹)")
        ax.grid(True, alpha=0.3)

        # ── Drawdown ──────────────────────────────────────────────────────────
        ax = axes[1]
        peak = eq.cummax()
        dd   = (eq - peak) / peak * 100
        ax.fill_between(range(len(dd)), dd.values, 0, color="red", alpha=0.4)
        ax.set_title(f"Drawdown  |  Max={metrics.get('max_drawdown',0):.1%}")
        ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)

        # ── Trade P&L distribution ────────────────────────────────────────────
        ax = axes[2]
        pnl = trades["net_pnl"]
        ax.hist(pnl[pnl > 0], bins=30, color="green", alpha=0.6, label="Wins")
        ax.hist(pnl[pnl < 0], bins=30, color="red",   alpha=0.6, label="Losses")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"P&L Distribution  |  WR={metrics.get('win_rate',0):.1%}  "
                     f"Expectancy=₹{metrics.get('expectancy',0):.0f}")
        ax.set_xlabel("Net P&L (₹)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"reports/backtest_{symbol.replace('.','_')}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"Report saved → {path}")

        # Text summary
        lines = [
            f"=== {symbol} Backtest Report ===",
            f"Trades:        {metrics.get('n_trades',0)}",
            f"Win Rate:      {metrics.get('win_rate',0):.1%}",
            f"Expectancy:    ₹{metrics.get('expectancy',0):.2f}",
            f"Profit Factor: {metrics.get('profit_factor',0):.2f}",
            f"Sharpe Ratio:  {metrics.get('sharpe',0):.3f}",
            f"Max Drawdown:  {metrics.get('max_drawdown',0):.1%}",
            f"Total Return:  {metrics.get('total_return',0):.1%}",
            f"Final Capital: ₹{metrics.get('final_capital',0):,.0f}",
        ]
        txt_path = f"reports/backtest_{symbol.replace('.','_')}.txt"
        with open(txt_path, "w") as fh:
            fh.write("\n".join(lines))
        log.info(f"Text report → {txt_path}")
