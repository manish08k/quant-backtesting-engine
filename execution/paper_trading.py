"""execution/paper_trading.py — Paper trading loop (runs every 5 min during market hours)."""
import time
import datetime
import pandas as pd
from data.data_loader  import YFinanceLoader
from data.data_cleaning import DataCleaner
from features.feature_engineering import FeatureEngineer
from features.feature_selection    import FeatureSelector
from strategy.signal_filter        import SignalGenerator
from risk.risk_manager             import RiskManager
from utils.config import CFG
from utils.logger import get_logger

log = get_logger(__name__)


class PaperTrader:
    def __init__(self, model, symbol: str, bench=None):
        self.model   = model
        self.symbol  = symbol
        self.bench   = bench
        self.rm      = RiskManager()
        self.eng     = FeatureEngineer()
        self.loader  = YFinanceLoader()
        self.cleaner = DataCleaner()
        self.sel     = FeatureSelector(top_k=CFG.features.top_k)
        self.pnl_log = []

    def _is_market_open(self) -> bool:
        now = datetime.datetime.now()
        if now.weekday() >= 5:
            return False
        mkt_open  = now.replace(hour=9,  minute=15, second=0)
        mkt_close = now.replace(hour=15, minute=30, second=0)
        return mkt_open <= now <= mkt_close

    def run(self, poll_seconds: int = 300):
        log.info(f"Paper trader started for {self.symbol}")
        while True:
            if not self._is_market_open():
                log.info("Market closed — sleeping 60s")
                time.sleep(60)
                continue

            try:
                raw = self.loader.fetch(self.symbol)
                df  = self.cleaner.clean(raw, self.symbol)

                bench_input = pd.DataFrame()
                if self.bench is not None and not self.bench.empty:
                    bi = self.bench.reindex(df.index, method="ffill").copy()
                    if "volume" not in bi.columns:
                        bi["volume"] = 1_000_000
                    bench_input = bi

                X = self.eng.build(df, benchmark=bench_input)
                # Use last bar only for signal
                X_last = X.iloc[[-1]]
                prob   = self.model.predict_proba(X_last)
                sig    = SignalGenerator().generate(
                    prob, features=X_last, threshold=self.model.threshold
                )
                last_sig = sig.iloc[-1]
                last_bar = df.index[-1]
                atr_val  = float(X.get("atr", pd.Series([df["close"].iloc[-1]*0.005])).iloc[-1])

                log.info(f"{last_bar}  price={df['close'].iloc[-1]:.2f}  "
                         f"prob={float(prob.iloc[-1]):.3f}  signal={last_sig}")

                if last_sig in ("BUY", "SELL") and self.rm.can_trade(last_bar.date()):
                    price  = df["close"].iloc[-1]
                    shares = self.rm.position_size(atr_val, price)
                    sl, tp = self.rm.sl_tp(price, atr_val, last_sig)
                    log.info(f"PAPER ORDER → {last_sig} {shares}×{self.symbol} "
                             f"@ {price:.2f}  SL={sl:.2f}  TP={tp:.2f}")

            except Exception as e:
                log.error(f"Paper loop error: {e}")

            time.sleep(poll_seconds)
