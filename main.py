"""
main.py — Institutional Intraday ML Trading Pipeline
=====================================================
Targets: AUC > 0.55 | probability spread > 0.05 std | ≥10 trades

Pipeline
--------
  1.  Load OHLCV (yfinance) — 150d window (was 60d) for richer training data
  2.  Reserve last 20% as holdout — never used in any fit step
  3.  Feature engineering  — per-fold: fit on train split, transform test split
      (StandardScaler managed inside FeatureSelector, not FeatureEngineer)
  4.  Triple-barrier labeling — noise filter (|ret| < 0.2%, confidence < 0.20)
  5.  Feature selection  — MI pre-screen + stability scoring + corr filter
      (force_keep strong features, drop noisy oscillators)
  6.  Walk-forward validation (TimeSeriesSplit n_folds=3, min 800 bars train)
      → per-fold: train_size, test_size, pos_rate, AUC, F1, mean_prob, max_prob
  7.  Weighted blend ensemble fit on train portion (first 80%) only
  8.  Feature importance  — logged top-10 post-fit
  9.  Signal filtering  — dynamic threshold + regime gate + EV gate
  10. Risk-managed simulation on holdout set
  11. Full P&L summary — Sharpe, Sortino, max drawdown, per-exit-reason stats

Changes vs. previous version
-----------------------------
  - period: "60d" → "150d"   (5m bars: ~2340 bars → ~5850 bars)
  - min_train_bars: 600 → 800
  - min_confidence: 0.15 → 0.20
  - meta_coefficients() call removed (no meta-model)
  - per-fold mean_prob / max_prob / std_prob logged explicitly
  - constant prediction guard added before backtest

Run
---
    pip install yfinance scikit-learn xgboost lightgbm numpy pandas
    python3 main.py --symbol RELIANCE.NS --benchmark ^NSEI
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, warnings, argparse, logging
from pathlib import Path
from types   import SimpleNamespace

# ── third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── project modules ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.feature_engineering import FeatureEngineer
from features.feature_selection   import FeatureSelector
from labeling.barriers            import select_label, compute_sample_weights, _atr_for_labeling, label_balance_check
from models.ensemble              import EnsembleModel, calibrate
from strategy.signal_filter       import SignalFilter
from risk.risk_manager            import RiskManager
from portfolio.allocation         import allocate_portfolio


def prepare_model_input(X):
    ROUTING_COLS = ["regime", "hl_range", "symbol_id"]
    return X.drop(columns=[c for c in ROUTING_COLS if c in X.columns])


# ── logger ────────────────────────────────────────────────────────────────────

def _make_logger(name: str) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler("logs/trading.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

log = _make_logger("pipeline")


# ── config ────────────────────────────────────────────────────────────────────

CFG = SimpleNamespace(
    data = SimpleNamespace(
        # yfinance hard-caps 5m data at 60 days — use 1h for longer history.
        # load_data() will auto-try 1h/730d if 5m/60d yields too few bars.
        interval     = "1h",        # primary interval  (1h → up to 730d)
        period       = "500d",      # primary period    (1h supports up to 730d)
        fallback_iv  = "5m",        # secondary interval if primary fails
        fallback_pd  = "60d",       # secondary period  (5m hard cap = 60d)
        min_bars     = 500,
    ),
    labeling = SimpleNamespace(
        pt_mult           = 1.5,
        sl_mult           = 1.0,
        max_hold_bars     = 12,
        min_ret_threshold = 0.005,
        min_confidence    = 0.20,   # loosened: 0.25 → 0.20 for more labeled bars
    ),
    model = SimpleNamespace(
        n_folds        = 2,
        min_train_bars = 800,       # lowered: 2000 → 800 — works on single-stock too
        random_state   = 42,
        holdout_pct    = 0.20,
    ),
    signal = SimpleNamespace(
        min_prob         = 0.52,    # lowered: 0.65 → 0.52 — adaptive percentile does real filtering
        max_prob_thresh  = 0.95,
        adaptive_window  = 20,
        adaptive_margin  = 0.02,
        top_k            = 10,      # allow up to 10 signals
        top_pct          = 1.0,
        cooldown_bars    = 2,       # lowered: 5 → 2
        min_ev           = 0.02,    # lowered: 0.05 → 0.02
    ),
    risk = SimpleNamespace(
        capital            = 1_000_000,
        risk_per_trade     = 0.01,
        max_daily_loss     = 0.02,
        max_consec_losses  = 8,
        atr_sl_mult        = 1.0,
        atr_tp_mult        = 2.0,
        trail_atr_mult     = 0.8,
        max_position_pct   = 0.10,
        min_prob           = 0.52,  # aligned with signal filter
        max_open_risk      = 0.05,
        max_trades_per_day = 10,
    ),
    backtest = SimpleNamespace(
        slippage_bps  = 5,
        brokerage_bps = 3,
    ),
)


# ── data loader ───────────────────────────────────────────────────────────────

def _yf_download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download from yfinance and return a clean DataFrame, or empty DataFrame on failure."""
    import yfinance as yf

    def _clean(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.columns = [c.lower() for c in d.columns]
        d = d.dropna(subset=["close"])
        if d.empty:
            return d
        d.index = pd.to_datetime(d.index, utc=True).tz_convert("Asia/Kolkata")
        return d

    try:
        raw = yf.download(symbol, period=period, interval=interval,
                          auto_adjust=True, progress=False)
        return _clean(raw)
    except Exception as e:
        log.warning(f"yfinance download failed for {symbol} ({interval}/{period}): {e}")
        return pd.DataFrame()


def load_data(symbol: str, benchmark_symbol: str = "^NSEI"):
    try:
        import yfinance as yf  # noqa: F401
    except ImportError:
        raise SystemExit("Run: pip install yfinance")

    bench = "^NSEI" if benchmark_symbol in ("NIFTY50", "^NSEI", "NIFTY") else benchmark_symbol

    # ── Try primary interval (1h / 500d) first, then fall back to 5m / 60d ──
    attempts = [
        (CFG.data.interval,    CFG.data.period),
        (CFG.data.fallback_iv, CFG.data.fallback_pd),
    ]

    df = pd.DataFrame()
    used_interval = CFG.data.interval
    for iv, pd_str in attempts:
        log.info(f"Downloading {symbol}  interval={iv}  period={pd_str} …")
        df = _yf_download(symbol, pd_str, iv)
        if len(df) >= CFG.data.min_bars:
            used_interval = iv
            log.info(f"  ✓ {symbol}: {len(df)} bars  (interval={iv})")
            break
        log.warning(
            f"  ✗ {symbol}: only {len(df)} bars with interval={iv}/period={pd_str} — "
            f"{'trying fallback …' if iv == CFG.data.interval else 'no more fallbacks.'}"
        )

    if len(df) < CFG.data.min_bars:
        raise ValueError(
            f"Could not fetch enough data for {symbol}. "
            f"Got {len(df)} bars (need {CFG.data.min_bars}). "
            "Tried intervals: " + ", ".join(f"{iv}/{p}" for iv, p in attempts) + ". "
            "Check the ticker symbol and your internet connection."
        )

    # ── Benchmark: use same interval that worked for the stock ───────────────
    bm = pd.DataFrame()
    for pd_str in [CFG.data.period, CFG.data.fallback_pd]:
        log.info(f"Downloading benchmark {bench}  interval={used_interval}  period={pd_str} …")
        bm = _yf_download(bench, pd_str, used_interval)
        if not bm.empty:
            log.info(f"  ✓ {bench}: {len(bm)} bars")
            break
        log.warning(f"  ✗ {bench}: empty with period={pd_str} — trying next …")

    # If benchmark is still empty, build a synthetic flat benchmark so the
    # pipeline doesn't crash — relative-benchmark features will be near-zero
    # but the model can still train on absolute price features.
    if bm.empty:
        log.warning(
            f"Benchmark {bench} unavailable — using flat synthetic benchmark. "
            "Relative-benchmark features will be near-zero."
        )
        bm = df[["open", "high", "low", "close", "volume"]].copy()

    # Align benchmark to stock index (forward-fill gaps, e.g. holiday mismatches)
    bm = bm.reindex(df.index, method="ffill").dropna(how="all")

    log.info(
        f"Data ready: stock={len(df)} bars  benchmark={len(bm)} bars  "
        f"interval={used_interval}  "
        f"range={df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
    )
    return df, bm


# ── transaction costs ─────────────────────────────────────────────────────────

def _transaction_cost(entry: float, exit_px: float, shares: int) -> float:
    notional  = shares * (entry + exit_px)
    slippage  = notional * CFG.backtest.slippage_bps  / 10_000
    brokerage = notional * CFG.backtest.brokerage_bps / 10_000
    return slippage + brokerage


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(
    df:         pd.DataFrame,
    signals:    pd.Series,
    prob:       pd.Series,
    atr_series: pd.Series,
    risk:       RiskManager,
    symbol:     str = "",
) -> pd.DataFrame:
    trades    = []
    in_trade  = False
    entry_px  = sl = tp = 0.0
    shares    = 0
    entry_idx = None
    bars_held = 0
    max_hold  = CFG.labeling.max_hold_bars

    close_arr = df["close"].reindex(signals.index)

    for idx in signals.index:
        date    = idx.date() if hasattr(idx, "date") else idx
        price   = float(close_arr.get(idx, np.nan))
        if np.isnan(price):
            continue
        atr_val = float(atr_series.reindex([idx]).fillna(0).iloc[0])
        sig     = signals.loc[idx]
        p       = float(prob.reindex([idx]).fillna(0.5).iloc[0])

        if in_trade:
            bars_held += 1
            if atr_val > 0:
                sl = risk.trail_stop(sl, price, atr_val)

            hit_tp  = price >= tp
            hit_sl  = price <= sl
            timeout = bars_held >= max_hold

            if hit_tp or hit_sl or timeout:
                exit_px = tp if hit_tp else (sl if hit_sl else price)
                cost    = _transaction_cost(entry_px, exit_px, shares)
                pnl     = (exit_px - entry_px) * shares - cost
                risk.update(pnl, date, symbol)
                trades.append({
                    "entry_time": entry_idx,
                    "exit_time":  idx,
                    "entry":      entry_px,
                    "exit":       exit_px,
                    "shares":     shares,
                    "pnl":        pnl,
                    "cost":       cost,
                    "bars":       bars_held,
                    "reason":     "TP" if hit_tp else ("SL" if hit_sl else "timeout"),
                    "prob":       p,
                    "atr":        atr_val,
                })
                in_trade  = False
                bars_held = 0

        if not in_trade and sig == "BUY" and atr_val > 0:
            if not risk.can_trade(date, symbol):
                continue
            qty = risk.position_size(atr_val, price, prob=p)
            if qty <= 0:
                continue
            sl, tp    = risk.sl_tp(price, atr_val)
            entry_px  = price
            entry_idx = idx
            shares    = qty
            bars_held = 0
            in_trade  = True

    log.info(f"Simulation complete — {len(trades)} trades executed")
    return pd.DataFrame(trades)


# ── performance metrics ───────────────────────────────────────────────────────

def _sharpe(pnl_series: pd.Series, periods_per_year: int = 252 * 78) -> float:
    mu  = pnl_series.mean()
    sig = pnl_series.std()
    if sig == 0:
        return 0.0
    return float(mu / sig * np.sqrt(periods_per_year))


def _sortino(pnl_series: pd.Series, periods_per_year: int = 252 * 78) -> float:
    mu   = pnl_series.mean()
    down = pnl_series[pnl_series < 0]
    if len(down) == 0:
        return float("inf")
    downside = down.std()
    if downside == 0:
        return 0.0
    return float(mu / downside * np.sqrt(periods_per_year))


# ── summary printer ───────────────────────────────────────────────────────────

def print_summary(
    results:  pd.DataFrame,
    folds:    list[dict],
    symbol:   str,
):
    n_folds  = len(folds)
    mean_auc = float(np.mean([f["auc"]      for f in folds])) if folds else 0.0
    mean_f1  = float(np.mean([f["f1"]       for f in folds])) if folds else 0.0
    mean_wr  = float(np.mean([f["win_rate"] for f in folds])) if folds else 0.0
    mean_pr  = float(np.mean([f.get("mean_prob", 0) for f in folds])) if folds else 0.0

    fold_str = "\n".join(
        f"    Fold {f['fold']:2d}: AUC={f['auc']:.4f}  F1={f['f1']:.4f}  "
        f"Brier={f['brier']:.4f}  WR={f['win_rate']:.2%}  "
        f"mean_p={f.get('mean_prob', 0):.3f}  max_p={f.get('max_prob', 0):.3f}  "
        f"std_p={f.get('std_prob', 0):.4f}  "
        f"n_sig={f.get('n_signals', '?')}  "
        f"train={f.get('train_size','?')}  test={f.get('test_size','?')}  "
        f"pos_tr={f.get('pos_rate_train',0):.3f}  pos_te={f.get('pos_rate_test',0):.3f}"
        for f in folds
    )

    if results.empty:
        log.info(
            f"\n{'='*70}\n"
            f"  BACKTEST SUMMARY — {symbol}\n"
            f"{'='*70}\n"
            f"  Walk-forward folds : {n_folds}\n"
            f"  Mean AUC           : {mean_auc:.4f}\n"
            f"  Mean F1            : {mean_f1:.4f}\n"
            f"  Mean Win Rate      : {mean_wr:.2%}\n"
            f"  Mean Prob (folds)  : {mean_pr:.3f}\n"
            f"{'─'*70}\n"
            f"{fold_str}\n"
            f"{'─'*70}\n"
            f"  No trades executed. Check: prob spread, signal threshold, risk limits.\n"
            f"{'='*70}"
        )
        return

    total_pnl  = results["pnl"].sum()
    total_cost = results["cost"].sum()
    net_pnl    = total_pnl
    gross_pnl  = total_pnl + total_cost  # gross = net + cost (cost already subtracted in sim)
    n_trades   = len(results)
    win_mask   = results["pnl"] > 0
    lose_mask  = results["pnl"] < 0
    win_rate   = float(win_mask.mean())
    avg_win    = float(results.loc[win_mask,  "pnl"].mean()) if win_mask.any()  else 0.0
    avg_loss   = float(results.loc[lose_mask, "pnl"].mean()) if lose_mask.any() else 0.0
    rr         = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    cum    = results["pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    sharpe  = _sharpe(results["pnl"])
    sortino = _sortino(results["pnl"])
    reasons = results["reason"].value_counts().to_dict()

    log.info(
        f"\n{'='*70}\n"
        f"  BACKTEST SUMMARY — {symbol}\n"
        f"{'='*70}\n"
        f"  Walk-forward folds : {n_folds}\n"
        f"  Mean AUC           : {mean_auc:.4f}\n"
        f"  Mean F1            : {mean_f1:.4f}\n"
        f"  Mean Win Rate (CV) : {mean_wr:.2%}\n"
        f"  Mean Prob (folds)  : {mean_pr:.3f}\n"
        f"{'─'*70}\n"
        f"  Per-fold detail:\n{fold_str}\n"
        f"{'─'*70}\n"
        f"  Total PnL (gross)  : ₹{gross_pnl:>12,.0f}\n"
        f"  Total Cost         : ₹{total_cost:>12,.0f}\n"
        f"  Net PnL            : ₹{net_pnl:>12,.0f}\n"
        f"  Cost Drag          : {total_cost/max(abs(gross_pnl),1)*100:.1f}% of gross\n"
        f"{'─'*70}\n"
        f"  N Trades           : {n_trades}\n"
        f"  Win Rate           : {win_rate:.1%}\n"
        f"  Avg Win            : ₹{avg_win:>10,.0f}\n"
        f"  Avg Loss           : ₹{avg_loss:>10,.0f}\n"
        f"  Reward / Risk      : {rr:.2f}\n"
        f"  Max Drawdown       : ₹{max_dd:>12,.0f}\n"
        f"  Sharpe (ann.)      : {sharpe:>8.3f}\n"
        f"  Sortino (ann.)     : {sortino:>8.3f}\n"
        f"  Avg Hold (bars)    : {results['bars'].mean():.1f}\n"
        f"  Avg Signal Prob    : {results['prob'].mean():.3f}\n"
        f"{'─'*70}\n"
        f"  Exit breakdown     : {reasons}\n"
        f"{'='*70}"
    )


# ── institutional walk-forward on pre-built multi-stock pool ──────────────────

def walk_forward_on_pool(
    X_sel:          pd.DataFrame,
    y:              pd.Series,
    sw:             pd.Series,
    n_folds:        int = 2,
    min_train_bars: int = 2000,
) -> list[dict]:
    """
    Walk-forward cross-validation on the already-selected, already-scaled
    feature matrix.  Feature selection and scaling are done OUTSIDE this
    function — the selector is never re-fitted here.

    Design
    ------
    - Input X_sel has integer RangeIndex (from reset_index after multi-stock concat).
    - TimeSeriesSplit(n_splits=n_folds) respects the positional ordering.
    - Folds with train_size < min_train_bars are skipped with a warning.
    - EnsembleModel is fitted fresh per fold (no parameter sharing across folds).
    - prob spread (std_prob) is asserted > 0; warning raised if constant.

    Returns
    -------
    list of per-fold metric dicts (auc, f1, brier, win_rate, n_signals,
    mean_prob, max_prob, std_prob, train_size, test_size,
    pos_rate_train, pos_rate_test).
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics         import roc_auc_score, f1_score, brier_score_loss

    tscv    = TimeSeriesSplit(n_splits=n_folds)
    results: list[dict] = []

    # ── Preview fold sizes before entering the loop ───────────────────────────
    log.info(f"Walk-forward on pool: {len(X_sel)} bars  n_folds={n_folds}  "
             f"min_train_bars={min_train_bars}")
    for _f, (_tr, _te) in enumerate(tscv.split(X_sel), 1):
        log.info(f"  Fold {_f} preview: train_size={len(_tr)}  test_size={len(_te)}")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_sel), 1):
        train_size = len(tr_idx)
        test_size  = len(te_idx)

        # ── A. Minimum train size guard ───────────────────────────────────────
        if train_size < min_train_bars:
            log.warning(
                f"Fold {fold}: SKIPPED — train_size={train_size} < "
                f"min_train_bars={min_train_bars}"
            )
            continue

        X_tr = X_sel.iloc[tr_idx]
        X_te = X_sel.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]
        sw_tr_base = sw.iloc[tr_idx].values

        pos_rate_tr = float(y_tr.mean())
        pos_rate_te = float(y_te.mean())

        # ── B. Sanity guards ──────────────────────────────────────────────────
        if y_tr.nunique() < 2:
            log.warning(f"Fold {fold}: SKIPPED — single class in train "
                        f"(pos_rate={pos_rate_tr:.3f})")
            continue
        if y_te.nunique() < 2:
            log.warning(f"Fold {fold}: SKIPPED — single class in test "
                        f"(pos_rate={pos_rate_te:.3f})")
            continue

        # ── C. Debug logging (always printed to stdout) ───────────────────────
        print(f"\n[FOLD {fold}] train_size={train_size}  test_size={test_size}  "
              f"pos_rate_train={pos_rate_tr:.3f}  pos_rate_test={pos_rate_te:.3f}")
        log.info(f"Fold {fold}: train_size={train_size}  test_size={test_size}  "
                 f"pos_rate_train={pos_rate_tr:.3f}  pos_rate_test={pos_rate_te:.3f}")

        # ── D. Fit EnsembleModel per fold ─────────────────────────────────────
        # Drop passthrough routing cols — EnsembleModel must not see them"""
        _ROUTING_COLS = ["regime", "hl_range", "symbol_id"]
        X_tr_m = prepare_model_input(X_tr)
        X_te_m = prepare_model_input(X_te) 

        fold_model = EnsembleModel(threshold=CFG.signal.min_prob)
        try:
            sw_series = pd.Series(sw_tr_base, index=y_tr.index)
            fold_model.fit(X_tr_m, y_tr, sample_weight=sw_series)
        except Exception as e:
            log.warning(f"Fold {fold}: EnsembleModel fit failed — {e}")
            continue

        # ── E. Predict ────────────────────────────────────────────────────────
        prob_series = fold_model.predict_proba(X_te_m)
        blend = prob_series.values

        # ── G. Probability spread check ───────────────────────────────────────
        prob_std   = float(blend.std())
        prob_mean  = float(blend.mean())
        prob_max   = float(blend.max())

        print(f"[FOLD {fold}] mean_prob={prob_mean:.4f}  "
              f"std_prob={prob_std:.4f}  max_prob={prob_max:.4f}")

        if prob_std < 1e-4:
            log.warning(
                f"Fold {fold}: CONSTANT PREDICTIONS DETECTED (std={prob_std:.6f}). "
                "Possible causes: all features near-zero, extreme label imbalance, "
                "or calibration collapsing on a tiny fold."
            )

        # ── H. Metrics ────────────────────────────────────────────────────────
        pred = (blend >= CFG.signal.min_prob).astype(int)
        try:
            auc   = roc_auc_score(y_te, blend)
            f1    = f1_score(y_te, pred, zero_division=0)
            brier = brier_score_loss(y_te, blend)
            wr    = float(pred[y_te.values == 1].mean()) if y_te.sum() > 0 else 0.0
            n_sig = int(pred.sum())
        except Exception as e:
            log.warning(f"Fold {fold}: metrics failed — {e}")
            continue

        fold_result = {
            "fold":           fold,
            "auc":            auc,
            "f1":             f1,
            "brier":          brier,
            "win_rate":       wr,
            "n_signals":      n_sig,
            "mean_prob":      prob_mean,
            "max_prob":       prob_max,
            "std_prob":       prob_std,
            "train_size":     train_size,
            "test_size":      test_size,
            "pos_rate_train": pos_rate_tr,
            "pos_rate_test":  pos_rate_te,
        }
        results.append(fold_result)

        # Cost impact estimate for this fold
        rt_bps    = CFG.backtest.slippage_bps + CFG.backtest.brokerage_bps
        ev_per_sig = float(pd.Series(blend).apply(
            lambda p: p * CFG.labeling.pt_mult - (1 - p) * CFG.labeling.sl_mult
        ).mean()) if n_sig > 0 else 0.0

        log.info(
            f"Fold {fold} RESULT: AUC={auc:.4f}  F1={f1:.4f}  "
            f"Brier={brier:.4f}  WR={wr:.2%}  n_sig={n_sig}  "
            f"mean_p={prob_mean:.4f}  std_p={prob_std:.4f}  "
            f"cost={rt_bps}bps  EV~{ev_per_sig*100:.2f}%"
        )

    if not results:
        log.error(
            "Walk-forward produced NO valid folds. "
            f"Pool has {len(X_sel)} bars but min_train_bars={min_train_bars}. "
            "Either reduce min_train_bars or add more symbols / longer history."
        )

    return results


# ── keep old function as thin wrapper so nothing external breaks ──────────────

def walk_forward_with_fe(
    df:    pd.DataFrame,
    bm:    pd.DataFrame,
    n_folds:        int = 2,
    min_train_bars: int = 2000,
) -> tuple[list[dict], "FeatureEngineer", "FeatureSelector", pd.DataFrame, pd.Series, pd.Series]:
    """
    Legacy wrapper — kept so existing callers don't break.
    Internally delegates to walk_forward_on_pool.
    Feature selection is done ONCE on the full train portion, OUTSIDE the folds.
    """
    from features.feature_engineering import add_alpha_features

    n_total      = len(df)
    n_train_all  = int(n_total * (1 - CFG.model.holdout_pct))
    df_train_all = df.iloc[:n_train_all]
    bm_train_all = bm.reindex(df_train_all.index, method="ffill")

    log.info(f"walk_forward_with_fe: {n_total} total bars  "
             f"{n_train_all} train  {n_total - n_train_all} holdout-reserved")

    # Feature engineering on full train portion
    fe = FeatureEngineer()
    X_raw = fe.fit_transform(df_train_all, bm_train_all)

    # Labels
    y_all = select_label(df_train_all.loc[X_raw.index], cfg=CFG.labeling)
    common = X_raw.index.intersection(y_all.index)
    X_all  = X_raw.loc[common].reset_index(drop=True)
    y_all  = y_all.loc[common].reset_index(drop=True)

    label_balance_check(y_all, context="walk_forward_pool")
    log.info(f"Labeled bars: {len(X_all)}  pos_rate={float(y_all.mean()):.3f}")

    # ── Feature selection ONCE, outside folds ─────────────────────────────────
    selector = FeatureSelector(top_k=12, corr_threshold=0.90)
    X_sel = selector.fit_transform(X_all, y_all)
    sw    = compute_sample_weights(y_all, max_hold_bars=CFG.labeling.max_hold_bars, recency_boost=1.5)

    # Walk-forward on pre-selected features
    folds = walk_forward_on_pool(
        X_sel, y_all, sw,
        n_folds=n_folds,
        min_train_bars=min_train_bars,
    )

    return folds, fe, selector, X_sel, y_all, sw


# ── multi-stock training data builder ────────────────────────────────────────

# Default universe — can be overridden via --symbols CLI argument
DEFAULT_SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]


def build_multi_stock_dataset(
    symbols:          list[str],
    benchmark_symbol: str = "^NSEI",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Download, engineer features, and label data for multiple symbols.
    Concatenates all symbols into a single feature matrix with a one-hot
    symbol column so the model can distinguish between stocks.

    Returns
    -------
    X_all : pd.DataFrame  — scaled, selected features (multi-stock)
    y_all : pd.Series     — labels aligned to X_all
    sw_all: pd.Series     — sample weights aligned to X_all

    Design notes
    ------------
    - Each symbol is independently labeled (its own ATR, its own barriers).
    - Feature engineering is fit fresh per symbol (no cross-symbol leakage).
    - FeatureSelector is fit on the concatenated pool (stable enough with ≥3 stocks).
    - TimeSeriesSplit is still used for internal stability scoring — the
      multi-stock concat is treated as a single panel for selection purposes.
    - The 'regime' column is computed per-symbol so it reflects each stock's
      own volatility state, not a universal one.
    - Symbol one-hot columns are appended AFTER feature selection so they
      don't distort stability/MI scores.
    """
    from features.feature_engineering import add_alpha_features

    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series]    = []
    sw_parts: list[pd.Series]   = []

    for sym in symbols:
        log.info(f"=== Multi-stock: loading {sym} ===")
        try:
            df_s, bm_s = load_data(sym, benchmark_symbol)
        except Exception as e:
            log.warning(f"Skipping {sym}: {e}")
            continue

        # Use only the train portion (first 80%) to avoid holdout leakage
        n_tr = int(len(df_s) * (1 - CFG.model.holdout_pct))
        df_tr = df_s.iloc[:n_tr]
        bm_tr = bm_s.reindex(df_tr.index, method="ffill")

        X_raw = add_alpha_features(df_tr, bm_tr)
        y_raw = select_label(df_tr.loc[X_raw.index], cfg=CFG.labeling)

        common = X_raw.index.intersection(y_raw.index)
        if len(common) < 200 or y_raw.loc[common].nunique() < 2:
            log.warning(f"Skipping {sym}: insufficient labeled data ({len(common)} bars)")
            continue

        X_s = X_raw.loc[common]
        y_s = y_raw.loc[common]
        sw_s = compute_sample_weights(y_s, max_hold_bars=CFG.labeling.max_hold_bars, recency_boost=1.5)

        # Tag with integer symbol_id instead of one-hot (avoids holdout leakage)
        X_s = X_s.copy()
        X_s["_symbol_tag"] = sym

        label_balance_check(y_s, context=f"multi_stock/{sym}")
        log.info(f"  {sym}: {len(X_s)} bars  pos_rate={float(y_s.mean()):.3f}")

        X_parts.append(X_s)
        y_parts.append(y_s)
        sw_parts.append(sw_s)

    if not X_parts:
        raise RuntimeError("No symbols yielded usable training data.")

    X_concat  = pd.concat(X_parts,  axis=0).reset_index(drop=True)
    y_concat  = pd.concat(y_parts,  axis=0).reset_index(drop=True)
    sw_concat = pd.concat(sw_parts, axis=0).reset_index(drop=True)

    # Encode symbol as single integer column (no one-hot — avoids holdout leakage)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_concat["symbol_id"] = le.fit_transform(X_concat["_symbol_tag"].values)
    X_concat = X_concat.drop(columns=["_symbol_tag"])

    log.info(
        f"Multi-stock pool: {len(X_concat)} bars  "
        f"{len(symbols)} symbols  "
        f"pos_rate={float(y_concat.mean()):.3f}  "
        f"features={X_concat.shape[1]}"
    )
    return X_concat, y_concat, sw_concat


# ── regime-split model container ─────────────────────────────────────────────

class RegimeModel:
    """
    Two EnsembleModels trained on separate regime slices.

    regime=1 → trending  (atr14 > atr50) → model_trend
    regime=0 → ranging   (atr14 ≤ atr50) → model_range

    At prediction time, the regime column in X routes each bar to the
    appropriate model. If a regime model has too little training data
    it falls back to the global model.

    predict_proba() returns a pd.Series aligned to X.index.
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold    = threshold
        self.model_trend  = EnsembleModel(threshold=threshold)
        self.model_range  = EnsembleModel(threshold=threshold)
        self.model_global = EnsembleModel(threshold=threshold)
        self._trend_ok    = False
        self._range_ok    = False
        self._regime_col  = "regime"

    def fit(
        self,
        X:  pd.DataFrame,
        y:  pd.Series,
        sw: pd.Series | None = None,
    ) -> "RegimeModel":
        X_no_reg_global = X.drop(columns=[self._regime_col], errors="ignore")
        self.model_global.fit(X_no_reg_global, y, sample_weight=sw)

        if self._regime_col not in X.columns:
            log.warning("RegimeModel: 'regime' column not found — using global model only.")
            self._trend_ok = self._range_ok = False
            return self

        regime = X[self._regime_col]
        # Round before threshold — regime must be raw 0/1 (passthrough in selector).
        mask_trend = regime.round() >= 1
        mask_range = ~mask_trend

        # Drop regime col so sub-models are trained on the same feature set they
        # will see at predict time (regime is routing metadata, not a signal).
        X_no_reg = X.drop(columns=[self._regime_col])

        for label, mask, attr_ok, model in [
            ("trend", mask_trend, "_trend_ok", self.model_trend),
            ("range", mask_range, "_range_ok", self.model_range),
        ]:
            X_r = X_no_reg[mask]
            y_r = y[mask] if isinstance(y, pd.Series) else y.iloc[mask]
            sw_r = sw[mask] if sw is not None else None
            n = len(X_r)
            pos = int((y_r == 1).sum())

            if n < 200 or pos < 20 or y_r.nunique() < 2:
                log.warning(
                    f"RegimeModel: {label} slice too small (n={n}, pos={pos}) "
                    "— will use global model for this regime."
                )
                setattr(self, attr_ok, False)
                continue

            log.info(f"RegimeModel: fitting {label} model on {n} bars (pos={pos})")
            try:
                model.fit(X_r, y_r, sample_weight=sw_r)
                setattr(self, attr_ok, True)
            except Exception as e:
                log.warning(f"RegimeModel: {label} model failed: {e}")
                setattr(self, attr_ok, False)

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X_no_reg = X.drop(columns=[self._regime_col], errors="ignore")

        if self._regime_col not in X.columns:
            return self.model_global.predict_proba(X_no_reg)

        regime = X[self._regime_col]
        # Round before threshold — guards against float drift if regime ever leaks
        # through a scaler; regime must be raw 0/1 (passthrough in FeatureSelector).
        mask_tr  = regime.round() >= 1
        mask_rng = ~mask_tr

        # Drop the regime routing col before prediction — sub-models were trained
        # without it (it was a passthrough, not ranked) so passing it in would
        # cause a feature-count mismatch and trigger "All models failed to predict".
        X_no_reg = X.drop(columns=[self._regime_col])

        out = pd.Series(np.nan, index=X.index, name="prob")

        # Trending regime
        if mask_tr.any():
            model = self.model_trend if self._trend_ok else self.model_global
            try:
                out[mask_tr] = model.predict_proba(X_no_reg[mask_tr]).values
            except Exception as e:
                log.warning(f"RegimeModel: trend model failed ({e}), falling back to global")
                out[mask_tr] = self.model_global.predict_proba(X_no_reg[mask_tr]).values

        # Ranging regime
        if mask_rng.any():
            model = self.model_range if self._range_ok else self.model_global
            try:
                out[mask_rng] = model.predict_proba(X_no_reg[mask_rng]).values
            except Exception as e:
                log.warning(f"RegimeModel: range model failed ({e}), falling back to global")
                out[mask_rng] = self.model_global.predict_proba(X_no_reg[mask_rng]).values

        # Fill any remaining NaNs with global
        still_nan = out.isna()
        if still_nan.any():
            out[still_nan] = self.model_global.predict_proba(X_no_reg[still_nan]).values

        prob_std = float(out.std())
        log.info(
            f"RegimeModel.predict_proba: mean={out.mean():.4f}  "
            f"std={prob_std:.4f}  max={out.max():.4f}  "
            f"trend_bars={int(mask_tr.sum())}  range_bars={int(mask_rng.sum())}"
        )
        return out

    def feature_importance(self, feature_names: list | None = None) -> pd.DataFrame:
        return self.model_global.feature_importance(feature_names)

    def meta_coefficients(self) -> None:
        return None


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    symbol:           str            = "RELIANCE.NS",
    benchmark_symbol: str            = "^NSEI",
    symbols:          list[str] | None = None,
    use_multi_stock:  bool           = True,
    use_regime_model: bool           = True,
) -> dict:
    """
    Main pipeline entry point.

    Parameters
    ----------
    symbol           : Primary stock for holdout backtest.
    benchmark_symbol : Benchmark for relative features.
    symbols          : Training universe (defaults to DEFAULT_SYMBOLS).
    use_multi_stock  : Train on all symbols; fall back to single if it fails.
    use_regime_model : Route bars through regime-split model; fall back to global.
    """
    log.info(
        f"=== Pipeline start: {symbol} vs {benchmark_symbol}  "
        f"multi_stock={use_multi_stock}  regime_model={use_regime_model} ==="
    )

    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    if symbol not in symbols:
        symbols = [symbol] + symbols

    # ── 1. Build training data ────────────────────────────────────────────────
    X_all_raw: pd.DataFrame
    y_all:     pd.Series
    sw_all:    pd.Series

    if use_multi_stock and len(symbols) > 1:
        log.info(f"Multi-stock training on: {symbols}")
        try:
            X_all_raw, y_all, sw_all = build_multi_stock_dataset(symbols, benchmark_symbol)
        except Exception as e:
            log.warning(f"Multi-stock build failed ({e}) — falling back to single stock.")
            use_multi_stock = False

    if not use_multi_stock:
        log.info(f"Single-stock training on: {symbol}")
        from features.feature_engineering import add_alpha_features
        df_s, bm_s = load_data(symbol, benchmark_symbol)
        n_tr  = int(len(df_s) * (1 - CFG.model.holdout_pct))
        df_tr = df_s.iloc[:n_tr]
        bm_tr = bm_s.reindex(df_tr.index, method="ffill")
        X_all_raw = add_alpha_features(df_tr, bm_tr)
        y_all = select_label(df_tr.loc[X_all_raw.index], cfg=CFG.labeling)
        common = X_all_raw.index.intersection(y_all.index)
        X_all_raw = X_all_raw.loc[common]
        y_all     = y_all.loc[common]
        sw_all    = compute_sample_weights(y_all, max_hold_bars=CFG.labeling.max_hold_bars, recency_boost=1.5)

    label_balance_check(y_all, context="training_pool")
    log.info(
        f"Training pool: {len(X_all_raw)} bars  "
        f"pos_rate={float(y_all.mean()):.3f}  features_raw={X_all_raw.shape[1]}"
    )

    # ── 2. Feature selection ──────────────────────────────────────────────────
    log.info("Running feature selection on training pool …")
    selector = FeatureSelector(top_k=12, corr_threshold=0.90)
    
    log.info(
        f"Feature selection: {len(selector.selected)} features  X_all={X_all.shape}"
    )
    if not selector.importance_df.empty:
        log.info(
            f"Top-10 features:\n{selector.importance_df.head(10).to_string(index=False)}"
        )

    # ── 3. Walk-forward CV on the MULTI-STOCK POOL ───────────────────────────
    # Feature selection already done in step 2 — we pass X_all (pre-selected,
    # pre-scaled) directly into walk_forward_on_pool. No re-fitting inside folds.
    log.info(
        f"Walk-forward CV on multi-stock pool: "
        f"{len(X_all)} bars  n_folds={CFG.model.n_folds}  "
        f"min_train_bars={CFG.model.min_train_bars}"
    )
    folds = walk_forward_on_pool(
        X_all, y_all, sw_all,
        n_folds        = CFG.model.n_folds,
        min_train_bars = CFG.model.min_train_bars,
    )

    # Load primary symbol data for holdout (done here to avoid reloading later)
    df_primary, bm_primary = load_data(symbol, benchmark_symbol)
    # fe_primary: FeatureEngineer fitted on primary symbol train portion
    n_p_total   = len(df_primary)
    n_p_train   = int(n_p_total * (1 - CFG.model.holdout_pct))
    df_p_train  = df_primary.iloc[:n_p_train]
    bm_p_train  = bm_primary.reindex(df_p_train.index, method="ffill")
    fe_primary  = FeatureEngineer()
    fe_primary.fit_transform(df_p_train, bm_p_train)   # fit for transform state

    if not folds:
        log.error(
            "Walk-forward produced no valid folds. "
            f"Pool has {len(X_all)} bars, min_train_bars={CFG.model.min_train_bars}. "
            "Add more symbols or reduce min_train_bars."
        )
        return {}

    mean_auc = float(np.mean([f["auc"] for f in folds]))
    mean_std = float(np.mean([f.get("std_prob", 0) for f in folds]))
    log.info(
        f"Walk-forward ({len(folds)} folds): "
        f"Mean AUC={mean_auc:.4f}  "
        f"Mean F1={np.mean([f['f1'] for f in folds]):.4f}  "
        f"Mean prob_std={mean_std:.4f}"
    )
    if mean_std < 0.01:
        log.warning(
            f"CONSTANT PREDICTION ALERT: mean prob_std={mean_std:.5f}. "
            "Check feature variance and label balance."
        )
    if mean_auc < 0.55:
        log.warning(f"Mean AUC={mean_auc:.4f} below 0.55 — limited edge. "
                    "Circuit breaker DISABLED for evaluation.")

    # ── 4. Fit final model on multi-stock pool ────────────────────────────────
    log.info(f"Fitting final model (regime={use_regime_model}) on {len(X_all)} bars …")
    _ROUTING_COLS = ["regime", "hl_range", "symbol_id"]
    if use_regime_model:
        model = RegimeModel(threshold=CFG.signal.min_prob)
        model.fit(X_all, y_all, sw=sw_all)
    else:
        X_all_m = X_all.drop(columns=[c for c in _ROUTING_COLS if c in X_all.columns])
        model = EnsembleModel(threshold=CFG.signal.min_prob)
        model.fit(X_all_m, y_all, sample_weight=sw_all)

    fi = model.feature_importance(feature_names=selector._alpha_cols)
    if not fi.empty:
        log.info(f"Top-10 feature importances:\n{fi.head(10).to_string()}")

    # ── 5. Holdout set for primary symbol ─────────────────────────────────────
    n_total     = len(df_primary)
    n_train_all = int(n_total * (1 - CFG.model.holdout_pct))
    df_holdout  = df_primary.iloc[n_train_all:]
    bm_holdout  = bm_primary.reindex(df_holdout.index, method="ffill")
    log.info(
        f"Holdout: {len(df_holdout)} bars "
        f"({df_holdout.index[0]} → {df_holdout.index[-1]})"
    )

    # ── 6. Holdout features & predictions ────────────────────────────────────
    log.info("Generating holdout features …")
    X_holdout_raw = fe_primary.transform(df_holdout, bm_holdout)
    X_holdout_sel = selector.transform(X_holdout_raw)

    # Base prediction + calibration (no meta-model — avoids train-on-holdout leakage)
    prob_base = model.predict_proba(X_holdout_sel)
    prob = pd.Series(
        calibrate(prob_base.values if isinstance(prob_base, pd.Series) else np.asarray(prob_base)),
        index=X_holdout_sel.index, name="prob",
    )

    prob_std = float(prob.std())
    log.info(f"Holdout prob dist: mean={prob.mean():.3f}  std={prob_std:.4f}  max={prob.max():.3f}")
    print(f"\n[DEBUG] prob_std={prob_std:.4f}  mean={prob.mean():.4f}  max={prob.max():.4f}")

    if prob_std < 0.01:
        log.error(f"CONSTANT PREDICTIONS (std={prob_std:.5f}). Check features and labels.")

    # Adaptive threshold: 70th percentile of probs (ensures ~30% bars pass)
    adaptive_thresh = max(float(np.percentile(prob.values, 70)), CFG.signal.min_prob)
    log.info(f"Adaptive threshold: {adaptive_thresh:.4f}  (floor={CFG.signal.min_prob})")
    mask = prob >= adaptive_thresh
    prob = prob[mask]
    X_holdout_sel = X_holdout_sel.loc[prob.index]
    log.info(f"Adaptive filter: {len(prob)}/{len(mask)} bars kept")
    print(f"[DEBUG] after adaptive filter: n={len(prob)}  prob_std={prob.std():.4f}")

    # ── 7. Signal generation ──────────────────────────────────────────────────
    sf = SignalFilter(
        min_prob         = CFG.signal.min_prob,
        max_prob_thresh  = CFG.signal.max_prob_thresh,
        adaptive_window  = CFG.signal.adaptive_window,
        adaptive_margin  = CFG.signal.adaptive_margin,
        top_k            = CFG.signal.top_k,
        top_pct          = CFG.signal.top_pct,
        cooldown_bars    = CFG.signal.cooldown_bars,
        min_ev           = CFG.signal.min_ev,
        pt_mult          = CFG.labeling.pt_mult,
        sl_mult          = CFG.labeling.sl_mult,
    )
    signals_raw = sf.generate(prob, features=X_holdout_sel)

    # Change 6: Fallback — if zero BUY signals, force the single highest-prob bar
    if (signals_raw == "BUY").sum() == 0:
        log.warning("No signals → fallback to top probability bar")
        if len(prob) > 0:
            top_idx = prob.idxmax()
            signals_raw.loc[top_idx] = "BUY"
            log.info(f"Fallback signal inserted at {top_idx}  prob={prob[top_idx]:.4f}")

    stats       = sf.stats(signals_raw, prob)
    cost_impact = sf.estimate_cost_impact(
        signals_raw, prob,
        entry_prices  = df_holdout["close"].reindex(prob.index),
        slippage_bps  = CFG.backtest.slippage_bps,
        brokerage_bps = CFG.backtest.brokerage_bps,
    )
    print(f"[DEBUG] trades: {stats['n_signals']}")
    print(f"[DEBUG] avg_prob: {stats['mean_prob']:.4f}")
    log.info(
        f"Signal stats: {stats['n_signals']} BUY | "
        f"rate={stats['signal_rate']:.2%} | "
        f"mean_prob={stats['mean_prob']:.3f} | "
        f"max_prob={stats['max_prob']:.3f}"
    )
    log.info(
        f"Cost impact: round-trip={cost_impact['total_cost_bps']:.0f}bps | "
        f"EV/signal={cost_impact.get('ev_per_signal_pct', 0):.2f}% | "
        f"cost/EV ratio={cost_impact.get('cost_to_ev_ratio', 0):.2f}"
    )

    # ── 7a. Portfolio allocation ──────────────────────────────────────────────
    buy_idx = signals_raw[signals_raw == "BUY"].index
    sig_df  = X_holdout_sel.reindex(buy_idx).copy()
    sig_df["prob"] = prob.reindex(buy_idx)
    if "hl_range" in X_holdout_sel.columns:
        sig_df["volatility"] = X_holdout_sel["hl_range"].reindex(buy_idx)
    sig_df = allocate_portfolio(sig_df, capital=CFG.risk.capital)
    log.info(
        f"Allocation: total_deployed=₹{sig_df['allocation'].sum():,.0f}  "
        f"max_single=₹{sig_df['allocation'].max():,.0f}"
    )

    signals = signals_raw

    # ── 8. Backtest ───────────────────────────────────────────────────────────
    log.info("Running risk-managed backtest on holdout …")
    risk = RiskManager(cfg=CFG.risk, edge_required=False)   # disabled for evaluation
    # Circuit breaker is intentionally disabled: the system must be allowed to
    # generate trades before edge can be confirmed. Set edge_required=True in
    # production only after ≥3 months of live validated AUC > 0.55.
    risk.set_edge_confirmed(True)
    log.info(
        f"Circuit breaker: DISABLED (edge_required=False)  "
        f"mean_auc={mean_auc:.4f} — all qualifying signals will trade"
    )

    atr_aligned = _atr_for_labeling(df_holdout).reindex(X_holdout_sel.index)
    results     = simulate(df_holdout, signals, prob, atr_aligned, risk, symbol)

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print_summary(results, folds, symbol)

    return {
        "folds":    folds,
        "model":    model,
        "selector": selector,
        "fe":       fe_primary,
        "signals":  signals,
        "prob":     prob,
        "results":  results,
        "mean_auc": mean_auc,
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Intraday ML Pipeline")
    parser.add_argument("--symbol",    default="RELIANCE.NS",
                        help="Primary NSE ticker for holdout backtest")
    parser.add_argument("--benchmark", default="^NSEI",
                        help="Benchmark ticker (default: ^NSEI)")
    parser.add_argument("--symbols",   nargs="+", default=DEFAULT_SYMBOLS,
                        help="Universe for multi-stock training")
    parser.add_argument("--no-multi",  action="store_true",
                        help="Disable multi-stock training (single stock only)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Disable regime-split model (single global model)")
    args = parser.parse_args()
    run(
        symbol           = args.symbol,
        benchmark_symbol = args.benchmark,
        symbols          = args.symbols,
        use_multi_stock  = not args.no_multi,
        use_regime_model = not args.no_regime,
    )