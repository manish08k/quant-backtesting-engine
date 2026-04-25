"""utils/config.py — Central config. Edit here only."""
from types import SimpleNamespace

# ── Data ──────────────────────────────────────────────────────────────────────
data = SimpleNamespace(
    symbols        = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
                      "AXISBANK.NS","KOTAKBANK.NS","SBIN.NS","WIPRO.NS","LT.NS"],
    benchmark      = "^NSEI",           # NIFTY 50
    sector_etfs    = {"IT":"INFY.NS","BANK":"HDFCBANK.NS"},
    interval       = "5m",
    period         = "60d",             # yfinance rolling window
    min_bars       = 200,               # lowered: 5m × 60d yields ~2400 bars but gaps reduce it
    cache_dir      = "cache/",
)

# ── Features ──────────────────────────────────────────────────────────────────
features = SimpleNamespace(
    top_k          = 30,
    swing_window   = 5,
    vwap_session   = True,
    mtf_intervals  = ["15m","1h"],
)

# ── Labeling ──────────────────────────────────────────────────────────────────
labeling = SimpleNamespace(
    pt_mult        = 1.5,   # profit-take = pt_mult × ATR
    sl_mult        = 1.0,   # stop-loss   = sl_mult  × ATR
    max_hold_bars  = 12,    # timeout
    min_pos_rate   = 0.10,
    meta_threshold = 0.50,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = SimpleNamespace(
    n_folds        = 5,
    min_train_bars = 300,
    blend_weights  = [0.50, 0.35, 0.15],   # XGB / LGB / LR
    calibrate      = True,
    random_state   = 42,
)

# ── Strategy ──────────────────────────────────────────────────────────────────
strategy = SimpleNamespace(
    base_threshold     = 0.55,
    trend_threshold    = 0.52,
    ranging_threshold  = 0.62,
    adx_trend_thresh   = 25,
    cooldown_bars      = 3,
    min_ev             = 0.0002,
    vol_filter_mult    = 0.5,
)

# ── Risk ──────────────────────────────────────────────────────────────────────
risk = SimpleNamespace(
    capital            = 1_000_000,
    risk_per_trade     = 0.01,
    max_daily_loss     = 0.02,
    max_consec_losses  = 4,
    atr_sl_mult        = 1.0,
    atr_tp_mult        = 1.5,
    trail_atr_mult     = 0.8,
    max_position_pct   = 0.10,
)

# ── Backtest ──────────────────────────────────────────────────────────────────
backtest = SimpleNamespace(
    slippage_bps   = 5,
    brokerage_bps  = 3,
    latency_bars   = 1,
    capital        = risk.capital,
)

# ── Execution ─────────────────────────────────────────────────────────────────
execution = SimpleNamespace(
    paper          = True,
    kite_api_key   = "",
    kite_api_secret= "",
    order_type     = "MARKET",
)

CFG = SimpleNamespace(
    data=data, features=features, labeling=labeling,
    model=model, strategy=strategy, risk=risk,
    backtest=backtest, execution=execution,
)