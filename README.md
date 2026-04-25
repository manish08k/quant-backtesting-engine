# Institutional Intraday Trading System

## Architecture

```
intraday_trader/
├── data/
│   ├── data_loader.py        OHLCV (yfinance + NIFTYBEES volume proxy)
│   └── data_cleaning.py      IST filter, OHLC fix, volume clip
├── features/
│   ├── feature_engineering.py  65 institutional features (no lookahead)
│   └── feature_selection.py    MI + XGBoost blended rank selection
├── labeling/
│   └── barriers.py           Triple-barrier (vol-adjusted) + meta-labeling
├── models/
│   ├── ensemble.py           XGBoost + LightGBM + LR blend (0.50/0.35/0.15)
│   ├── evaluation.py         Walk-forward fold summary
│   ├── train.py              Standalone training script
│   ├── predict.py            Standalone inference script
│   ├── labeling.py           Re-export of labeling primitives
│   └── saved/                Serialised model + features.json
├── strategy/
│   ├── signal_filter.py      Regime-aware threshold + EV + MTF + cooldown
│   ├── signal_generator.py   Meta-signal layer (gates primary signals)
│   ├── position_sizing.py    ADX-regime routing (vol-target vs fixed-frac)
│   └── risk_management.py    Pre-trade risk gate
├── risk/
│   ├── risk_manager.py       Sizing, trailing stop, daily halt, consec losses
│   ├── position_sizing.py    Fixed-fraction / vol-target / half-Kelly
│   ├── slippage.py           Base bps + volume-impact model
│   └── transaction_costs.py  Full Zerodha NSE intraday cost breakdown
├── backtest/
│   ├── backtester.py         Event-driven, slippage+brokerage+latency, trail stop
│   ├── performance.py        Sharpe, drawdown, expectancy, equity curve
│   ├── multi_asset.py        Portfolio Sharpe across universe
│   └── stress_test.py        Monte Carlo (200 permutations)
├── execution/
│   └── paper_trading.py      5-min poll loop, live signal + order sizing
├── reports/
│   └── report_generator.py   PNG equity curve + text summary
├── notebooks/
│   └── research.ipynb        EDA, feature correlation, model diagnostics
├── utils/
│   ├── config.py             Central config (all params here)
│   └── logger.py             Dual stdout + file logger
├── main.py                   Orchestrator (backtest|multi|diagnose|stress|paper|train)
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Diagnose first — check data quality and label distribution
python3 main.py --mode diagnose --symbol RELIANCE.NS

# 2. Full backtest with report
python3 main.py --mode backtest --symbol RELIANCE.NS

# 3. Multi-asset portfolio backtest
python3 main.py --mode multi

# 4. Monte Carlo stress test
python3 main.py --mode stress --symbol RELIANCE.NS

# 5. Paper trading (live 5-min loop)
python3 main.py --mode paper --symbol RELIANCE.NS

# 6. Train only (saves model)
python3 models/train.py --symbol RELIANCE.NS

# 7. Predict latest bars
python3 models/predict.py --symbol RELIANCE.NS --bars 10
```

## Key Design Decisions

### Data Pipeline
- **NIFTY volume fix**: `^NSEI` has zero volume from yfinance. Fixed by fetching
  `NIFTYBEES.NS` (liquid ETF) as volume proxy — aligned and scaled 10×.
- **IST timestamp filter**: only 09:15–15:30 bars kept; pre/post market stripped.
- **No leakage**: all features use `.shift(1)` or bar-t only values.

### Signal Assignment (symmetric dead zone)
- `BUY`  if prob ≥ (1 − dynamic_threshold)
- `SELL` if prob ≤ dynamic_threshold
- `HOLD` otherwise
- Threshold is ADX-regime dependent: lower in trend, higher in range.

### Features (65 total → 30 selected)
- **Price structure**: swing H/L, breakout strength, consolidation range, trend persistence
- **Volume**: spike follow-through, VW momentum, order-flow imbalance, exhaustion
- **Volatility**: ATR percentile, ratio (expansion), Hurst proxy, hvol
- **Microstructure**: OFI (close-low / range), tick imbalance, candle delta
- **Context**: VWAP deviation + slope, MTF EMA alignment (9/21/50/200), session fraction
- **Market-wide**: NIFTY return, rolling beta (30-bar), corr30, relative strength, idx_trending

### Labeling
- Triple-barrier: `pt_mult=1.5×ATR`, `sl_mult=1.0×ATR`, `max_hold=12 bars`
- Timeout bars → NaN (not forced to 0) — cleaner signal
- Fallback to 3-bar forward return if positive rate < 10%
- Meta-labeling available: primary model → direction; meta model → whether to trade

### Ensemble
- XGBoost + LightGBM + Logistic Regression (blend 0.50/0.35/0.15)
- Per-fold `scale_pos_weight` from actual class ratio
- Threshold tuned per fold via F1 maximisation (range 0.40–0.80)
- Isotonic calibration on 15% holdout (if ≥500 samples)

### Regime-aware filtering
- ADX ≥ 25 → trend mode → threshold = 0.52 (more trades)
- ADX < 25 → range mode → threshold = 0.62 (more selective)
- MTF alignment score must be ≥ 2/3 for BUY, ≤ 1/3 for SELL
- EV filter: `prob × 1.5 − (1−prob) × 1.0 > min_ev`
- Volume filter: `vol_ratio ≥ 0.5 × 20-bar avg`
- Cooldown: 3 bars after each trade

### Risk Management
- Position size: `(capital × 1%) / (1.0 × ATR)`; capped at 10% of capital
- Trailing stop: ratchets up by `0.8 × ATR` as price moves favourably
- Daily halt: triggers at −2% daily P&L
- Consecutive loss halt: triggers at 4 losses in a row
- EOD close-out: all positions closed at session end

### Transaction Costs (Backtester)
- Slippage: 5 bps base + volume-impact `k√(order/ADV)`
- Brokerage: min(₹20, 0.03%) per side (Zerodha)
- STT: 0.025% on sell side; Stamp: 0.003% on buy side
- Execution latency: 1 bar (signal on bar T → fill on bar T+1 open)

### Performance Metrics Output
- Sharpe ratio (annualised, daily P&L basis)
- Max drawdown
- Win rate
- Expectancy (₹ per trade)
- Profit factor
- Equity curve PNG + text report in `reports/`
