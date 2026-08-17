# DEMA-MACD Confirmation & Calibration Methodology v0.07

**By Murat Konuklar — MK FinTECH LabGEN @2026 ATELIER ISTANBUL**

## 1. Core DEMA

For a source series `x` and lookback `n`:

`DEMA(x,n) = 2 × EMA(x,n) − EMA(EMA(x,n),n)`

The engine forms:

- `Fast DEMA = DEMA(source, fast)`
- `Slow DEMA = DEMA(source, slow)`
- `DEMA MACD = Fast DEMA − Slow DEMA`
- `Signal = DEMA(DEMA MACD, signal)` by default (EMA is supported)
- `Histogram = DEMA MACD − Signal`

## 2. Reference vs MK Layer

The public TradingView concept is treated as a continuation-confirmation reference, not as a complete trading system. v0.07 does not copy Pine source. The MK layer adds explicit regime, lifecycle, risk and calibration rules.

## 3. BUY Score

Weighted components include:

- recent bullish DEMA crossover;
- DEMA MACD above signal;
- rising signal line;
- positive histogram;
- expanding histogram;
- price above DEMA trend;
- rising DEMA trend;
- classic MACD bullish confirmation;
- ADX/+DI directional confirmation;
- optional NW bullish regime;
- no-chase ATR extension.

Disabled/unavailable optional components disappear from the score denominator rather than being treated as failures.

## 4. SELL Score

SELL is intentionally specified independently. Components include:

- recent bearish DEMA crossover;
- DEMA MACD below signal;
- falling signal line;
- negative histogram;
- deteriorating histogram;
- price below DEMA trend;
- falling DEMA trend;
- classic MACD bearish confirmation;
- ADX/-DI directional confirmation;
- optional NW bearish/path-lost state;
- prior swing-low breach.

## 5. Capital-Protection Override

While long, `RISK EXIT` has priority over the normal score hierarchy when one of these confirmed completed-bar conditions occurs:

- hard stop: close <= entry price × (1 − hard stop %)
- ATR trail: close <= position peak close − ATR multiplier × ATR
- close below the rolling **prior** swing low

The current bar cannot create its own prior-swing stop because the rolling swing series is shifted by one bar.

## 6. State Machine

When flat:

- `WAIT / CASH`
- `BUY WATCH`
- `BUY`

When long:

- `HOLD`
- `SELL WATCH`
- `REDUCE`
- `SELL`
- `RISK EXIT`

The score thresholds and persistence are configurable. Re-entry cooldown creates hysteresis to reduce one-bar BUY/SELL oscillation.

`REDUCE` is advisory by default. Optional partial execution belongs only to the v0.07 DEMA strategy. It does not alter Legacy Fidelity.

## 7. Execution Timing

For row `i`:

- indicator and action classification are read from row `i−1`;
- transaction price is row `i` adjusted open;
- row `i` close is used only for row `i` mark-to-market and future decisions.

This prevents same-bar execution look-ahead.

## 8. Nadaraya-Watson Integration

When the NW research layer is enabled and valid, DEMA can use:

- NW direction;
- price relative to NW trend path;
- NW bullish/bearish regime.

NW is a confirmation component, not a replacement for DEMA-MACD.

## 9. Calibration

Parameters searched include fast/slow/signal lengths and BUY/SELL score thresholds.

Selection sequence:

1. run every candidate causally;
2. rank on training data;
3. retain top training quartile;
4. select within shortlist using train + validation robustness;
5. favor stable parameter plateaus;
6. report final OOS metrics **without using them for selection**;
7. run expanding walk-forward folds, each reselecting from data available before the fold test interval.

Training score combines CAGR, Calmar, Sortino, Sharpe, Max Drawdown, Win Rate, Whipsaw Rate and Exposure, with a penalty for insufficient closed trades.

## 10. SELL / Exit Quality

For each full exit, the engine measures a forward-horizon diagnostic:

- worst forward return after exit;
- best forward return after exit;
- downside-avoided proxy;
- upside-foregone proxy;
- net exit utility = downside avoided − upside foregone;
- false-exit proxy.

These are diagnostic proxies, not causal counterfactual estimates.

## 11. Data Governance

The module itself does not download market data. The parent Streamlit application retains Yahoo Finance-only live ingestion and explicit data-stop behavior. No synthetic observations, alternate-provider fallback or missing-price filling are added.
