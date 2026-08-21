# v0.08.8 — 15m Intraday Tactical Lab

**By Murat Konuklar**

## Objective

Extend the existing causal Nadaraya-Watson + Institutional Tactical architecture into a dedicated 15-minute execution-quality layer while preserving the project's strict market-data governance.

## Market-data governance

- Yahoo Finance / `yfinance.download` remains the only live market-data source.
- The Intraday Tactical Lab makes no additional provider request.
- No forward fill, back fill, interpolation, synthetic OHLCV or alternate-provider substitution is used.
- 15-minute requests must remain inside Yahoo's current 60-day intraday availability window.
- If the latest 15-minute Yahoo timestamp is timezone-aware and its bar has not finished, that bar is explicitly withheld before NW, benchmark-relative, Tactical, risk and intraday calculations.

## Session models

`Auto` selects:

- `.IS` → BIST Cash, 10:00–18:00 Europe/Istanbul.
- `*-USD` crypto → UTC 24/7 day, 00:00 reset.
- `=F` metals futures → CME/COMEX metals session, 18:00–17:00 America/New_York with daily break.
- other tickers → US Cash, 09:30–16:00 America/New_York.

The session model is user-overridable in 15m mode.

## Causal calculations

### Session VWAP

For each session and completed bar `t`:

`VWAP_t = cumulative(sum(TypicalPrice_i × Volume_i)) / cumulative(sum(Volume_i))`

Only bars up to `t` enter the value.

### Opening range

The opening range evolves causally during the first `N` completed 15m bars. After bar `N` closes, the high/low freezes for the rest of that session. Breakout markers can therefore only occur after the opening range is finalized.

### Same-slot relative volume

A bar is compared with the median volume of the **same session slot** over prior sessions only. Current/future session volume never enters the baseline. This is materially more meaningful intraday than a generic rolling-volume ratio.

### Intraday risk

- rolling true-range ATR;
- rolling annualized realized volatility;
- session gap;
- session return;
- session peak-to-current drawdown.

### Intraday Confirmation Score

The score combines available causal evidence from:

- NW direction;
- MK momentum warnings;
- price vs session VWAP;
- finalized opening-range position;
- benchmark residual drift z-score;
- same-slot RVOL directional climax.

The score is clipped to `[-100, +100]` and classified into strong/down/neutral/up/strong-up confirmation states. It is a transparent research diagnostic only in v0.08.8; it does not mutate Tactical target exposure.

## 15m NW presets

The 15-minute presets are bar-based MK research settings, not claimed QuantAlgo proprietary defaults. Crypto uses longer bar lookbacks because it trades continuously 24/7.

## Execution causality

The existing Institutional Tactical engine remains:

**completed bar signal → next available adjusted open execution**.

The Intraday Tactical Lab does not introduce same-bar execution.
