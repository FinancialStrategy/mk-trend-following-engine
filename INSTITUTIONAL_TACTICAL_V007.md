# MK Trend Following Engine v0.07 — Institutional Tactical Upgrade

## Why this version exists
The previous engine could remain HOLD while a price was already showing short-term exhaustion,
because the primary live decision still leaned too heavily on the legacy trend/stop architecture.

v0.07 changes the hierarchy.

## Primary live decision
`MK Institutional Tactical` is now the primary decision layer.

Legacy ATR/Bollinger/ATR-Trailing logic is retained only as a reproducible audit/reference layer.

## Fast Nadaraya-Watson envelope response
The public QuantAlgo methodology explicitly exposes:
- Source Cross Above Upper Band
- Source Cross Below Lower Band
- bullish / bearish kernel slope reversals

v0.07 uses those states as tactical portfolio-risk events:
- Upper-band cross -> immediate early trim
- Re-entry below upper band -> stronger exhaustion confirmation
- Bearish reversal / path loss -> additional de-risk
- Lower-band failure -> hard exit candidate

## Benchmark-relative intelligence
Every curated instrument is mapped to an explicit Yahoo benchmark and can be manually overridden.

Examples:
- AKBNK.IS -> XBANK.IS
- ASTOR.IS -> XUSIN.IS
- TUPRS.IS -> XKMYA.IS
- THYAO.IS -> XULAS.IS
- NVDA -> SMH
- JPM -> XLF
- XOM -> XLE

The engine estimates rolling beta using prior-window information, then calculates:
- beta-adjusted residual return
- standardized residual impulse
- cumulative residual drift z-score
- relative price-ratio z-score
- multi-bar asset minus benchmark return

Exact timestamp inner alignment only. Missing benchmark observations are never filled.

## Exposure ladder
The model can target:
- 100%
- 75%
- 50%
- 25%
- 0%

A signal is evaluated on a completed bar and the exposure rebalance executes at the next adjusted open.

## Intraday 15-minute mode
Yahoo supports 15m data but limits intraday history to the last 60 days.
The application hard-stops a larger request rather than silently truncating it.

## No-data-manufacturing policy
- Yahoo Finance only
- no synthetic market observations
- no alternate market-data provider
- no forward fill
- no backfill
- no silent benchmark substitution
