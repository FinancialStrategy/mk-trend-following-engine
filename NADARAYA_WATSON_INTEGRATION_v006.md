# MK Trend Following Engine v0.06 — Nadaraya-Watson Trend Integration

**By Murat Konuklar**  
**MK FinTECH LabGEN @2026 ATELIER ISTANBUL**

## Source methodology

Reference indicator: **Nadaraya-Watson Trend [QuantAlgo]**, published as an open-source script on TradingView.

The integration is an **independent Python implementation** of the public methodology. The Pine Script source is not redistributed verbatim.

Publicly described architecture:

1. One-sided / causal Nadaraya-Watson endpoint regression.
2. Current and historical bars only; no centered future-looking fit.
3. Six selectable kernels:
   - Gaussian
   - Rational Quadratic
   - Epanechnikov
   - Triangular
   - Quartic
   - Cosine
4. Effective bandwidth = bandwidth × multiplier.
5. NW trend = normalized kernel-weighted price estimate.
6. Residual envelope = NW estimate ± band multiplier × kernel-weighted mean absolute residual.
7. Bullish/bearish path state determined from the bar-to-bar NW slope.
8. Reversal events occur when slope direction flips.

## Causality discipline

The implementation uses an inclusive lag set `0..lookback`, matching the public Pine loop semantics. The default strict mode waits until the entire requested lookback exists before emitting the NW estimate. No shorter hidden fallback window is substituted.

A dedicated validation recomputes truncated price histories and verifies that the NW estimate at each cutoff is identical to the value computed using the full history. This proves that appending future observations does not rewrite historical NW values.

## Strategy layer

The QuantAlgo publication is an **indicator**, not a portfolio backtest specification. Therefore the MK Engine keeps the indicator and strategy layers separate.

### Public-Methodology Reversal Translation

Long entry candidate:
- bullish NW slope reversal;
- price/source above NW trend path;
- optional no-chase filter requiring price not to be above the upper residual band.

Exit candidate:
- bearish NW slope reversal; or
- price/source falls below NW trend path.

### MK Confirmed NW Trend

Long entry candidate:
- NW direction remains bullish for a configurable confirmation period;
- price/source is above NW trend path;
- optional no-chase filter blocks entry above the upper residual band.

Exit candidate:
- price/source loses the NW trend path; or
- bearish NW direction persists for the configured exit confirmation period.

### Execution

A signal is read only from the **prior completed bar**. Any BUY or SELL executes at the **next bar's adjusted open**. The system remains long/cash and all-in/all-out, consistent with the existing Trend Following Engine architecture.

## MK research presets

The numeric presets in Streamlit are **MK research presets**, not claimed QuantAlgo defaults:

- MK Institutional Balanced
- Public-Methodology Gaussian
- MK Fast Research
- MK Smooth Position
- Custom

The user can independently select source, kernel, lookback, bandwidth, Rational Quadratic relative weight, residual-band multiplier, confirmation bars, and strategy mode.

## New interactive visuals

The new `Nadaraya-Watson Trend` tab contains:

- adjusted candlesticks;
- bullish/bearish NW path;
- upper/lower residual bands;
- bullish/bearish kernel reversal markers;
- actual next-open strategy BUY/SELL executions;
- causal decision matrix;
- NW strategy vs Buy & Hold equity curve;
- market-exposure strip;
- kernel weight profile;
- normalized NW slope;
- residual-band-width diagnostics;
- strategy comparison against the active legacy system and Buy & Hold;
- NW trade ledger;
- dedicated standalone interactive NW HTML export.

## Data governance

Unchanged:

- Yahoo Finance only for live market data.
- No synthetic market observations.
- No alternate-provider fallback.
- No price forward-fill or back-fill.
- No silent substitution.
- Failed or incomplete Yahoo data stops the requested run.

## Validation

The original 1,043-observation GE Golden Master remains unchanged and still passes the legacy replication suite. The NW module is additionally tested for:

- all six kernel weight functions;
- non-negative finite weights;
- normalized weights summing to one;
- residual-band ordering;
- full-lookback warm-up discipline;
- causal prefix invariance;
- next-open execution accounting;
- compatibility with existing performance and trade-ledger functions.
