# v0.08 — Institutional Client Surface Rebuild

## Removed from client UI
- all Legacy Reference / Audit panels
- Price & Legacy Signals wording and audit chart
- all legacy-threshold Trend Diagnostics
- legacy decision fallback and legacy performance snippets

The workbook-replication engine remains only as code-level regression infrastructure.

## Nadaraya-Watson rulebook
The NW tab now publishes the exact causal rule hierarchy: estimator, slope regime, bullish/bearish reversals, upper/lower envelope events, confirmed entry/exit rules, benchmark filter, and next-open execution.

## Risk Analytics
VaR is now computed for Asset, selected Yahoo Benchmark, Beta-Adjusted Active Residual, and Primary Strategy using the selected timeframe and risk calibration window. Methods:
- Historical VaR
- Parametric Normal VaR
- Monte Carlo empirical-bootstrap VaR

Monte Carlo scenarios are in-memory risk scenarios sampled from observed returns. They are never appended to Yahoo history and never substitute missing market observations.

## Trend Diagnostics
Client diagnostics are now only:
- Nadaraya-Watson trend + residual envelope
- NW slope
- beta-adjusted benchmark residual drift z-score
- relative volume
- staged target exposure

No legacy threshold appears in the client diagnostic.
