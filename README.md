# MK Trend Following Analytics Engine v0.06

**By Murat Konuklar**  
MK FinTECH LabGEN @2026 ATELIER ISTANBUL

Institutional Streamlit trend-following research engine with validated legacy Excel replication plus a causal Nadaraya-Watson Trend research layer.

## Live market-data governance

- Yahoo Finance via `yfinance` only.
- No synthetic market observations.
- No alternate market-data fallback.
- No price forward-fill / back-fill.
- Incomplete or malformed Yahoo responses stop the requested run.

## Strategy architecture

Legacy validated systems remain intact:

- ATR
- Bollinger
- ATR Trailing Stop

Additive v0.06 research layer:

- Nadaraya-Watson Trend public-methodology implementation
- Six kernels
- Kernel-weighted residual bands
- Causal slope-reversal states
- MK Confirmed NW Trend strategy
- Public-Methodology Reversal Translation
- Next-open execution backtest
- Dedicated interactive charts and HTML export

See `NADARAYA_WATSON_INTEGRATION_v006.md` for methodology and attribution.

## Streamlit Cloud

Main file:

```text
app.py
```

Python: `3.12`

No secrets are required in this Yahoo-only build.

## Validation

GitHub Actions runs:

- original GE Golden Master replication
- decision/report integration
- rolling-risk tests
- entry-gate tests
- cold-start regression
- Nadaraya-Watson six-kernel + causal prefix-invariance + next-open execution tests
- Streamlit headless health check
