# v0.05 — ASTOR Equity-Curve Flat-Line Root Cause & Fix

## What the screenshot proves
The dotted Buy & Hold line moves continuously while the solid Trend Strategy line is flat.
Therefore the underlying ASTOR price dataset is not frozen. The flat series is the strategy's
cash portfolio.

## Root cause
The validated legacy engine uses:

`BUY if prior adjusted close >= prior rolling maximum`

The rolling maximum uses `max_buy_weeks`, which is actually an observation count.

The old default is 2000 observations. For a newly listed stock with fewer than 2000 daily
observations, this becomes an all-history-high entry gate. After a SELL, the system cannot
re-enter until a new historical high is reached.

## v0.05
The engine itself is NOT changed.

The Streamlit layer now offers:
- Frequency-Aware entry mode (default)
- Legacy Exact 2000-observation mode
- Custom observation mode

Default Frequency-Aware horizon:
- Daily: 12M = 252 observations
- Weekly: 12M = 52 observations
- Monthly: 12M = 12 observations

The Equity Curve now also:
- plots market exposure below the strategy curve;
- shades cash regimes;
- reports the longest cash regime;
- reports the underlying stock return during that cash interval;
- warns when the configured entry gate is effectively an all-history-high rule.

No synthetic data.
No fallback market-data provider.
No forward fill.
No backfill.
