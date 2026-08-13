# v0.04 — Flat Strategy Risk Line Presentation Fix

The v0.03 calculation correction was valid: underlying risk was separated from
strategy risk. However, the strategy chart could still contain a horizontal 0%
line when the strategy spent an entire rolling window in cash.

That horizontal segment is mathematically valid for a cash-only portfolio, but
it is visually misleading next to an equity-risk chart.

v0.04 therefore:
- preserves the TRUE strategy rolling return / volatility values;
- masks ONLY the chart-display series during 100% cash rolling windows;
- leaves a visual gap rather than drawing a continuous 0% risk line;
- shades those intervals as CASH REGIME;
- adds rolling market exposure (0% to 100%);
- keeps underlying ASTOR risk sourced only from adjusted prices;
- adds an integrity hard-stop if moving underlying prices somehow generate
  both effectively constant underlying rolling-return and rolling-volatility
  histories.

No Yahoo observations are modified or filled.
No synthetic market data.
No fallback market-data provider.
