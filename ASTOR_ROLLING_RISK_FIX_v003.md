# ASTOR Rolling Return & Volatility — Root Cause and v0.03 Fix
**By Murat Konuklar**

## Root cause
v0.02 calculated both rolling return and rolling volatility from:

`df["Portfolio"]`

That is the **trend strategy equity curve**, not ASTOR's own adjusted-price series.

When the legacy all-in/all-out strategy is in CASH after a SELL, Portfolio remains constant.
Therefore:
- one-period strategy return = 0%
- rolling strategy return can become 0%
- realized strategy volatility can become 0%

A long cash regime consequently appears as a perfectly flat line.

This is not evidence that Yahoo returned a frozen ASTOR price series. It is a presentation /
risk-source classification error in v0.02.

## v0.03 correction
Risk Analytics now separates:

1. **Underlying Asset Rolling Risk**
   - Source: `AdjCloseCalc`
   - Rolling Return: adjusted price change over selected rolling window
   - Annualized Volatility: standard deviation of adjusted-price returns × sqrt(periods/year)

2. **Strategy Rolling Risk**
   - Source: `Portfolio`
   - Shows the actual risk of the trend-following portfolio
   - Cash regimes may legitimately produce flat 0% return / 0% volatility segments

## No data-policy change
- Yahoo Finance only
- No synthetic data
- No fallback vendor
- No forward fill
- No backfill
- Early rolling-window NaNs remain NaN and are not filled

## Streamlit update
Replace / add these files in GitHub:
- `app.py` — replace
- `MK_Trend_Following_HTML_Report_v003.py` — add
- `MK_Trend_Following_Risk_Analytics_v003.py` — add
- `.github/workflows/validate.yml` — replace

Do **not** remove or modify the validated:
- `MK_Trend_Following_Engine_v001.py`
- `MK_Trend_Following_Decision_Engine_v002.py`
- `MK_Trend_Following_Universe_v002.py`

After Commit, Streamlit Community Cloud should redeploy automatically.
