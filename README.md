# MK Trend Following Analytics Engine — Streamlit Cloud v0.02

**By Murat Konuklar**  
**MK FinTECH LabGEN @2026 ATELIER ISTANBUL**

Validated legacy trend-following engine with strict Yahoo Finance data governance, institutional Streamlit UI, decision causality, interactive analytics and standalone HTML export.

## v0.02 highlights

- BIST / US Stocks / Precious Metals curated instrument selector.
- Manual Yahoo ticker remains available.
- BUY / HOLD / SELL / WAIT-CASH portfolio decision layer.
- Explicit Decision Causality Matrix.
- Interactive Plotly candlestick, volume, stops, BUY/SELL executions and time-range controls.
- Strategy vs Buy & Hold, drawdown, rolling risk and trend diagnostics.
- Trade Ledger and trade statistics.
- Standalone interactive HTML v0.02 export.
- Validated legacy core retained without changing Golden Master mathematics.

## Strict Data Governance

- Yahoo Finance via `yfinance` is the only live market-data source.
- No synthetic data.
- No alternate vendor fallback.
- No forward-fill / back-fill of missing market prices.
- Malformed or incomplete Yahoo data produces a hard stop.
- `auto_adjust=False`, `repair=False`.

## App entry point

`app.py`

## Required root modules

- `app.py`
- `MK_Trend_Following_Engine_v001.py`
- `MK_Trend_Following_Decision_Engine_v002.py`
- `MK_Trend_Following_Universe_v002.py`
- `MK_Trend_Following_HTML_Report_v002.py`
- `requirements.txt`

## Local validation

```bash
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

## Streamlit Community Cloud

- Branch: `main`
- Main file path: `app.py`
- Recommended Python: `3.12`
- No secrets required in this Yahoo-only version.

See `UPDATE_TO_V002_TR.md` for the Turkish update steps.
