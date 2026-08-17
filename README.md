# MK Trend Following Analytics Engine v0.07 — DEMA-MACD BUY / SELL Confirmation

**By Murat Konuklar**  
**MK FinTECH LabGEN @2026 ATELIER ISTANBUL**

Streamlit Community Cloud-ready continuation of the validated TREND FOLLOW ALGO. v0.07 preserves the existing Legacy Fidelity, Decision Engine, Universe, Risk Analytics, Entry Gate, Nadaraya-Watson v0.06 and existing export architecture, then adds a separate DEMA-MACD full-cycle confirmation layer.

## What v0.07 Adds

- DEMA-MACD BUY confirmation engine.
- Independent SELL framework; SELL is not implemented as a simple inverse BUY crossover.
- Lifecycle states: `WAIT / CASH → BUY WATCH → BUY → HOLD → SELL WATCH → REDUCE → SELL / RISK EXIT`.
- BUY Score and SELL Score (0–100) with component-level diagnostics.
- DEMA fast/slow line, signal, histogram, slope and crossover persistence.
- Classic MACD cross-confirmation.
- DEMA trend regime and price/trend structure.
- ADX / +DI / -DI directional filter.
- Optional Nadaraya-Watson regime confirmation.
- no-chase extension control measured in ATR units.
- SELL deterioration persistence.
- hard-stop, ATR trailing-stop and prior swing-low capital-protection overrides.
- REDUCE is advisory by default; optional partial next-open execution is isolated inside the new DEMA layer.
- re-entry cooldown / hysteresis.
- event ledger, trade ledger and exit-quality diagnostics.
- calibration ranking with parameter-plateau robustness.
- train / validation / untouched final OOS split.
- expanding walk-forward re-selection using prior data only.
- standalone DEMA-MACD interactive HTML and CSV export.

## Data Governance

The live application retains the strict existing policy:

- Yahoo Finance through the existing adapter is the **only live market-data source**.
- no synthetic market observations.
- no alternate-provider fallback.
- no forward-fill / back-fill of missing market prices.
- `auto_adjust=False`; raw OHLC and Adjusted Close remain separate.
- failed/incomplete Yahoo response stops the requested run.

The DEMA-MACD module downloads no data by itself.

## Execution Governance

All executable DEMA actions are causal:

1. compute the signal from the **completed prior bar**;
2. classify the position-aware lifecycle state;
3. execute BUY / SELL / RISK EXIT, and optional REDUCE, at the **next adjusted open**.

No centered filter and no future bar is used.

## SELL Architecture

SELL is deliberately asymmetric to BUY. A bearish crossover is only one component. The engine separately evaluates:

- recent bearish DEMA-MACD crossover;
- DEMA-MACD below signal;
- falling signal slope;
- negative / deteriorating histogram;
- price below DEMA trend;
- falling DEMA trend;
- classic MACD bearish confirmation;
- ADX bearish directional regime;
- optional NW bearish/path-lost state;
- prior swing-low breach;
- ATR trailing stop;
- hard stop.

Action hierarchy while long:

`HOLD → SELL WATCH → REDUCE → SELL`, with `RISK EXIT` as the capital-protection override.

## Calibration Governance

Calibration does **not** select the single historical return peak.

- candidates are ranked on training metrics;
- only a train shortlist is validation-eligible;
- robust score combines train and validation only;
- parameter plateau score favors stable neighboring parameter sets;
- final OOS metrics are reported but never used in selection or tie-breaking;
- expanding walk-forward folds re-select parameters using information available before each fold.

Focused / Balanced / Deep grids are available. Focused is recommended for Streamlit Cloud interactive use; larger grids require materially more CPU.

## TradingView Methodology Reference

The new module is an independent Python implementation inspired by the public methodology/description of **DEMA MACD BUY signal confirmation** by `bilguut` on TradingView (published 2026-01-09). The Pine source is **not redistributed verbatim**. The MK SELL lifecycle, risk controls, score system, calibration, walk-forward framework and portfolio execution rules are separately specified extensions.

Reference page:
`https://www.tradingview.com/script/JNJuMP41-dema-macd-buy-signal-confirmation/`

## Repository Layout

```text
.
├── app.py
├── MK_DEMA_MACD_Confirmation_v007.py
├── MK_DEMA_MACD_HTML_Report_v007.py
├── dema_macd_validation_v007.py
├── MK_Nadaraya_Watson_Trend_v006.py
├── MK_Nadaraya_Watson_HTML_Report_v006.py
├── nadaraya_watson_validation_v006.py
├── MK_Trend_Following_Engine_v001.py
├── MK_Trend_Following_Decision_Engine_v002.py
├── MK_Trend_Following_Entry_Gate_v005.py
├── MK_Trend_Following_Risk_Analytics_v004.py
├── MK_Trend_Following_Risk_Analytics_v003.py
├── MK_Trend_Following_Universe_v002.py
├── MK_Trend_Following_HTML_Report_v003.py
├── MK_Trend_Following_Golden_Master_Validation_v001.txt
├── LOCAL_VALIDATION_RESULT_v006.txt
├── LOCAL_VALIDATION_RESULT_v007.txt
├── DEMA_MACD_METHODOLOGY_v007.md
├── requirements.txt
├── smoke_test.py
├── .streamlit/config.toml
└── .github/workflows/validate.yml
```

## Streamlit Cloud Deployment

1. Replace the repository root files with the contents of this package while preserving the hidden `.streamlit` and `.github` directories.
2. Main file path: `app.py`.
3. Python: 3.12.
4. Deploy / reboot the existing app.
5. In the sidebar, enable **DEMA-MACD BUY / SELL Layer**.
6. Start with **MK Institutional Balanced**.
7. Run without calibration first to verify the requested Yahoo history.
8. Then enable **Calibration + Walk-Forward** and start with **Focused**.

No new API secret is required.

## Validation

`python smoke_test.py` performs static/import QA without downloading market data.

`python dema_macd_validation_v007.py` runs the full DEMA QA suite against Matplotlib's bundled historical GOOG OHLCV sample. This is an offline real historical **QA fixture only** and is never a live/fallback source for the application. It verifies causal prefix invariance, next-open execution, SELL lifecycle coverage, calibration governance, walk-forward output and standalone HTML generation.

The included `LOCAL_VALIDATION_RESULT_v007.txt` records the current PASS result.

## Important Interpretation

This project is an analytics/research engine. A BUY/SELL/REDUCE/RISK EXIT state is a rule-based model output, not personalized investment advice. Calibration results are historical robustness diagnostics, not guarantees of future performance.
