# MK Trend Following Analytics Engine v0.01
**By Murat Konuklar**  
MK FinTECH LabGEN @2026 ATELIER ISTANBUL

## Project Objective
Re-engineer the legacy `trend-following.xls` workbook into a strict-data, institutional-style
Python analytics engine with interactive HTML output and a live Yahoo Finance application layer.

## Non-Negotiable Data Governance
- **No synthetic data**
- **No fallback data vendor**
- **No forward-fill or back-fill of missing market prices**
- Yahoo Finance is the only live market-data source in v0.01.
- Any empty, malformed, duplicate, non-positive, incomplete or insufficient Yahoo response stops the run.
- `auto_adjust=False` and `repair=False` are explicit.
- Raw OHLC and Adjusted Close remain distinct.
- A user-facing end date is treated as inclusive; the adapter requests one extra calendar day because
  `yfinance.download(..., end=...)` uses an exclusive end boundary.

## Legacy Workbook Forensics
The binary Excel workbook contains a VBA-driven data workflow, not merely a table.

### Legacy input map
| Cell | Meaning |
|---|---|
| B2 | Start Date |
| B3 | End Date |
| B4 | Yahoo stock symbol |
| C3 | Frequency (`d`, `w`, `m`) |
| E3 | ATR window parameter |
| F4 | ATR multiplier |
| Q4 | Bollinger window parameter |
| S4 | Bollinger standard-deviation multiplier |
| S5 | Maximum-price BUY lookback |
| V3 | Strategy selector (`0=ATR`, `1=Bolli`, `2=Tr. Stop`) |
| Y3 | Initial capital |

### Original Yahoo mechanism
The `GetData()` VBA macro constructs the obsolete:
`http://chart.yahoo.com/table.csv?...`
endpoint and imports CSV through an Excel QueryTable.

### Strategy choices recovered
1. ATR
2. Bollinger
3. ATR Trailing Stop

### Core accounting / execution mechanics recovered
- `Scale = Adj Close / Close`
- Adjusted OHLC = raw OHLC × Scale
- True Range uses adjusted OHLC and previous adjusted close.
- A signal is determined from the **prior completed bar**.
- A triggered transaction executes at the **current bar's adjusted open**.
- BUY converts prior cash into shares.
- SELL converts prior shares into cash.
- Portfolio = shares × current adjusted close + cash.
- Buy & Hold = initial capital × current adjusted close / first adjusted close.

This prior-bar/next-open convention is preserved because it avoids same-bar execution look-ahead.

## Legacy Fidelity Quirks Preserved
The Excel workbook uses inclusive `OFFSET` ranges. Therefore, after saturation:
- ATR parameter `8` uses **9 True Range observations**.
- Bollinger parameter `40` uses **41 adjusted-close observations**.
- Max-price lookback behaves similarly.

These are intentionally preserved in `legacy_inclusive_windows=True` so that the Python engine
reproduces the old workbook rather than silently changing its mathematics.

## Golden Master
The workbook contains 1,043 weekly GE observations:
- Data start: 1988-12-16
- Data end: 2008-12-08
- Initial capital: 100,000
- Strategy selector: ATR Trailing Stop
- Legacy final strategy portfolio: 1,449,095.2235
- Legacy final Buy & Hold: 747,161.5721

The extracted workbook data is stored in:
`MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv`

## Golden Master Validation
The recreated Python engine was tested against every available legacy calculation field.
All major numerical columns and all 1,043 signal states pass at a tolerance of `1e-8`.
The remaining numerical differences are floating-point noise only.

Audit file:
`MK_Trend_Following_Golden_Master_Validation_v001.txt`

## Deliverables

### 1. Core Engine
`MK_Trend_Following_Engine_v001.py`

Contains:
- strict Yahoo Finance adapter
- strict market-data validation
- legacy calculation engine
- ATR / Bollinger / ATR Trailing Stop strategies
- portfolio accounting
- Buy & Hold comparator
- performance summary
- Golden Master validator

### 2. Standalone HTML Generator
`MK_Trend_Following_HTML_Report_v001.py`

Generates a self-contained interactive HTML report with:
- institutional thin-font layout
- adjusted candlestick chart
- executed BUY / SELL markers
- rolling max and active stop
- strategy vs Buy & Hold equity curve
- drawdown diagnostics
- ATR / Bollinger / trailing-stop diagnostics
- filterable and sortable calculation ledger
- methodology and data-governance disclosure

### 3. Live Streamlit Application
`MK_Trend_Following_App_v001.py`

User controls:
- Yahoo ticker
- start / end date
- daily / weekly / monthly interval
- strategy
- initial capital
- ATR window
- ATR multiplier
- Bollinger window
- Bollinger standard deviation
- max-price BUY lookback
- Legacy Fidelity switch

The live application does **not** switch source if Yahoo fails. It displays `STRICT DATA STOP`.

### 4. Legacy Demonstration HTML
`MK_Trend_Following_LEGACY_GE_v001.html`

Generated exclusively from the actual historical GE dataset embedded in the original workbook.
No synthetic or fallback observations were introduced.

## Local Application Run
```bash
pip install -r requirements.txt
streamlit run MK_Trend_Following_App_v001.py
```

## Direct Engine Example
```bash
python MK_Trend_Following_Engine_v001.py \
  --ticker AAPL \
  --start 2021-01-01 \
  --end 2026-08-13 \
  --interval 1d \
  --strategy ATR_TRAILING_STOP
```

## Golden Master Test
```bash
python MK_Trend_Following_Engine_v001.py \
  --golden-test MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv
```

## Design Standard
The visual language intentionally uses:
- white / neutral institutional background
- thin typography
- restrained color
- full-width analytical charts
- explicit risk and methodology disclosure
- no decorative trading-game aesthetics

## Important Boundary
The standalone HTML is an interactive **report** generated from a specific downloaded dataset.
For a browser user to type a new ticker and request fresh Yahoo data, a Python backend is required.
That role is provided by the Streamlit application or the Colab workflow; the exported HTML remains
portable and interactive after generation.
