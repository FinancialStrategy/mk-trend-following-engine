# MK Trend Following Analytics Engine — Streamlit Cloud v0.01

**By Murat Konuklar**  
**MK FinTECH LabGEN @2026 ATELIER ISTANBUL**

A Streamlit Community Cloud-ready deployment of the validated legacy Trend Following workbook engine.

## Data Governance

This repository is intentionally strict:

- Yahoo Finance via `yfinance` is the **only live market-data source**.
- **No synthetic data.**
- **No alternate market-data fallback.**
- **No forward-fill / back-fill of missing market prices.**
- Missing or malformed Yahoo data causes an explicit **STRICT DATA STOP**.
- `auto_adjust=False` and `repair=False`.
- Raw OHLC and Adjusted Close remain distinct.
- The user-facing end date is inclusive; the adapter compensates for yfinance's exclusive `end` boundary.

## Application Entry Point

```text
app.py
```

## Repository Layout

```text
MK_Trend_Following_Streamlit_Cloud_v001/
├── app.py
├── MK_Trend_Following_Engine_v001.py
├── MK_Trend_Following_HTML_Report_v001.py
├── MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv
├── MK_Trend_Following_Golden_Master_Validation_v001.txt
├── MK_Trend_Following_Forensic_and_Architecture_v001.md
├── requirements.txt
├── smoke_test.py
├── .gitignore
├── .streamlit/
│   └── config.toml
└── .github/
    └── workflows/
        └── validate.yml
```

## Streamlit Community Cloud Deployment

1. Create a GitHub repository, for example:
   `mk-trend-following-engine`.
2. Upload the **contents** of this folder to the repository root.
3. Preserve the hidden `.streamlit` and `.github` folders.
4. Commit to the `main` branch.
5. Open Streamlit Community Cloud and choose **Create app**.
6. Choose the GitHub repository and branch `main`.
7. Set **Main file path** to:
   `app.py`
8. Open **Advanced settings**.
9. Select **Python 3.12**.
10. No secrets are required for this v0.01 Yahoo-only build.
11. Deploy.

## Local Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

## Golden Master

The offline test uses the real historical GE dataset embedded in the original legacy workbook.
It does **not** request Yahoo data and it does **not** generate synthetic data.

Expected control values:

- Observations: `1,043`
- Final strategy portfolio: `1,449,095.2235`
- Final Buy & Hold: `747,161.5721`
- Signal-state replication: complete
- Overall Golden Master result: PASS

## Cloud Behavior

The live app requests Yahoo Finance only when **RUN ANALYSIS** is pressed.

If Yahoo refuses, rate-limits, returns malformed data, or required fields are missing, the application stops that run. It will **not** silently move to another source.

## HTML Export

After a successful live analysis, the user can export:

- standalone interactive HTML report
- full calculation ledger CSV

## Security / Secrets

This version does not need API keys or `secrets.toml`.

If secrets are added in a future version, do **not** commit `.streamlit/secrets.toml`.
The repository `.gitignore` already excludes it.

## Research Use

This is a research and analytics tool. It is not investment advice.
