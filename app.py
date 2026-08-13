
from __future__ import annotations
import pandas as pd
import streamlit as st

from MK_Trend_Following_Engine_v001 import (
    EngineConfig,
    YahooFinanceAdapter,
    run_legacy_engine,
    performance_summary,
    DataIntegrityError,
    MarketDataError,
)
from MK_Trend_Following_HTML_Report_v001 import build_html

st.set_page_config(
    page_title="MK Trend Following Analytics Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
html, body, [class*="css"] {
    font-family: "Arial Narrow","Helvetica Neue",Arial,sans-serif;
    font-weight: 300;
}
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1600px;}
h1,h2,h3 {font-weight: 300 !important; letter-spacing: .02em;}
[data-testid="stMetricLabel"] {font-size: .78rem;}
[data-testid="stMetricValue"] {font-weight: 300; font-size: 1.55rem;}
hr {border: 0; border-top: 1px solid #d9dde3;}
.small-note {font-size:.79rem; color:#5f6368;}
.governance {
    border:1px solid #d9dde3; padding:10px 12px; font-size:.78rem;
    background:#fafafa; margin: 4px 0 10px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.title("MK Trend Following Analytics Engine")
st.caption("By Murat Konuklar  |  v0.01 Streamlit Cloud  |  Strict Yahoo Finance Data Policy  |  No Synthetic Data  |  No Fallback")

with st.sidebar:
    st.subheader("Analysis Controls")
    ticker = st.text_input("Yahoo Finance Ticker", value="AAPL").strip().upper()
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start Date", value=(pd.Timestamp.today().normalize() - pd.DateOffset(years=5)).date())
    with c2:
        end = st.date_input("End Date", value=pd.Timestamp.today().date())

    interval_label = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    interval = {"Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}[interval_label]

    strategy_label = st.selectbox(
        "Strategy",
        ["ATR Trailing Stop", "ATR", "Bollinger"],
        index=0,
    )
    strategy_name = {"ATR":"ATR", "Bollinger":"BOLLINGER", "ATR Trailing Stop":"ATR_TRAILING_STOP"}[strategy_label]

    st.divider()
    st.subheader("Legacy Parameters")
    initial_capital = st.number_input("Initial Capital", min_value=1.0, value=100000.0, step=10000.0)
    atr_weeks = st.number_input("ATR Window", min_value=1, value=8, step=1)
    atr_multiplier = st.number_input("ATR Multiplier", min_value=0.1, value=10.0, step=0.5)
    bollinger_weeks = st.number_input("Bollinger Window", min_value=2, value=40, step=1)
    bollinger_sd = st.number_input("Bollinger Std Dev", min_value=0.1, value=2.5, step=0.1)
    max_buy_weeks = st.number_input("Max-Price BUY Window", min_value=2, value=2000, step=10)

    legacy_fidelity = st.toggle(
        "Legacy Fidelity (inclusive OFFSET windows)",
        value=True,
        help="Preserves the original workbook's inclusive Excel OFFSET range semantics exactly."
    )

    run_clicked = st.button("RUN ANALYSIS", type="primary", use_container_width=True)

st.markdown(
    '<div class="governance"><b>DATA GOVERNANCE:</b> Yahoo Finance is the only market-data source in this build. '
    'Missing, malformed, duplicate, non-positive or insufficient observations stop the run. '
    'No synthetic observations, interpolation, alternate vendor, cached substitution or silent fallback is permitted.</div>',
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.summary = None
    st.session_state.config = None
    st.session_state.raw = None

if run_clicked:
    if not ticker:
        st.error("Ticker is required.")
        st.stop()
    if start >= end:
        st.error("End Date must be later than Start Date.")
        st.stop()

    cfg = EngineConfig(
        initial_capital=float(initial_capital),
        atr_weeks=int(atr_weeks),
        atr_multiplier=float(atr_multiplier),
        bollinger_weeks=int(bollinger_weeks),
        bollinger_sd=float(bollinger_sd),
        max_buy_weeks=int(max_buy_weeks),
        strategy=strategy_name,
        legacy_inclusive_windows=bool(legacy_fidelity),
    )

    try:
        with st.spinner("Requesting Yahoo Finance and calculating the strategy..."):
            raw = YahooFinanceAdapter.fetch(
                ticker=ticker,
                start=str(start),
                end=str(end),
                interval=interval,
            )
            result = run_legacy_engine(raw, cfg)
            summary = performance_summary(result, initial_capital=cfg.initial_capital)
        st.session_state.result = result
        st.session_state.summary = summary
        st.session_state.config = cfg
        st.session_state.raw = raw
        st.session_state.ticker = ticker
        st.session_state.interval = interval
    except (DataIntegrityError, MarketDataError) as exc:
        st.error(f"STRICT DATA STOP — {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"RUN STOPPED — {type(exc).__name__}: {exc}")
        st.stop()

result = st.session_state.result
summary = st.session_state.summary
cfg = st.session_state.config

if result is None:
    st.info("Set the parameters and run the analysis. No data is loaded until RUN ANALYSIS is pressed.")
    st.stop()

ticker_used = st.session_state.get("ticker", ticker)
interval_used = st.session_state.get("interval", interval)

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Last Adj. Close", f"{float(result['AdjCloseCalc'].iloc[-1]):,.2f}")
k2.metric("Current Signal", summary["current_signal"])
k3.metric("Strategy CAGR", f"{summary['strategy_cagr']:.2%}")
k4.metric("Buy & Hold CAGR", f"{summary['buyhold_cagr']:.2%}")
k5.metric("Strategy Max DD", f"{summary['max_drawdown']:.2%}")
k6.metric("Ann. Volatility", f"{summary['annualized_volatility']:.2%}")

tabs = st.tabs([
    "Executive",
    "Price & Signals",
    "Strategy vs Buy & Hold",
    "Drawdown",
    "Trend Diagnostics",
    "Calculation Ledger",
    "Methodology & Governance",
])

import plotly.graph_objects as go

with tabs[0]:
    a,b,c,d = st.columns(4)
    a.metric("Observations", f"{summary['observations']:,}")
    b.metric("Final Strategy Value", f"{summary['portfolio_final']:,.2f}")
    c.metric("Final Buy & Hold", f"{summary['buyhold_final']:,.2f}")
    d.metric("Completed Sells", f"{summary['first_sells']:,}")
    st.write(
        "The dashboard reproduces the legacy workbook's signal timing discipline: "
        "the completed prior bar determines the signal and any resulting trade executes at the next bar's adjusted open."
    )

with tabs[1]:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=result.index,
        open=result["AdjOpen"], high=result["AdjHigh"], low=result["AdjLow"], close=result["AdjCloseCalc"],
        name="Adjusted OHLC"
    ))
    fig.add_trace(go.Scatter(x=result.index, y=result["MaxPrice"], name="Rolling Max", mode="lines"))
    stop_col = {"ATR":"ATR_Stop","BOLLINGER":"LowerBollinger","ATR_TRAILING_STOP":"ATRTrailingStop"}[cfg.strategy]
    fig.add_trace(go.Scatter(x=result.index, y=result[stop_col], name="Active Stop", mode="lines"))
    buys = result["FirstBuy"].fillna(0).astype(bool)
    sells = result["FirstSell"].fillna(0).astype(bool)
    fig.add_trace(go.Scatter(
        x=result.index[buys], y=result.loc[buys, "AdjOpen"], mode="markers", name="Executed BUY",
        marker=dict(symbol="triangle-up", size=10)
    ))
    fig.add_trace(go.Scatter(
        x=result.index[sells], y=result.loc[sells, "AdjOpen"], mode="markers", name="Executed SELL",
        marker=dict(symbol="triangle-down", size=10)
    ))
    fig.update_layout(
        height=650, template="plotly_white", hovermode="x unified",
        xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=35,b=20),
        legend=dict(orientation="h", y=1.03)
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["Portfolio"], mode="lines", name="Trend Strategy"))
    fig.add_trace(go.Scatter(x=result.index, y=result["BuyHold"], mode="lines", name="Buy & Hold"))
    fig.update_layout(
        height=560, template="plotly_white", hovermode="x unified",
        margin=dict(l=20,r=20,t=35,b=20), legend=dict(orientation="h", y=1.03)
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    strategy_dd = result["Portfolio"] / result["Portfolio"].cummax() - 1.0
    buyhold_dd = result["BuyHold"] / result["BuyHold"].cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=strategy_dd, mode="lines", name="Strategy DD"))
    fig.add_trace(go.Scatter(x=result.index, y=buyhold_dd, mode="lines", name="Buy & Hold DD"))
    fig.update_layout(
        height=520, template="plotly_white", hovermode="x unified",
        yaxis_tickformat=".0%", margin=dict(l=20,r=20,t=35,b=20),
        legend=dict(orientation="h", y=1.03)
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["AdjCloseCalc"], mode="lines", name="Adj Close"))
    fig.add_trace(go.Scatter(x=result.index, y=result["ATR_Stop"], mode="lines", name="ATR Stop"))
    fig.add_trace(go.Scatter(x=result.index, y=result["LowerBollinger"], mode="lines", name="Lower Bollinger"))
    fig.add_trace(go.Scatter(x=result.index, y=result["ATRTrailingStop"], mode="lines", name="ATR Trailing Stop"))
    fig.update_layout(
        height=600, template="plotly_white", hovermode="x unified",
        margin=dict(l=20,r=20,t=35,b=20), legend=dict(orientation="h", y=1.03)
    )
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    show_cols = [
        "Open","High","Low","Close","Volume","Adj Close","Return","TrueRange","ATR_Stop",
        "MaxPrice","Signal","Shares","Cash","Portfolio","BuyHold","LowerBollinger","ATRTrailingStop"
    ]
    st.dataframe(
        result[show_cols].sort_index(ascending=False),
        use_container_width=True,
        height=650,
    )
    csv_bytes = result.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Calculation Ledger CSV",
        csv_bytes,
        file_name=f"MK_Trend_Following_{ticker_used}_ledger.csv",
        mime="text/csv",
    )

with tabs[6]:
    st.markdown("""
### Legacy Fidelity
The original Excel mechanics are preserved in the fidelity mode, including inclusive `OFFSET`-style rolling windows.
A nominal 8-period ATR setting therefore contains 9 observations after the window saturates, exactly as in the legacy workbook.

### Signal Timing
Signals are determined from the prior completed bar. Trades are executed at the next bar's adjusted open.
This removes same-bar execution look-ahead from the legacy strategy.

### Strict Market-Data Governance
This application accepts Yahoo Finance market data only. It does not fabricate observations and it does not switch to another vendor.
A failed or incomplete Yahoo response terminates the run with an explicit error.

### Price Adjustment
Yahoo is requested with `auto_adjust=False`. Raw OHLC and `Adj Close` remain separately available.
The legacy engine derives a scale factor from `Adj Close / Close` and applies it to OHLC, matching the workbook's mechanics.
""")

html = build_html(
    result,
    cfg,
    ticker=ticker_used,
    source_note=f"Yahoo Finance via yfinance | {interval_used} | Strict no-fallback policy",
)
st.download_button(
    "Export Standalone Interactive HTML",
    data=html.encode("utf-8"),
    file_name=f"MK_Trend_Following_{ticker_used}_{interval_used}_v001.html",
    mime="text/html",
    use_container_width=True,
)

st.caption("MK FinTECH LabGEN @2026 ATELIER ISTANBUL")
