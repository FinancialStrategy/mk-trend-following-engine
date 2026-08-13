from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from MK_Trend_Following_Engine_v001 import (
    EngineConfig,
    YahooFinanceAdapter,
    run_legacy_engine,
    performance_summary,
    DataIntegrityError,
    MarketDataError,
)
from MK_Trend_Following_Decision_Engine_v002 import (
    decision_snapshot,
    trade_ledger,
    trade_statistics,
    active_stop_column,
)
from MK_Trend_Following_Universe_v002 import (
    market_names,
    groups_for,
    instruments_for,
    flat_universe_rows,
)
from MK_Trend_Following_Risk_Analytics_v004 import (
    rolling_window_options,
    rolling_risk_frame,
    risk_state_snapshot,
    validate_underlying_risk_dynamics,
    cash_regimes,
)
from MK_Trend_Following_HTML_Report_v003 import build_html


APP_VERSION = "v0.04"
PLOT_CFG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
RANGE_SELECTOR = dict(
    buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ],
    x=0,
    xanchor="left",
    y=1.12,
    yanchor="top",
    bgcolor="#FFFFFF",
    activecolor="#E2E8F0",
    bordercolor="#CBD5E1",
    borderwidth=1,
    font=dict(size=10),
)

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
.block-container {padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1720px;}
h1,h2,h3,h4 {font-weight: 300 !important; letter-spacing: .015em;}
[data-testid="stMetricLabel"] {font-size: .72rem; letter-spacing:.04em; text-transform:uppercase;}
[data-testid="stMetricValue"] {font-weight: 300; font-size: 1.45rem;}
hr {border: 0; border-top: 1px solid #d9dde3;}
.governance {border:1px solid #d9dde3; padding:10px 12px; font-size:.76rem; background:#fafafa; margin:4px 0 12px 0;}
.decision-card {border:1px solid #cbd5e1; border-left:4px solid #334155; padding:16px 18px; background:#fff; min-height:155px;}
.decision-label {font-size:.68rem; letter-spacing:.16em; color:#64748b; text-transform:uppercase;}
.decision-value {font-size:2rem; font-weight:300; margin:3px 0 8px; color:#0f172a;}
.decision-reason {font-size:.85rem; line-height:1.55; color:#334155;}
.micro {font-size:.75rem; color:#64748b; line-height:1.45;}
.section-note {border-left:2px solid #cbd5e1; padding-left:10px; color:#475569; font-size:.82rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------- Formatting helpers ----------------------------
def fmt_pct(v):
    return "—" if v is None or not np.isfinite(float(v)) else f"{float(v):.2%}"


def fmt_num(v, d=2):
    return "—" if v is None or not np.isfinite(float(v)) else f"{float(v):,.{d}f}"


def _base_layout(fig, title, height=560, ytitle=None):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color="#0F172A")),
        template="plotly_white",
        height=height,
        margin=dict(l=52, r=24, t=76, b=38),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif", size=11, color="#334155"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right", yanchor="bottom"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1", title_text=ytitle or "")
    return fig


def make_price_chart(df, cfg, chart_mode="Candlestick"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.035,
    )
    if chart_mode == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
            name="Adjusted OHLC", increasing_line_color="#334155", decreasing_line_color="#94A3B8",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["AdjCloseCalc"], mode="lines", name="Adjusted Close",
            line=dict(width=1.4, color="#0F172A"),
        ), row=1, col=1)

    stop_col = active_stop_column(cfg)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MaxPrice"], mode="lines", name="Rolling Max / Entry Gate",
        line=dict(width=1.0, color="#64748B", dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[stop_col], mode="lines", name=f"Active Stop — {stop_col}",
        line=dict(width=1.6, color="#B45309"),
    ), row=1, col=1)

    buys = df[df["FirstBuy"] > 0]
    sells = df[df["FirstSell"] > 0]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["AdjOpen"], mode="markers", name="Executed BUY",
        marker=dict(symbol="triangle-up", size=11, color="#166534", line=dict(width=1, color="#FFFFFF")),
        customdata=np.column_stack([buys["Portfolio"]]) if len(buys) else None,
        hovertemplate="BUY<br>%{x}<br>Open: %{y:,.2f}<br>Portfolio: %{customdata[0]:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["AdjOpen"], mode="markers", name="Executed SELL",
        marker=dict(symbol="triangle-down", size=11, color="#991B1B", line=dict(width=1, color="#FFFFFF")),
        customdata=np.column_stack([sells["Portfolio"]]) if len(sells) else None,
        hovertemplate="SELL<br>%{x}<br>Open: %{y:,.2f}<br>Portfolio: %{customdata[0]:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume", marker_color="#CBD5E1", opacity=.7,
        hovertemplate="%{x}<br>Volume: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    _base_layout(fig, "Interactive Price, Entry Gate, Active Stop and Executed Trades", 680)
    fig.update_yaxes(title_text="Adjusted Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False)))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def make_equity_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Portfolio"], mode="lines", name="Trend Strategy", line=dict(width=1.8, color="#0F172A")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BuyHold"], mode="lines", name="Buy & Hold", line=dict(width=1.4, color="#64748B", dash="dot")))
    _base_layout(fig, "Strategy Equity Curve vs Buy & Hold", 560, "Portfolio Value")
    fig.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False)))
    return fig


def make_drawdown_chart(df):
    dd = df["Portfolio"] / df["Portfolio"].cummax() - 1.0
    bhdd = df["BuyHold"] / df["BuyHold"].cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=dd, mode="lines", name="Strategy Drawdown", fill="tozeroy", line=dict(width=1.3, color="#991B1B")))
    fig.add_trace(go.Scatter(x=df.index, y=bhdd, mode="lines", name="Buy & Hold Drawdown", line=dict(width=1.1, color="#64748B")))
    _base_layout(fig, "Drawdown and Capital Preservation", 500, "Drawdown")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False)))
    return fig


def make_trend_diagnostics(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["AdjCloseCalc"], mode="lines", name="Adjusted Close", line=dict(width=1.2, color="#0F172A")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MaxPrice"], mode="lines", name="Rolling Max", line=dict(width=1.0, color="#64748B", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR_Stop"], mode="lines", name="ATR Stop", line=dict(width=1.0, color="#7C3AED")))
    fig.add_trace(go.Scatter(x=df.index, y=df["LowerBollinger"], mode="lines", name="Lower Bollinger", line=dict(width=1.0, color="#2563EB")))
    fig.add_trace(go.Scatter(x=df.index, y=df["ATRTrailingStop"], mode="lines", name="ATR Trailing Stop", line=dict(width=1.5, color="#B45309")))
    _base_layout(fig, "Trend Diagnostics — All Legacy Thresholds", 590, "Adjusted Price / Threshold")
    fig.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False)))
    return fig


def make_underlying_rolling_risk_chart(df, rolling, spec, instrument_label):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rolling["AssetRollingReturn"], mode="lines",
            name=f"{spec.label} Asset Return",
            line=dict(width=1.5, color="#0F172A"),
            hovertemplate="%{x}<br>Rolling Return: %{y:.2%}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rolling["AssetAnnualizedVolatility"], mode="lines",
            name=f"{spec.label} Ann. Asset Volatility",
            line=dict(width=1.4, color="#B45309"),
            hovertemplate="%{x}<br>Ann. Volatility: %{y:.2%}<extra></extra>",
        ),
        secondary_y=True,
    )
    _base_layout(
        fig,
        f"{instrument_label} — Underlying Asset Rolling Return & Volatility",
        520,
    )
    fig.update_yaxes(title_text="Underlying Rolling Return", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Underlying Annualized Volatility", tickformat=".0%", secondary_y=True)
    fig.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False)))
    return fig


def make_strategy_rolling_risk_chart(df, rolling, spec):
    # Two-row institutional view:
    # top = strategy risk only while it has market exposure,
    # bottom = rolling exposure. Pure-cash windows are shaded.
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rolling["StrategyRollingReturnDisplay"],
            mode="lines",
            name=f"{spec.label} Strategy Return",
            line=dict(width=1.5, color="#334155"),
            connectgaps=False,
            hovertemplate="%{x}<br>Strategy Rolling Return: %{y:.2%}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rolling["StrategyAnnualizedVolatilityDisplay"],
            mode="lines",
            name=f"{spec.label} Ann. Strategy Volatility",
            line=dict(width=1.4, color="#7C3AED"),
            connectgaps=False,
            hovertemplate="%{x}<br>Strategy Ann. Volatility: %{y:.2%}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rolling["RollingExposure"],
            mode="lines",
            name=f"{spec.label} Rolling Market Exposure",
            line=dict(width=1.35, color="#64748B"),
            fill="tozeroy",
            fillcolor="rgba(100,116,139,0.10)",
            connectgaps=False,
            hovertemplate="%{x}<br>Rolling Exposure: %{y:.1%}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Mark pure-cash windows explicitly instead of drawing a misleading
    # horizontal 0% strategy-risk line.
    for x0, x1 in cash_regimes(rolling):
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(148,163,184,0.12)",
            line_width=0,
            layer="below",
            row="all", col=1,
        )

    fig.update_layout(
        title=dict(
            text="Trend Strategy Rolling Risk & Market Exposure",
            x=0.01, xanchor="left",
            font=dict(size=16, family="Arial, sans-serif", color="#111827"),
        ),
        height=650,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=45, r=45, t=60, b=30),
        legend=dict(orientation="h", y=1.04, x=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        title_text="Strategy Rolling Return",
        tickformat=".0%",
        row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Strategy Ann. Volatility",
        tickformat=".0%",
        row=1, col=1, secondary_y=True
    )
    fig.update_yaxes(
        title_text="Market Exposure",
        tickformat=".0%",
        range=[0, 1.02],
        row=2, col=1
    )
    fig.update_xaxes(
        rangeselector=RANGE_SELECTOR,
        rangeslider=dict(visible=False),
        row=1, col=1,
    )
    return fig


# ---------------------------- Header ----------------------------
st.title("MK Trend Following Analytics Engine")
st.caption(f"By Murat Konuklar  |  {APP_VERSION} Streamlit Cloud  |  Institutional Trend Systems")
st.markdown(
    '<div class="governance"><b>STRICT DATA GOVERNANCE:</b> Yahoo Finance is the only live market-data source. '
    'No synthetic observations, no alternate provider, no forward-fill/back-fill, no silent substitution. '
    'A failed or incomplete Yahoo response stops the requested analysis.</div>',
    unsafe_allow_html=True,
)


# ---------------------------- Sidebar controls ----------------------------
with st.sidebar:
    st.subheader("Instrument Universe")
    market = st.selectbox("Market / Asset Universe", market_names(), index=0)

    selected_name = "Manual"
    selected_group = "Manual"
    if market == "Manual Yahoo Ticker":
        ticker = st.text_input("Yahoo Finance Ticker", value="AAPL").strip().upper()
    else:
        groups = groups_for(market)
        selected_group = st.selectbox("Sector / Group", groups, index=0)
        instruments = instruments_for(market, selected_group)
        selected = st.selectbox(
            "Instrument",
            instruments,
            format_func=lambda x: f"{x[0]}  |  {x[1]}",
            index=0,
        )
        selected_name, ticker = selected
        st.caption(f"Yahoo ticker in use: `{ticker}`")

    st.divider()
    st.subheader("Analysis Controls")
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start Date", value=(pd.Timestamp.today().normalize() - pd.DateOffset(years=5)).date())
    with c2:
        end = st.date_input("End Date", value=pd.Timestamp.today().date())

    interval_label = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    interval = {"Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}[interval_label]

    strategy_label = st.selectbox("Strategy", ["ATR Trailing Stop", "ATR", "Bollinger"], index=0)
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
        "Legacy Fidelity",
        value=True,
        help="Preserves the original workbook's inclusive Excel OFFSET window semantics exactly.",
    )

    run_clicked = st.button("RUN ANALYSIS", type="primary", use_container_width=True)


# ---------------------------- State ----------------------------
for key, default in {
    "result": None, "summary": None, "config": None, "raw": None,
    "decision": None, "trades": None, "trade_stats": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
        with st.spinner(f"Requesting Yahoo Finance for {ticker} and running the validated legacy engine..."):
            raw = YahooFinanceAdapter.fetch(
                ticker=ticker, start=str(start), end=str(end), interval=interval,
                minimum_observations=cfg.minimum_observations,
            )
            result = run_legacy_engine(raw, cfg)
            summary = performance_summary(result, initial_capital=cfg.initial_capital)
            decision = decision_snapshot(result, cfg)
            trades = trade_ledger(result)
            tstats = trade_statistics(trades)

        st.session_state.result = result
        st.session_state.summary = summary
        st.session_state.config = cfg
        st.session_state.raw = raw
        st.session_state.decision = decision
        st.session_state.trades = trades
        st.session_state.trade_stats = tstats
        st.session_state.ticker = ticker
        st.session_state.instrument_name = selected_name
        st.session_state.market = market
        st.session_state.group = selected_group
        st.session_state.interval = interval
        st.session_state.interval_label = interval_label
    except (DataIntegrityError, MarketDataError) as exc:
        st.error(f"STRICT DATA STOP — {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"RUN STOPPED — {type(exc).__name__}: {exc}")
        st.stop()

result = st.session_state.result
summary = st.session_state.summary
cfg = st.session_state.config
decision = st.session_state.decision
trades = st.session_state.trades
tstats = st.session_state.trade_stats

if result is None:
    st.info("Choose an instrument from BIST, US Stocks, Precious Metals, or enter a manual Yahoo ticker; then press RUN ANALYSIS.")
    u = pd.DataFrame(flat_universe_rows())
    st.subheader("Curated Instrument Universe")
    st.dataframe(u, use_container_width=True, hide_index=True, height=460)
    st.caption("The universe is a convenience selector only. Live price history is requested from Yahoo Finance after RUN ANALYSIS; the list itself is not a fallback data source.")
    st.stop()

# Persisted run identity
ticker_used = st.session_state.ticker
name_used = st.session_state.instrument_name
market_used = st.session_state.market
group_used = st.session_state.group
interval_used = st.session_state.interval
interval_label_used = st.session_state.interval_label

# ---------------------------- Executive strip ----------------------------
st.markdown(f"### {name_used if name_used != 'Manual' else ticker_used}  ·  `{ticker_used}`")
st.caption(f"{market_used} / {group_used}  |  {interval_label_used}  |  {summary['start'].date()} → {summary['end'].date()}  |  Active strategy: {cfg.strategy.replace('_',' ')}")

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("Decision", decision["decision"])
k2.metric("Position", decision["position"].replace(" / LONG", ""))
k3.metric("Last Adj. Close", fmt_num(result["AdjCloseCalc"].iloc[-1]))
k4.metric("Price / Stop Gap", fmt_pct(decision["price_stop_gap"]))
k5.metric("Strategy CAGR", fmt_pct(summary["strategy_cagr"]))
k6.metric("Buy & Hold CAGR", fmt_pct(summary["buyhold_cagr"]))
k7.metric("Strategy Max DD", fmt_pct(summary["max_drawdown"]))
k8.metric("Ann. Volatility", fmt_pct(summary["annualized_volatility"]))

# ---------------------------- Main tabs ----------------------------
tabs = st.tabs([
    "Executive & Decision",
    "Price & Signals",
    "Strategy vs Buy & Hold",
    "Risk Analytics",
    "Trend Diagnostics",
    "Trade Ledger",
    "Calculation Ledger",
    "Instrument Universe",
    "Methodology & Governance",
])

with tabs[0]:
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            f'''<div class="decision-card">
                <div class="decision-label">Current Strategy Decision</div>
                <div class="decision-value">{decision['decision']}</div>
                <div class="decision-reason">{decision['rationale']}</div>
            </div>''',
            unsafe_allow_html=True,
        )
        st.markdown("#### Decision Causality Matrix")
        st.dataframe(decision["gates"], use_container_width=True, hide_index=True, height=248)
        st.markdown(
            f"<div class='section-note'><b>Important:</b> {decision['legacy_scope_note']} "
            "Therefore the valid strategy actions are BUY, HOLD, SELL and WAIT / CASH. "
            "A REDUCE decision would be a new methodology and is deliberately not invented inside Legacy Fidelity.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Trigger Diagnostics")
        d1,d2 = st.columns(2)
        d1.metric("Raw Legacy Trigger", decision["raw_trigger"])
        d2.metric("Active Threshold", decision["active_stop_column"])
        d3,d4 = st.columns(2)
        d3.metric("Prior Close", fmt_num(decision["prior_close"]))
        d4.metric("Prior Rolling Max", fmt_num(decision["prior_max"]))
        d5,d6 = st.columns(2)
        d5.metric("Prior Active Stop", fmt_num(decision["prior_active_stop"]))
        d6.metric("Prior Close / Stop", fmt_pct(decision["prior_stop_gap"]))
        d7,d8 = st.columns(2)
        d7.metric("Breakout Gap", fmt_pct(decision["breakout_gap"]))
        d8.metric("Completed Exits", f"{summary['first_sells']:,}")
        st.markdown(
            "<div class='micro'>Decision timing: the prior completed bar determines the trigger. "
            "If an executable BUY or SELL is generated, the transaction occurs at the current bar's adjusted open. "
            "A raw BUY while already invested becomes portfolio-level HOLD because the legacy engine does not pyramid.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Performance Snapshot")
    p1,p2,p3,p4,p5,p6 = st.columns(6)
    p1.metric("Final Strategy Value", f"{summary['portfolio_final']:,.0f}")
    p2.metric("Final Buy & Hold", f"{summary['buyhold_final']:,.0f}")
    p3.metric("Closed Trades", f"{tstats['closed_trades']:,}")
    p4.metric("Win Rate", fmt_pct(tstats["win_rate"]))
    p5.metric("Avg Trade", fmt_pct(tstats["avg_trade"]))
    p6.metric("Avg Holding Days", fmt_num(tstats["avg_holding_days"], 0))

with tabs[1]:
    c1, c2 = st.columns([1, 3])
    with c1:
        chart_mode = st.radio("Price display", ["Candlestick", "Adjusted Close Line"], horizontal=False)
        st.caption("Use the range buttons above the chart, mouse-wheel zoom, legend trace toggles and hover inspection.")
    with c2:
        st.markdown(
            "<div class='section-note'><b>Entry gate:</b> prior adjusted close must reach the prior rolling maximum. "
            "<b>Exit gate:</b> prior adjusted close must breach the selected stop. "
            "BUY/SELL markers show actual executions at the next bar's adjusted open.</div>",
            unsafe_allow_html=True,
        )
    st.plotly_chart(make_price_chart(result, cfg, chart_mode), use_container_width=True, config=PLOT_CFG)

with tabs[2]:
    st.plotly_chart(make_equity_chart(result), use_container_width=True, config=PLOT_CFG)
    e1,e2,e3,e4 = st.columns(4)
    total_strategy = summary["portfolio_final"] / cfg.initial_capital - 1.0
    total_bh = summary["buyhold_final"] / cfg.initial_capital - 1.0
    e1.metric("Total Strategy Return", fmt_pct(total_strategy))
    e2.metric("Total Buy & Hold Return", fmt_pct(total_bh))
    e3.metric("Cumulative Excess", fmt_pct(total_strategy-total_bh))
    e4.metric("CAGR Spread", fmt_pct(summary["strategy_cagr"]-summary["buyhold_cagr"]))

with tabs[3]:
    st.markdown(
        "<div class='section-note'><b>Risk-source separation:</b> Underlying Asset risk is calculated from the instrument's "
        "adjusted-close series. Strategy risk is calculated separately from the portfolio equity curve. "
        "Cash periods can legitimately flatten Strategy return/volatility, but they must never flatten the underlying instrument's risk chart.</div>",
        unsafe_allow_html=True,
    )

    window_specs = rolling_window_options(result.index)
    default_idx = next((i for i, x in enumerate(window_specs) if x.label == "3M"), 0)
    selected_spec = st.selectbox(
        "Rolling Risk Window",
        window_specs,
        index=default_idx,
        format_func=lambda x: f"{x.label}  |  {x.observations} {x.frequency_label.lower()} observations",
    )
    rolling, used_spec = rolling_risk_frame(result, selected_spec.observations)
    risk_state = risk_state_snapshot(result, rolling, used_spec)
    risk_integrity = validate_underlying_risk_dynamics(result, rolling)

    if risk_integrity["impossible_flatness"]:
        st.error(
            "RISK INTEGRITY STOP — The underlying adjusted-close series moves, but both "
            "underlying rolling return and rolling volatility are effectively constant. "
            "This is not accepted as a valid chart state. Review the deployed files / data pipeline."
        )
        st.stop()

    st.plotly_chart(make_drawdown_chart(result), use_container_width=True, config=PLOT_CFG)

    st.markdown("#### Underlying Asset Risk — Market Data")
    st.plotly_chart(
        make_underlying_rolling_risk_chart(result, rolling, used_spec, f"{name_used} ({ticker_used})"),
        use_container_width=True,
        config=PLOT_CFG,
    )
    a1,a2,a3,a4 = st.columns(4)
    a1.metric(f"{used_spec.label} Asset Rolling Return", fmt_pct(risk_state["asset_rolling_return"]))
    a2.metric(f"{used_spec.label} Asset Ann. Volatility", fmt_pct(risk_state["asset_annualized_volatility"]))
    a3.metric("Underlying Source", "Adjusted Close")
    a4.metric("Risk-Series Unique Points", f"{risk_integrity['rolling_return_unique']:,}")

    with st.expander("Underlying Risk Integrity Diagnostics", expanded=False):
        st.dataframe(
            pd.DataFrame([{
                "Adjusted Close Unique Prices": risk_integrity["price_unique"],
                "Adjusted Close Range": risk_integrity["price_range"],
                "Rolling Return Unique Values": risk_integrity["rolling_return_unique"],
                "Rolling Return Range": risk_integrity["rolling_return_range"],
                "Rolling Vol Unique Values": risk_integrity["rolling_vol_unique"],
                "Rolling Vol Range": risk_integrity["rolling_vol_range"],
                "Impossible Flatness": risk_integrity["impossible_flatness"],
            }]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Strategy Risk — Portfolio Exposure")
    st.plotly_chart(
        make_strategy_rolling_risk_chart(result, rolling, used_spec),
        use_container_width=True,
        config=PLOT_CFG,
    )
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric(f"{used_spec.label} Strategy Rolling Return", fmt_pct(risk_state["strategy_rolling_return"]))
    s2.metric(f"{used_spec.label} Strategy Ann. Volatility", fmt_pct(risk_state["strategy_annualized_volatility"]))
    s3.metric(f"{used_spec.label} Rolling Exposure", fmt_pct(risk_state["rolling_exposure"]))
    s4.metric("Current Position", risk_state["current_position"])
    s5.metric("Full-History Cash Exposure", fmt_pct(risk_state["cash_exposure_ratio"]))

    if risk_state["strategy_flat_reason"]:
        st.info(risk_state["strategy_flat_reason"])
    else:
        st.caption(
            "If the strategy graph contains flat segments, inspect Current Position and cash exposure. "
            "Flat strategy volatility during a cash regime is mathematically valid and is not interpreted as zero volatility for the underlying stock."
        )

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Strategy Max DD", fmt_pct(summary["max_drawdown"]))
    r2.metric("Buy & Hold Max DD", fmt_pct(summary["buyhold_max_drawdown"]))
    r3.metric("Best Closed Trade", fmt_pct(tstats["best_trade"]))
    r4.metric("Worst Closed Trade", fmt_pct(tstats["worst_trade"]))

with tabs[4]:
    st.plotly_chart(make_trend_diagnostics(result), use_container_width=True, config=PLOT_CFG)
    st.markdown(
        "<div class='section-note'>All three legacy thresholds are shown simultaneously for diagnosis. "
        "Only the strategy selected in the sidebar controls the exit decision. Legacy ATR is reproduced exactly; "
        "it is not silently replaced by a conventional modern ATR-price stop.</div>",
        unsafe_allow_html=True,
    )

with tabs[5]:
    if trades is None or trades.empty:
        st.info("No completed or open trades were generated in the selected history.")
    else:
        tdisplay = trades.copy()
        if "Trade Return" in tdisplay:
            tdisplay["Trade Return"] = tdisplay["Trade Return"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        for c in ["Entry Price", "Exit Price"]:
            if c in tdisplay:
                tdisplay[c] = tdisplay[c].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
        st.dataframe(tdisplay.sort_values("Entry Date", ascending=False), use_container_width=True, hide_index=True, height=520)
        st.caption("Trade returns use executed adjusted-open prices; an open trade is marked to the latest adjusted close.")

with tabs[6]:
    show_cols = [
        "Open","High","Low","Close","Volume","Adj Close","Return","TrueRange","ATR_Stop",
        "MaxPrice","Signal","Shares","Cash","Portfolio","BuyHold","LowerBollinger","ATRTrailingStop"
    ]
    st.dataframe(result[show_cols].sort_index(ascending=False), use_container_width=True, height=650)
    csv_bytes = result.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Calculation Ledger CSV", csv_bytes,
        file_name=f"MK_Trend_Following_{ticker_used}_ledger_{APP_VERSION.replace('.','')}.csv",
        mime="text/csv",
    )

with tabs[7]:
    universe_df = pd.DataFrame(flat_universe_rows())
    uc1, uc2 = st.columns(2)
    with uc1:
        market_filter = st.multiselect("Filter market", sorted(universe_df["Market"].unique()), default=[])
    with uc2:
        group_options = sorted(universe_df["Group"].unique())
        group_filter = st.multiselect("Filter sector / group", group_options, default=[])
    filt = universe_df.copy()
    if market_filter:
        filt = filt[filt["Market"].isin(market_filter)]
    if group_filter:
        filt = filt[filt["Group"].isin(group_filter)]
    st.dataframe(filt, use_container_width=True, hide_index=True, height=580)
    st.caption("This is a curated convenience universe, not an exhaustive exchange constituent list. Manual Yahoo ticker input remains available for instruments outside the list.")

with tabs[8]:
    st.markdown(f"""
### Decision hierarchy
**1. Entry gate:** the prior completed adjusted close is compared with the prior rolling maximum.  
**2. Exit gate:** the prior completed adjusted close is compared with the active strategy threshold.  
**3. Position gate:** the engine checks whether the portfolio is already invested or in cash.  
**4. Execution:** executable transactions occur at the current bar's adjusted open.

### Portfolio-level decisions
- **BUY:** breakout gate is triggered while the portfolio is in cash; shares are purchased at the current adjusted open.
- **HOLD:** the portfolio is already invested and no executable exit occurs. A repeated raw BUY trigger also remains HOLD because the legacy engine does not pyramid.
- **SELL:** the active stop gate is breached while the portfolio is invested; shares are liquidated at the current adjusted open.
- **WAIT / CASH:** the portfolio is in cash and no executable entry occurs.

### Why REDUCE is absent
The original workbook is an all-in/all-out trend-following system. Partial position reduction is not part of its validated legacy mathematics. Adding REDUCE would require a separately specified modern strategy layer and is intentionally excluded from Legacy Fidelity.

### Legacy Fidelity
The original Excel mechanics are preserved, including inclusive `OFFSET`-style rolling windows. A nominal ATR parameter of 8 therefore contains 9 observations after saturation, exactly as in the legacy workbook.

### Strict market-data governance
Yahoo Finance is the only live market-data source in this build. The application does not fabricate observations, fill missing market prices, or switch to another vendor. A failed or incomplete Yahoo response terminates that requested run.

### Price adjustment
Yahoo is requested with `auto_adjust=False`. Raw OHLC and `Adj Close` stay distinct. The legacy engine derives `Scale = Adj Close / Close` and applies it to OHLC to reproduce the workbook methodology.

### Research boundary
The displayed BUY / HOLD / SELL / WAIT labels are deterministic **strategy states**, not discretionary investment recommendations.
""")

html_doc = build_html(
    result,
    cfg,
    ticker=ticker_used,
    instrument_name=name_used,
    market_label=f"{market_used} / {group_used}",
    source_note=f"Yahoo Finance via yfinance | {interval_used} | Strict no-fallback policy",
)
st.download_button(
    "Export Standalone Interactive HTML",
    data=html_doc.encode("utf-8"),
    file_name=f"MK_Trend_Following_{ticker_used}_{interval_used}_v004.html",
    mime="text/html",
    use_container_width=True,
)

st.caption("MK FinTECH LabGEN @2026 ATELIER ISTANBUL  |  By Murat Konuklar")
