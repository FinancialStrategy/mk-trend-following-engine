from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from MK_Trend_Following_Engine_v001 import (
    EngineConfig,
    run_legacy_engine,
    performance_summary,
    DataIntegrityError,
    MarketDataError,
)
try:
    from MK_Yahoo_Original_Fetch_v0085 import YahooFinanceAdapter
except ModuleNotFoundError:
    # Deployment-resilience fallback only.
    # This is NOT an alternate market-data provider:
    # it uses the original Yahoo/yfinance adapter already embedded in the legacy engine module.
    from MK_Trend_Following_Engine_v001 import YahooFinanceAdapter

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
from MK_Institutional_Risk_Analytics_v008 import (
    VaRConfig,
    rolling_window_options,
    rolling_risk_frame,
    risk_state_snapshot,
    validate_underlying_risk_dynamics,
    cash_regimes,
    build_var_table,
)
from MK_Trend_Following_Entry_Gate_v005 import (
    horizon_options,
    resolve_entry_lookback,
    effective_gate_state,
    portfolio_cash_regimes,
    longest_cash_regime,
    latest_execution_events,
)
from MK_Trend_Following_HTML_Report_v003 import build_html
from MK_Nadaraya_Watson_Trend_v006 import (
    NWConfig,
    NWStrategyConfig,
    KERNELS as NW_KERNELS,
    compute_nadaraya_watson,
    run_nw_strategy,
    nw_decision_snapshot,
    kernel_weight_profile,
    strategy_mode_label,
)
from MK_Nadaraya_Watson_HTML_Report_v006 import build_nw_html_report
from MK_Benchmark_Relative_v007 import (
    RelativeConfig, default_benchmark, benchmark_name,
    compute_relative_analytics, relative_snapshot,
)
from MK_Institutional_Tactical_v007 import (
    TacticalConfig, run_tactical_strategy, tactical_snapshot,
)


APP_VERSION = "v0.08.5.2"
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
    y=1.22,
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


def _yahoo_cache_bucket(interval: str) -> int:
    """
    Prevent routine Streamlit reruns from repeatedly hitting Yahoo.

    15m data: refresh at most once/minute for identical requests.
    Daily:    refresh at most once/5 minutes.
    Weekly:   refresh at most once/15 minutes.
    Monthly:  refresh at most once/30 minutes.
    """
    seconds = {
        "15m": 60,
        "1d": 300,
        "1wk": 900,
        "1mo": 1800,
    }.get(str(interval), 300)
    return int(time.time() // seconds)


@st.cache_data(ttl=3600, max_entries=256, show_spinner=False)
def _cached_yahoo_fetch(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    minimum_observations: int,
    cache_bucket: int,
) -> pd.DataFrame:
    # cache_bucket is intentionally part of the cache key.
    # Market data are still fetched only from the active Yahoo adapter.
    _ = cache_bucket
    return YahooFinanceAdapter.fetch(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
        minimum_observations=minimum_observations,
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
        title=dict(
            text=title, x=0.01, xanchor="left", y=0.955, yanchor="top",
            font=dict(size=15, color="#0F172A"), pad=dict(t=4, b=4)
        ),
        template="plotly_white",
        height=height,
        margin=dict(l=52, r=24, t=116, b=38),
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


def make_equity_chart(df, entry_label=""):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.82, 0.18], vertical_spacing=0.045,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Portfolio"], mode="lines", name="Trend Strategy",
            line=dict(width=1.8, color="#0F172A"),
            hovertemplate="%{x}<br>Strategy Portfolio: %{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["BuyHold"], mode="lines", name="Buy & Hold",
            line=dict(width=1.4, color="#64748B", dash="dot"),
            hovertemplate="%{x}<br>Buy & Hold: %{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    exposure = pd.to_numeric(df["Shares"], errors="coerce").fillna(0.0).gt(0).astype(float)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=exposure, mode="lines", name="Market Exposure",
            line=dict(width=1.2, color="#64748B"),
            fill="tozeroy", fillcolor="rgba(100,116,139,0.10)",
            hovertemplate="%{x}<br>Exposure: %{y:.0%}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Explicitly shade full cash regimes so a flat strategy curve cannot
    # be mistaken for missing underlying market data.
    for reg in portfolio_cash_regimes(df):
        fig.add_vrect(
            x0=reg["Start"], x1=reg["End"],
            fillcolor="rgba(148,163,184,0.10)",
            line_width=0, layer="below",
            row="all", col=1,
        )

    title = "Strategy Equity Curve vs Buy & Hold"
    if entry_label:
        title += f" — {entry_label}"

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", y=0.955, yanchor="top", font=dict(size=15, color="#0F172A"), pad=dict(t=4, b=4)),
        template="plotly_white",
        height=660,
        margin=dict(l=52, r=24, t=124, b=38),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif", size=11, color="#334155"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right", yanchor="bottom"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    fig.update_yaxes(title_text="Portfolio Value", gridcolor="#E2E8F0", row=1, col=1)
    fig.update_yaxes(title_text="Exposure", tickformat=".0%", range=[0,1.02], gridcolor="#E2E8F0", row=2, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(
        rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False),
        row=1, col=1,
    )
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
            x=0.01, xanchor="left", y=0.955, yanchor="top",
            font=dict(size=16, family="Arial, sans-serif", color="#111827"),
            pad=dict(t=4, b=4),
        ),
        height=650,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=45, r=45, t=118, b=30),
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


def make_nw_overlay_chart(df, nw_cfg):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
        name="Adjusted OHLC",
        increasing_line_color="#334155", decreasing_line_color="#94A3B8",
        increasing_fillcolor="#FFFFFF", decreasing_fillcolor="#E2E8F0",
        whiskerwidth=0.25,
    ))

    bull = df["NWTrend"].where(df["NWDirection"] > 0)
    bear = df["NWTrend"].where(df["NWDirection"] < 0)
    flat = df["NWTrend"].where(df["NWDirection"] == 0)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["NWLower"], mode="lines", name="NW Lower Residual Band",
        line=dict(width=0.9, color="#94A3B8"), hovertemplate="%{x}<br>Lower: %{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["NWUpper"], mode="lines", name="NW Upper Residual Band",
        line=dict(width=0.9, color="#94A3B8"), fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
        hovertemplate="%{x}<br>Upper: %{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bull, mode="lines", name="NW Bullish Path",
        line=dict(width=2.1, color="#0F766E"), connectgaps=False,
        hovertemplate="%{x}<br>NW Trend: %{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bear, mode="lines", name="NW Bearish Path",
        line=dict(width=2.1, color="#B91C1C"), connectgaps=False,
        hovertemplate="%{x}<br>NW Trend: %{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=flat, mode="lines", name="NW Flat Path",
        line=dict(width=1.4, color="#64748B"), connectgaps=False,
    ))

    bull_rev = df["NWBullishReversal"].fillna(False).astype(bool)
    bear_rev = df["NWBearishReversal"].fillna(False).astype(bool)
    fig.add_trace(go.Scatter(
        x=df.index[bull_rev], y=df.loc[bull_rev, "NWTrend"], mode="markers", name="Bullish Kernel Reversal",
        marker=dict(symbol="triangle-up", size=10, color="#0F766E", line=dict(width=0.5, color="#FFFFFF")),
    ))
    fig.add_trace(go.Scatter(
        x=df.index[bear_rev], y=df.loc[bear_rev, "NWTrend"], mode="markers", name="Bearish Kernel Reversal",
        marker=dict(symbol="triangle-down", size=10, color="#B91C1C", line=dict(width=0.5, color="#FFFFFF")),
    ))

    buys = df["FirstBuy"].fillna(0).gt(0)
    sells = df["FirstSell"].fillna(0).gt(0)
    fig.add_trace(go.Scatter(
        x=df.index[buys], y=df.loc[buys, "AdjOpen"], mode="markers", name="NW Strategy BUY",
        marker=dict(symbol="triangle-up", size=13, color="#111827"),
        hovertemplate="%{x}<br>Executed BUY: %{y:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index[sells], y=df.loc[sells, "AdjOpen"], mode="markers", name="NW Strategy SELL",
        marker=dict(symbol="triangle-down", size=13, color="#7C2D12"),
        hovertemplate="%{x}<br>Executed SELL: %{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Nadaraya-Watson Trend — {nw_cfg.kernel} Kernel | Lookback {nw_cfg.lookback} | h={nw_cfg.effective_bandwidth:g}",
            x=0.01, xanchor="left", y=0.955, yanchor="top",
            font=dict(size=15, color="#0F172A"), pad=dict(t=4, b=4)
        ),
        template="plotly_white", height=690, hovermode="x unified",
        margin=dict(l=45, r=25, t=126, b=35),
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right", yanchor="bottom"),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif", size=11, color="#334155"),
    )
    fig.update_xaxes(rangeselector=RANGE_SELECTOR, showgrid=False)
    fig.update_yaxes(title_text="Adjusted Price / NW Estimate", gridcolor="#E2E8F0")
    return fig


def make_nw_equity_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Portfolio"], mode="lines", name="NW Trend Strategy",
        line=dict(width=1.8, color="#0F172A")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BuyHold"], mode="lines", name="Buy & Hold",
        line=dict(width=1.3, color="#64748B", dash="dot")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["NWExposure"], mode="lines", name="NW Market Exposure",
        line=dict(width=1.2, color="#475569"), fill="tozeroy", fillcolor="rgba(71,85,105,0.10)"
    ), row=2, col=1)
    fig.update_layout(
        title=dict(text="MK Causal Nadaraya-Watson Strategy vs Buy & Hold", x=0.01, xanchor="left", y=0.955, yanchor="top", font=dict(size=15), pad=dict(t=4, b=4)),
        template="plotly_white", height=620, hovermode="x unified",
        margin=dict(l=45, r=25, t=118, b=30),
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif", size=11),
    )
    fig.update_yaxes(title_text="Portfolio Value", gridcolor="#E2E8F0", row=1, col=1)
    fig.update_yaxes(title_text="Exposure", tickformat=".0%", range=[0,1.02], row=2, col=1)
    fig.update_xaxes(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False), row=1, col=1)
    return fig


def make_nw_kernel_chart(nw_cfg):
    profile = kernel_weight_profile(nw_cfg)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=profile["Lag"], y=profile["NormalizedWeight"], name="Normalized Kernel Weight",
        marker_color="#475569",
        hovertemplate="Lag %{x}<br>Weight %{y:.5f}<extra></extra>",
    ))
    _base_layout(fig, f"Kernel Weight Profile — {nw_cfg.kernel}", 430, "Normalized Weight")
    fig.update_xaxes(title_text="Lag (bars)")
    return fig


def make_nw_state_chart(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    slope_pct = df["NWSlope"] / df["NWTrend"].shift(1)
    fig.add_trace(go.Scatter(
        x=df.index, y=slope_pct, mode="lines", name="NW Normalized Slope",
        line=dict(width=1.4, color="#0F172A")
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["NWBandWidthPct"], mode="lines", name="Residual Band Width / NW",
        line=dict(width=1.2, color="#B45309")
    ), secondary_y=True)
    _base_layout(fig, "NW State Diagnostics — Slope and Residual Dispersion", 470)
    fig.update_yaxes(title_text="NW Slope / Prior NW", tickformat=".2%", secondary_y=False)
    fig.update_yaxes(title_text="Residual Envelope Width", tickformat=".1%", secondary_y=True)
    fig.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR, rangeslider=dict(visible=False)))
    return fig



def make_tactical_envelope_chart(df, benchmark_ticker):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72,0.28], vertical_spacing=0.07,
        specs=[[{}],[{"secondary_y":True}]],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
        name="Adjusted OHLC", increasing_line_color="#334155", decreasing_line_color="#94A3B8",
        increasing_fillcolor="#FFFFFF", decreasing_fillcolor="#E2E8F0",
    ), row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["NWUpper"],mode="lines",name="NW Upper Band",
        line=dict(width=1.0,color="#B45309")
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["NWTrend"],mode="lines",name="NW Trend",
        line=dict(width=1.8,color="#0F172A")
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["NWLower"],mode="lines",name="NW Lower Band",
        line=dict(width=1.0,color="#64748B")
    ),row=1,col=1)

    ua=df["NWCrossAboveUpper"].fillna(False)
    ur=df["NWReenterBelowUpper"].fillna(False)
    lb=df["NWCrossBelowLower"].fillna(False)
    fig.add_trace(go.Scatter(
        x=df.index[ua],y=df.loc[ua,"AdjCloseCalc"],mode="markers",name="Upper Band Cross — Early De-risk",
        marker=dict(symbol="triangle-down",size=11,color="#B45309")
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index[ur],y=df.loc[ur,"AdjCloseCalc"],mode="markers",name="Upper Band Re-entry — Confirmed Exhaustion",
        marker=dict(symbol="triangle-down",size=13,color="#7C2D12")
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index[lb],y=df.loc[lb,"AdjCloseCalc"],mode="markers",name="Lower Band Break",
        marker=dict(symbol="x",size=10,color="#991B1B")
    ),row=1,col=1)

    fig.add_trace(go.Scatter(
        x=df.index,y=df["ResidualDriftZ"],mode="lines",name=f"Beta-Adjusted Relative Drift Z vs {benchmark_ticker}",
        line=dict(width=1.5,color="#1D4ED8")
    ),row=2,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["RelativeVolume"],mode="lines",name="Relative Volume",
        line=dict(width=1.0,color="#64748B",dash="dot")
    ),row=2,col=1,secondary_y=True)

    if tactical_cfg is not None:
        for y in [tactical_cfg.weak_z,tactical_cfg.strong_z,tactical_cfg.extreme_z,
                  -tactical_cfg.weak_z,-tactical_cfg.strong_z,-tactical_cfg.extreme_z]:
            fig.add_hline(y=y,line_width=0.7,line_dash="dot",line_color="#CBD5E1",row=2,col=1)

    fig.update_layout(
        title=dict(text="Institutional Tactical Envelope + Benchmark Relative Deviation",
                   x=0.01,xanchor="left",y=0.955,yanchor="top",font=dict(size=15),pad=dict(t=4,b=4)),
        template="plotly_white",height=760,hovermode="x unified",
        margin=dict(l=50,r=35,t=125,b=35),
        legend=dict(orientation="h",y=1.035,x=1,xanchor="right"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11,color="#334155"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Adjusted Price / NW Envelope",row=1,col=1,gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="Relative Drift Z",row=2,col=1,secondary_y=False,gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="RVOL",row=2,col=1,secondary_y=True)
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False),row=1,col=1)
    return fig


def make_tactical_portfolio_chart(df):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.76,0.24],vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["TacticalPortfolio"],mode="lines",name="Institutional Tactical Portfolio",
        line=dict(width=1.8,color="#0F172A")
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["BuyHold"],mode="lines",name="Buy & Hold",
        line=dict(width=1.2,color="#64748B",dash="dot")
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df.index,y=df["TacticalTargetExposure"],mode="lines",name="Target Exposure",
        line=dict(width=1.3,color="#334155"),fill="tozeroy",fillcolor="rgba(51,65,85,.10)"
    ),row=2,col=1)
    fig.update_layout(
        title=dict(text="Institutional Tactical Portfolio & Staged Exposure",
                   x=.01,xanchor="left",y=.955,yanchor="top",font=dict(size=15),pad=dict(t=4,b=4)),
        template="plotly_white",height=620,hovermode="x unified",
        margin=dict(l=50,r=30,t=118,b=35),
        legend=dict(orientation="h",y=1.03,x=1,xanchor="right"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11,color="#334155"),
    )
    fig.update_yaxes(title_text="Portfolio Value",row=1,col=1,gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="Target Exposure",tickformat=".0%",range=[0,1.02],row=2,col=1)
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False),row=1,col=1)
    return fig


def make_primary_price_chart(df, ticker_label):
    """Client-facing chart: price + NW/tactical information only."""
    source_df = tactical_result if tactical_result is not None else (nw_result if nw_result is not None else df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=source_df.index, open=source_df["AdjOpen"], high=source_df["AdjHigh"],
        low=source_df["AdjLow"], close=source_df["AdjCloseCalc"], name="Adjusted OHLC",
        increasing_line_color="#334155", decreasing_line_color="#94A3B8",
        increasing_fillcolor="#FFFFFF", decreasing_fillcolor="#E2E8F0",
    ))
    if "NWTrend" in source_df.columns:
        fig.add_trace(go.Scatter(x=source_df.index,y=source_df["NWTrend"],mode="lines",name="Nadaraya-Watson Trend",line=dict(width=1.8,color="#0F172A")))
        fig.add_trace(go.Scatter(x=source_df.index,y=source_df["NWUpper"],mode="lines",name="NW Upper Band",line=dict(width=1.0,color="#B45309")))
        fig.add_trace(go.Scatter(x=source_df.index,y=source_df["NWLower"],mode="lines",name="NW Lower Band",line=dict(width=1.0,color="#64748B")))
    if "NWCrossAboveUpper" in source_df.columns:
        m=source_df["NWCrossAboveUpper"].fillna(False)
        fig.add_trace(go.Scatter(x=source_df.index[m],y=source_df.loc[m,"AdjCloseCalc"],mode="markers",name="Upper Band Cross — Early De-risk",marker=dict(symbol="triangle-down",size=11,color="#B45309")))
    if "NWReenterBelowUpper" in source_df.columns:
        m=source_df["NWReenterBelowUpper"].fillna(False)
        fig.add_trace(go.Scatter(x=source_df.index[m],y=source_df.loc[m,"AdjCloseCalc"],mode="markers",name="Upper Band Re-entry — Confirmed Exhaustion",marker=dict(symbol="triangle-down",size=13,color="#7C2D12")))
    if "NWCrossBelowLower" in source_df.columns:
        m=source_df["NWCrossBelowLower"].fillna(False)
        fig.add_trace(go.Scatter(x=source_df.index[m],y=source_df.loc[m,"AdjCloseCalc"],mode="markers",name="Lower Band Break",marker=dict(symbol="x",size=10,color="#991B1B")))
    fig.update_layout(
        title=dict(text=f"{ticker_label} — Primary Price Structure & Tactical Signals",x=.01,xanchor="left",y=.955,yanchor="top",font=dict(size=15,color="#0F172A"),pad=dict(t=4,b=4)),
        template="plotly_white",height=680,hovermode="x unified",margin=dict(l=48,r=25,t=122,b=35),
        legend=dict(orientation="h",y=1.035,x=1,xanchor="right"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11,color="#334155"),
        xaxis_rangeslider_visible=False,paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,showgrid=False)
    fig.update_yaxes(title_text="Adjusted Price",gridcolor="#E2E8F0")
    return fig



def make_institutional_trend_diagnostics(df, benchmark_ticker, tactical_cfg):
    fig=make_subplots(
        rows=3,cols=1,shared_xaxes=True,
        row_heights=[0.56,0.24,0.20],vertical_spacing=0.055,
        specs=[[{}],[{"secondary_y":True}],[{"secondary_y":True}]],
    )
    fig.add_trace(go.Scatter(x=df.index,y=df["AdjCloseCalc"],mode="lines",name="Adjusted Close",line=dict(width=1.4,color="#0F172A")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["NWTrend"],mode="lines",name="NW Trend",line=dict(width=1.8,color="#334155")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["NWUpper"],mode="lines",name="NW Upper Band",line=dict(width=1.0,color="#B45309")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["NWLower"],mode="lines",name="NW Lower Band",line=dict(width=1.0,color="#64748B")),row=1,col=1)
    ua=df["NWCrossAboveUpper"].fillna(False); ur=df["NWReenterBelowUpper"].fillna(False); lb=df["NWCrossBelowLower"].fillna(False)
    fig.add_trace(go.Scatter(x=df.index[ua],y=df.loc[ua,"AdjCloseCalc"],mode="markers",name="Upper Band Cross",marker=dict(symbol="triangle-down",size=10,color="#B45309")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index[ur],y=df.loc[ur,"AdjCloseCalc"],mode="markers",name="Upper Band Re-entry",marker=dict(symbol="triangle-down",size=12,color="#7C2D12")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index[lb],y=df.loc[lb,"AdjCloseCalc"],mode="markers",name="Lower Band Break",marker=dict(symbol="x",size=10,color="#991B1B")),row=1,col=1)

    fig.add_trace(go.Scatter(x=df.index,y=df["NWNormalizedSlope"],mode="lines",name="NW Normalized Slope",line=dict(width=1.3,color="#0F172A")),row=2,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["ResidualDriftZ"],mode="lines",name=f"Relative Drift Z vs {benchmark_ticker}",line=dict(width=1.4,color="#1D4ED8")),row=2,col=1,secondary_y=True)
    for y in [tactical_cfg.weak_z,tactical_cfg.strong_z,tactical_cfg.extreme_z,-tactical_cfg.weak_z,-tactical_cfg.strong_z,-tactical_cfg.extreme_z]:
        fig.add_hline(y=y,row=2,col=1,line_width=.7,line_dash="dot",line_color="#CBD5E1")

    fig.add_trace(go.Scatter(x=df.index,y=df["TacticalTargetExposure"],mode="lines",name="Target Exposure",line=dict(width=1.3,color="#334155"),fill="tozeroy",fillcolor="rgba(51,65,85,.08)"),row=3,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["RelativeVolume"],mode="lines",name="Relative Volume",line=dict(width=1.0,color="#B45309")),row=3,col=1,secondary_y=True)

    fig.update_layout(
        title=dict(text="Institutional Trend Diagnostics — NW Structure, Relative Regime & Exposure",x=.01,xanchor="left",y=.965,yanchor="top",font=dict(size=15),pad=dict(t=4,b=4)),
        template="plotly_white",height=840,hovermode="x unified",margin=dict(l=52,r=45,t=130,b=35),
        legend=dict(orientation="h",y=1.035,x=1,xanchor="right"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11,color="#334155"),
    )
    fig.update_yaxes(title_text="Price / NW",row=1,col=1,gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="NW Slope",row=2,col=1,secondary_y=False,gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="Relative Z",row=2,col=1,secondary_y=True)
    fig.update_yaxes(title_text="Exposure",tickformat=".0%",range=[0,1.02],row=3,col=1,secondary_y=False)
    fig.update_yaxes(title_text="RVOL",row=3,col=1,secondary_y=True)
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False),row=1,col=1)
    return fig


def make_var_comparison_chart(var_table, confidence):
    d=var_table[(var_table["Status"]=="OK") & np.isclose(var_table["Confidence"],confidence)].copy()
    fig=go.Figure()
    for method in ["Historical","Parametric Normal","Monte Carlo Bootstrap"]:
        x=d[d["Method"]==method]
        fig.add_trace(go.Bar(x=x["Series"],y=x["VaR"],name=method,hovertemplate="%{x}<br>VaR: %{y:.2%}<extra>"+method+"</extra>"))
    fig.update_layout(
        title=dict(text=f"{confidence:.0%} VaR — Method Comparison",x=.01,xanchor="left",y=.955,yanchor="top",font=dict(size=15),pad=dict(t=4,b=4)),
        barmode="group",template="plotly_white",height=460,margin=dict(l=55,r=25,t=105,b=55),
        legend=dict(orientation="h",y=1.04,x=1,xanchor="right"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11,color="#334155"),
    )
    fig.update_yaxes(title_text="VaR (loss %)",tickformat=".1%",gridcolor="#E2E8F0")
    return fig


def _risk_horizon_text(interval,bars):
    bars=int(bars)
    if interval=="15m": return f"{bars} bar(s) = {bars*15} minutes"
    if interval=="1d": return f"{bars} trading day(s)"
    if interval=="1wk": return f"{bars} week(s)"
    if interval=="1mo": return f"{bars} month(s)"
    return f"{bars} bar(s)"

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

    interval_label = st.selectbox("Frequency", ["15 Minutes", "Daily", "Weekly", "Monthly"], index=1)
    interval = {"15 Minutes":"15m", "Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}[interval_label]
    if interval == "15m":
        st.caption("Yahoo 15-minute mode: strict maximum historical span is 60 days. The engine will not silently truncate an older request.")

    initial_capital = st.number_input(
        "Initial Capital", min_value=1.0, value=100000.0, step=10000.0,
        help="Starting capital for the primary Institutional Tactical and research strategy layers."
    )

    # Internal workbook-replication defaults remain code-only for regression tests.
    # They are never shown as client controls or client decisions.
    strategy_name = "ATR_TRAILING_STOP"
    atr_weeks = 8
    atr_multiplier = 10.0
    bollinger_weeks = 40
    bollinger_sd = 2.5
    legacy_fidelity = True
    entry_mode_label = "Frequency-Aware"
    _entry_options = horizon_options(interval)
    _default_horizon = "12M" if "12M" in _entry_options else _entry_options[min(2, len(_entry_options)-1)]
    max_buy_weeks, entry_gate_label = resolve_entry_lookback(
        interval, "Frequency-Aware", horizon=_default_horizon
    )

    st.divider()
    st.subheader("Nadaraya-Watson Trend Module")
    nw_enabled = st.toggle(
        "Enable Nadaraya-Watson Research Layer",
        value=True,
        help="Independent causal Python implementation of the public QuantAlgo Nadaraya-Watson Trend methodology.",
    )

    nw_preset = st.selectbox(
        "NW Research Preset",
        ["MK Institutional Balanced", "Public-Methodology Gaussian", "MK Fast Research", "MK Smooth Position", "Custom"],
        index=0,
        disabled=not nw_enabled,
        help="Numeric presets are MK Engine research presets; they are not claimed to be QuantAlgo's proprietary/default parameter values.",
    )

    preset_map = {
        "MK Institutional Balanced": dict(lookback=100, bandwidth=12.0, kernel="Rational Quadratic", relative_weight=1.0, band_multiplier=2.0, confirmation=2, exit_confirmation=2),
        "Public-Methodology Gaussian": dict(lookback=100, bandwidth=8.0, kernel="Gaussian", relative_weight=1.0, band_multiplier=2.0, confirmation=1, exit_confirmation=1),
        "MK Fast Research": dict(lookback=50, bandwidth=8.0, kernel="Gaussian", relative_weight=1.0, band_multiplier=1.8, confirmation=1, exit_confirmation=1),
        "MK Smooth Position": dict(lookback=150, bandwidth=20.0, kernel="Rational Quadratic", relative_weight=1.5, band_multiplier=2.2, confirmation=3, exit_confirmation=2),
    }
    _nw_defaults = preset_map.get(nw_preset, preset_map["MK Institutional Balanced"])

    nw_kernel = st.selectbox(
        "Kernel", list(NW_KERNELS),
        index=list(NW_KERNELS).index(_nw_defaults["kernel"]),
        disabled=(not nw_enabled) or nw_preset != "Custom",
    ) if nw_preset == "Custom" else _nw_defaults["kernel"]
    if nw_preset != "Custom" and nw_enabled:
        st.caption(f"Kernel: **{nw_kernel}**")

    if nw_preset == "Custom":
        nw_lookback = st.number_input("NW Lookback", min_value=2, value=100, step=1, disabled=not nw_enabled)
        nw_bandwidth = st.number_input("NW Bandwidth", min_value=0.1, value=8.0, step=0.5, disabled=not nw_enabled)
        nw_h_mult = st.number_input("Bandwidth Multiplier", min_value=0.1, value=1.0, step=0.1, disabled=not nw_enabled)
        nw_relative_weight = st.number_input("RQ Relative Weight", min_value=0.05, value=1.0, step=0.25, disabled=not nw_enabled)
        nw_band_mult = st.number_input("Residual Band Multiplier", min_value=0.1, value=2.0, step=0.1, disabled=not nw_enabled)
        nw_confirmation = st.number_input("Bullish Confirmation Bars", min_value=1, value=2, step=1, disabled=not nw_enabled)
        nw_exit_confirmation = st.number_input("Bearish Exit Confirmation Bars", min_value=1, value=1, step=1, disabled=not nw_enabled)
    else:
        nw_lookback = int(_nw_defaults["lookback"])
        nw_bandwidth = float(_nw_defaults["bandwidth"])
        nw_h_mult = 1.0
        nw_relative_weight = float(_nw_defaults["relative_weight"])
        nw_band_mult = float(_nw_defaults["band_multiplier"])
        nw_confirmation = int(_nw_defaults["confirmation"])
        nw_exit_confirmation = int(_nw_defaults["exit_confirmation"])
        if nw_enabled:
            st.caption(
                f"Lookback {nw_lookback} | Bandwidth {nw_bandwidth:g} | Residual Band ×{nw_band_mult:g} | "
                f"Confirm {nw_confirmation}/{nw_exit_confirmation}"
            )

    nw_source = st.selectbox("NW Price Source", ["Adjusted Close", "HLC3", "OHLC4"], index=0, disabled=not nw_enabled)
    nw_strategy_mode_label = st.selectbox(
        "NW Strategy Logic",
        ["MK Confirmed Trend", "Public-Methodology Reversal Translation"],
        index=0, disabled=not nw_enabled,
    )
    nw_strategy_mode = {
        "MK Confirmed Trend": "MK_CONFIRMED_TREND",
        "Public-Methodology Reversal Translation": "QUANTALGO_REVERSAL_TRANSLATION",
    }[nw_strategy_mode_label]
    nw_avoid_chase = st.toggle(
        "Avoid Entry Above Upper Residual Band", value=True, disabled=not nw_enabled,
        help="MK strategy risk filter: do not initiate a new long when price is already beyond the upper residual envelope.",
    )


    st.divider()
    st.subheader("Institutional Tactical Layer")
    tactical_enabled = st.toggle(
        "Enable Primary Tactical Decision Engine",
        value=True,
        help="Nadaraya-Watson envelope excursions + benchmark-relative deviation + staged exposure reduction.",
    )

    auto_benchmark = default_benchmark(ticker)
    benchmark_mode = st.selectbox(
        "Benchmark Selection",
        ["Auto Mapped", "Manual Yahoo Benchmark"],
        index=0,
        disabled=not tactical_enabled,
    )
    if benchmark_mode == "Auto Mapped":
        benchmark_ticker = auto_benchmark or ""
        if benchmark_ticker:
            st.caption(f"Primary benchmark: **{benchmark_name(benchmark_ticker)}** (`{benchmark_ticker}`)")
        else:
            st.warning("No curated benchmark mapping exists for this ticker. Select Manual Yahoo Benchmark.")
    else:
        benchmark_ticker = st.text_input("Benchmark Yahoo Ticker", value=(auto_benchmark or "XU100.IS")).strip().upper()

    sensitivity = st.selectbox(
        "Tactical Sensitivity",
        ["High Sensitivity", "Institutional Balanced", "Conservative"],
        index=0,
        disabled=not tactical_enabled,
    )
    sensitivity_map = {
        "High Sensitivity": dict(weak=1.25,strong=1.75,extreme=2.5,vol=1.35),
        "Institutional Balanced": dict(weak=1.5,strong=2.0,extreme=3.0,vol=1.5),
        "Conservative": dict(weak=2.0,strong=2.5,extreme=3.5,vol=1.75),
    }
    _sens = sensitivity_map[sensitivity]

    if interval == "15m":
        default_beta, default_drift = 40, 8
    elif interval == "1d":
        default_beta, default_drift = 60, 10
    elif interval == "1wk":
        default_beta, default_drift = 26, 4
    else:
        default_beta, default_drift = 12, 3

    rc1, rc2 = st.columns(2)
    with rc1:
        relative_beta_window = st.number_input(
            "Relative Beta Window", min_value=10, value=int(default_beta), step=1,
            disabled=not tactical_enabled,
        )
    with rc2:
        relative_drift_horizon = st.number_input(
            "Relative Drift Horizon", min_value=1, value=int(default_drift), step=1,
            disabled=not tactical_enabled,
        )
    st.caption(
        "Primary tactical hierarchy: upper-band excursion → early trim; upper-band re-entry / bearish reversal → deeper reduction; "
        "benchmark-relative breakdown → forced de-risking; lower-band recovery + bullish confirmation → staged re-entry."
    )

    run_clicked = st.button("RUN ANALYSIS", type="primary", width="stretch")


# ---------------------------- State ----------------------------
STATE_SCHEMA_VERSION = 4
_previous_schema = st.session_state.get("_state_schema_version")
if _previous_schema != STATE_SCHEMA_VERSION:
    # Clear only computed analysis objects from an older deployed code schema.
    # This prevents stale dataclass/session objects from surviving a hot redeploy.
    for _k in [
        "result", "summary", "config", "raw", "decision", "trades", "trade_stats",
        "ticker", "instrument_name", "market", "group", "interval", "interval_label",
        "entry_mode", "entry_gate_label", "entry_lookback",
        "tactical_enabled","benchmark_ticker","benchmark_market","relative_config","relative_result",
        "relative_snapshot","tactical_config","tactical_result","tactical_snapshot","tactical_sensitivity",
        "nw_enabled", "nw_result", "nw_indicator", "nw_config", "nw_strategy_config",
        "nw_summary", "nw_decision", "nw_trades", "nw_trade_stats", "nw_preset",
        "asset_yahoo_audit","benchmark_yahoo_audit",
    ]:
        st.session_state.pop(_k, None)
    st.session_state["_state_schema_version"] = STATE_SCHEMA_VERSION

for key, default in {
    "result": None, "summary": None, "config": None, "raw": None,
    "decision": None, "trades": None, "trade_stats": None,
    "nw_result": None, "nw_indicator": None, "nw_config": None, "nw_strategy_config": None,
    "nw_summary": None, "nw_decision": None, "nw_trades": None, "nw_trade_stats": None,
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
    if interval == "15m" and (pd.Timestamp(end) - pd.Timestamp(start)).days > 59:
        st.error(
            "STRICT INTRADAY DATA STOP — Yahoo Finance 15-minute history cannot extend beyond the last 60 days. "
            "Reduce the requested date span; the engine will not silently truncate or substitute data."
        )
        st.stop()
    if tactical_enabled and not benchmark_ticker:
        st.error("BENCHMARK GOVERNANCE STOP — Primary Tactical Layer requires an explicit Yahoo benchmark ticker.")
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
        with st.spinner(f"Requesting Yahoo Finance for {ticker} and running the institutional analytics pipeline..."):
            raw = _cached_yahoo_fetch(
                ticker=ticker, start=str(start), end=str(end), interval=interval,
                minimum_observations=cfg.minimum_observations,
                cache_bucket=_yahoo_cache_bucket(interval),
            )
            asset_yahoo_audit = dict(raw.attrs.get("yahoo_audit", {}))
            result = run_legacy_engine(raw, cfg)
            summary = performance_summary(result, initial_capital=cfg.initial_capital)
            decision = decision_snapshot(result, cfg)
            trades = trade_ledger(result)
            tstats = trade_statistics(trades)

            if nw_enabled:
                nw_cfg = NWConfig(
                    lookback=int(nw_lookback),
                    bandwidth=float(nw_bandwidth),
                    bandwidth_multiplier=float(nw_h_mult),
                    kernel=nw_kernel,
                    relative_weight=float(nw_relative_weight),
                    band_multiplier=float(nw_band_mult),
                    source=nw_source,
                    minimum_observations=cfg.minimum_observations,
                )
                nw_scfg = NWStrategyConfig(
                    mode=nw_strategy_mode,
                    confirmation_bars=int(nw_confirmation),
                    exit_confirmation_bars=int(nw_exit_confirmation),
                    avoid_upper_band_chase=bool(nw_avoid_chase),
                    initial_capital=float(initial_capital),
                )
                nw_indicator = compute_nadaraya_watson(result, nw_cfg)
                nw_result = run_nw_strategy(result, nw_indicator, nw_scfg)
                nw_summary = performance_summary(nw_result, initial_capital=nw_scfg.initial_capital)
                nw_decision = nw_decision_snapshot(nw_result, nw_scfg)
                nw_trades = trade_ledger(nw_result)
                nw_tstats = trade_statistics(nw_trades)
            else:
                nw_cfg = nw_scfg = nw_indicator = nw_result = None
                nw_summary = nw_decision = nw_trades = nw_tstats = None

            if tactical_enabled:
                if not nw_enabled or nw_result is None:
                    raise ValueError("Institutional Tactical Layer requires the Nadaraya-Watson layer to be enabled.")
                benchmark_market = _cached_yahoo_fetch(
                    ticker=benchmark_ticker, start=str(start), end=str(end), interval=interval,
                    minimum_observations=max(30, int(relative_beta_window) + int(relative_drift_horizon) + 3),
                    cache_bucket=_yahoo_cache_bucket(interval),
                )
                benchmark_yahoo_audit = dict(benchmark_market.attrs.get("yahoo_audit", {}))
                rel_cfg = RelativeConfig(
                    beta_window=int(relative_beta_window),
                    drift_horizon=int(relative_drift_horizon),
                    weak_z=float(_sens["weak"]),
                    strong_z=float(_sens["strong"]),
                    extreme_z=float(_sens["extreme"]),
                )
                relative_result = compute_relative_analytics(nw_result, benchmark_market, rel_cfg)
                rel_snapshot = relative_snapshot(relative_result, rel_cfg)
                tactical_cfg = TacticalConfig(
                    weak_z=float(_sens["weak"]),
                    strong_z=float(_sens["strong"]),
                    extreme_z=float(_sens["extreme"]),
                    volume_climax=float(_sens["vol"]),
                    initial_capital=float(initial_capital),
                )
                tactical_result = run_tactical_strategy(nw_result, relative_result, tactical_cfg)
                tactical_decision = tactical_snapshot(tactical_result, tactical_cfg)
            else:
                benchmark_market = rel_cfg = relative_result = rel_snapshot = None
                benchmark_yahoo_audit = {}
                tactical_cfg = tactical_result = tactical_decision = None

        st.session_state.result = result
        st.session_state.summary = summary
        st.session_state.config = cfg
        st.session_state.raw = raw
        st.session_state.asset_yahoo_audit = asset_yahoo_audit
        st.session_state.benchmark_yahoo_audit = benchmark_yahoo_audit
        st.session_state.decision = decision
        st.session_state.trades = trades
        st.session_state.trade_stats = tstats
        st.session_state.ticker = ticker
        st.session_state.instrument_name = selected_name
        st.session_state.market = market
        st.session_state.group = selected_group
        st.session_state.interval = interval
        st.session_state.interval_label = interval_label
        st.session_state.entry_mode = entry_mode_label
        st.session_state.entry_gate_label = entry_gate_label
        st.session_state.entry_lookback = int(max_buy_weeks)
        st.session_state.nw_enabled = bool(nw_enabled)
        st.session_state.nw_result = nw_result
        st.session_state.nw_indicator = nw_indicator
        st.session_state.nw_config = nw_cfg
        st.session_state.nw_strategy_config = nw_scfg
        st.session_state.nw_summary = nw_summary
        st.session_state.nw_decision = nw_decision
        st.session_state.nw_trades = nw_trades
        st.session_state.nw_trade_stats = nw_tstats
        st.session_state.nw_preset = nw_preset
        st.session_state.tactical_enabled = bool(tactical_enabled)
        st.session_state.benchmark_ticker = benchmark_ticker
        st.session_state.benchmark_market = benchmark_market
        st.session_state.relative_config = rel_cfg
        st.session_state.relative_result = relative_result
        st.session_state.relative_snapshot = rel_snapshot
        st.session_state.tactical_config = tactical_cfg
        st.session_state.tactical_result = tactical_result
        st.session_state.tactical_snapshot = tactical_decision
        st.session_state.tactical_sensitivity = sensitivity
    except (DataIntegrityError, MarketDataError) as exc:
        st.error(f"STRICT DATA STOP — {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"RUN STOPPED — {type(exc).__name__}: {exc}")
        st.stop()

result = st.session_state.get("result")

# Cold-start and redeploy guard:
# Never dereference config/decision/trade objects before a completed run exists.
if result is None:
    st.info("Choose an instrument from BIST, US Stocks, Precious Metals, or enter a manual Yahoo ticker; then press RUN ANALYSIS.")
    u = pd.DataFrame(flat_universe_rows())
    st.subheader("Curated Instrument Universe")
    st.dataframe(u, width="stretch", hide_index=True, height=460)
    st.caption("The universe is a convenience selector only. Live price history is requested from Yahoo Finance after RUN ANALYSIS; the list itself is not a fallback data source.")
    st.stop()

summary = st.session_state.get("summary")
cfg = st.session_state.get("config")
decision = st.session_state.get("decision")
trades = st.session_state.get("trades")
tstats = st.session_state.get("trade_stats")
nw_enabled_used = bool(st.session_state.get("nw_enabled", False))
nw_result = st.session_state.get("nw_result")
nw_indicator = st.session_state.get("nw_indicator")
nw_cfg = st.session_state.get("nw_config")
nw_scfg = st.session_state.get("nw_strategy_config")
nw_summary = st.session_state.get("nw_summary")
nw_decision = st.session_state.get("nw_decision")
nw_trades = st.session_state.get("nw_trades")
nw_tstats = st.session_state.get("nw_trade_stats")
nw_preset_used = st.session_state.get("nw_preset", "")
tactical_enabled_used = bool(st.session_state.get("tactical_enabled", False))
benchmark_ticker_used = st.session_state.get("benchmark_ticker")
benchmark_market = st.session_state.get("benchmark_market")
rel_cfg = st.session_state.get("relative_config")
relative_result = st.session_state.get("relative_result")
rel_snapshot = st.session_state.get("relative_snapshot")
tactical_cfg = st.session_state.get("tactical_config")
tactical_result = st.session_state.get("tactical_result")
tactical_decision = st.session_state.get("tactical_snapshot")
tactical_sensitivity_used = st.session_state.get("tactical_sensitivity", "")
asset_yahoo_audit = st.session_state.get("asset_yahoo_audit", {})
benchmark_yahoo_audit = st.session_state.get("benchmark_yahoo_audit", {})

entry_gate_label_used = st.session_state.get("entry_gate_label", "")
entry_lookback_state = st.session_state.get("entry_lookback")
if entry_lookback_state is not None:
    entry_lookback_used = int(entry_lookback_state)
else:
    # Safe compatibility fallback only after a completed result exists.
    entry_lookback_used = int(getattr(cfg, "max_buy_weeks", 2000))
entry_mode_used = st.session_state.get("entry_mode", "Legacy Exact")

# Persisted run identity
ticker_used = st.session_state.ticker
name_used = st.session_state.instrument_name
market_used = st.session_state.market
group_used = st.session_state.group
interval_used = st.session_state.interval
interval_label_used = st.session_state.interval_label

# ---------------------------- Executive strip ----------------------------
st.markdown(f"### {name_used if name_used != 'Manual' else ticker_used}  ·  `{ticker_used}`")
st.caption(f"{market_used} / {group_used}  |  {interval_label_used}  |  {summary['start'].date()} → {summary['end'].date()}  |  Primary engine: MK Institutional Tactical")

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
if tactical_enabled_used and tactical_decision is not None:
    k1.metric("PRIMARY Decision", tactical_decision["decision"])
    k2.metric("Target Exposure", fmt_pct(tactical_decision["target_exposure"]))
    k3.metric("Last Adj. Close", fmt_num(result["AdjCloseCalc"].iloc[-1]))
    k4.metric("Relative Drift Z", fmt_num(tactical_decision["relative_z"]))
    k5.metric("Rolling Beta", fmt_num(tactical_decision["beta"]))
    k6.metric("NW Envelope Z", fmt_num(tactical_decision["envelope_z"]))
    k7.metric("Benchmark", benchmark_ticker_used or "—")
    k8.metric("Decision Source", "Tactical v0.07.1")
else:
    k1.metric("PRIMARY Decision", "NO DECISION")
    k2.metric("Target Exposure", "—")
    k3.metric("Last Adj. Close", fmt_num(result["AdjCloseCalc"].iloc[-1]))
    k4.metric("Relative Drift Z", "—")
    k5.metric("Rolling Beta", "—")
    k6.metric("NW Envelope Z", "—")
    k7.metric("Benchmark", benchmark_ticker_used or "—")
    k8.metric("Decision Source", "TACTICAL DISABLED")
    st.warning("The Institutional Tactical Layer is disabled. No secondary decision engine will be substituted.")

# ---------------------------- Main tabs ----------------------------
tabs = st.tabs([
    "Executive & Primary Decision",
    "Institutional Tactical",
    "Price & Signals",
    "Nadaraya-Watson Trend",
    "Strategy vs Buy & Hold",
    "Risk Analytics",
    "Trend Diagnostics",
    "Trade Ledger",
    "Calculation Ledger",
    "Instrument Universe",
    "Methodology & Governance",
])

with tabs[0]:
    if tactical_enabled_used and tactical_decision is not None:
        st.markdown(
            f"""<div class="decision-card">
                <div class="decision-label">PRIMARY INSTITUTIONAL DECISION</div>
                <div class="decision-value">{tactical_decision['decision']}</div>
                <div class="decision-reason">{tactical_decision['rationale']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        p1,p2,p3,p4,p5 = st.columns(5)
        p1.metric("Target Exposure", fmt_pct(tactical_decision["target_exposure"]))
        p2.metric("Relative Drift Z", fmt_num(tactical_decision["relative_z"]))
        p3.metric("NW Envelope Z", fmt_num(tactical_decision["envelope_z"]))
        p4.metric("Rolling Beta", fmt_num(tactical_decision["beta"]))
        p5.metric("Relative Volume", fmt_num(tactical_decision["relative_volume"]))
        st.markdown("#### Primary Decision Causality")
        st.dataframe(tactical_decision["gates"],width="stretch",hide_index=True,height=300)
        st.caption(tactical_decision["timing_note"])

        with st.expander("Yahoo Data Quality & Completed-Session Audit", expanded=False):
            _audit_rows = []
            for _role, _a in [("Asset", asset_yahoo_audit), ("Benchmark", benchmark_yahoo_audit)]:
                if not _a:
                    continue
                _audit_rows.append({
                    "Role": _role,
                    "Ticker": _a.get("ticker", ""),
                    "Interval": _a.get("interval", ""),
                    "Effective Completed Cutoff": _a.get("effective_completed_cutoff", ""),
                    "Accepted Mode": _a.get("accepted_mode", ""),
                    "Yahoo Routes With Data": len(_a.get("routes_with_data", []) or []),
                    "Same-Source Reconciled": bool(_a.get("reconciled", False)),
                    "Recovered Sessions": ", ".join(_a.get("recovered_sessions", []) or []) or "—",
                    "Precision-Normalized Matches": int(_a.get("precision_normalized_matches", 0) or 0),
                    "Max Precision Spread (bps)": float(_a.get("max_precision_spread_bps", 0.0) or 0.0),
                    "Non-Session Placeholders": len(_a.get("non_session_placeholders", []) or []),
                    "Unfinished Sessions Withheld": len(_a.get("unfinished_session_rows", []) or []),
                    "Observations": _a.get("observations", ""),
                })
            if _audit_rows:
                st.dataframe(pd.DataFrame(_audit_rows), width="stretch", hide_index=True)

            for _role, _a in [("Asset", asset_yahoo_audit), ("Benchmark", benchmark_yahoo_audit)]:
                if not _a:
                    continue
                if _a.get("non_session_placeholders"):
                    st.caption(
                        f"{_role} — exchange-calendar non-session Yahoo placeholders quarantined: "
                        + ", ".join(_a.get("non_session_placeholders", []))
                    )
                if _a.get("unfinished_session_rows"):
                    st.caption(
                        f"{_role} — unfinished sessions withheld from completed-bar analysis: "
                        + ", ".join(_a.get("unfinished_session_rows", []))
                    )
                if _a.get("recovered_sessions"):
                    st.info(
                        f"{_role} — recent completed sessions recovered by targeted Yahoo-only retry: "
                        + ", ".join(_a.get("recovered_sessions", []))
                    )
                if _a.get("historical_missing_sessions"):
                    _hist = _a.get("historical_missing_sessions", [])
                    st.warning(
                        f"{_role} — {len(_hist)} older historical XIST session gap(s) detected in Yahoo. "
                        "They are disclosed for backtest/risk-quality review but do not block the current live decision. "
                        "No values were filled. Examples: " + ", ".join(_hist[:8])
                    )

            st.caption(
                "Governance: Yahoo Finance only. The stable primary fetch again uses yfinance.download with keepna=False, "
                "matching the original working acquisition behavior. Returned rows remain strictly validated. "
                "For BIST daily data, recent completed-session coverage is enforced; older historical gaps are disclosed "
                "as audit warnings rather than blocking the live decision. No fill, interpolation, Close→Adj Close "
                "substitution, averaging, or alternate-provider fallback is permitted."
            )
        st.markdown(
            f"<div class='section-note'><b>Benchmark:</b> {benchmark_name(benchmark_ticker_used)} "
            f"(`{benchmark_ticker_used}`) · <b>Sensitivity:</b> {tactical_sensitivity_used}. "
            "Decision source: Institutional Tactical.</div>",
            unsafe_allow_html=True,
        )


with tabs[1]:
    if not tactical_enabled_used or tactical_result is None:
        st.info("Institutional Tactical Layer was disabled for this run.")
    else:
        st.markdown(
            "<div class='section-note'><b>Fast tactical objective:</b> react to envelope exhaustion and relative deterioration before a deep loss develops. "
            "An upside NW-envelope excursion can trigger an immediate staged trim; re-entry below the upper band, "
            "bearish NW reversal, abnormal relative volume and beta-adjusted benchmark divergence can deepen the reduction. "
            "Severe relative breakdown or lower-band failure can force a full exit.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_tactical_envelope_chart(tactical_result, benchmark_ticker_used),
            width="stretch",config=PLOT_CFG
        )
        t1,t2,t3,t4,t5,t6 = st.columns(6)
        t1.metric("Decision",tactical_decision["decision"])
        t2.metric("Target Exposure",fmt_pct(tactical_decision["target_exposure"]))
        t3.metric("Relative Drift Z",fmt_num(tactical_decision["relative_z"]))
        t4.metric("Rolling Beta",fmt_num(tactical_decision["beta"]))
        t5.metric("NW Envelope Z",fmt_num(tactical_decision["envelope_z"]))
        t6.metric("Benchmark",benchmark_ticker_used)

        st.markdown("#### Tactical Gate Matrix")
        st.dataframe(tactical_decision["gates"],width="stretch",hide_index=True,height=300)

        st.markdown("#### Staged Exposure & Capital Path")
        st.plotly_chart(make_tactical_portfolio_chart(tactical_result),width="stretch",config=PLOT_CFG)

        with st.expander("Tactical Action Ledger",expanded=False):
            cols=["AdjCloseCalc","NWTrend","NWUpper","NWLower","NWEnvelopeZ","ResidualDriftZ",
                  "RollingBeta","RelativeVolume","TacticalAction","TacticalTargetExposure",
                  "TacticalExposure","TacticalPortfolio","TacticalRationale"]
            st.dataframe(tactical_result[cols].sort_index(ascending=False),width="stretch",height=560)

with tabs[2]:
    st.markdown(
        "<div class='section-note'><b>Client-facing signal chart:</b> adjusted price, Nadaraya-Watson path/envelope "
        "and Institutional Tactical events only.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_primary_price_chart(result, f"{name_used} ({ticker_used})"), width="stretch", config=PLOT_CFG)

with tabs[3]:
    if not nw_enabled_used or nw_result is None:
        st.info("Nadaraya-Watson research layer was disabled for this run. Enable it in the sidebar and run the analysis again.")
    else:
        st.markdown(
            "<div class='section-note'><b>Nadaraya-Watson rulebook:</b> the estimator is causal and one-sided. "
            "Every signal is formed from a completed bar; portfolio actions execute at the next adjusted open. "
            "The tactical layer treats envelope excursions as early risk events rather than waiting for a distant hard stop.</div>",
            unsafe_allow_html=True,
        )

        _nw_rules = pd.DataFrame([
            {"Layer":"Estimator","Rule":"NW(t) = kernel-weighted estimate using current/past bars only","Action":"State estimation","Current Setting":f"{nw_cfg.kernel}; lookback {nw_cfg.lookback}; h={nw_cfg.effective_bandwidth:g}"},
            {"Layer":"Trend Regime","Rule":"NW slope > 0 = bullish; NW slope < 0 = bearish","Action":"Directional state","Current Setting":nw_decision["trend_direction"]},
            {"Layer":"Bullish Reversal","Rule":"NW slope flips non-positive → positive","Action":"BUY / restore candidate","Current Setting":"TRIGGERED" if nw_decision["bullish_reversal"] else "No"},
            {"Layer":"Bearish Reversal","Rule":"NW slope flips non-negative → negative","Action":"REDUCE / SELL candidate","Current Setting":"TRIGGERED" if nw_decision["bearish_reversal"] else "No"},
            {"Layer":"Upper Envelope","Rule":"Source crosses above NW Upper residual band","Action":"Immediate tactical trim; overextension alert","Current Setting":f"Upper={nw_decision['upper']:.2f}"},
            {"Layer":"Upper Re-entry","Rule":"Source returns below Upper band after excursion","Action":"Deeper reduction / exhaustion confirmation","Current Setting":"Primary Tactical rule"},
            {"Layer":"Lower Envelope","Rule":"Source crosses below NW Lower band + bearish slope","Action":"SELL 100% candidate","Current Setting":f"Lower={nw_decision['lower']:.2f}"},
            {"Layer":"MK Confirmed Entry","Rule":f"{nw_scfg.confirmation_bars} bullish bar(s) + Source > NW" + (" + Source <= Upper" if nw_scfg.avoid_upper_band_chase else ""),"Action":"BUY / restore","Current Setting":nw_decision["entry_reason"]},
            {"Layer":"MK Confirmed Exit","Rule":f"Source < NW OR {nw_scfg.exit_confirmation_bars} bearish bar(s)","Action":"Exit candidate","Current Setting":nw_decision["exit_reason"]},
            {"Layer":"Benchmark Filter","Rule":f"Beta-adjusted residual drift vs {benchmark_ticker_used}; z thresholds ±{tactical_cfg.weak_z:g}/±{tactical_cfg.strong_z:g}/±{tactical_cfg.extreme_z:g}","Action":"Escalate BUY/REDUCE/SELL","Current Setting":f"z={tactical_decision['relative_z']:.2f}" if tactical_decision else "N/A"},
            {"Layer":"Execution","Rule":"Completed bar signal → next adjusted open","Action":"No look-ahead","Current Setting":"ENFORCED"},
        ])
        st.markdown("#### Nadaraya-Watson + Tactical Rulebook")
        st.dataframe(_nw_rules,width="stretch",hide_index=True,height=430)
        st.caption(
            "Priority matters: severe lower-band / extreme relative-breakdown exits are evaluated before weaker trim signals. "
            "Upper-band crosses are deliberately treated as early de-risking events in High Sensitivity mode."
        )

        n1,n2,n3,n4,n5,n6,n7 = st.columns(7)
        n1.metric("NW Strategy Decision", nw_decision["decision"])
        n2.metric("NW Trend Regime", nw_decision["trend_direction"])
        n3.metric("Price / NW Gap", fmt_pct(nw_decision["price_trend_gap"]))
        n4.metric("Residual Band Width", fmt_pct(nw_decision["band_width_pct"]))
        n5.metric("NW Strategy CAGR", fmt_pct(nw_summary["strategy_cagr"]))
        n6.metric("NW Max Drawdown", fmt_pct(nw_summary["max_drawdown"]))
        n7.metric("NW Closed Trades", f"{nw_tstats['closed_trades']:,}")

        st.markdown("#### Nadaraya-Watson Price Structure")
        st.plotly_chart(make_nw_overlay_chart(nw_result, nw_cfg), width="stretch", config=PLOT_CFG)

        left,right = st.columns([1.2,1])
        with left:
            st.markdown("#### NW Decision Causality")
            st.markdown(
                f'''<div class="decision-card">
                    <div class="decision-label">Current NW Strategy State</div>
                    <div class="decision-value">{nw_decision['decision']}</div>
                    <div class="decision-reason">{nw_decision['rationale']}</div>
                </div>''',
                unsafe_allow_html=True,
            )
            st.dataframe(nw_decision["gates"], width="stretch", hide_index=True, height=270)
        with right:
            st.markdown("#### Causal Strategy Rules")
            st.markdown(
                f"**Strategy:** {strategy_mode_label(nw_scfg.mode)}  \n"
                f"**Preset:** {nw_preset_used}  \n"
                f"**Kernel:** {nw_cfg.kernel}  \n"
                f"**Lookback:** {nw_cfg.lookback} bars  \n"
                f"**Effective bandwidth:** {nw_cfg.effective_bandwidth:g}  \n"
                f"**Residual bands:** ± {nw_cfg.band_multiplier:g} × kernel-weighted mean absolute residual  \n"
                f"**Entry condition:** {nw_decision['entry_reason']}  \n"
                f"**Exit condition:** {nw_decision['exit_reason']}"
            )
            st.caption(nw_decision["timing_note"])
            st.warning(
                "The QuantAlgo publication is an indicator. The BUY/SELL portfolio rules shown here are the explicitly defined MK research translation / strategy layer, not a claim that QuantAlgo publishes or endorses this backtest strategy."
            )

        st.markdown("#### NW Strategy Performance")
        st.plotly_chart(make_nw_equity_chart(nw_result), width="stretch", config=PLOT_CFG)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("NW Final Value", f"{nw_summary['portfolio_final']:,.0f}")
        c2.metric("Buy & Hold Final", f"{nw_summary['buyhold_final']:,.0f}")
        c3.metric("CAGR Spread", fmt_pct(nw_summary["strategy_cagr"]-nw_summary["buyhold_cagr"]))
        c4.metric("Win Rate", fmt_pct(nw_tstats["win_rate"]))
        c5.metric("Avg Holding Days", fmt_num(nw_tstats["avg_holding_days"],0))

        c_left,c_right = st.columns(2)
        with c_left:
            st.plotly_chart(make_nw_kernel_chart(nw_cfg), width="stretch", config=PLOT_CFG)
        with c_right:
            st.plotly_chart(make_nw_state_chart(nw_result), width="stretch", config=PLOT_CFG)

        st.markdown("#### Strategy Research Comparison — Same Market Data")
        _rows=[]
        if tactical_enabled_used and tactical_result is not None:
            _tp=pd.to_numeric(tactical_result["TacticalPortfolio"],errors="coerce")
            _years=max((tactical_result.index[-1]-tactical_result.index[0]).days/365.25,1e-9)
            _rows.append({"System":"MK Institutional Tactical","CAGR":(_tp.iloc[-1]/float(initial_capital))**(1/_years)-1,"Max Drawdown":(_tp/_tp.cummax()-1).min(),"Final Value":float(_tp.iloc[-1]),"Closed Trades":int((tactical_result["TacticalTradedValue"]>0).sum())})
        _rows.extend([
            {"System":strategy_mode_label(nw_scfg.mode),"CAGR":nw_summary["strategy_cagr"],"Max Drawdown":nw_summary["max_drawdown"],"Final Value":nw_summary["portfolio_final"],"Closed Trades":nw_tstats["closed_trades"]},
            {"System":"Buy & Hold","CAGR":nw_summary["buyhold_cagr"],"Max Drawdown":nw_summary["buyhold_max_drawdown"],"Final Value":nw_summary["buyhold_final"],"Closed Trades":0},
        ])
        compare=pd.DataFrame(_rows)
        st.dataframe(
            compare.style.format({"CAGR":"{:.2%}","Max Drawdown":"{:.2%}","Final Value":"{:,.0f}"}),
            width="stretch", hide_index=True,
        )

        with st.expander("NW Strategy Trade Ledger", expanded=False):
            st.dataframe(nw_trades.sort_values("Entry Date", ascending=False) if len(nw_trades) else nw_trades, width="stretch", hide_index=True, height=460)

        nw_html = build_nw_html_report(
            nw_result, nw_cfg, nw_scfg,
            ticker=ticker_used,
            instrument_name=name_used,
            market_label=f"{market_used} / {group_used}",
            source_note=f"Yahoo Finance via yfinance | {interval_used} | strict no-fallback policy",
        )
        st.download_button(
            "Export Nadaraya-Watson Interactive HTML",
            data=nw_html.encode("utf-8"),
            file_name=f"MK_Nadaraya_Watson_{ticker_used}_{interval_used}_v006.html",
            mime="text/html",
            width="stretch",
        )


with tabs[4]:
    if tactical_enabled_used and tactical_result is not None:
        st.markdown("<div class='section-note'><b>Primary strategy comparison:</b> Institutional Tactical portfolio versus Buy & Hold.</div>", unsafe_allow_html=True)
        st.plotly_chart(make_tactical_portfolio_chart(tactical_result), width="stretch", config=PLOT_CFG)
        _tp=pd.to_numeric(tactical_result["TacticalPortfolio"],errors="coerce")
        _bh=pd.to_numeric(tactical_result["BuyHold"],errors="coerce")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Tactical Total Return",fmt_pct(_tp.iloc[-1]/float(initial_capital)-1))
        c2.metric("Buy & Hold Return",fmt_pct(_bh.iloc[-1]/float(initial_capital)-1))
        c3.metric("Tactical Max DD",fmt_pct((_tp/_tp.cummax()-1).min()))
        c4.metric("Buy & Hold Max DD",fmt_pct((_bh/_bh.cummax()-1).min()))
    elif nw_enabled_used and nw_result is not None:
        st.info("Tactical Layer disabled; showing Nadaraya-Watson research strategy versus Buy & Hold.")
        st.plotly_chart(make_nw_equity_chart(nw_result), width="stretch", config=PLOT_CFG)
    else:
        st.warning("No primary strategy comparison is available.")

with tabs[5]:
    st.markdown(
        "<div class='section-note'><b>Institutional risk frame:</b> the selected dashboard timeframe, selected rolling calibration window, "
        "and the active Yahoo benchmark are used consistently. No missing asset/benchmark observations are filled.</div>",
        unsafe_allow_html=True,
    )

    _risk_source=result.copy()
    _risk_label="Nadaraya-Watson Research"
    if tactical_enabled_used and tactical_result is not None:
        _risk_source["Portfolio"]=pd.to_numeric(tactical_result["TacticalPortfolio"],errors="coerce")
        _risk_source["Shares"]=np.where(pd.to_numeric(tactical_result["TacticalTargetExposure"],errors="coerce")>0,1.0,0.0)
        _risk_label="MK Institutional Tactical"
    elif nw_enabled_used and nw_result is not None:
        _risk_source["Portfolio"]=pd.to_numeric(nw_result["Portfolio"],errors="coerce")
        _risk_source["Shares"]=pd.to_numeric(nw_result["NWExposure"],errors="coerce")

    window_specs=rolling_window_options(_risk_source.index)
    default_idx=max(0,len(window_specs)-1)
    selected_spec=st.selectbox(
        "Risk Calibration Window",
        window_specs,index=default_idx,
        format_func=lambda x:f"{x.label} | {x.observations} {x.frequency_label.lower()} observations",
    )
    rolling,used_spec=rolling_risk_frame(_risk_source,selected_spec.observations)
    risk_state=risk_state_snapshot(_risk_source,rolling,used_spec)
    risk_integrity=validate_underlying_risk_dynamics(result,rolling)
    if risk_integrity["impossible_flatness"]:
        st.error("RISK INTEGRITY STOP — moving Yahoo prices produced an impossible flat underlying risk state. Analysis stopped.")
        st.stop()

    st.markdown("### Value at Risk — Asset, Benchmark, Active Residual & Strategy")
    vh1,vh2=st.columns([1,1])
    with vh1:
        default_h=4 if interval_used=="15m" else 1
        var_horizon=st.number_input("VaR Horizon (bars)",min_value=1,max_value=max(1,min(20,used_spec.observations//3)),value=min(default_h,max(1,min(20,used_spec.observations//3))),step=1)
    with vh2:
        mc_scenarios=st.selectbox("Monte Carlo Scenarios",[10000,25000,50000],index=1)

    if relative_result is None or benchmark_ticker_used is None:
        st.error("BENCHMARK RISK STOP — VaR comparison requires the selected Yahoo benchmark and exact-timestamp relative dataset.")
    else:
        _asset_ret=pd.to_numeric(result["AdjCloseCalc"],errors="coerce").pct_change(fill_method=None)
        _bench_ret=pd.to_numeric(relative_result["BenchmarkPrice"],errors="coerce").pct_change(fill_method=None)
        _strategy_ret=pd.to_numeric(_risk_source["Portfolio"],errors="coerce").pct_change(fill_method=None)
        _active_resid=np.expm1(pd.to_numeric(relative_result["ResidualReturn"],errors="coerce"))
        _series={
            f"Asset — {ticker_used}":_asset_ret,
            f"Benchmark — {benchmark_ticker_used}":_bench_ret,
            "Beta-Adjusted Active Residual":_active_resid,
            f"Strategy — {_risk_label}":_strategy_ret,
        }
        _var_cfg=VaRConfig(horizon_bars=int(var_horizon),mc_scenarios=int(mc_scenarios),min_observations=max(20,min(30,used_spec.observations//2)))
        var_table=build_var_table(_series,calibration_observations=used_spec.observations,config=_var_cfg)
        _vt=var_table.copy()
        _vt["Confidence"]=_vt["Confidence"].map(lambda x:f"{x:.0%}" if pd.notna(x) else "—")
        _vt["VaR"]=_vt["VaR"].map(lambda x:f"{x:.2%}" if pd.notna(x) else "—")

        v1,v2,v3,v4=st.columns(4)
        v1.metric("Timeframe",interval_label_used)
        v2.metric("Calibration",f"{used_spec.label} / {used_spec.observations} bars")
        v3.metric("VaR Horizon",_risk_horizon_text(interval_used,var_horizon))
        v4.metric("Benchmark",benchmark_ticker_used)
        st.dataframe(_vt,width="stretch",hide_index=True,height=620)
        conf_view=st.radio("VaR Chart Confidence",["95%","99%"],horizontal=True,index=1)
        st.plotly_chart(make_var_comparison_chart(var_table,0.95 if conf_view=="95%" else 0.99),width="stretch",config=PLOT_CFG)
        st.caption(
            "Historical VaR = empirical lower-tail quantile of observed compounded returns. Parametric VaR = Normal model fitted to observed log returns. "
            "Monte Carlo VaR = empirical bootstrap scenarios drawn only from observed returns. Simulation scenarios are risk calculations only; they are never appended to Yahoo history or used as replacement market observations."
        )

    st.markdown("### Rolling Risk & Drawdown")
    st.plotly_chart(make_drawdown_chart(_risk_source),width="stretch",config=PLOT_CFG)
    st.markdown("#### Underlying Asset Risk")
    st.plotly_chart(make_underlying_rolling_risk_chart(result,rolling,used_spec,f"{name_used} ({ticker_used})"),width="stretch",config=PLOT_CFG)
    a1,a2,a3,a4=st.columns(4)
    a1.metric(f"{used_spec.label} Asset Rolling Return",fmt_pct(risk_state["asset_rolling_return"]))
    a2.metric(f"{used_spec.label} Asset Ann. Volatility",fmt_pct(risk_state["asset_annualized_volatility"]))
    a3.metric("Risk Frequency",used_spec.frequency_label)
    a4.metric("Risk-Series Unique Points",f"{risk_integrity['rolling_return_unique']:,}")

    st.markdown("#### Strategy Risk — Primary Portfolio")
    st.plotly_chart(make_strategy_rolling_risk_chart(_risk_source,rolling,used_spec),width="stretch",config=PLOT_CFG)
    s1,s2,s3,s4=st.columns(4)
    s1.metric(f"{used_spec.label} Strategy Rolling Return",fmt_pct(risk_state["strategy_rolling_return"]))
    s2.metric(f"{used_spec.label} Strategy Ann. Volatility",fmt_pct(risk_state["strategy_annualized_volatility"]))
    s3.metric(f"{used_spec.label} Rolling Exposure",fmt_pct(risk_state["rolling_exposure"]))
    s4.metric("Current Position",risk_state["current_position"])

    with st.expander("Risk Integrity Diagnostics",expanded=False):
        st.dataframe(pd.DataFrame([{
            "Adjusted Close Unique Prices":risk_integrity["price_unique"],
            "Adjusted Close Range":risk_integrity["price_range"],
            "Rolling Return Unique Values":risk_integrity["rolling_return_unique"],
            "Rolling Return Range":risk_integrity["rolling_return_range"],
            "Rolling Vol Unique Values":risk_integrity["rolling_vol_unique"],
            "Rolling Vol Range":risk_integrity["rolling_vol_range"],
            "Impossible Flatness":risk_integrity["impossible_flatness"],
        }]),width="stretch",hide_index=True)

with tabs[6]:
    st.markdown(
        "<div class='section-note'><b>Trend Diagnostics:</b> Nadaraya-Watson structure, benchmark-relative regime, relative volume, and target exposure. "
        "No historical workbook threshold is used in this client-facing diagnostic.</div>",
        unsafe_allow_html=True,
    )
    if tactical_enabled_used and tactical_result is not None and tactical_cfg is not None:
        st.plotly_chart(make_institutional_trend_diagnostics(tactical_result,benchmark_ticker_used,tactical_cfg),width="stretch",config=PLOT_CFG)
        _last=tactical_result.iloc[-1]
        d1,d2,d3,d4,d5,d6=st.columns(6)
        d1.metric("NW Regime","BULLISH" if int(_last["NWDirection"])>0 else "BEARISH" if int(_last["NWDirection"])<0 else "FLAT")
        d2.metric("NW Envelope Z",fmt_num(_last["NWEnvelopeZ"]))
        d3.metric("Relative Drift Z",fmt_num(_last["ResidualDriftZ"]))
        d4.metric("Rolling Beta",fmt_num(_last["RollingBeta"]))
        d5.metric("Relative Volume",fmt_num(_last["RelativeVolume"]))
        d6.metric("Target Exposure",fmt_pct(_last["TacticalTargetExposure"]))
        st.markdown("#### Diagnostic Interpretation")
        st.dataframe(tactical_decision["gates"],width="stretch",hide_index=True,height=310)
    elif nw_enabled_used and nw_result is not None:
        st.plotly_chart(make_nw_state_chart(nw_result),width="stretch",config=PLOT_CFG)
        st.info("Benchmark-relative tactical diagnostics are unavailable because the Tactical Layer was disabled.")
    else:
        st.info("No institutional trend diagnostic is available for this run.")

with tabs[7]:
    if tactical_enabled_used and tactical_result is not None:
        _a=tactical_result.copy(); _prev=_a["TacticalTargetExposure"].shift(1)
        _mask=_a["TacticalTargetExposure"].ne(_prev) & _a["TacticalAction"].ne("")
        _ledger=_a.loc[_mask,["AdjOpen","AdjCloseCalc","TacticalAction","TacticalTargetExposure","TacticalExposure","TacticalTradedValue","ResidualDriftZ","NWEnvelopeZ","RelativeVolume","TacticalRationale"]].copy()
        _ledger.index.name="Execution Date"
        st.markdown("#### Institutional Tactical Action Ledger")
        st.dataframe(_ledger.sort_index(ascending=False),width="stretch",height=560)
        st.caption("Exposure changes use the completed prior bar and execute at the next adjusted open.")
    elif nw_enabled_used and nw_trades is not None:
        st.markdown("#### Nadaraya-Watson Research Trade Ledger")
        st.dataframe(nw_trades.sort_values("Entry Date",ascending=False) if len(nw_trades) else nw_trades,width="stretch",hide_index=True,height=520)
    else:
        st.info("No primary trade/action ledger is available.")

with tabs[8]:
    if tactical_enabled_used and tactical_result is not None:
        _cols=["Open","High","Low","Close","Volume","Adj Close","AdjCloseCalc","NWTrend","NWUpper","NWLower","NWEnvelopeZ","NWSlope","NWDirection","NWCrossAboveUpper","NWReenterBelowUpper","NWCrossBelowLower","BenchmarkPrice","RollingBeta","ResidualZ","ResidualDriftZ","PriceRatioZ","RelativeVolume","TacticalAction","TacticalTargetExposure","TacticalExposure","TacticalPortfolio","TacticalRationale"]
        _cols=[c for c in _cols if c in tactical_result.columns]
        st.dataframe(tactical_result[_cols].sort_index(ascending=False),width="stretch",height=650)
        _csv=tactical_result[_cols].reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Export Institutional Tactical Ledger CSV",_csv,file_name=f"MK_Institutional_Tactical_{ticker_used}_{APP_VERSION.replace('.','')}.csv",mime="text/csv")
    elif nw_enabled_used and nw_result is not None:
        st.dataframe(nw_result.sort_index(ascending=False),width="stretch",height=650)
    else:
        st.info("No primary calculation ledger is available.")

with tabs[9]:
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
    st.dataframe(filt, width="stretch", hide_index=True, height=580)
    st.caption("This is a curated convenience universe, not an exhaustive exchange constituent list. Manual Yahoo ticker input remains available for instruments outside the list.")

with tabs[10]:
    st.markdown("""
### Nadaraya-Watson Trend research layer
The NW module independently implements the public QuantAlgo methodology: a **one-sided causal endpoint estimator** using only the current and historical bars; Gaussian, Rational Quadratic, Epanechnikov, Triangular, Quartic and Cosine kernels; an effective bandwidth `h = bandwidth × multiplier`; and optional residual bands based on a kernel-weighted mean absolute residual around the current NW estimate.

Trend direction is the bar-to-bar slope of the NW estimate. A bullish reversal is a non-positive → positive slope transition; a bearish reversal is a non-negative → negative transition. Because only past/current observations are used, appended future bars cannot rewrite a historical NW estimate.

**Attribution:** methodology reference is *Nadaraya-Watson Trend [QuantAlgo]* on TradingView. The Pine source is not redistributed verbatim in this project. Numeric MK presets and the portfolio strategy rules are our own research layer.

### MK causal NW strategy
The indicator and the strategy are deliberately separated. The strategy reads a signal only from a **completed prior bar** and executes any trade at the **next adjusted open**. `MK Confirmed NW Trend` requires persistent bullish NW direction plus price above the path, with an optional upper-residual-band chase filter; exits occur when price loses the path or bearish direction confirms. `Public-Methodology Reversal Translation` is a simpler research translation built around bullish/bearish slope reversals and price relative to the NW path. Neither is presented as a QuantAlgo trading recommendation.

""")
    st.markdown(f"""
### Client decision-governance hierarchy
The client-facing primary decision source is **MK Institutional Tactical**. It supports staged exposure
`100% → 75% → 50% → 25% → 0%` and combines Nadaraya-Watson envelope events with benchmark-relative
deviation and relative volume. If Tactical is disabled, the application displays **NO PRIMARY DECISION**;
it does not substitute another decision engine.


### Institutional Tactical Layer
The primary live decision is the Institutional Tactical engine. Its hierarchy is:
1. **NW upper-band cross:** immediate early de-risk / staged trim.
2. **Upper-band re-entry:** confirms failed upside excursion and deepens the reduction.
3. **NW reversal / path loss:** directional deterioration.
4. **Beta-adjusted benchmark residual drift:** identifies abnormal relative weakness or overextension.
5. **Relative volume:** flags potential intraday climax.
6. **Staged target exposure:** 100% → 75% → 50% → 25% → 0%, executed at the next adjusted open.

Relative analytics use exact-timestamp inner alignment only. Missing benchmark bars are not filled.

### 15-Minute Intraday Governance
Yahoo Finance supports 15-minute data, but intraday history is restricted to the most recent 60 days.
The engine hard-stops requests beyond that horizon instead of silently truncating the analysis.

### Strict market-data governance
Yahoo Finance is the only live market-data source in this build. The application does not fabricate observations, fill missing market prices, or switch to another vendor. A failed or incomplete Yahoo response terminates that requested run.

### Price adjustment
Yahoo is requested with `auto_adjust=False`. Raw OHLC and `Adj Close` stay distinct. Adjusted OHLC is derived consistently from Yahoo `Adj Close / Close`; no alternate vendor or filled prices are introduced.

### Research boundary
The displayed BUY / HOLD / SELL / WAIT labels are deterministic **strategy states**, not discretionary investment recommendations.
""")

st.caption("MK FinTECH LabGEN @2026 ATELIER ISTANBUL  |  By Murat Konuklar")
