from __future__ import annotations

from dataclasses import replace
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
from MK_DEMA_MACD_Confirmation_v007 import (
    DEMAMACDConfig, DEMAStrategyConfig, DEMACalibrationConfig,
    compute_dema_macd, run_dema_macd_strategy, dema_decision_snapshot,
    dema_event_ledger, dema_trade_ledger, exit_quality_metrics,
    calibrate_dema_macd, dema_preset, lifecycle_explanation,
)
from MK_DEMA_MACD_HTML_Report_v007 import build_dema_macd_html_report


APP_VERSION = "v0.07"
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
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color="#0F172A")),
        template="plotly_white",
        height=660,
        margin=dict(l=52, r=24, t=92, b=38),
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
            x=0.01, xanchor="left", font=dict(size=15, color="#0F172A")
        ),
        template="plotly_white", height=690, hovermode="x unified",
        margin=dict(l=45, r=25, t=85, b=35),
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
        title=dict(text="MK Causal Nadaraya-Watson Strategy vs Buy & Hold", x=0.01, xanchor="left", font=dict(size=15)),
        template="plotly_white", height=620, hovermode="x unified",
        margin=dict(l=45, r=25, t=75, b=30),
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



# ---------------------------- DEMA-MACD charts ----------------------------
def make_dema_price_chart(df, dema_cfg):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
        name="Adjusted OHLC", increasing_line_color="#0F766E", decreasing_line_color="#B91C1C",
    ))
    for c,label,color,width in [
        ("DEMAFast","Fast DEMA","#2563EB",1.15),("DEMASlow","Slow DEMA","#7C3AED",1.15),("DEMATrend","Trend DEMA","#334155",1.55)
    ]:
        if c in df:
            fig.add_trace(go.Scatter(x=df.index,y=df[c],mode="lines",name=label,line=dict(color=color,width=width)))
    buys=df["FirstBuy"].fillna(0).gt(0); sells=df["FirstSell"].fillna(0).gt(0)
    fig.add_trace(go.Scatter(x=df.index[buys],y=df.loc[buys,"AdjOpen"],mode="markers",name="BUY",marker=dict(symbol="triangle-up",size=13,color="#065F46")))
    fig.add_trace(go.Scatter(x=df.index[sells],y=df.loc[sells,"AdjOpen"],mode="markers",name="SELL / RISK EXIT",marker=dict(symbol="triangle-down",size=13,color="#991B1B")))
    if "DEMAReduceMarker" in df:
        red=pd.to_numeric(df["DEMAReduceMarker"],errors="coerce"); m=red.notna()
        fig.add_trace(go.Scatter(x=df.index[m],y=red[m],mode="markers",name="REDUCE",marker=dict(symbol="diamond",size=9,color="#B45309")))
    fig.update_layout(template="plotly_white",height=690,hovermode="x unified",margin=dict(l=45,r=25,t=85,b=35),
        title=dict(text=f"DEMA-MACD Price Structure | {dema_cfg.fast_length}/{dema_cfg.slow_length}/{dema_cfg.signal_length}",x=.01,font=dict(size=15)),
        legend=dict(orientation="h",y=1.03,x=1,xanchor="right"),xaxis_rangeslider_visible=False,
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11))
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,showgrid=False); fig.update_yaxes(title_text="Adjusted Price",gridcolor="#E2E8F0")
    return fig


def make_dema_oscillator_chart(df):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.62,.38],vertical_spacing=.07)
    fig.add_trace(go.Bar(x=df.index,y=df["DEMAHistogram"],name="DEMA Histogram",marker_color="#94A3B8"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["DEMAMACD"],mode="lines",name="DEMA MACD",line=dict(width=1.4,color="#0F172A")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["DEMASignal"],mode="lines",name="Signal",line=dict(width=1.2,color="#B45309")),row=1,col=1)
    fig.add_hline(y=0,line_width=1,line_color="#CBD5E1",row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BuyScore"],mode="lines",name="BUY Score",line=dict(width=1.4,color="#0F766E")),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["SellScore"],mode="lines",name="SELL Score",line=dict(width=1.4,color="#B91C1C")),row=2,col=1)
    fig.update_layout(template="plotly_white",height=650,hovermode="x unified",margin=dict(l=45,r=25,t=75,b=40),title=dict(text="DEMA-MACD Momentum & Confirmation Scores",x=.01,font=dict(size=15)),legend=dict(orientation="h",y=1.03,x=1,xanchor="right"))
    fig.update_yaxes(title_text="Momentum",row=1,col=1,gridcolor="#E2E8F0"); fig.update_yaxes(title_text="Score",range=[0,100],row=2,col=1,gridcolor="#E2E8F0")
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False),row=1,col=1)
    return fig


def make_dema_equity_chart(df):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.78,.22],vertical_spacing=.06)
    fig.add_trace(go.Scatter(x=df.index,y=df["Portfolio"],mode="lines",name="DEMA-MACD Strategy",line=dict(width=1.8,color="#0F172A")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BuyHold"],mode="lines",name="Buy & Hold",line=dict(width=1.3,color="#64748B",dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["DEMAExposure"],mode="lines",name="Exposure",line=dict(width=1.2,color="#475569"),fill="tozeroy",fillcolor="rgba(71,85,105,.10)"),row=2,col=1)
    fig.update_layout(template="plotly_white",height=600,hovermode="x unified",margin=dict(l=45,r=25,t=75,b=35),title=dict(text="DEMA-MACD Strategy vs Buy & Hold",x=.01,font=dict(size=15)),legend=dict(orientation="h",y=1.03,x=1,xanchor="right"))
    fig.update_yaxes(title_text="Portfolio Value",row=1,col=1,gridcolor="#E2E8F0"); fig.update_yaxes(title_text="Exposure",tickformat=".0%",range=[0,1.02],row=2,col=1)
    return fig


def make_dema_calibration_chart(calibration):
    rank=calibration.get("ranking",pd.DataFrame())
    if rank is None or rank.empty: return go.Figure()
    top=rank.head(30).copy(); top["Parameter Set"]=top.apply(lambda r:f"{int(r.Fast)}/{int(r.Slow)}/{int(r.Signal)} | B{int(r.BuyThreshold)} S{int(r.SellThreshold)}",axis=1)
    fig=go.Figure(go.Bar(x=top["InstitutionalScore"],y=top["Parameter Set"],orientation="h",marker_color="#475569"))
    fig.update_layout(template="plotly_white",height=max(460,22*len(top)),title=dict(text="Calibration Robustness Ranking — Train + Validation Selection; OOS Untouched",x=.01,font=dict(size=15)),margin=dict(l=150,r=25,t=70,b=40),yaxis=dict(autorange="reversed"),xaxis_title="Institutional Robustness Score")
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
    st.markdown("##### Entry Gate Governance")
    entry_mode_label = st.selectbox(
        "Entry Breakout Mode",
        ["Frequency-Aware", "Legacy Exact", "Custom"],
        index=0,
        help=(
            "Frequency-Aware maps a calendar horizon to the selected data frequency. "
            "Legacy Exact preserves the original 2000-observation breakout gate."
        ),
    )

    entry_horizon = "12M"
    custom_entry_observations = 252
    if entry_mode_label == "Frequency-Aware":
        entry_horizon = st.selectbox(
            "Entry Breakout Horizon",
            horizon_options(interval),
            index=horizon_options(interval).index("12M"),
            help="BUY requires the prior adjusted close to reach the rolling maximum over this horizon.",
        )
        max_buy_weeks, entry_gate_label = resolve_entry_lookback(
            interval, "Frequency-Aware", horizon=entry_horizon
        )
        st.caption(f"Effective entry lookback: **{max_buy_weeks} observations**.")
    elif entry_mode_label == "Legacy Exact":
        max_buy_weeks, entry_gate_label = resolve_entry_lookback(interval, "Legacy Exact")
        st.warning(
            "Legacy Exact uses 2000 observations. For recently listed stocks such as ASTOR, "
            "this can become an effective all-history-high entry gate and can keep the strategy in cash for long periods."
        )
    else:
        custom_entry_observations = st.number_input(
            "Custom Entry Lookback (observations)",
            min_value=2, value=252, step=1,
        )
        max_buy_weeks, entry_gate_label = resolve_entry_lookback(
            interval, "Custom", custom_observations=int(custom_entry_observations)
        )

    legacy_fidelity = st.toggle(
        "Legacy OFFSET Fidelity",
        value=True,
        help="Preserves the original workbook's inclusive Excel OFFSET window semantics exactly.",
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
    st.subheader("DEMA-MACD Confirmation Engine")
    dema_enabled = st.toggle("Enable DEMA-MACD BUY / SELL Layer", value=True,
        help="Independent causal confirmation engine with BUY WATCH, BUY, HOLD, SELL WATCH, REDUCE, SELL and RISK EXIT lifecycle states.")
    dema_preset_name = st.selectbox("DEMA-MACD Research Preset",
        ["MK Institutional Balanced","Reference-Style Continuation","MK Fast Confirmation","MK Smooth Position","Custom"],
        index=0, disabled=not dema_enabled)
    _dema_i0,_dema_s0=dema_preset(dema_preset_name)
    if dema_preset_name == "Custom":
        dc1,dc2,dc3=st.columns(3)
        dema_fast=dc1.number_input("Fast",min_value=2,value=12,step=1,disabled=not dema_enabled)
        dema_slow=dc2.number_input("Slow",min_value=3,value=26,step=1,disabled=not dema_enabled)
        dema_signal=dc3.number_input("Signal",min_value=1,value=9,step=1,disabled=not dema_enabled)
        dema_buy_thr=st.slider("BUY Confirmation Score",0,100,65,1,disabled=not dema_enabled)
        dema_watch_thr=st.slider("SELL WATCH Score",0,80,35,1,disabled=not dema_enabled)
        dema_reduce_thr=st.slider("REDUCE Score",int(dema_watch_thr),90,max(50,int(dema_watch_thr)),1,disabled=not dema_enabled)
        dema_sell_thr=st.slider("SELL Confirmation Score",int(dema_reduce_thr),100,max(65,int(dema_reduce_thr)),1,disabled=not dema_enabled)
        dema_sell_persist=st.number_input("SELL Persistence Bars",min_value=1,value=2,step=1,disabled=not dema_enabled)
        dema_cooldown=st.number_input("Re-entry Cooldown Bars",min_value=0,value=3,step=1,disabled=not dema_enabled)
        dema_atr_mult=st.number_input("ATR Risk-Exit Multiplier",min_value=.5,value=3.0,step=.1,disabled=not dema_enabled)
        dema_hard_stop=st.number_input("Hard Stop (%)",min_value=1.0,max_value=50.0,value=12.0,step=.5,disabled=not dema_enabled)/100.0
    else:
        dema_fast,dema_slow,dema_signal=_dema_i0.fast_length,_dema_i0.slow_length,_dema_i0.signal_length
        dema_buy_thr=_dema_s0.buy_threshold; dema_watch_thr=_dema_s0.sell_watch_threshold; dema_reduce_thr=_dema_s0.reduce_threshold; dema_sell_thr=_dema_s0.sell_threshold
        dema_sell_persist=_dema_s0.sell_persistence_bars; dema_cooldown=_dema_s0.cooldown_bars; dema_atr_mult=_dema_s0.atr_trailing_multiplier; dema_hard_stop=_dema_s0.hard_stop_pct
        if dema_enabled: st.caption(f"DEMA {dema_fast}/{dema_slow}/{dema_signal} | BUY ≥ {dema_buy_thr:.0f} | WATCH/REDUCE/SELL {dema_watch_thr:.0f}/{dema_reduce_thr:.0f}/{dema_sell_thr:.0f}")
    dema_execute_reduce=st.toggle("Execute REDUCE as Partial Sale",value=False,disabled=not dema_enabled,
        help="Off by default: REDUCE remains advisory. When enabled, the new DEMA layer sells the configured fraction at next adjusted open. Legacy engine is never changed.")
    dema_reduce_fraction=st.slider("REDUCE Fraction",10,90,50,5,disabled=(not dema_enabled) or (not dema_execute_reduce))/100.0
    dema_calibrate=st.toggle("Run Calibration + Walk-Forward",value=False,disabled=not dema_enabled,
        help="Parameter search uses train/validation selection and leaves a final OOS holdout untouched.")
    dema_grid=st.selectbox("Calibration Grid",["Focused","Balanced","Deep"],index=0,disabled=(not dema_enabled) or (not dema_calibrate))
    dema_folds=st.number_input("Walk-Forward Folds",min_value=1,max_value=6,value=3,step=1,disabled=(not dema_enabled) or (not dema_calibrate))

    run_clicked = st.button("RUN ANALYSIS", type="primary", use_container_width=True)


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
        "nw_enabled", "nw_result", "nw_indicator", "nw_config", "nw_strategy_config",
        "nw_summary", "nw_decision", "nw_trades", "nw_trade_stats", "nw_preset",
        "dema_enabled", "dema_result", "dema_indicator", "dema_config", "dema_strategy_config",
        "dema_summary", "dema_decision", "dema_trades", "dema_trade_stats", "dema_events",
        "dema_exit_detail", "dema_exit_summary", "dema_calibration", "dema_preset",
    ]:
        st.session_state.pop(_k, None)
    st.session_state["_state_schema_version"] = STATE_SCHEMA_VERSION

for key, default in {
    "result": None, "summary": None, "config": None, "raw": None,
    "decision": None, "trades": None, "trade_stats": None,
    "nw_result": None, "nw_indicator": None, "nw_config": None, "nw_strategy_config": None,
    "nw_summary": None, "nw_decision": None, "nw_trades": None, "nw_trade_stats": None,
    "dema_result": None, "dema_indicator": None, "dema_config": None, "dema_strategy_config": None,
    "dema_summary": None, "dema_decision": None, "dema_trades": None, "dema_trade_stats": None,
    "dema_events": None, "dema_exit_detail": None, "dema_exit_summary": None, "dema_calibration": None,
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

            if dema_enabled:
                base_i,base_s=dema_preset(dema_preset_name)
                dema_cfg=replace(base_i,
                    fast_length=int(dema_fast),slow_length=int(dema_slow),signal_length=int(dema_signal),
                    use_nw_filter=bool(base_i.use_nw_filter and nw_enabled),minimum_observations=cfg.minimum_observations)
                dema_scfg=replace(base_s,
                    initial_capital=float(initial_capital),buy_threshold=float(dema_buy_thr),
                    sell_watch_threshold=float(dema_watch_thr),reduce_threshold=float(dema_reduce_thr),sell_threshold=float(dema_sell_thr),
                    sell_persistence_bars=int(dema_sell_persist),cooldown_bars=int(dema_cooldown),
                    atr_trailing_multiplier=float(dema_atr_mult),hard_stop_pct=float(dema_hard_stop),
                    execute_reduce=bool(dema_execute_reduce),reduce_fraction=float(dema_reduce_fraction))
                dema_indicator=compute_dema_macd(result,dema_cfg,nw_indicator if nw_enabled else None)
                dema_result=run_dema_macd_strategy(result,dema_indicator,dema_scfg)
                dema_summary=performance_summary(dema_result,initial_capital=dema_scfg.initial_capital)
                dema_decision=dema_decision_snapshot(dema_result,dema_scfg)
                dema_trades=dema_trade_ledger(dema_result); dema_tstats=trade_statistics(dema_trades)
                dema_events=dema_event_ledger(dema_result)
                dema_exit_detail,dema_exit_summary=exit_quality_metrics(dema_result)
                if dema_calibrate:
                    dema_calibration=calibrate_dema_macd(result,dema_cfg,dema_scfg,nw_indicator if nw_enabled else None,
                        DEMACalibrationConfig(grid_depth=dema_grid,walk_forward_folds=int(dema_folds)))
                else:
                    dema_calibration=None
            else:
                dema_cfg=dema_scfg=dema_indicator=dema_result=None
                dema_summary=dema_decision=dema_trades=dema_tstats=dema_events=dema_exit_detail=dema_exit_summary=dema_calibration=None

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
        st.session_state.dema_enabled = bool(dema_enabled)
        st.session_state.dema_result = dema_result
        st.session_state.dema_indicator = dema_indicator
        st.session_state.dema_config = dema_cfg
        st.session_state.dema_strategy_config = dema_scfg
        st.session_state.dema_summary = dema_summary
        st.session_state.dema_decision = dema_decision
        st.session_state.dema_trades = dema_trades
        st.session_state.dema_trade_stats = dema_tstats
        st.session_state.dema_events = dema_events
        st.session_state.dema_exit_detail = dema_exit_detail
        st.session_state.dema_exit_summary = dema_exit_summary
        st.session_state.dema_calibration = dema_calibration
        st.session_state.dema_preset = dema_preset_name
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
    st.dataframe(u, use_container_width=True, hide_index=True, height=460)
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
dema_enabled_used = bool(st.session_state.get("dema_enabled", False))
dema_result = st.session_state.get("dema_result")
dema_indicator = st.session_state.get("dema_indicator")
dema_cfg = st.session_state.get("dema_config")
dema_scfg = st.session_state.get("dema_strategy_config")
dema_summary = st.session_state.get("dema_summary")
dema_decision = st.session_state.get("dema_decision")
dema_trades = st.session_state.get("dema_trades")
dema_tstats = st.session_state.get("dema_trade_stats")
dema_events = st.session_state.get("dema_events")
dema_exit_detail = st.session_state.get("dema_exit_detail")
dema_exit_summary = st.session_state.get("dema_exit_summary") or {}
dema_calibration = st.session_state.get("dema_calibration")
dema_preset_used = st.session_state.get("dema_preset", "")

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

if dema_enabled_used and dema_result is not None and dema_decision is not None:
    st.markdown("#### DEMA-MACD Confirmation State")
    dm1,dm2,dm3,dm4,dm5,dm6 = st.columns(6)
    dm1.metric("DEMA Decision", dema_decision["decision"])
    dm2.metric("DEMA Position", dema_decision["position"])
    dm3.metric("BUY Score", f"{dema_decision['buy_score']:.1f}")
    dm4.metric("SELL Score", f"{dema_decision['sell_score']:.1f}")
    dm5.metric("DEMA CAGR", fmt_pct(dema_summary["strategy_cagr"]))
    dm6.metric("DEMA Max DD", fmt_pct(dema_summary["max_drawdown"]))

# ---------------------------- Main tabs ----------------------------
tabs = st.tabs([
    "Executive & Decision",
    "Price & Signals",
    "Nadaraya-Watson Trend",
    "DEMA-MACD Confirmation",
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
    if not nw_enabled_used or nw_result is None:
        st.info("Nadaraya-Watson research layer was disabled for this run. Enable it in the sidebar and run the analysis again.")
    else:
        st.markdown(
            "<div class='section-note'><b>Methodology attribution:</b> QuantAlgo's public/open-source TradingView "
            "Nadaraya-Watson Trend describes a one-sided causal kernel estimator, six selectable kernels, kernel-weighted "
            "absolute-residual bands, and slope-reversal markers. The Python implementation here is an independent "
            "reimplementation of that public methodology; the MK strategy layer is separately specified and uses next-open execution.</div>",
            unsafe_allow_html=True,
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
        st.plotly_chart(make_nw_overlay_chart(nw_result, nw_cfg), use_container_width=True, config=PLOT_CFG)

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
            st.dataframe(nw_decision["gates"], use_container_width=True, hide_index=True, height=270)
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
        st.plotly_chart(make_nw_equity_chart(nw_result), use_container_width=True, config=PLOT_CFG)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("NW Final Value", f"{nw_summary['portfolio_final']:,.0f}")
        c2.metric("Buy & Hold Final", f"{nw_summary['buyhold_final']:,.0f}")
        c3.metric("CAGR Spread", fmt_pct(nw_summary["strategy_cagr"]-nw_summary["buyhold_cagr"]))
        c4.metric("Win Rate", fmt_pct(nw_tstats["win_rate"]))
        c5.metric("Avg Holding Days", fmt_num(nw_tstats["avg_holding_days"],0))

        c_left,c_right = st.columns(2)
        with c_left:
            st.plotly_chart(make_nw_kernel_chart(nw_cfg), use_container_width=True, config=PLOT_CFG)
        with c_right:
            st.plotly_chart(make_nw_state_chart(nw_result), use_container_width=True, config=PLOT_CFG)

        st.markdown("#### Strategy Research Comparison — Same Market Data")
        compare = pd.DataFrame([
            {
                "System":"Active Legacy Strategy",
                "CAGR":summary["strategy_cagr"],
                "Max Drawdown":summary["max_drawdown"],
                "Final Value":summary["portfolio_final"],
                "Closed Trades":tstats["closed_trades"],
            },
            {
                "System":strategy_mode_label(nw_scfg.mode),
                "CAGR":nw_summary["strategy_cagr"],
                "Max Drawdown":nw_summary["max_drawdown"],
                "Final Value":nw_summary["portfolio_final"],
                "Closed Trades":nw_tstats["closed_trades"],
            },
            {
                "System":"Buy & Hold",
                "CAGR":nw_summary["buyhold_cagr"],
                "Max Drawdown":nw_summary["buyhold_max_drawdown"],
                "Final Value":nw_summary["buyhold_final"],
                "Closed Trades":0,
            },
        ])
        st.dataframe(
            compare.style.format({"CAGR":"{:.2%}","Max Drawdown":"{:.2%}","Final Value":"{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

        with st.expander("NW Strategy Trade Ledger", expanded=False):
            st.dataframe(nw_trades.sort_values("Entry Date", ascending=False) if len(nw_trades) else nw_trades, use_container_width=True, hide_index=True, height=460)

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
            use_container_width=True,
        )


with tabs[3]:
    if not dema_enabled_used or dema_result is None:
        st.info("DEMA-MACD confirmation layer was disabled for this run. Enable it in the sidebar and run the analysis again.")
    else:
        st.markdown("#### Full-Cycle DEMA-MACD BUY / SELL Confirmation")
        st.markdown(
            "<div class='section-note'>The reference continuation concept is preserved as an auditable momentum layer, while the MK engine adds a separately calibrated lifecycle. "
            "SELL is not a simple inverse BUY: deterioration score, persistence, DEMA trend structure, standard MACD confirmation, ADX/regime filters, Nadaraya-Watson context when enabled, ATR trailing risk, hard stop and swing-low breach are evaluated independently. "
            "Executable actions use the prior completed bar and the next adjusted open.</div>", unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("Current Decision",dema_decision["decision"]); c2.metric("Position",dema_decision["position"])
        c3.metric("BUY Score",f"{dema_decision['buy_score']:.1f}"); c4.metric("SELL Score",f"{dema_decision['sell_score']:.1f}")
        c5.metric("Closed Trades",f"{int((dema_tstats or {}).get('closed_trades',0) or 0)}"); c6.metric("Net Exit Utility",fmt_pct(dema_exit_summary.get("avg_net_exit_utility")))
        st.plotly_chart(make_dema_price_chart(dema_result,dema_cfg),use_container_width=True,config=PLOT_CFG)
        left,right=st.columns([1.15,1])
        with left:
            st.markdown("##### Current Decision Gates")
            st.dataframe(dema_decision["gates"],use_container_width=True,hide_index=True,height=390)
        with right:
            st.markdown("##### Position Lifecycle")
            st.dataframe(lifecycle_explanation(),use_container_width=True,hide_index=True,height=390)
        st.plotly_chart(make_dema_oscillator_chart(dema_result),use_container_width=True,config=PLOT_CFG)
        st.plotly_chart(make_dema_equity_chart(dema_result),use_container_width=True,config=PLOT_CFG)
        st.markdown("##### DEMA-MACD Event Ledger")
        st.dataframe(dema_events.sort_values("Date",ascending=False) if dema_events is not None and not dema_events.empty else pd.DataFrame(),use_container_width=True,hide_index=True,height=430)
        e1,e2=st.columns(2)
        with e1:
            st.markdown("##### DEMA-MACD Trade Ledger")
            st.dataframe(dema_trades.sort_values("Entry Date",ascending=False) if dema_trades is not None and not dema_trades.empty else pd.DataFrame(),use_container_width=True,hide_index=True,height=430)
        with e2:
            st.markdown("##### SELL / Exit Quality")
            st.dataframe(dema_exit_detail.sort_values("Exit Date",ascending=False) if dema_exit_detail is not None and not dema_exit_detail.empty else pd.DataFrame(),use_container_width=True,hide_index=True,height=430)
        if dema_calibration:
            st.markdown("#### Calibration Lab — Robustness Before Peak Return")
            sp=dema_calibration["splits"]; bp=dema_calibration["best_params"]
            st.caption(f"Train {sp['train_start'].date()} → {sp['train_end'].date()} | Validation {sp['validation_start'].date()} → {sp['validation_end'].date()} | Untouched OOS {sp['oos_start'].date()} → {sp['oos_end'].date()}")
            q1,q2,q3,q4,q5=st.columns(5)
            q1.metric("Best Fast",bp["fast_length"]); q2.metric("Best Slow",bp["slow_length"]); q3.metric("Best Signal",bp["signal_length"]); q4.metric("BUY Threshold",f"{bp['buy_threshold']:.0f}"); q5.metric("SELL Threshold",f"{bp['sell_threshold']:.0f}")
            st.plotly_chart(make_dema_calibration_chart(dema_calibration),use_container_width=True,config=PLOT_CFG)
            st.markdown("##### Top Parameter Sets")
            st.dataframe(dema_calibration["ranking"].head(30),use_container_width=True,hide_index=True,height=520)
            st.markdown("##### Expanding Walk-Forward")
            st.dataframe(dema_calibration["walk_forward"],use_container_width=True,hide_index=True,height=360)
            st.caption(dema_calibration["selection_note"])
        st.markdown("#### DEMA-MACD Export")
        dema_html=build_dema_macd_html_report(dema_result,ticker_used,interval_used,dema_decision,dema_summary,dema_tstats,dema_events,dema_trades,dema_exit_detail,dema_exit_summary,dema_calibration)
        dcol1,dcol2=st.columns(2)
        dcol1.download_button("Export DEMA-MACD Interactive HTML",dema_html.encode("utf-8"),file_name=f"MK_DEMA_MACD_{ticker_used}_{interval_used}_v007.html",mime="text/html",use_container_width=True)
        dcol2.download_button("Export DEMA-MACD Ledger CSV",dema_result.reset_index().to_csv(index=False).encode("utf-8"),file_name=f"MK_DEMA_MACD_{ticker_used}_{interval_used}_v007.csv",mime="text/csv",use_container_width=True)

with tabs[4]:
    st.plotly_chart(make_equity_chart(result, entry_gate_label_used), use_container_width=True, config=PLOT_CFG)
    e1,e2,e3,e4 = st.columns(4)
    total_strategy = summary["portfolio_final"] / cfg.initial_capital - 1.0
    total_bh = summary["buyhold_final"] / cfg.initial_capital - 1.0
    e1.metric("Total Strategy Return", fmt_pct(total_strategy))
    e2.metric("Total Buy & Hold Return", fmt_pct(total_bh))
    e3.metric("Cumulative Excess", fmt_pct(total_strategy-total_bh))
    e4.metric("CAGR Spread", fmt_pct(summary["strategy_cagr"]-summary["buyhold_cagr"]))

    gate_state = effective_gate_state(result, entry_lookback_used)
    events = latest_execution_events(result)
    longest_cash = longest_cash_regime(result)

    st.markdown("#### Entry Gate & Cash-Regime Diagnostics")
    g1,g2,g3,g4,g5 = st.columns(5)
    g1.metric("Entry Lookback", f"{entry_lookback_used:,} obs")
    g2.metric("Effective Gate", "ALL-HISTORY HIGH" if gate_state["effective_all_history"] else "ROLLING HIGH")
    g3.metric("Latest Entry Threshold", fmt_num(gate_state["latest_entry_gate"]))
    g4.metric("Gap to Entry Gate", fmt_pct(gate_state["gap_to_entry_gate"]))
    g5.metric("Last Executed BUY", events["last_buy"].date().isoformat() if events["last_buy"] is not None else "—")

    if gate_state["effective_all_history"]:
        st.warning(
            f"ENTRY GATE DIAGNOSTIC — The selected lookback ({entry_lookback_used:,}) is at least as long as the "
            f"available dataset ({gate_state['observations']:,} observations). The BUY gate therefore behaves like an "
            "all-history-high breakout rule. A strategy that has sold can remain in CASH until the stock reaches a new historical high."
        )

    if longest_cash is not None:
        st.markdown(
            f"<div class='section-note'><b>Longest cash regime:</b> "
            f"{longest_cash['Start'].date()} → {longest_cash['End'].date()} "
            f"({longest_cash['Observations']:,} observations / {longest_cash['Calendar Days']:,} calendar days). "
            f"During that interval the strategy portfolio is flat by construction because Shares = 0 and Cash is unchanged, "
            f"while the underlying stock itself returned {longest_cash['Underlying Return During Cash']:.2%}.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Cash Regime Ledger", expanded=False):
        cash_df = pd.DataFrame(portfolio_cash_regimes(result))
        if cash_df.empty:
            st.write("No cash regimes in the selected history.")
        else:
            cash_df = cash_df.sort_values("Start", ascending=False)
            cash_df["Underlying Return During Cash"] = cash_df["Underlying Return During Cash"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )
            st.dataframe(cash_df, use_container_width=True, hide_index=True)

with tabs[5]:
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

with tabs[6]:
    st.plotly_chart(make_trend_diagnostics(result), use_container_width=True, config=PLOT_CFG)
    st.markdown(
        "<div class='section-note'>All three legacy thresholds are shown simultaneously for diagnosis. "
        "Only the strategy selected in the sidebar controls the exit decision. Legacy ATR is reproduced exactly; "
        "it is not silently replaced by a conventional modern ATR-price stop.</div>",
        unsafe_allow_html=True,
    )

with tabs[7]:
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

with tabs[8]:
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
    st.dataframe(filt, use_container_width=True, hide_index=True, height=580)
    st.caption("This is a curated convenience universe, not an exhaustive exchange constituent list. Manual Yahoo ticker input remains available for instruments outside the list.")

with tabs[10]:
    st.markdown("""
### DEMA-MACD BUY / SELL confirmation layer
The v0.07 module is an independent Python implementation of the publicly described **DEMA MACD BUY signal confirmation** concept and does not redistribute the Pine source verbatim. The MK extension deliberately separates candidate momentum from portfolio action. The lifecycle is **WAIT / CASH → BUY WATCH → BUY → HOLD → SELL WATCH → REDUCE → SELL / RISK EXIT**.

**SELL governance is asymmetric by design.** A bearish DEMA-MACD event alone does not automatically become an exit. SELL strength aggregates bearish cross recency, negative signal slope, histogram deterioration, standard-MACD confirmation, DEMA trend structure, regime filters and optional NW confirmation. Capital-protection overrides additionally evaluate ATR trailing stop, hard stop and prior swing-low breach. REDUCE is advisory by default and can be enabled as a separately governed partial next-open execution; the validated Legacy Fidelity engine remains all-in/all-out and is never modified.

### Calibration governance
Calibration searches parameter neighborhoods rather than selecting one historical peak. Candidates are ranked on training data, validation is used only among a train shortlist, a final OOS segment is untouched by selection, and expanding walk-forward folds re-select parameters using prior data only. Parameter-plateau scoring favors stable neighboring configurations.

### Nadaraya-Watson Trend research layer
The NW module independently implements the public QuantAlgo methodology: a **one-sided causal endpoint estimator** using only the current and historical bars; Gaussian, Rational Quadratic, Epanechnikov, Triangular, Quartic and Cosine kernels; an effective bandwidth `h = bandwidth × multiplier`; and optional residual bands based on a kernel-weighted mean absolute residual around the current NW estimate.

Trend direction is the bar-to-bar slope of the NW estimate. A bullish reversal is a non-positive → positive slope transition; a bearish reversal is a non-negative → negative transition. Because only past/current observations are used, appended future bars cannot rewrite a historical NW estimate.

**Attribution:** methodology reference is *Nadaraya-Watson Trend [QuantAlgo]* on TradingView. The Pine source is not redistributed verbatim in this project. Numeric MK presets and the portfolio strategy rules are our own research layer.

### MK causal NW strategy
The indicator and the strategy are deliberately separated. The strategy reads a signal only from a **completed prior bar** and executes any trade at the **next adjusted open**. `MK Confirmed NW Trend` requires persistent bullish NW direction plus price above the path, with an optional upper-residual-band chase filter; exits occur when price loses the path or bearish direction confirms. `Public-Methodology Reversal Translation` is a simpler research translation built around bullish/bearish slope reversals and price relative to the NW path. Neither is presented as a QuantAlgo trading recommendation.

""")
    st.markdown(f"""
### Entry-gate governance
The engine field historically named `max_buy_weeks` is actually an **observation-count lookback**.  
- **Frequency-Aware (default):** 12M maps to 252 daily, 52 weekly, or 12 monthly observations.
- **Legacy Exact:** preserves the original 2000-observation rule.
- **Custom:** directly specifies the observation count.

For a recently listed stock, a 2000-observation lookback can exceed the entire available history. In that case the entry gate becomes an effective **all-history-high breakout** and long cash plateaus are expected after an exit.

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
    file_name=f"MK_Trend_Following_{ticker_used}_{interval_used}_v006.html",
    mime="text/html",
    use_container_width=True,
)

st.caption("MK FinTECH LabGEN @2026 ATELIER ISTANBUL  |  By Murat Konuklar")
