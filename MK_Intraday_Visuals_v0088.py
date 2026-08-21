"""Plotly visuals for MK Intraday Tactical Lab v0.08.8.1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _split_regime_path(df: pd.DataFrame):
    trend = pd.to_numeric(df.get("NWTrend"), errors="coerce") if "NWTrend" in df.columns else pd.Series(np.nan, index=df.index)
    direction = pd.to_numeric(df.get("NWDirection"), errors="coerce").fillna(0).astype(int) if "NWDirection" in df.columns else pd.Series(0, index=df.index)
    bull = pd.Series(np.nan, index=df.index)
    bear = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        if not np.isfinite(trend.iloc[i]) or not np.isfinite(trend.iloc[i-1]):
            continue
        target = bull if direction.iloc[i] > 0 else bear if direction.iloc[i] < 0 else None
        if target is not None:
            target.iloc[i-1] = trend.iloc[i-1]
            target.iloc[i] = trend.iloc[i]
    return bull, bear


def build_intraday_tactical_figure(df: pd.DataFrame, ticker: str, session_label: str):
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.43, 0.16, 0.15, 0.13, 0.13], vertical_spacing=0.030,
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}], [{}], [{}]],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
        name="Adjusted 15m OHLC", increasing_line_color="#334155", decreasing_line_color="#94A3B8",
        increasing_fillcolor="#CBD5E1", decreasing_fillcolor="#E2E8F0",
    ), row=1, col=1)

    if "SessionVWAP" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SessionVWAP"], mode="lines", name="Session VWAP", line=dict(color="#2563EB", width=1.5)), row=1, col=1)
    if "OpeningRangeHigh" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["OpeningRangeHigh"], mode="lines", name="Opening Range High", line=dict(color="#64748B", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["OpeningRangeLow"], mode="lines", name="Opening Range Low", line=dict(color="#64748B", width=1, dash="dot")), row=1, col=1)

    bull, bear = _split_regime_path(df)
    if bull.notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=bull, mode="lines", name="NW Bullish Trend", line=dict(color="#16A34A", width=3)), row=1, col=1)
    if bear.notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=bear, mode="lines", name="NW Bearish Trend", line=dict(color="#DC2626", width=3)), row=1, col=1)

    markers = [
        ("OpeningRangeBreakoutUp", "OR Breakout Up", "triangle-up", "#16A34A", "OpeningRangeHigh"),
        ("OpeningRangeBreakoutDown", "OR Breakout Down", "triangle-down", "#DC2626", "OpeningRangeLow"),
        ("VWAPCrossUp", "VWAP Cross Up", "circle", "#0EA5E9", "SessionVWAP"),
        ("VWAPCrossDown", "VWAP Cross Down", "circle-open", "#7C3AED", "SessionVWAP"),
    ]
    for flag, name, symbol, color, ycol in markers:
        if flag in df.columns and ycol in df.columns:
            mask = df[flag].fillna(False).astype(bool)
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.index[mask], y=df.loc[mask, ycol], mode="markers", name=name,
                    marker=dict(symbol=symbol, size=9, color=color, line=dict(width=1, color="#FFFFFF")),
                ), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#CBD5E1", opacity=0.75), row=2, col=1, secondary_y=False)
    if "SlotRelativeVolume" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SlotRelativeVolume"], mode="lines", name="Same-Slot RVOL", line=dict(color="#EA580C", width=1.4)), row=2, col=1, secondary_y=True)

    if "IntradayRealizedVol" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["IntradayRealizedVol"], mode="lines", name="Rolling Realized Vol", line=dict(color="#0F172A", width=1.4)), row=3, col=1, secondary_y=False)
    if "SessionDrawdownPct" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SessionDrawdownPct"], mode="lines", name="Session Drawdown", line=dict(color="#B91C1C", width=1.2)), row=3, col=1, secondary_y=True)

    if "IntradayConfirmationScore" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["IntradayConfirmationScore"], mode="lines",
            name="Intraday Confirmation Score", line=dict(color="#111827", width=1.9)
        ), row=4, col=1)
        fig.add_hline(y=0, line_width=1, line_color="#CBD5E1", row=4, col=1)
        fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="#86EFAC", row=4, col=1)
        fig.add_hline(y=-50, line_width=1, line_dash="dot", line_color="#FCA5A5", row=4, col=1)

    if "TacticalTargetExposure" in df.columns:
        _target = pd.to_numeric(df["TacticalTargetExposure"], errors="coerce") * 100.0
        fig.add_trace(go.Scatter(
            x=df.index, y=_target, mode="lines", name="TACTICAL TARGET EXPOSURE",
            line_shape="hv", line=dict(color="#0F172A", width=3.0),
            fill="tozeroy", fillcolor="rgba(15,23,42,0.10)",
            hovertemplate="%{x}<br>Target Exposure: %{y:.0f}%<extra></extra>",
        ), row=5, col=1)
        _chg = _target.ne(_target.shift(1)) & _target.notna()
        if len(_chg):
            _chg.iloc[0] = False
        if _chg.any():
            fig.add_trace(go.Scatter(
                x=df.index[_chg], y=_target[_chg], mode="markers", name="Exposure Change",
                marker=dict(size=8, symbol="diamond", color="#0F172A", line=dict(width=1, color="#FFFFFF")),
                hovertemplate="%{x}<br>New Target: %{y:.0f}%<extra></extra>",
            ), row=5, col=1)
        for _tier in [0, 25, 50, 75, 100]:
            fig.add_hline(y=_tier, line_width=0.7, line_dash="dot", line_color="#CBD5E1", row=5, col=1)

    fig.update_layout(
        title=dict(text=f"15-Minute Intraday Tactical Lab — {ticker}<br><sup>{session_label}</sup>", x=0.01, xanchor="left"),
        template="plotly_white", height=1120, hovermode="x unified",
        margin=dict(l=55, r=55, t=100, b=35),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif", size=11, color="#334155"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False, gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="RVOL", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Realized Vol", tickformat=".1%", row=3, col=1, secondary_y=False, gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="Session DD", tickformat=".1%", row=3, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Confirmation", range=[-105, 105], row=4, col=1, gridcolor="#E2E8F0")
    fig.update_yaxes(
        title_text="Target Exposure", range=[-5, 105], tickvals=[0,25,50,75,100],
        ticktext=["0%","25%","50%","75%","100%"], row=5, col=1, gridcolor="#E2E8F0"
    )
    return fig
