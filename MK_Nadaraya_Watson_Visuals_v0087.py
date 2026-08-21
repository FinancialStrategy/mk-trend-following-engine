"""QuantAlgo-style Nadaraya-Watson visual layer v0.08.7.
By Murat Konuklar

This module reproduces the publicly documented visual semantics of QuantAlgo's
Nadaraya-Watson Trend without redistributing Pine source code:
- slope-coloured bullish/bearish NW path
- layered glow / gradient-like visual depth
- residual envelope
- reversal markers located below/above the trend path
- upper/lower band-cross alert markers
- optional trend-coloured bars/background

MK Momentum Upward / Downward warnings are additive and explicitly separate
from QuantAlgo's five publicly documented built-in alert families.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go


THEMES = {
    "Classic": {"bull":"#00E676","bear":"#FF1744","neutral":"#94A3B8","band":"#A3A3A3","upwarn":"#22D3EE","downwarn":"#F59E0B"},
    "Aqua": {"bull":"#00E5FF","bear":"#FF6E80","neutral":"#94A3B8","band":"#64748B","upwarn":"#67E8F9","downwarn":"#FDBA74"},
    "Cosmic": {"bull":"#8B5CF6","bear":"#EC4899","neutral":"#94A3B8","band":"#7C3AED","upwarn":"#22D3EE","downwarn":"#F59E0B"},
    "Cyber": {"bull":"#00FFB3","bear":"#FF3D71","neutral":"#64748B","band":"#00B8D9","upwarn":"#00E5FF","downwarn":"#FFAB00"},
    "Neon": {"bull":"#39FF14","bear":"#FF073A","neutral":"#94A3B8","band":"#A3A3A3","upwarn":"#00FFFF","downwarn":"#FFD700"},
    "Institutional Light": {"bull":"#0F766E","bear":"#B91C1C","neutral":"#64748B","band":"#94A3B8","upwarn":"#0369A1","downwarn":"#B45309"},
}


@dataclass(frozen=True)
class NWVisualConfig:
    theme: str = "Classic"
    dark_background: bool = True
    show_glow: bool = True
    show_residual_bands: bool = True
    show_reversal_markers: bool = True
    show_band_alerts: bool = True
    show_momentum_warnings: bool = True
    color_bars_by_trend: bool = True
    tint_background_by_trend: bool = False
    custom_bull: str = "#00E676"
    custom_bear: str = "#FF1744"

    def palette(self):
        if self.theme == "Custom":
            p = dict(THEMES["Classic"])
            p["bull"] = self.custom_bull
            p["bear"] = self.custom_bear
            return p
        return dict(THEMES.get(self.theme, THEMES["Classic"]))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def _masked_ohlc(df, mask):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    return [df[c].where(mask) for c in ["AdjOpen","AdjHigh","AdjLow","AdjCloseCalc"]]


def regime_path_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return slope-coloured NW path series with Pine-style segment timing.

    The color of segment t-1 -> t is determined by NWDirection[t].  To make
    that segment visible in Plotly, the new regime trace contains both the
    previous and current endpoint on each transition.  This avoids the common
    one-bar visual gap created by simply masking NWTrend by current direction.
    Signal values are not modified.
    """
    trend = pd.to_numeric(df["NWTrend"], errors="coerce").astype(float)
    direction = pd.to_numeric(df["NWDirection"], errors="coerce").fillna(0).astype(int)
    bull = pd.Series(np.nan, index=df.index, dtype=float)
    bear = pd.Series(np.nan, index=df.index, dtype=float)
    flat = pd.Series(np.nan, index=df.index, dtype=float)

    vals = trend.to_numpy(float)
    dirs = direction.to_numpy(int)
    for i in range(len(df)):
        if not np.isfinite(vals[i]):
            continue
        target = bull if dirs[i] > 0 else bear if dirs[i] < 0 else flat
        target.iat[i] = vals[i]
        if i > 0 and np.isfinite(vals[i - 1]):
            # Current slope owns the segment from the prior endpoint to here.
            target.iat[i - 1] = vals[i - 1]
    return bull, bear, flat


def _add_regime_background(fig, df, direction, palette, opacity=0.035):
    if len(df) == 0:
        return
    d = pd.Series(direction, index=df.index).fillna(0).astype(int)
    start = 0
    vals = d.to_numpy()
    for i in range(1, len(vals)+1):
        if i == len(vals) or vals[i] != vals[start]:
            state = vals[start]
            if state != 0:
                color = palette["bull"] if state > 0 else palette["bear"]
                x0 = df.index[start]
                x1 = df.index[i-1]
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity, line_width=0, layer="below")
            start = i


def build_nw_price_figure(df, nw_cfg, visual_cfg: NWVisualConfig, range_selector=None, title=None):
    p = visual_cfg.palette()
    dark = bool(visual_cfg.dark_background)
    bg = "#050505" if dark else "#FFFFFF"
    fg = "#E5E7EB" if dark else "#334155"
    grid = "#1F2937" if dark else "#E2E8F0"
    candle_neutral_up = "#CBD5E1" if dark else "#334155"
    candle_neutral_down = "#6B7280" if dark else "#94A3B8"

    fig = go.Figure()
    if visual_cfg.tint_background_by_trend:
        _add_regime_background(fig, df, df["NWDirection"], p)

    if visual_cfg.color_bars_by_trend:
        bull_mask = df["NWDirection"].gt(0)
        bear_mask = df["NWDirection"].lt(0)
        flat_mask = ~(bull_mask | bear_mask)
        for label, mask, color in [
            ("Bullish Trend Bars", bull_mask, p["bull"]),
            ("Bearish Trend Bars", bear_mask, p["bear"]),
            ("Neutral Bars", flat_mask, p["neutral"]),
        ]:
            o,h,l,c = _masked_ohlc(df, mask)
            fig.add_trace(go.Candlestick(
                x=df.index, open=o, high=h, low=l, close=c, name=label,
                increasing_line_color=color, decreasing_line_color=color,
                increasing_fillcolor=_rgba(color,0.25 if dark else 0.12),
                decreasing_fillcolor=_rgba(color,0.62 if dark else 0.25),
                whiskerwidth=0.25,
            ))
    else:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
            name="Adjusted OHLC", increasing_line_color=candle_neutral_up, decreasing_line_color=candle_neutral_down,
            increasing_fillcolor=bg, decreasing_fillcolor=candle_neutral_down, whiskerwidth=0.25,
        ))

    bull, bear, flat = regime_path_series(df)

    if visual_cfg.show_residual_bands:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["NWLower"], mode="lines", name="NW Lower Residual Band",
            line=dict(width=0.85, color=_rgba(p["band"],0.70)),
            hovertemplate="%{x}<br>Lower: %{y:,.4f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["NWUpper"], mode="lines", name="NW Upper Residual Band",
            line=dict(width=0.85, color=_rgba(p["band"],0.70)),
            fill="tonexty", fillcolor=_rgba(p["band"],0.08 if dark else 0.07),
            hovertemplate="%{x}<br>Upper: %{y:,.4f}<extra></extra>"
        ))

    # Layered line widths imitate the public TradingView glow/gradient aesthetic
    # while keeping the actual signal path unchanged.
    if visual_cfg.show_glow:
        for width, alpha in [(12,0.035),(8,0.055),(5,0.09)]:
            fig.add_trace(go.Scatter(x=df.index,y=bull,mode="lines",showlegend=False,hoverinfo="skip",
                                     line=dict(width=width,color=_rgba(p["bull"],alpha)),connectgaps=False))
            fig.add_trace(go.Scatter(x=df.index,y=bear,mode="lines",showlegend=False,hoverinfo="skip",
                                     line=dict(width=width,color=_rgba(p["bear"],alpha)),connectgaps=False))

    fig.add_trace(go.Scatter(
        x=df.index,y=bull,mode="lines",name="NW Bullish Path",
        line=dict(width=2.8,color=p["bull"]),connectgaps=False,
        hovertemplate="%{x}<br>NW Trend: %{y:,.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index,y=bear,mode="lines",name="NW Bearish Path",
        line=dict(width=2.8,color=p["bear"]),connectgaps=False,
        hovertemplate="%{x}<br>NW Trend: %{y:,.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df.index,y=flat,mode="lines",name="NW Flat Path",
        line=dict(width=1.6,color=p["neutral"]),connectgaps=False,
        hovertemplate="%{x}<br>NW Trend: %{y:,.4f}<extra></extra>"
    ))

    if visual_cfg.show_reversal_markers:
        bu = df["NWBullishReversal"].fillna(False).astype(bool)
        bd = df["NWBearishReversal"].fillna(False).astype(bool)
        buy_y = df["NWBullishMarkerY"] if "NWBullishMarkerY" in df else df["NWTrend"]
        sell_y = df["NWBearishMarkerY"] if "NWBearishMarkerY" in df else df["NWTrend"]
        fig.add_trace(go.Scatter(
            x=df.index[bu], y=buy_y[bu], mode="markers", name="Bullish Kernel Reversal",
            marker=dict(symbol="triangle-up",size=11,color=p["bull"],line=dict(width=0.6,color=bg)),
            hovertemplate="%{x}<br>Bullish Kernel Reversal<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[bd], y=sell_y[bd], mode="markers", name="Bearish Kernel Reversal",
            marker=dict(symbol="triangle-down",size=11,color=p["bear"],line=dict(width=0.6,color=bg)),
            hovertemplate="%{x}<br>Bearish Kernel Reversal<extra></extra>"
        ))

    if visual_cfg.show_band_alerts:
        ua = df.get("NWCrossAboveUpper", pd.Series(False,index=df.index)).fillna(False).astype(bool)
        lb = df.get("NWCrossBelowLower", pd.Series(False,index=df.index)).fillna(False).astype(bool)
        fig.add_trace(go.Scatter(
            x=df.index[ua],y=df.loc[ua,"NWSource"],mode="markers",name="Source Cross Above Upper Band",
            marker=dict(symbol="diamond-open",size=9,color=p["downwarn"],line=dict(width=1.5,color=p["downwarn"])),
            hovertemplate="%{x}<br>Source Cross Above Upper Band<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[lb],y=df.loc[lb,"NWSource"],mode="markers",name="Source Cross Below Lower Band",
            marker=dict(symbol="diamond-open",size=9,color=p["upwarn"],line=dict(width=1.5,color=p["upwarn"])),
            hovertemplate="%{x}<br>Source Cross Below Lower Band<extra></extra>"
        ))

    if visual_cfg.show_momentum_warnings:
        mu = df.get("NWMomentumUpwardWarning", pd.Series(False,index=df.index)).fillna(False).astype(bool)
        md = df.get("NWMomentumDownwardWarning", pd.Series(False,index=df.index)).fillna(False).astype(bool)
        uy = df.get("NWMomentumUpMarkerY", df["NWTrend"])
        dy = df.get("NWMomentumDownMarkerY", df["NWTrend"])
        fig.add_trace(go.Scatter(
            x=df.index[mu],y=uy[mu],mode="markers+text",name="Momentum Upward — MK Warning",
            text=["M↑"]*int(mu.sum()),textposition="bottom center",
            textfont=dict(size=9,color=p["upwarn"]),
            marker=dict(symbol="arrow-up",size=10,color=p["upwarn"]),
            hovertemplate="%{x}<br>Momentum Upward warning<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[md],y=dy[md],mode="markers+text",name="Momentum Downward — MK Warning",
            text=["M↓"]*int(md.sum()),textposition="top center",
            textfont=dict(size=9,color=p["downwarn"]),
            marker=dict(symbol="arrow-down",size=10,color=p["downwarn"]),
            hovertemplate="%{x}<br>Momentum Downward warning<extra></extra>"
        ))

    if "FirstBuy" in df and "AdjOpen" in df:
        buys = df["FirstBuy"].fillna(0).gt(0)
        sells = df["FirstSell"].fillna(0).gt(0)
        fig.add_trace(go.Scatter(
            x=df.index[buys],y=df.loc[buys,"AdjOpen"],mode="markers",name="Executed NW BUY",
            marker=dict(symbol="circle",size=7,color="#FFFFFF" if dark else "#111827",line=dict(width=1.5,color=p["bull"])),
            hovertemplate="%{x}<br>Executed NW BUY: %{y:,.4f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[sells],y=df.loc[sells,"AdjOpen"],mode="markers",name="Executed NW SELL",
            marker=dict(symbol="circle",size=7,color="#FFFFFF" if dark else "#111827",line=dict(width=1.5,color=p["bear"])),
            hovertemplate="%{x}<br>Executed NW SELL: %{y:,.4f}<extra></extra>"
        ))

    if title is None:
        title = f"Nadaraya-Watson Trend — {nw_cfg.kernel} | Lookback {nw_cfg.lookback} | h={nw_cfg.effective_bandwidth:g}"
    fig.update_layout(
        title=dict(text=title,x=0.01,xanchor="left",y=0.965,yanchor="top",font=dict(size=15,color=fg),pad=dict(t=4,b=4)),
        template=None,height=720,hovermode="x unified",margin=dict(l=48,r=28,t=128,b=38),
        legend=dict(orientation="h",y=1.03,x=1,xanchor="right",yanchor="bottom",font=dict(color=fg)),
        xaxis_rangeslider_visible=False,paper_bgcolor=bg,plot_bgcolor=bg,
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif",size=11,color=fg),
    )
    fig.update_xaxes(showgrid=False,zeroline=False,color=fg)
    if range_selector is not None:
        rs = dict(range_selector)
        if dark:
            rs.update(bgcolor="#111827",activecolor="#374151",bordercolor="#4B5563",font=dict(color=fg))
        fig.update_xaxes(rangeselector=rs)
    fig.update_yaxes(title_text="Adjusted Price / NW Estimate",gridcolor=grid,zerolinecolor=grid,color=fg)
    return fig
