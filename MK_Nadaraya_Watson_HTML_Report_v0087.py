"""Standalone institutional HTML report for MK Nadaraya-Watson Trend v0.08.7.
By Murat Konuklar
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from MK_Trend_Following_Engine_v001 import performance_summary
from MK_Trend_Following_Decision_Engine_v002 import trade_ledger, trade_statistics
from MK_Institutional_Risk_Analytics_v0087 import infer_periodicity
from MK_Nadaraya_Watson_Visuals_v0087 import NWVisualConfig, build_nw_price_figure
from MK_Nadaraya_Watson_Trend_v0087 import (
    NWConfig, NWStrategyConfig, nw_decision_snapshot, kernel_weight_profile,
    strategy_mode_label, nw_alert_ledger,
)

RANGE_SELECTOR = dict(buttons=[
    dict(count=1,label="1M",step="month",stepmode="backward"),
    dict(count=3,label="3M",step="month",stepmode="backward"),
    dict(count=6,label="6M",step="month",stepmode="backward"),
    dict(count=1,label="YTD",step="year",stepmode="todate"),
    dict(count=1,label="1Y",step="year",stepmode="backward"),
    dict(count=3,label="3Y",step="year",stepmode="backward"),
    dict(step="all",label="ALL"),
], bgcolor="#FFFFFF", activecolor="#E2E8F0", bordercolor="#CBD5E1", borderwidth=1)


def _layout(fig, title, height=520):
    fig.update_layout(
        title=dict(text=title,x=0.01,xanchor="left",font=dict(size=16,color="#0F172A")),
        template="plotly_white",height=height,hovermode="x unified",
        margin=dict(l=50,r=30,t=80,b=35),
        legend=dict(orientation="h",y=1.03,x=1,xanchor="right"),
        paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, Helvetica, sans-serif",size=11,color="#334155"),
    )


def _price_fig(df, cfg):
    return build_nw_price_figure(
        df, cfg,
        NWVisualConfig(theme="Classic", dark_background=True, show_glow=True,
                       show_residual_bands=True, show_reversal_markers=True,
                       show_band_alerts=True, show_momentum_warnings=True,
                       color_bars_by_trend=True, tint_background_by_trend=False),
        range_selector=RANGE_SELECTOR,
        title=f"Nadaraya-Watson Trend — {cfg.kernel} | Lookback {cfg.lookback} | h={cfg.effective_bandwidth:g}",
    )


def _equity_fig(df):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.78,.22],vertical_spacing=.06)
    fig.add_trace(go.Scatter(x=df.index,y=df["Portfolio"],mode="lines",name="NW Trend Strategy",line=dict(width=1.8,color="#0F172A")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BuyHold"],mode="lines",name="Buy & Hold",line=dict(width=1.2,color="#64748B",dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["NWExposure"],mode="lines",name="Market Exposure",line=dict(width=1.1,color="#475569"),fill="tozeroy",fillcolor="rgba(71,85,105,.10)"),row=2,col=1)
    _layout(fig,"MK Causal NW Strategy vs Buy & Hold",600)
    fig.update_yaxes(title_text="Portfolio Value",row=1,col=1)
    fig.update_yaxes(title_text="Exposure",tickformat=".0%",range=[0,1.02],row=2,col=1)
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False),row=1,col=1)
    return fig


def _state_fig(df):
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    slope=df["NWSlope"]/df["NWTrend"].shift(1)
    fig.add_trace(go.Scatter(x=df.index,y=slope,mode="lines",name="NW normalized slope",line=dict(width=1.3,color="#0F172A")),secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index,y=df["NWBandWidthPct"],mode="lines",name="Residual band width",line=dict(width=1.2,color="#B45309")),secondary_y=True)
    _layout(fig,"NW State Diagnostics — Slope & Residual Dispersion",460)
    fig.update_yaxes(tickformat=".2%",secondary_y=False)
    fig.update_yaxes(tickformat=".1%",secondary_y=True)
    fig.update_xaxes(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False))
    return fig


def _kernel_fig(cfg):
    p=kernel_weight_profile(cfg)
    fig=go.Figure(go.Bar(x=p["Lag"],y=p["NormalizedWeight"],marker_color="#475569",name="Normalized Weight"))
    _layout(fig,f"Kernel Weight Profile — {cfg.kernel}",410)
    fig.update_xaxes(title_text="Lag (bars)");fig.update_yaxes(title_text="Normalized Weight")
    return fig


def _fmt(v, typ="num"):
    if v is None or not np.isfinite(float(v)): return "—"
    if typ=="pct": return f"{float(v):.2%}"
    return f"{float(v):,.2f}"


def build_nw_html_report(df: pd.DataFrame, cfg: NWConfig, scfg: NWStrategyConfig, *, ticker: str, instrument_name: str="", market_label: str="", source_note: str="") -> str:
    summary=performance_summary(df,scfg.initial_capital,periods_per_year=infer_periodicity(df.index)[0])
    decision=nw_decision_snapshot(df,scfg)
    trades=trade_ledger(df); stats=trade_statistics(trades)
    figs=[_price_fig(df,cfg),_equity_fig(df),_state_fig(df),_kernel_fig(cfg)]
    fragments=[f.to_html(full_html=False,include_plotlyjs=True if i==0 else False,config={"displaylogo":False,"responsive":True,"scrollZoom":True}) for i,f in enumerate(figs)]

    gate_html=decision["gates"].to_html(index=False,border=0,classes="data-table",float_format=lambda x:f"{x:.6f}")
    if len(trades):
        trade_html=trades.sort_values("Entry Date",ascending=False).to_html(index=False,border=0,classes="data-table")
    else:
        trade_html="<p>No closed NW trades in selected history.</p>"

    title=f"{instrument_name or ticker} — Nadaraya-Watson Trend Research"
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;font-weight:300;color:#111827;background:#fff;margin:0}}
.wrap{{max-width:1600px;margin:auto;padding:24px 32px 50px}} h1,h2,h3{{font-weight:400}}
.meta,.note{{color:#64748B;font-size:12px;line-height:1.55}} .governance{{border:1px solid #CBD5E1;background:#F8FAFC;padding:12px;margin:15px 0}}
.kpis{{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:8px;margin:18px 0}} .kpi{{border:1px solid #E2E8F0;padding:10px 12px}}
.kpi .l{{font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:.05em}} .kpi .v{{font-size:20px;font-weight:300;margin-top:5px}}
.section{{border-top:1px solid #E2E8F0;padding-top:12px;margin-top:22px}} .data-table{{border-collapse:collapse;width:100%;font-size:11px}}
.data-table th,.data-table td{{border-bottom:1px solid #E2E8F0;padding:7px;text-align:left}} .data-table th{{background:#F8FAFC;font-weight:500}}
@media(max-width:900px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.wrap{{padding:15px}}}}
</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="meta">By Murat Konuklar | MK FinTECH LabGEN @2026 ATELIER ISTANBUL<br>{html.escape(market_label)} | {html.escape(source_note)}</div>
<div class="governance"><b>STRICT DATA GOVERNANCE:</b> Yahoo Finance is the only live market-data source in the parent engine. No synthetic observations, alternate provider, forward-fill/back-fill or silent substitution.</div>
<div class="note"><b>Methodology attribution:</b> independent Python implementation of the public/open-source methodology described by QuantAlgo's TradingView “Nadaraya-Watson Trend” indicator. Pine source is not redistributed verbatim. Numeric MK presets and portfolio rules are separate MK research methodology.</div>
<div class="kpis">
<div class="kpi"><div class="l">Decision</div><div class="v">{html.escape(decision['decision'])}</div></div>
<div class="kpi"><div class="l">NW Regime</div><div class="v">{html.escape(decision['trend_direction'])}</div></div>
<div class="kpi"><div class="l">Price / NW Gap</div><div class="v">{_fmt(decision['price_trend_gap'],'pct')}</div></div>
<div class="kpi"><div class="l">Strategy CAGR</div><div class="v">{_fmt(summary['strategy_cagr'],'pct')}</div></div>
<div class="kpi"><div class="l">Buy & Hold CAGR</div><div class="v">{_fmt(summary['buyhold_cagr'],'pct')}</div></div>
<div class="kpi"><div class="l">Max Drawdown</div><div class="v">{_fmt(summary['max_drawdown'],'pct')}</div></div>
<div class="kpi"><div class="l">Closed Trades</div><div class="v">{stats['closed_trades']}</div></div>
</div>
<div class="section"><h2>Price Structure & Signals</h2>{fragments[0]}</div>
<div class="section"><h2>Decision Causality</h2><p>{html.escape(decision['rationale'])}</p>{gate_html}<p class="note">{html.escape(decision['timing_note'])}</p></div>
<div class="section"><h2>Strategy Performance</h2>{fragments[1]}</div>
<div class="section"><h2>NW State Diagnostics</h2>{fragments[2]}</div>
<div class="section"><h2>Kernel Weight Profile</h2>{fragments[3]}</div>
<div class="section"><h2>Trade Ledger</h2>{trade_html}</div>
<div class="section"><h2>Research Methodology</h2><p>Kernel: {html.escape(cfg.kernel)} | Lookback: {cfg.lookback} | Effective bandwidth: {cfg.effective_bandwidth:g} | Residual band multiplier: {cfg.band_multiplier:g} | Strategy: {html.escape(strategy_mode_label(scfg.mode))}.</p><p class="note">The underlying QuantAlgo publication is an indicator. The portfolio rules in this report are an MK research translation and are not presented as a QuantAlgo recommendation or endorsement.</p></div>
</div></body></html>'''


def write_nw_html_report(df, cfg, scfg, output_path, **kwargs):
    doc=build_nw_html_report(df,cfg,scfg,**kwargs)
    Path(output_path).write_text(doc,encoding="utf-8")
    return doc
