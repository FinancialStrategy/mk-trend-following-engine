"""
MK DEMA-MACD Confirmation HTML Report v0.07
By Murat Konuklar
MK FinTECH LabGEN @2026 ATELIER ISTANBUL

Standalone institutional report for the DEMA-MACD confirmation layer.
No market data is downloaded here; it renders already-validated parent-engine data.
"""
from __future__ import annotations

from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def _fmt(v, kind="num"):
    try:
        x = float(v)
    except Exception:
        return "—"
    if not np.isfinite(x):
        return "—"
    if kind == "pct":
        return f"{x:.2%}"
    if kind == "score":
        return f"{x:.1f}"
    if kind == "money":
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def _table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or len(df) == 0:
        return "<div class='empty'>No rows for this section.</div>"
    x = df.head(max_rows).copy()
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{float(v):,.4f}")
    return x.to_html(index=False, escape=True, border=0, classes="tbl")


def _figure_html(fig: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )


def _price_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["AdjOpen"], high=df["AdjHigh"], low=df["AdjLow"], close=df["AdjCloseCalc"],
        name="Adjusted OHLC", increasing_line_width=1, decreasing_line_width=1,
    ))
    for c, label, width in [("DEMAFast","Fast DEMA",1.2),("DEMASlow","Slow DEMA",1.2),("DEMATrend","Trend DEMA",1.5)]:
        if c in df:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=label, line=dict(width=width)))
    if "BuyMarker" in df:
        m = pd.to_numeric(df["BuyMarker"], errors="coerce")
        sel = m.notna()
        fig.add_trace(go.Scatter(x=df.index[sel], y=m[sel], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=10)))
    if "SellMarker" in df:
        m = pd.to_numeric(df["SellMarker"], errors="coerce")
        sel = m.notna()
        fig.add_trace(go.Scatter(x=df.index[sel], y=m[sel], mode="markers", name="SELL / RISK EXIT", marker=dict(symbol="triangle-down", size=10)))
    if "DEMAReduceMarker" in df:
        m = pd.to_numeric(df["DEMAReduceMarker"], errors="coerce")
        sel = m.notna()
        fig.add_trace(go.Scatter(x=df.index[sel], y=m[sel], mode="markers", name="REDUCE", marker=dict(symbol="diamond", size=8)))
    fig.update_layout(
        template="plotly_white", height=560,
        title=dict(text=f"{html.escape(ticker)} — DEMA-MACD Price Structure & Executed Actions", x=.01, font=dict(size=16)),
        margin=dict(l=55,r=25,t=80,b=55), legend=dict(orientation="h", y=-.15),
        xaxis=dict(rangeslider=dict(visible=False)), yaxis_title="Adjusted Price",
    )
    return fig


def _osc_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.62,.38])
    fig.add_trace(go.Bar(x=df.index, y=df["DEMAHistogram"], name="Histogram"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DEMAMACD"], mode="lines", name="DEMA MACD", line=dict(width=1.4)), row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DEMASignal"], mode="lines", name="Signal", line=dict(width=1.2)), row=1,col=1)
    fig.add_hline(y=0, line_width=1, row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BuyScore"], mode="lines", name="BUY Score", line=dict(width=1.4)), row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SellScore"], mode="lines", name="SELL Score", line=dict(width=1.4)), row=2,col=1)
    fig.update_yaxes(title_text="Momentum", row=1,col=1)
    fig.update_yaxes(title_text="Score 0–100", range=[0,100], row=2,col=1)
    fig.update_layout(template="plotly_white", height=650, title=dict(text="DEMA-MACD Momentum & Confirmation Scores",x=.01,font=dict(size=16)), margin=dict(l=55,r=25,t=75,b=55), legend=dict(orientation="h",y=-.12))
    return fig


def _equity_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Portfolio"],mode="lines",name="DEMA-MACD Strategy",line=dict(width=1.7)))
    fig.add_trace(go.Scatter(x=df.index,y=df["BuyHold"],mode="lines",name="Buy & Hold",line=dict(width=1.2)))
    fig.update_layout(template="plotly_white",height=420,title=dict(text="DEMA-MACD Strategy vs Buy & Hold",x=.01,font=dict(size=16)),margin=dict(l=55,r=25,t=75,b=55),legend=dict(orientation="h",y=-.15),yaxis_title="Portfolio Value")
    return fig


def build_dema_macd_html_report(
    df: pd.DataFrame,
    ticker: str,
    interval: str,
    decision: dict,
    summary: dict,
    trade_stats: dict | None = None,
    event_ledger: pd.DataFrame | None = None,
    trade_ledger: pd.DataFrame | None = None,
    exit_detail: pd.DataFrame | None = None,
    exit_summary: dict | None = None,
    calibration: dict | None = None,
    output_path: str | Path | None = None,
) -> str:
    trade_stats = trade_stats or {}
    exit_summary = exit_summary or {}
    latest = df.iloc[-1]
    gates = decision.get("gates", pd.DataFrame())
    figs = [
        _figure_html(_price_figure(df, ticker), include_js=True),
        _figure_html(_osc_figure(df)),
        _figure_html(_equity_figure(df)),
    ]
    calibration_html = ""
    if calibration:
        ranking = calibration.get("ranking", pd.DataFrame())
        wf = calibration.get("walk_forward", pd.DataFrame())
        splits = calibration.get("splits", {})
        bp = calibration.get("best_params", {})
        calibration_html = f"""
        <section><h2>Calibration & Walk-Forward Validation</h2>
        <div class='note'>{html.escape(calibration.get('selection_note',''))}</div>
        <div class='grid4'>
          <div class='kpi'><b>Best Fast / Slow</b><span>{bp.get('fast_length','—')} / {bp.get('slow_length','—')}</span></div>
          <div class='kpi'><b>Signal Length</b><span>{bp.get('signal_length','—')}</span></div>
          <div class='kpi'><b>BUY Threshold</b><span>{bp.get('buy_threshold','—')}</span></div>
          <div class='kpi'><b>SELL Threshold</b><span>{bp.get('sell_threshold','—')}</span></div>
        </div>
        <p class='micro'>Train: {html.escape(str(splits.get('train_start','')))} → {html.escape(str(splits.get('train_end','')))} | Validation: {html.escape(str(splits.get('validation_start','')))} → {html.escape(str(splits.get('validation_end','')))} | Final OOS: {html.escape(str(splits.get('oos_start','')))} → {html.escape(str(splits.get('oos_end','')))}</p>
        <h3>Top Robust Parameter Sets</h3>{_table(ranking.head(20),20)}
        <h3>Expanding Walk-Forward</h3>{_table(wf,30)}
        </section>"""

    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>MK DEMA-MACD Confirmation — {html.escape(ticker)}</title>
    <style>
    body{{font-family:'Arial Narrow','Helvetica Neue',Arial,sans-serif;font-weight:300;background:#f7f8fa;color:#172033;margin:0}}
    .wrap{{max-width:1680px;margin:auto;padding:22px}} .hero{{background:#111827;color:white;padding:24px 28px;border-radius:12px}}
    .hero h1{{font-weight:300;margin:0 0 6px;font-size:28px}} .hero p{{margin:0;color:#cbd5e1;font-size:13px}}
    section{{background:white;border:1px solid #dce1e8;border-radius:10px;padding:20px 22px;margin:18px 0}}
    h2{{font-weight:300;font-size:21px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}} h3{{font-weight:400;font-size:15px}}
    .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .kpi{{border:1px solid #e2e8f0;padding:12px;border-radius:8px}}
    .kpi b{{display:block;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#64748b}} .kpi span{{display:block;font-size:21px;font-weight:300;margin-top:4px}}
    .note{{border-left:3px solid #64748b;background:#f8fafc;padding:10px 12px;font-size:12px;line-height:1.5}} .micro{{font-size:11px;color:#64748b}}
    .tbl{{border-collapse:collapse;width:100%;font-size:11px}} .tbl th{{text-align:left;background:#f1f5f9;border-bottom:1px solid #cbd5e1;padding:7px;position:sticky;top:0}} .tbl td{{padding:6px 7px;border-bottom:1px solid #edf2f7}} .empty{{font-size:12px;color:#64748b}}
    .scroll{{overflow:auto;max-height:520px}} footer{{text-align:center;color:#64748b;font-size:11px;padding:25px}}
    @media(max-width:900px){{.grid4{{grid-template-columns:1fr 1fr}}}}
    </style></head><body><div class='wrap'>
    <div class='hero'><h1>MK DEMA-MACD Confirmation & Calibration Engine v0.07</h1><p>By Murat Konuklar | MK FinTECH LabGEN @2026 ATELIER ISTANBUL | {html.escape(ticker)} | {html.escape(interval)}</p></div>
    <section><h2>Current Decision</h2><div class='grid4'>
      <div class='kpi'><b>Decision</b><span>{html.escape(str(decision.get('decision','—')))}</span></div>
      <div class='kpi'><b>Position</b><span>{html.escape(str(decision.get('position','—')))}</span></div>
      <div class='kpi'><b>BUY Score</b><span>{_fmt(decision.get('buy_score'), 'score')}</span></div>
      <div class='kpi'><b>SELL Score</b><span>{_fmt(decision.get('sell_score'), 'score')}</span></div>
    </div><div class='note' style='margin-top:12px'>{html.escape(str(decision.get('rationale','')))}</div></section>
    <section><h2>Performance & Exit Quality</h2><div class='grid4'>
      <div class='kpi'><b>Strategy CAGR</b><span>{_fmt(summary.get('strategy_cagr'),'pct')}</span></div>
      <div class='kpi'><b>Max Drawdown</b><span>{_fmt(summary.get('max_drawdown'),'pct')}</span></div>
      <div class='kpi'><b>Closed Trades</b><span>{int(trade_stats.get('closed_trades',0) or 0)}</span></div>
      <div class='kpi'><b>Net Exit Utility</b><span>{_fmt(exit_summary.get('avg_net_exit_utility'),'pct')}</span></div>
    </div></section>
    <section><h2>Price & Lifecycle</h2>{figs[0]}</section>
    <section><h2>Momentum Diagnostics</h2>{figs[1]}</section>
    <section><h2>Strategy Equity</h2>{figs[2]}</section>
    <section><h2>Decision Gates</h2><div class='scroll'>{_table(gates,60)}</div></section>
    <section><h2>Signal Event Ledger</h2><div class='scroll'>{_table(event_ledger,100)}</div></section>
    <section><h2>Trade Ledger</h2><div class='scroll'>{_table(trade_ledger,100)}</div></section>
    <section><h2>SELL / Exit Quality Diagnostics</h2><div class='scroll'>{_table(exit_detail,100)}</div></section>
    {calibration_html}
    <section><h2>Methodology & Governance</h2><div class='note'>This module independently implements the publicly described DEMA-MACD continuation-confirmation idea and extends it into a separate MK lifecycle with BUY WATCH, BUY, HOLD, SELL WATCH, REDUCE, SELL and RISK EXIT. Executable actions are based on the prior completed bar and occur at the next adjusted open. Calibration reserves a final out-of-sample holdout and adds expanding walk-forward refits. No market-data fallback, synthetic observations, or price filling is introduced by this module.</div></section>
    <footer>MK FinTECH LabGEN @2026 ATELIER ISTANBUL | By Murat Konuklar | Research / analytics tool; not investment advice.</footer>
    </div></body></html>"""
    if output_path:
        Path(output_path).write_text(doc, encoding="utf-8")
    return doc
