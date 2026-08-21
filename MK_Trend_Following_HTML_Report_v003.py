"""Institutional interactive HTML report builder for MK Trend Following Engine v0.03.
By Murat Konuklar
"""
from __future__ import annotations
from pathlib import Path
import html, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

from MK_Trend_Following_Engine_v001 import performance_summary, EngineConfig
from MK_Trend_Following_Decision_Engine_v002 import decision_snapshot, trade_ledger, trade_statistics, active_stop_column
from MK_Trend_Following_Risk_Analytics_v003 import rolling_risk_frame, risk_state_snapshot

PLOT_CFG = {"displaylogo":False,"responsive":True,"scrollZoom":True,"modeBarButtonsToRemove":["lasso2d","select2d"]}
FONT = "Arial Narrow, Helvetica Neue, Arial, sans-serif"
RANGE_SELECTOR = dict(
    buttons=[
        dict(count=1,label="1M",step="month",stepmode="backward"),
        dict(count=3,label="3M",step="month",stepmode="backward"),
        dict(count=6,label="6M",step="month",stepmode="backward"),
        dict(count=1,label="YTD",step="year",stepmode="todate"),
        dict(count=1,label="1Y",step="year",stepmode="backward"),
        dict(count=3,label="3Y",step="year",stepmode="backward"),
        dict(step="all",label="ALL"),
    ], x=0, y=1.22, xanchor="left", yanchor="top",
    bgcolor="#FFFFFF", activecolor="#E2E8F0", bordercolor="#CBD5E1", borderwidth=1,
)


def _fig_layout(fig,title,ytitle=None,height=430):
    fig.update_layout(
        title={"text":title,"x":0.01,"xanchor":"left","y":0.955,"yanchor":"top",
               "font":{"size":16,"family":FONT,"color":"#111827"},"pad":{"t":4,"b":4}},
        template="plotly_white",height=height,margin=dict(l=58,r=25,t=118,b=45),
        font=dict(family=FONT,size=11,color="#374151"),hovermode="x unified",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(255,255,255,0)"),
        paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB",zerolinecolor="#D1D5DB",title_text=ytitle or "")
    return fig


def build_figures(df: pd.DataFrame, cfg: EngineConfig):
    stop_col=active_stop_column(cfg)
    f1=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.78,.22],vertical_spacing=.035)
    f1.add_trace(go.Candlestick(x=df.index,open=df["AdjOpen"],high=df["AdjHigh"],low=df["AdjLow"],close=df["AdjCloseCalc"],name="Adjusted OHLC",increasing_line_color="#374151",decreasing_line_color="#9CA3AF"),row=1,col=1)
    f1.add_trace(go.Scatter(x=df.index,y=df["MaxPrice"],mode="lines",name="Rolling Max / Entry Gate",line=dict(width=1.1,color="#6B7280",dash="dot")),row=1,col=1)
    f1.add_trace(go.Scatter(x=df.index,y=df[stop_col],mode="lines",name=f"Active Stop — {stop_col}",line=dict(width=1.5,color="#B45309")),row=1,col=1)
    buys=df[df["FirstBuy"]>0]; sells=df[df["FirstSell"]>0]
    f1.add_trace(go.Scatter(x=buys.index,y=buys["AdjOpen"],mode="markers",name="Executed BUY",marker=dict(symbol="triangle-up",size=11,color="#166534")),row=1,col=1)
    f1.add_trace(go.Scatter(x=sells.index,y=sells["AdjOpen"],mode="markers",name="Executed SELL",marker=dict(symbol="triangle-down",size=11,color="#991B1B")),row=1,col=1)
    f1.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",marker_color="#CBD5E1",opacity=.7),row=2,col=1)
    _fig_layout(f1,"Interactive Price, Entry Gate, Active Stop and Executed Trades",height=610)
    f1.update_yaxes(title_text="Adjusted Price",row=1,col=1); f1.update_yaxes(title_text="Volume",row=2,col=1)
    f1.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False)))
    f1.update_xaxes(rangeslider_visible=False)

    f2=go.Figure()
    f2.add_trace(go.Scatter(x=df.index,y=df["Portfolio"],mode="lines",name="Trend Strategy",line=dict(width=1.8,color="#111827")))
    f2.add_trace(go.Scatter(x=df.index,y=df["BuyHold"],mode="lines",name="Buy & Hold",line=dict(width=1.4,color="#6B7280",dash="dot")))
    _fig_layout(f2,"Strategy Equity Curve vs Buy & Hold","Portfolio Value",450)
    f2.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False)))

    dd=df["Portfolio"]/df["Portfolio"].cummax()-1; bhdd=df["BuyHold"]/df["BuyHold"].cummax()-1
    f3=go.Figure()
    f3.add_trace(go.Scatter(x=df.index,y=dd,mode="lines",name="Strategy Drawdown",fill="tozeroy",line=dict(width=1.2,color="#991B1B")))
    f3.add_trace(go.Scatter(x=df.index,y=bhdd,mode="lines",name="Buy & Hold Drawdown",line=dict(width=1.1,color="#6B7280")))
    _fig_layout(f3,"Drawdown and Capital Preservation","Drawdown",390); f3.update_yaxes(tickformat=".0%")
    f3.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False)))

    f4=go.Figure()
    f4.add_trace(go.Scatter(x=df.index,y=df["AdjCloseCalc"],mode="lines",name="Adjusted Close",line=dict(width=1.0,color="#111827")))
    f4.add_trace(go.Scatter(x=df.index,y=df["MaxPrice"],mode="lines",name="Rolling Max",line=dict(width=1.0,color="#64748B",dash="dot")))
    f4.add_trace(go.Scatter(x=df.index,y=df["ATR_Stop"],mode="lines",name="ATR Stop",line=dict(width=1.0,color="#7C3AED")))
    f4.add_trace(go.Scatter(x=df.index,y=df["LowerBollinger"],mode="lines",name="Lower Bollinger",line=dict(width=1.0,color="#2563EB")))
    f4.add_trace(go.Scatter(x=df.index,y=df["ATRTrailingStop"],mode="lines",name="ATR Trailing Stop",line=dict(width=1.5,color="#B45309")))
    _fig_layout(f4,"Trend Diagnostics — All Legacy Thresholds","Adjusted Price / Threshold",450)
    f4.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False)))
    rolling, spec = rolling_risk_frame(df)
    risk_state = risk_state_snapshot(df, rolling, spec)

    f5=make_subplots(specs=[[{"secondary_y":True}]])
    f5.add_trace(go.Scatter(
        x=df.index,y=rolling["AssetRollingReturn"],mode="lines",
        name=f"{spec.label} Asset Return",line=dict(width=1.5,color="#111827")
    ),secondary_y=False)
    f5.add_trace(go.Scatter(
        x=df.index,y=rolling["AssetAnnualizedVolatility"],mode="lines",
        name=f"{spec.label} Ann. Asset Volatility",line=dict(width=1.4,color="#B45309")
    ),secondary_y=True)
    _fig_layout(f5,"Underlying Asset Rolling Return & Volatility",height=430)
    f5.update_yaxes(title_text="Underlying Rolling Return",tickformat=".0%",secondary_y=False)
    f5.update_yaxes(title_text="Underlying Annualized Volatility",tickformat=".0%",secondary_y=True)
    f5.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False)))

    f6=make_subplots(specs=[[{"secondary_y":True}]])
    f6.add_trace(go.Scatter(
        x=df.index,y=rolling["StrategyRollingReturn"],mode="lines",
        name=f"{spec.label} Strategy Return",line=dict(width=1.5,color="#334155")
    ),secondary_y=False)
    f6.add_trace(go.Scatter(
        x=df.index,y=rolling["StrategyAnnualizedVolatility"],mode="lines",
        name=f"{spec.label} Ann. Strategy Volatility",line=dict(width=1.4,color="#7C3AED")
    ),secondary_y=True)
    _fig_layout(f6,"Trend Strategy Rolling Return & Volatility",height=410)
    f6.update_yaxes(title_text="Strategy Rolling Return",tickformat=".0%",secondary_y=False)
    f6.update_yaxes(title_text="Strategy Annualized Volatility",tickformat=".0%",secondary_y=True)
    f6.update_layout(xaxis=dict(rangeselector=RANGE_SELECTOR,rangeslider=dict(visible=False)))

    return [f1,f2,f3,f4,f5,f6], risk_state, spec


def _fmt(v,kind="num"):
    if v is None or (isinstance(v,(float,np.floating)) and not math.isfinite(v)): return "—"
    if kind=="pct": return f"{v:.2%}"
    if kind=="money": return f"{v:,.0f}"
    return f"{v:,.2f}" if isinstance(v,(float,np.floating)) else str(v)


def build_html(df: pd.DataFrame, cfg: EngineConfig, ticker="", instrument_name="", market_label="", source_note="", report_date=None, output_path=None):
    summ=performance_summary(df,cfg.initial_capital)
    dec=decision_snapshot(df,cfg)
    ledger=trade_ledger(df); tstats=trade_statistics(ledger)
    report_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") if report_date is None else str(report_date)
    last=df.iloc[-1]
    total_strategy=summ["portfolio_final"]/cfg.initial_capital-1
    total_bh=summ["buyhold_final"]/cfg.initial_capital-1
    kpis=[
        ("DECISION",dec["decision"]),("POSITION",dec["position"]),("LAST ADJ. CLOSE",_fmt(float(last["AdjCloseCalc"]))),
        ("PRICE / STOP GAP",_fmt(dec["price_stop_gap"],"pct")),("STRATEGY CAGR",_fmt(summ["strategy_cagr"],"pct")),
        ("BUY & HOLD CAGR",_fmt(summ["buyhold_cagr"],"pct")),("MAX DRAWDOWN",_fmt(summ["max_drawdown"],"pct")),
        ("ANNUALIZED VOL",_fmt(summ["annualized_volatility"],"pct")),("CLOSED TRADES",str(tstats["closed_trades"])),
        ("WIN RATE",_fmt(tstats["win_rate"],"pct")),
    ]
    figs,risk_state,risk_spec=build_figures(df,cfg)
    fragments=[f.to_html(full_html=False,include_plotlyjs=False,config=PLOT_CFG) for f in figs]

    gate_rows="".join(
        "<tr>"+"".join(f"<td>{html.escape(str(row[c]))}</td>" for c in ["Gate","Rule","Observed","Status","Meaning"])+"</tr>"
        for _,row in dec["gates"].iterrows()
    )
    trade_rows=""
    if ledger is not None and not ledger.empty:
        for _,r in ledger.iloc[::-1].iterrows():
            vals=[
                pd.Timestamp(r["Entry Date"]).strftime("%Y-%m-%d"),
                _fmt(float(r["Entry Price"])),
                "" if pd.isna(r["Exit Date"]) else pd.Timestamp(r["Exit Date"]).strftime("%Y-%m-%d"),
                "" if pd.isna(r["Exit Price"]) else _fmt(float(r["Exit Price"])),
                str(int(r["Holding Days"])),
                _fmt(float(r["Trade Return"]),"pct"),
                str(r["Status"]),
            ]
            trade_rows += "<tr>"+"".join(f"<td>{html.escape(v)}</td>" for v in vals)+"</tr>"

    tcols=["AdjOpen","AdjHigh","AdjLow","AdjCloseCalc","Signal","ATR_Stop","LowerBollinger","ATRTrailingStop","Portfolio","BuyHold"]
    rows=[]
    for dt,r in df[tcols].iloc[::-1].iterrows():
        vals=[dt.strftime("%Y-%m-%d")]
        for c in tcols:
            v=r[c]
            if c=="Signal": vals.append(str(v or ""))
            elif c in ("Portfolio","BuyHold"): vals.append(f"{float(v):,.2f}" if pd.notna(v) else "")
            else: vals.append(f"{float(v):.6f}" if pd.notna(v) else "")
        rows.append("<tr>"+"".join(f"<td>{html.escape(x)}</td>" for x in vals)+"</tr>")

    table_headers=["Date","Adj Open","Adj High","Adj Low","Adj Close","Signal","ATR Stop","Lower Bollinger","ATR Trailing Stop","Portfolio","Buy & Hold"]
    kpi_html="".join(f'<div class="kpi"><div class="klabel">{html.escape(a)}</div><div class="kvalue">{html.escape(b)}</div></div>' for a,b in kpis)
    methodology=f"""
    <div class='method-grid'>
      <div><b>Data source</b><br>{html.escape(source_note or 'Yahoo Finance via yfinance; strict single-source mode')}</div>
      <div><b>Execution convention</b><br>Prior completed bar determines trigger; trade at current adjusted open</div>
      <div><b>Active strategy</b><br>{html.escape(cfg.strategy.replace('_',' '))}</div>
      <div><b>Legacy scope</b><br>All-in/all-out; REDUCE is not defined</div>
      <div><b>Entry gate</b><br>Prior adjusted close ≥ prior rolling maximum</div>
      <div><b>Exit gate</b><br>Prior adjusted close ≤ active strategy threshold</div>
      <div><b>Legacy window convention</b><br>{'Enabled: inclusive OFFSET-style windows' if cfg.legacy_inclusive_windows else 'Disabled: conventional window length'}</div>
      <div><b>Data governance</b><br>No synthetic data, no price filling, no alternate-source fallback</div>
    </div>"""

    title_name=instrument_name or ticker
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>MK Trend Following Analytics — {html.escape(ticker)}</title>
<style>
:root{{--ink:#111827;--muted:#64748B;--line:#E2E8F0;--soft:#F8FAFC;--accent:#334155;}}
*{{box-sizing:border-box}}body{{margin:0;background:#F3F4F6;color:var(--ink);font-family:{FONT};font-weight:300;letter-spacing:.01em}}
.shell{{max-width:1550px;margin:0 auto;padding:24px}}.hero{{background:#fff;border:1px solid var(--line);padding:26px 30px 22px;border-radius:7px}}
.eyebrow{{font-size:10px;letter-spacing:.16em;color:var(--muted);text-transform:uppercase}}h1{{font-size:28px;font-weight:300;margin:8px 0 6px}}.sub{{font-size:12px;color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(145px,1fr));gap:9px;margin-top:20px}}.kpi{{border-top:2px solid #CBD5E1;background:var(--soft);padding:13px 14px;min-height:72px}}.klabel{{font-size:9px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase}}.kvalue{{font-size:19px;font-weight:300;margin-top:7px}}
.section{{background:#fff;border:1px solid var(--line);border-radius:7px;margin-top:14px;padding:14px 16px 10px}}.section h2{{font-size:15px;font-weight:400;margin:4px 0 12px}}.decision{{border-left:4px solid var(--accent);background:#fff;padding:16px 18px;border:1px solid var(--line);border-left-width:4px}}.decision .big{{font-size:29px;font-weight:300;margin:4px 0 8px}}.decision p{{font-size:12px;line-height:1.55;color:#475569;margin:0}}
.method-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid var(--line);border-left:1px solid var(--line)}}.method-grid>div{{padding:13px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px;line-height:1.45}}.method-grid b{{font-weight:500}}
.controls{{display:flex;gap:10px;align-items:center;margin:8px 0 12px;flex-wrap:wrap}}input,select{{border:1px solid #D1D5DB;background:white;padding:8px 10px;font:300 11px {FONT};min-width:180px}}
.tablewrap{{max-height:520px;overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:10px;white-space:nowrap}}th{{position:sticky;top:0;background:#F3F4F6;text-align:right;padding:8px;border-bottom:1px solid #D1D5DB;font-weight:500;cursor:pointer}}td{{text-align:right;padding:7px 8px;border-bottom:1px solid #F1F5F9}}th:first-child,td:first-child{{text-align:left}}
.gates th,.gates td,.trades th,.trades td{{text-align:left}}.note{{font-size:10px;line-height:1.5;color:var(--muted);margin:8px 2px}}footer{{font-size:10px;color:#9CA3AF;text-align:center;padding:22px}}
@media(max-width:950px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.method-grid{{grid-template-columns:1fr 1fr}}.shell{{padding:10px}}}}@media(max-width:560px){{.method-grid{{grid-template-columns:1fr}}}}
</style><script>{get_plotlyjs()}</script></head><body><div class='shell'>
<div class='hero'><div class='eyebrow'>MK FINTECH LABGEN / INSTITUTIONAL TREND SYSTEMS / v0.02</div><h1>MK Trend Following Analytics Engine</h1><div class='sub'>By Murat Konuklar &nbsp; | &nbsp; {html.escape(title_name)} / {html.escape(ticker)} &nbsp; | &nbsp; {html.escape(market_label)} &nbsp; | &nbsp; {summ['start'].date()} → {summ['end'].date()} &nbsp; | &nbsp; Generated {html.escape(report_date)}</div><div class='kpis'>{kpi_html}</div></div>
<div class='section'><h2>Decision & Causality</h2><div class='decision'><div class='eyebrow'>Portfolio-Level Strategy State</div><div class='big'>{html.escape(dec['decision'])}</div><p>{html.escape(dec['rationale'])}</p></div><div class='tablewrap' style='max-height:300px;margin-top:12px'><table class='gates'><thead><tr><th>Gate</th><th>Rule</th><th>Observed</th><th>Status</th><th>Meaning</th></tr></thead><tbody>{gate_rows}</tbody></table></div><div class='note'>{html.escape(dec['legacy_scope_note'])}</div></div>
<div class='section'><h2>Price & Execution</h2>{fragments[0]}</div>
<div class='section'><h2>Portfolio & Benchmark</h2>{fragments[1]}<div class='note'>Total strategy return: {_fmt(total_strategy,'pct')} &nbsp; | &nbsp; Total Buy & Hold return: {_fmt(total_bh,'pct')} &nbsp; | &nbsp; CAGR spread: {_fmt(summ['strategy_cagr']-summ['buyhold_cagr'],'pct')}</div></div>
<div class='section'><h2>Risk — Drawdown</h2>{fragments[2]}</div>
<div class='section'><h2>Underlying Asset Rolling Risk</h2>{fragments[4]}<div class='note'>Calculated directly from adjusted-close market prices. Window: {html.escape(risk_spec.label)} / {risk_spec.observations} observations. Latest rolling return: {_fmt(risk_state['asset_rolling_return'],'pct')} &nbsp; | &nbsp; Latest annualized volatility: {_fmt(risk_state['asset_annualized_volatility'],'pct')}</div></div>
<div class='section'><h2>Strategy Rolling Risk</h2>{fragments[5]}<div class='note'>Calculated from the strategy Portfolio equity curve. Current position: {html.escape(risk_state['current_position'])} &nbsp; | &nbsp; Full-history cash exposure: {_fmt(risk_state['cash_exposure_ratio'],'pct')}. Flat strategy segments may reflect cash exposure and must not be interpreted as zero volatility in the underlying asset.</div></div>
<div class='section'><h2>Trend Diagnostics</h2>{fragments[3]}</div>
<div class='section'><h2>Trade Ledger</h2><div class='tablewrap'><table class='trades'><thead><tr><th>Entry Date</th><th>Entry Price</th><th>Exit Date</th><th>Exit Price</th><th>Holding Days</th><th>Trade Return</th><th>Status</th></tr></thead><tbody>{trade_rows}</tbody></table></div></div>
<div class='section'><h2>Methodology & Governance</h2>{methodology}<div class='note'>Legacy Fidelity deliberately preserves the original workbook's inclusive OFFSET-style windows and historical quirks for reproducibility. Strategy-state labels are deterministic outputs of the model, not discretionary investment recommendations.</div></div>
<div class='section'><h2>Calculation Ledger</h2><div class='controls'><input id='q' placeholder='Filter any visible value...' oninput='filterRows()'><select id='sig' onchange='filterRows()'><option value=''>All signals</option><option>BUY</option><option>SELL</option></select><span class='note' id='count'></span></div><div class='tablewrap'><table id='ledger'><thead><tr>{''.join(f'<th onclick="sortTable({i})">{h}</th>' for i,h in enumerate(table_headers))}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<footer>MK FinTECH LabGEN @2026 ATELIER ISTANBUL &nbsp; | &nbsp; By Murat Konuklar &nbsp; | &nbsp; Research / analytics tool; not investment advice.</footer></div>
<script>
function filterRows(){{let q=document.getElementById('q').value.toLowerCase(),s=document.getElementById('sig').value,rows=document.querySelectorAll('#ledger tbody tr'),n=0;rows.forEach(r=>{{let text=r.innerText.toLowerCase(),sig=r.children[5].innerText;let ok=(!q||text.includes(q))&&(!s||sig===s);r.style.display=ok?'':'none';if(ok)n++;}});document.getElementById('count').innerText=n+' rows visible';}}
let sortDir={{}};function sortTable(c){{let tb=document.querySelector('#ledger tbody'),rs=Array.from(tb.rows);sortDir[c]=!sortDir[c];rs.sort((a,b)=>{{let x=a.cells[c].innerText.replace(/,/g,''),y=b.cells[c].innerText.replace(/,/g,'');let nx=parseFloat(x),ny=parseFloat(y);let z=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return sortDir[c]?z:-z;}});rs.forEach(r=>tb.appendChild(r));}}filterRows();
</script></body></html>"""
    if output_path:
        Path(output_path).write_text(doc,encoding="utf-8")
    return doc
