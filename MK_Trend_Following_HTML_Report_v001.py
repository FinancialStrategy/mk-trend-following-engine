"""Institutional interactive HTML report builder for MK Trend Following Engine v0.01."""
from __future__ import annotations
from pathlib import Path
import html, json, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from MK_Trend_Following_Engine_v001 import performance_summary, EngineConfig

PLOT_CFG={"displaylogo":False,"responsive":True,"scrollZoom":True,"modeBarButtonsToRemove":["lasso2d","select2d"]}
FONT="Arial Narrow, Helvetica Neue, Arial, sans-serif"

def _fig_layout(fig,title,ytitle=None,height=430):
    fig.update_layout(
        title={"text":title,"x":0.01,"xanchor":"left","font":{"size":16,"family":FONT,"color":"#111827"}},
        template="plotly_white",height=height,margin=dict(l=58,r=25,t=62,b=45),
        font=dict(family=FONT,size=11,color="#374151"),hovermode="x unified",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,bgcolor="rgba(255,255,255,0)"),
        paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=False,rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#E5E7EB",zerolinecolor="#D1D5DB",title_text=ytitle or "")
    return fig

def build_figures(df: pd.DataFrame, cfg: EngineConfig):
    # Price / signal / selected stop
    f1=go.Figure()
    f1.add_trace(go.Candlestick(x=df.index,open=df["AdjOpen"],high=df["AdjHigh"],low=df["AdjLow"],close=df["AdjCloseCalc"],name="Adjusted OHLC",increasing_line_color="#374151",decreasing_line_color="#9CA3AF"))
    f1.add_trace(go.Scatter(x=df.index,y=df["MaxPrice"],mode="lines",name="Rolling Max",line=dict(width=1.1,color="#6B7280")))
    stop_col={"ATR":"ATR_Stop","BOLLINGER":"LowerBollinger","ATR_TRAILING_STOP":"ATRTrailingStop"}[cfg.strategy]
    f1.add_trace(go.Scatter(x=df.index,y=df[stop_col],mode="lines",name=stop_col,line=dict(width=1.4,color="#B45309")))
    buys=df[df["FirstBuy"]>0];sells=df[df["FirstSell"]>0]
    f1.add_trace(go.Scatter(x=buys.index,y=buys["AdjOpen"],mode="markers",name="Executed Buy",marker=dict(symbol="triangle-up",size=11,color="#166534")))
    f1.add_trace(go.Scatter(x=sells.index,y=sells["AdjOpen"],mode="markers",name="Executed Sell",marker=dict(symbol="triangle-down",size=11,color="#991B1B")))
    _fig_layout(f1,"Price, Signal Execution and Active Stop","Adjusted Price",500)
    f1.update_layout(xaxis_rangeslider_visible=False)

    f2=go.Figure()
    f2.add_trace(go.Scatter(x=df.index,y=df["Portfolio"],mode="lines",name="Trend Strategy",line=dict(width=1.8,color="#111827")))
    f2.add_trace(go.Scatter(x=df.index,y=df["BuyHold"],mode="lines",name="Buy & Hold",line=dict(width=1.4,color="#6B7280",dash="dot")))
    _fig_layout(f2,"Portfolio Growth: Trend Strategy vs Buy & Hold","Portfolio Value",430)

    dd=df["Portfolio"]/df["Portfolio"].cummax()-1
    bhdd=df["BuyHold"]/df["BuyHold"].cummax()-1
    f3=go.Figure()
    f3.add_trace(go.Scatter(x=df.index,y=dd,mode="lines",name="Strategy Drawdown",fill="tozeroy",line=dict(width=1.2,color="#991B1B")))
    f3.add_trace(go.Scatter(x=df.index,y=bhdd,mode="lines",name="Buy & Hold Drawdown",line=dict(width=1.1,color="#6B7280")))
    _fig_layout(f3,"Drawdown Diagnostics","Drawdown",360);f3.update_yaxes(tickformat=".0%")

    f4=go.Figure()
    f4.add_trace(go.Scatter(x=df.index,y=df["ATR_Stop"],mode="lines",name="Legacy ATR Stop",line=dict(width=1.2,color="#7C3AED")))
    f4.add_trace(go.Scatter(x=df.index,y=df["LowerBollinger"],mode="lines",name="Lower Bollinger",line=dict(width=1.2,color="#2563EB")))
    f4.add_trace(go.Scatter(x=df.index,y=df["ATRTrailingStop"],mode="lines",name="ATR Trailing Stop",line=dict(width=1.5,color="#B45309")))
    f4.add_trace(go.Scatter(x=df.index,y=df["AdjCloseCalc"],mode="lines",name="Adjusted Close",line=dict(width=1,color="#111827"),opacity=.7))
    _fig_layout(f4,"Trend Stop Diagnostics","Adjusted Price / Stop",420)
    return [f1,f2,f3,f4]

def _fmt(v,kind="num"):
    if v is None or (isinstance(v,float) and not math.isfinite(v)):return "—"
    if kind=="pct":return f"{v:.2%}"
    if kind=="money":return f"{v:,.0f}"
    return f"{v:,.2f}" if isinstance(v,(float,np.floating)) else str(v)

def build_html(df: pd.DataFrame, cfg: EngineConfig, ticker="LEGACY GE", source_note="", report_date=None, output_path=None):
    summ=performance_summary(df,cfg.initial_capital)
    report_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") if report_date is None else str(report_date)
    last=df.iloc[-1]; active={"ATR":"ATR_Stop","BOLLINGER":"LowerBollinger","ATR_TRAILING_STOP":"ATRTrailingStop"}[cfg.strategy]
    active_stop=float(last[active]) if pd.notna(last[active]) else np.nan
    distance=(float(last["AdjCloseCalc"])/active_stop-1) if np.isfinite(active_stop) and active_stop!=0 else np.nan
    kpis=[
        ("LAST ADJ. CLOSE",_fmt(float(last["AdjCloseCalc"]))),
        ("CURRENT SIGNAL",summ["current_signal"]),
        ("STRATEGY CAGR",_fmt(summ["strategy_cagr"],"pct")),
        ("BUY & HOLD CAGR",_fmt(summ["buyhold_cagr"],"pct")),
        ("MAX DRAWDOWN",_fmt(summ["max_drawdown"],"pct")),
        ("ANNUALIZED VOL",_fmt(summ["annualized_volatility"],"pct")),
        ("ACTIVE STOP",_fmt(active_stop)),
        ("PRICE / STOP GAP",_fmt(distance,"pct")),
        ("ROUND-TRIP EXITS",str(summ["first_sells"])),
        ("OBSERVATIONS",f"{summ['observations']:,}"),
    ]
    figs=build_figures(df,cfg)
    fragments=[f.to_html(full_html=False,include_plotlyjs=False,config=PLOT_CFG) for f in figs]
    # table rows
    tcols=["AdjOpen","AdjHigh","AdjLow","AdjCloseCalc","Signal","ATR_Stop","LowerBollinger","ATRTrailingStop","Portfolio","BuyHold"]
    rows=[]
    for dt,r in df[tcols].iloc[::-1].iterrows():
        vals=[dt.strftime("%Y-%m-%d")]
        for c in tcols:
            v=r[c]
            if c=="Signal":vals.append(str(v or ""))
            elif c in ("Portfolio","BuyHold"):vals.append(f"{float(v):,.2f}" if pd.notna(v) else "")
            else: vals.append(f"{float(v):.6f}" if pd.notna(v) else "")
        rows.append("<tr>"+"".join(f"<td>{html.escape(x)}</td>" for x in vals)+"</tr>")
    table_headers=["Date","Adj Open","Adj High","Adj Low","Adj Close","Signal","ATR Stop","Lower Bollinger","ATR Trailing Stop","Portfolio","Buy & Hold"]
    kpi_html="".join(f'<div class="kpi"><div class="klabel">{html.escape(a)}</div><div class="kvalue">{html.escape(b)}</div></div>' for a,b in kpis)
    methodology=f"""
    <div class='method-grid'>
      <div><b>Data source</b><br>{html.escape(source_note or 'Yahoo Finance via yfinance; strict single-source mode')}</div>
      <div><b>Execution convention</b><br>Signal from prior completed bar; trade at current adjusted open</div>
      <div><b>Active strategy</b><br>{html.escape(cfg.strategy.replace('_',' '))}</div>
      <div><b>Legacy window convention</b><br>{'Enabled: inclusive OFFSET-style windows' if cfg.legacy_inclusive_windows else 'Disabled: conventional window length'}</div>
      <div><b>ATR</b><br>{cfg.atr_multiplier:g} × rolling average True Range; parameter {cfg.atr_weeks}</div>
      <div><b>Bollinger</b><br>Population σ, window parameter {cfg.bollinger_weeks}, multiplier {cfg.bollinger_sd:g}</div>
      <div><b>Max-price lookback</b><br>{cfg.max_buy_weeks} parameter; legacy inclusive behavior retained</div>
      <div><b>Data governance</b><br>No synthetic data, no price filling, no alternate-source fallback</div>
    </div>"""
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>MK Trend Following Analytics — {html.escape(ticker)}</title>
<style>
:root{{--ink:#111827;--muted:#6B7280;--line:#E5E7EB;--soft:#F8FAFC;--accent:#B45309;}}
*{{box-sizing:border-box}}body{{margin:0;background:#F3F4F6;color:var(--ink);font-family:{FONT};font-weight:300;letter-spacing:.01em}}
.shell{{max-width:1500px;margin:0 auto;padding:24px}}.hero{{background:#fff;border:1px solid var(--line);padding:26px 30px 22px;border-radius:8px}}
.eyebrow{{font-size:11px;letter-spacing:.16em;color:var(--muted);text-transform:uppercase}}h1{{font-size:28px;font-weight:300;margin:8px 0 6px}}.sub{{font-size:12px;color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(145px,1fr));gap:10px;margin-top:20px}}.kpi{{border-top:2px solid #D1D5DB;background:var(--soft);padding:13px 14px;min-height:72px}}.klabel{{font-size:9px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase}}.kvalue{{font-size:20px;font-weight:300;margin-top:7px}}
.section{{background:#fff;border:1px solid var(--line);border-radius:8px;margin-top:14px;padding:14px 16px 8px}}.section h2{{font-size:15px;font-weight:400;margin:4px 0 12px}}
.method-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid var(--line);border-left:1px solid var(--line)}}.method-grid>div{{padding:13px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px;line-height:1.45}}.method-grid b{{font-weight:500}}
.controls{{display:flex;gap:10px;align-items:center;margin:8px 0 12px;flex-wrap:wrap}}input,select{{border:1px solid #D1D5DB;background:white;padding:8px 10px;font:300 11px {FONT};min-width:180px}}
.tablewrap{{max-height:520px;overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:10px;white-space:nowrap}}th{{position:sticky;top:0;background:#F3F4F6;text-align:right;padding:8px;border-bottom:1px solid #D1D5DB;font-weight:500;cursor:pointer}}th:first-child,td:first-child,th:nth-child(6),td:nth-child(6){{text-align:left}}td{{text-align:right;padding:7px 8px;border-bottom:1px solid #F1F5F9}}
.note{{font-size:10px;line-height:1.5;color:var(--muted);margin:8px 2px}}footer{{font-size:10px;color:#9CA3AF;text-align:center;padding:22px}}
@media(max-width:950px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.method-grid{{grid-template-columns:1fr 1fr}}.shell{{padding:10px}}}}@media(max-width:560px){{.method-grid{{grid-template-columns:1fr}}}}
</style><script>{get_plotlyjs()}</script></head><body><div class='shell'>
<div class='hero'><div class='eyebrow'>MK FINTECH LABGEN / TREND SYSTEMS</div><h1>MK Trend Following Analytics Engine</h1><div class='sub'>By Murat Konuklar &nbsp; | &nbsp; {html.escape(ticker)} &nbsp; | &nbsp; {html.escape(str(summ['start'].date()))} → {html.escape(str(summ['end'].date()))} &nbsp; | &nbsp; Generated {html.escape(report_date)}</div><div class='kpis'>{kpi_html}</div></div>
<div class='section'><h2>Price & Execution</h2>{fragments[0]}</div>
<div class='section'><h2>Portfolio & Benchmark</h2>{fragments[1]}</div>
<div class='section'><h2>Risk</h2>{fragments[2]}</div>
<div class='section'><h2>Trend Diagnostics</h2>{fragments[3]}</div>
<div class='section'><h2>Methodology & Governance</h2>{methodology}<div class='note'>Legacy Fidelity mode deliberately preserves the original workbook's inclusive OFFSET-style windows, including its historical lookback off-by-one behavior. This is retained for reproducibility, not presented as a universal technical-analysis convention.</div></div>
<div class='section'><h2>Calculation Ledger</h2><div class='controls'><input id='q' placeholder='Filter any visible value...' oninput='filterRows()'><select id='sig' onchange='filterRows()'><option value=''>All signals</option><option>BUY</option><option>SELL</option></select><span class='note' id='count'></span></div><div class='tablewrap'><table id='ledger'><thead><tr>{''.join(f'<th onclick="sortTable({i})">{h}</th>' for i,h in enumerate(table_headers))}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<footer>MK FinTECH LabGEN @2026 ATELIER ISTANBUL &nbsp; | &nbsp; By Murat Konuklar &nbsp; | &nbsp; Research / analytics tool; not investment advice.</footer></div>
<script>
function filterRows(){{let q=document.getElementById('q').value.toLowerCase(),s=document.getElementById('sig').value,rows=document.querySelectorAll('#ledger tbody tr'),n=0;rows.forEach(r=>{{let text=r.innerText.toLowerCase(),sig=r.children[5].innerText;let ok=(!q||text.includes(q))&&(!s||sig===s);r.style.display=ok?'':'none';if(ok)n++;}});document.getElementById('count').innerText=n+' rows visible';}}
let sortDir={{}};function sortTable(c){{let tb=document.querySelector('#ledger tbody'),rs=Array.from(tb.rows);sortDir[c]=!sortDir[c];rs.sort((a,b)=>{{let x=a.cells[c].innerText.replace(/,/g,''),y=b.cells[c].innerText.replace(/,/g,'');let nx=parseFloat(x),ny=parseFloat(y);let z=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return sortDir[c]?z:-z;}});rs.forEach(r=>tb.appendChild(r));}}filterRows();
</script></body></html>"""
    if output_path:
        Path(output_path).write_text(doc,encoding="utf-8")
    return doc

if __name__=='__main__':
    import argparse
    from MK_Trend_Following_Engine_v001 import load_legacy_golden_csv,run_legacy_engine
    ap=argparse.ArgumentParser();ap.add_argument('--golden',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    g=load_legacy_golden_csv(args.golden);m=g.rename(columns={'AdjClose':'Adj Close'})[['Open','High','Low','Close','Volume','Adj Close']]
    cfg=EngineConfig();r=run_legacy_engine(m,cfg,validate=False)
    build_html(r,cfg,ticker='GE / LEGACY GOLDEN MASTER',source_note='Original trend-following.xls embedded GE weekly history; no synthetic data; no fallback',output_path=args.out)
    print(args.out)
