"""
Offline validation for MK DEMA-MACD Confirmation Engine v0.07.
Uses Matplotlib's bundled historical GOOG OHLCV sample as a REAL historical QA fixture.
It is never used as live/fallback market data by the Streamlit application.
No synthetic market observations are generated.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
import matplotlib

from MK_Trend_Following_Engine_v001 import EngineConfig, run_legacy_engine, performance_summary
from MK_Nadaraya_Watson_Trend_v006 import NWConfig, compute_nadaraya_watson
from MK_DEMA_MACD_Confirmation_v007 import (
    DEMACalibrationConfig, compute_dema_macd, run_dema_macd_strategy,
    dema_decision_snapshot, dema_event_ledger, dema_trade_ledger,
    exit_quality_metrics, calibrate_dema_macd, dema_preset,
)
from MK_Trend_Following_Decision_Engine_v002 import trade_statistics
from MK_DEMA_MACD_HTML_Report_v007 import build_dema_macd_html_report

ROOT=Path(__file__).resolve().parent


def load_real_goog_fixture() -> pd.DataFrame:
    path=Path(matplotlib.__file__).resolve().parent / "mpl-data" / "sample_data" / "goog.npz"
    if not path.exists():
        raise FileNotFoundError(f"Matplotlib historical GOOG fixture not found: {path}")
    a=np.load(path,allow_pickle=True)["price_data"]
    return pd.DataFrame({
        "Open":a["open"],"High":a["high"],"Low":a["low"],"Close":a["close"],
        "Volume":a["volume"],"Adj Close":a["adj_close"],
    },index=pd.DatetimeIndex(a["date"].astype("datetime64[ns]"))).sort_index()


def validate() -> dict:
    raw=load_real_goog_fixture()
    if len(raw)!=1047 or raw.index.min()!=pd.Timestamp("2004-08-19") or raw.index.max()!=pd.Timestamp("2008-10-14"):
        raise AssertionError("Historical GOOG QA fixture identity changed unexpectedly")

    base=run_legacy_engine(raw,EngineConfig(initial_capital=100000,max_buy_weeks=252,legacy_inclusive_windows=True))
    nw_cfg=NWConfig(lookback=100,bandwidth=12.0,kernel="Rational Quadratic",relative_weight=1.0,
                    band_multiplier=2.0,source="Adjusted Close",minimum_observations=80)
    nw=compute_nadaraya_watson(base,nw_cfg)

    icfg,scfg=dema_preset("MK Institutional Balanced")
    icfg=replace(icfg,minimum_observations=80,use_nw_filter=True)
    ind=compute_dema_macd(base,icfg,nw)
    strat=run_dema_macd_strategy(base,ind,scfg)
    dec=dema_decision_snapshot(strat,scfg)
    events=dema_event_ledger(strat)
    trades=dema_trade_ledger(strat)
    tstats=trade_statistics(trades)
    exit_detail,exit_summary=exit_quality_metrics(strat)
    perf=performance_summary(strat,scfg.initial_capital)

    # Causal prefix invariance: future rows cannot alter historical indicator values.
    cut=800
    ind_prefix=compute_dema_macd(base.iloc[:cut],icfg,nw.iloc[:cut])
    causal_cols=["DEMAFast","DEMASlow","DEMAMACD","DEMASignal","DEMAHistogram","BuyScore","SellScore"]
    causal_err={}
    for c in causal_cols:
        a=ind.loc[ind_prefix.index,c].to_numpy(float); b=ind_prefix[c].to_numpy(float)
        mask=np.isfinite(a)&np.isfinite(b)
        causal_err[c]=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.0
    if max(causal_err.values())>1e-12:
        raise AssertionError(f"Causal prefix invariance failed: {causal_err}")

    # Next-open execution audit.
    marker_ok=True
    for i in range(1,len(strat)):
        if float(strat["FirstBuy"].iloc[i])>0:
            marker_ok &= bool(np.isclose(float(strat["BuyMarker"].iloc[i]),float(strat["AdjOpen"].iloc[i])))
        if float(strat["FirstSell"].iloc[i])>0:
            marker_ok &= bool(np.isclose(float(strat["SellMarker"].iloc[i]),float(strat["AdjOpen"].iloc[i])))
    if not marker_ok:
        raise AssertionError("Next-open execution marker audit failed")

    observed=set(events["Lifecycle Action"].astype(str)) if not events.empty else set()
    required={"BUY","SELL","SELL WATCH","REDUCE","RISK EXIT"}
    if not required.issubset(observed):
        raise AssertionError(f"SELL lifecycle coverage incomplete. Observed={sorted(observed)}")

    # Calibration: selection must be independent from OOS columns.
    ref_i,ref_s=dema_preset("Reference-Style Continuation")
    ref_i=replace(ref_i,minimum_observations=80)
    t0=time.time()
    cal=calibrate_dema_macd(base,ref_i,ref_s,None,DEMACalibrationConfig(
        grid_depth="Focused",walk_forward_folds=2,minimum_closed_trades=1))
    elapsed=time.time()-t0
    ranking=cal["ranking"]
    expected=ranking.sort_values(["InstitutionalScore","ValidationScore","TrainScore","PlateauScore"],ascending=False).iloc[0]
    if int(expected["CandidateID"])!=int(ranking.iloc[0]["CandidateID"]):
        raise AssertionError("Calibration selection ordering is inconsistent")
    if len(cal["walk_forward"])<1:
        raise AssertionError("Walk-forward validation produced no folds")

    html_doc=build_dema_macd_html_report(strat,"GOOG-QA","1d",dec,perf,tstats,events,trades,exit_detail,exit_summary,cal)
    if "DEMA-MACD Confirmation" not in html_doc or len(html_doc)<10000:
        raise AssertionError("Standalone HTML report generation failed")

    return {
        "fixture_rows":len(raw),"fixture_start":raw.index.min(),"fixture_end":raw.index.max(),
        "causal_max_error":max(causal_err.values()),"next_open_execution":marker_ok,
        "events":len(events),"closed_trades":tstats["closed_trades"],"risk_exits":int((events["Lifecycle Action"]=="RISK EXIT").sum()),
        "sell_events":int((events["Lifecycle Action"]=="SELL").sum()),"reduce_states":int((events["Lifecycle Action"]=="REDUCE").sum()),
        "current_decision":dec["decision"],"strategy_cagr":perf["strategy_cagr"],"strategy_max_dd":perf["max_drawdown"],
        "avg_net_exit_utility":exit_summary["avg_net_exit_utility"],"false_exit_rate":exit_summary["false_exit_rate"],
        "calibration_candidates":len(ranking),"calibration_seconds":elapsed,"best_params":cal["best_params"],"walk_forward_folds":len(cal["walk_forward"]),
    }


if __name__=="__main__":
    r=validate()
    text=f"""MK TREND FOLLOWING ANALYTICS ENGINE v0.07
DEMA-MACD BUY / SELL CONFIRMATION VALIDATION
By Murat Konuklar
============================================================
Historical QA fixture: Matplotlib bundled GOOG OHLCV (real historical sample; never live/fallback data)
Rows: {r['fixture_rows']:,}
Date range: {r['fixture_start'].date()} -> {r['fixture_end'].date()}
Synthetic market observations: NO
Alternate live market-data fallback: NO
DEMA causal prefix invariance: PASS (max error {r['causal_max_error']:.3e})
Prior-bar / next-adjusted-open execution: PASS
BUY/SELL lifecycle coverage: PASS
SELL events: {r['sell_events']}
RISK EXIT events: {r['risk_exits']}
REDUCE states: {r['reduce_states']}
Closed trades: {r['closed_trades']}
Current DEMA state: {r['current_decision']}
Strategy CAGR on QA fixture: {r['strategy_cagr']:.4%}
Strategy Max Drawdown on QA fixture: {r['strategy_max_dd']:.4%}
Avg Net Exit Utility (20-bar proxy): {r['avg_net_exit_utility']:.4%}
False Exit Proxy Rate: {r['false_exit_rate']:.4%}
Focused calibration candidates: {r['calibration_candidates']}
Final OOS used for parameter selection: NO
Walk-forward folds: {r['walk_forward_folds']}
Best robustness-selected params: {r['best_params']}
Standalone DEMA HTML generation: PASS
Overall DEMA v0.07 validation: PASS
"""
    out=ROOT/"LOCAL_VALIDATION_RESULT_v007.txt"
    out.write_text(text,encoding="utf-8")
    print(text)
