"""Decision causality layer for MK Trend Following Analytics Engine v0.02.
By Murat Konuklar

This layer interprets the validated legacy engine. It does not modify legacy signals or portfolio accounting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from MK_Trend_Following_Engine_v001 import EngineConfig


def active_stop_column(cfg: EngineConfig) -> str:
    return {"ATR":"ATR_Stop", "BOLLINGER":"LowerBollinger", "ATR_TRAILING_STOP":"ATRTrailingStop"}[cfg.strategy]


def _f(v, digits=2):
    return "N/A" if v is None or not np.isfinite(float(v)) else f"{float(v):,.{digits}f}"


def decision_snapshot(df: pd.DataFrame, cfg: EngineConfig) -> dict:
    if len(df) < 2:
        raise ValueError("At least two observations are required for decision causality.")

    last = df.iloc[-1]
    prior = df.iloc[-2]
    stop_col = active_stop_column(cfg)

    prior_close = float(prior["AdjCloseCalc"])
    prior_max = float(prior["MaxPrice"])
    prior_stop = float(prior[stop_col]) if pd.notna(prior[stop_col]) else np.nan
    current_close = float(last["AdjCloseCalc"])
    current_stop = float(last[stop_col]) if pd.notna(last[stop_col]) else np.nan
    prior_shares = float(prior["Shares"])
    current_shares = float(last["Shares"])

    breakout = bool(prior_close >= prior_max)
    stop_breach = bool(np.isfinite(prior_stop) and prior_close <= prior_stop)
    raw_trigger = str(last["Signal"]).strip() if pd.notna(last["Signal"]) else ""
    executed_buy = bool(float(last["FirstBuy"]) > 0)
    executed_sell = bool(float(last["FirstSell"]) > 0)

    if executed_buy:
        decision = "BUY"
        position = "INVESTED / LONG"
        rationale = (
            f"The prior completed bar closed at {_f(prior_close)}, at or above its rolling maximum of {_f(prior_max)}. "
            f"The portfolio was in cash, so the breakout trigger converted cash into shares at the current bar's adjusted open of {_f(last['AdjOpen'])}."
        )
    elif executed_sell:
        decision = "SELL"
        position = "CASH"
        rationale = (
            f"The prior completed bar closed at {_f(prior_close)}, at or below the active {stop_col} threshold of {_f(prior_stop)}. "
            f"The portfolio was invested, so the position was liquidated at the current bar's adjusted open of {_f(last['AdjOpen'])}."
        )
    elif current_shares > 0:
        decision = "HOLD"
        position = "INVESTED / LONG"
        if raw_trigger == "BUY":
            rationale = (
                f"The breakout condition remains active because the prior close {_f(prior_close)} is at or above the rolling maximum {_f(prior_max)}, "
                "but the legacy portfolio is already fully invested. The model does not pyramid or add to an existing position, so the portfolio remains HOLD."
            )
        else:
            stop_text = _f(prior_stop)
            rationale = (
                f"The portfolio is already invested and the exit gate is not breached: prior close {_f(prior_close)} remains above the active "
                f"{stop_col} threshold {stop_text}. No sell execution is permitted, therefore the position remains HOLD."
            )
    else:
        decision = "WAIT / CASH"
        position = "CASH"
        if raw_trigger == "SELL":
            rationale = (
                f"The stop condition is active, but the portfolio was already in cash. There are no shares to sell, so the system remains WAIT / CASH."
            )
        else:
            rationale = (
                f"The portfolio is in cash and the breakout gate is not satisfied: prior close {_f(prior_close)} is below the rolling maximum {_f(prior_max)}. "
                "No entry is executed, so the system remains WAIT / CASH."
            )

    price_stop_gap = current_close/current_stop - 1.0 if np.isfinite(current_stop) and current_stop != 0 else np.nan
    prior_stop_gap = prior_close/prior_stop - 1.0 if np.isfinite(prior_stop) and prior_stop != 0 else np.nan
    breakout_gap = prior_close/prior_max - 1.0 if prior_max != 0 else np.nan

    gates = pd.DataFrame([
        {
            "Gate": "Breakout / Entry",
            "Rule": "Prior Adj Close >= Prior Rolling Max",
            "Observed": f"{_f(prior_close)} vs {_f(prior_max)}",
            "Status": "TRIGGERED" if breakout else "NOT TRIGGERED",
            "Meaning": "Permits BUY trigger" if breakout else "No new entry trigger",
        },
        {
            "Gate": "Stop / Exit",
            "Rule": f"Prior Adj Close <= Prior {stop_col}",
            "Observed": f"{_f(prior_close)} vs {_f(prior_stop)}",
            "Status": "TRIGGERED" if stop_breach else "NOT TRIGGERED",
            "Meaning": "Permits SELL trigger" if stop_breach else "Exit condition not met",
        },
        {
            "Gate": "Position State",
            "Rule": "Position before current execution",
            "Observed": "LONG" if prior_shares > 0 else "CASH",
            "Status": "INVESTED" if prior_shares > 0 else "CASH",
            "Meaning": "Can sell / cannot buy again" if prior_shares > 0 else "Can buy / cannot sell",
        },
        {
            "Gate": "Execution",
            "Rule": "Prior-bar signal; current adjusted-open execution",
            "Observed": f"Raw trigger: {raw_trigger or 'NONE'}",
            "Status": decision,
            "Meaning": "Final portfolio action",
        },
    ])

    return {
        "decision": decision,
        "position": position,
        "raw_trigger": raw_trigger or "NONE",
        "strategy": cfg.strategy,
        "active_stop_column": stop_col,
        "trigger_bar_date": df.index[-2],
        "execution_bar_date": df.index[-1],
        "prior_close": prior_close,
        "prior_max": prior_max,
        "prior_active_stop": prior_stop,
        "current_close": current_close,
        "current_active_stop": current_stop,
        "price_stop_gap": price_stop_gap,
        "prior_stop_gap": prior_stop_gap,
        "breakout_gap": breakout_gap,
        "breakout_triggered": breakout,
        "stop_triggered": stop_breach,
        "rationale": rationale,
        "gates": gates,
        "legacy_scope_note": "Legacy strategy is all-in/all-out. REDUCE / partial-position sizing is not defined in the original workbook.",
    }


def trade_ledger(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    open_trade=None
    for dt, r in df.iterrows():
        if float(r.get("FirstBuy", 0) or 0) > 0:
            open_trade={
                "Entry Date": pd.Timestamp(dt),
                "Entry Price": float(r["AdjOpen"]),
                "Entry Portfolio": float(r["Portfolio"]),
            }
        elif float(r.get("FirstSell", 0) or 0) > 0 and open_trade is not None:
            exit_price=float(r["AdjOpen"])
            entry_price=open_trade["Entry Price"]
            rows.append({
                "Entry Date": open_trade["Entry Date"],
                "Entry Price": entry_price,
                "Exit Date": pd.Timestamp(dt),
                "Exit Price": exit_price,
                "Holding Days": int((pd.Timestamp(dt)-open_trade["Entry Date"]).days),
                "Trade Return": exit_price/entry_price-1.0,
                "Status": "CLOSED",
            })
            open_trade=None

    if open_trade is not None:
        last_dt=pd.Timestamp(df.index[-1]); last_close=float(df["AdjCloseCalc"].iloc[-1])
        rows.append({
            "Entry Date": open_trade["Entry Date"],
            "Entry Price": open_trade["Entry Price"],
            "Exit Date": pd.NaT,
            "Exit Price": np.nan,
            "Holding Days": int((last_dt-open_trade["Entry Date"]).days),
            "Trade Return": last_close/open_trade["Entry Price"]-1.0,
            "Status": "OPEN / MARK-TO-MARKET",
        })
    return pd.DataFrame(rows)


def trade_statistics(ledger: pd.DataFrame) -> dict:
    if ledger is None or ledger.empty:
        return {"closed_trades":0,"win_rate":np.nan,"avg_trade":np.nan,"best_trade":np.nan,"worst_trade":np.nan,"avg_holding_days":np.nan}
    closed=ledger[ledger["Status"]=="CLOSED"].copy()
    if closed.empty:
        return {"closed_trades":0,"win_rate":np.nan,"avg_trade":np.nan,"best_trade":np.nan,"worst_trade":np.nan,"avg_holding_days":np.nan}
    r=closed["Trade Return"].astype(float)
    return {
        "closed_trades":int(len(closed)),
        "win_rate":float((r>0).mean()),
        "avg_trade":float(r.mean()),
        "best_trade":float(r.max()),
        "worst_trade":float(r.min()),
        "avg_holding_days":float(closed["Holding Days"].mean()),
    }
