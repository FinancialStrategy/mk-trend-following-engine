"""Entry-gate governance and cash-regime diagnostics for v0.05.

The validated legacy engine remains unchanged.
This layer controls how the UI translates an investment horizon into the
engine's existing max_buy_weeks observation-count parameter.

No synthetic data. No market-data fallback. No price filling.

By Murat Konuklar
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


HORIZON_MAP = {
    "15m": {
        "32B": 32,
        "64B": 64,
        "128B": 128,
        "256B": 256,
    },
    "1d": {
        "3M": 63,
        "6M": 126,
        "12M": 252,
        "24M": 504,
        "36M": 756,
    },
    "1wk": {
        "3M": 13,
        "6M": 26,
        "12M": 52,
        "24M": 104,
        "36M": 156,
    },
    "1mo": {
        "3M": 3,
        "6M": 6,
        "12M": 12,
        "24M": 24,
        "36M": 36,
    },
}


def horizon_options(interval: str) -> list[str]:
    if interval not in HORIZON_MAP:
        raise ValueError(f"Unsupported interval: {interval}")
    return list(HORIZON_MAP[interval].keys())


def horizon_to_observations(interval: str, horizon: str) -> int:
    try:
        return int(HORIZON_MAP[interval][horizon])
    except KeyError as exc:
        raise ValueError(f"Unsupported interval/horizon: {interval}/{horizon}") from exc


def resolve_entry_lookback(
    interval: str,
    mode: str,
    horizon: str = "12M",
    custom_observations: int = 252,
) -> tuple[int, str]:
    """
    Returns (observation_count, descriptive_label).

    Modes
    -----
    Frequency-Aware:
        horizon is mapped to observations according to Daily/Weekly/Monthly data.
    Legacy Exact:
        uses 2000 observations exactly, reproducing the old workbook default.
    Custom:
        user supplies the observation count directly.
    """
    if mode == "Frequency-Aware":
        obs = horizon_to_observations(interval, horizon)
        return obs, f"{horizon} frequency-aware breakout ({obs} observations)"
    if mode == "Legacy Exact":
        return 2000, "Legacy exact breakout (2000 observations)"
    if mode == "Custom":
        obs = int(custom_observations)
        if obs < 2:
            raise ValueError("Custom entry lookback must be at least 2 observations.")
        return obs, f"Custom breakout ({obs} observations)"
    raise ValueError(f"Unsupported entry mode: {mode}")


def effective_gate_state(df: pd.DataFrame, lookback: int) -> dict:
    """
    Diagnose whether the configured rolling-max entry rule is effectively
    an all-history-high gate for the selected dataset.
    """
    n = int(len(df))
    effective_all_history = bool(lookback >= n)
    latest_close = float(df["AdjCloseCalc"].iloc[-1])
    latest_gate = float(df["MaxPrice"].iloc[-1])
    gap = latest_close / latest_gate - 1.0 if latest_gate != 0 else np.nan

    return {
        "observations": n,
        "entry_lookback": int(lookback),
        "effective_all_history": effective_all_history,
        "latest_close": latest_close,
        "latest_entry_gate": latest_gate,
        "gap_to_entry_gate": gap,
    }


def portfolio_cash_regimes(df: pd.DataFrame) -> list[dict]:
    """
    Contiguous periods in which the strategy holds zero shares.
    Uses engine output only; no fabricated observations.
    """
    if "Shares" not in df.columns:
        raise KeyError("Shares is required.")
    shares = pd.to_numeric(df["Shares"], errors="coerce")
    mask = shares.fillna(0.0).le(0.0)

    regimes = []
    start = None
    prev = None
    for dt, is_cash in mask.items():
        if bool(is_cash) and start is None:
            start = pd.Timestamp(dt)
        elif not bool(is_cash) and start is not None:
            end = pd.Timestamp(prev)
            regimes.append(_regime_row(df, start, end))
            start = None
        prev = dt
    if start is not None:
        regimes.append(_regime_row(df, start, pd.Timestamp(prev)))
    return regimes


def _regime_row(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    segment = df.loc[start:end]
    return {
        "Start": start,
        "End": end,
        "Calendar Days": int((end - start).days),
        "Observations": int(len(segment)),
        "Portfolio Start": float(segment["Portfolio"].iloc[0]),
        "Portfolio End": float(segment["Portfolio"].iloc[-1]),
        "Underlying Start": float(segment["AdjCloseCalc"].iloc[0]),
        "Underlying End": float(segment["AdjCloseCalc"].iloc[-1]),
        "Underlying Return During Cash": float(
            segment["AdjCloseCalc"].iloc[-1] / segment["AdjCloseCalc"].iloc[0] - 1.0
        ) if len(segment) > 0 else np.nan,
    }


def longest_cash_regime(df: pd.DataFrame) -> dict | None:
    regimes = portfolio_cash_regimes(df)
    if not regimes:
        return None
    return max(regimes, key=lambda x: x["Observations"])


def latest_execution_events(df: pd.DataFrame) -> dict:
    buys = df.index[pd.to_numeric(df["FirstBuy"], errors="coerce").fillna(0).gt(0)]
    sells = df.index[pd.to_numeric(df["FirstSell"], errors="coerce").fillna(0).gt(0)]
    return {
        "last_buy": pd.Timestamp(buys[-1]) if len(buys) else None,
        "last_sell": pd.Timestamp(sells[-1]) if len(sells) else None,
        "buy_count": int(len(buys)),
        "sell_count": int(len(sells)),
    }
