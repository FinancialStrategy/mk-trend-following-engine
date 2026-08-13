"""Risk analytics layer for MK Trend Following Analytics Engine v0.03.
Fixes the v0.02 semantic issue where "Rolling Return and Volatility" used the
strategy Portfolio series and could therefore appear flat during cash periods.

By Murat Konuklar
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingWindowSpec:
    label: str
    observations: int
    periods_per_year: int
    frequency_label: str


def infer_periodicity(index: pd.DatetimeIndex) -> tuple[int, str]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252, "Daily"
    gaps = pd.Series(index).diff().dt.days.dropna()
    med = float(gaps.median()) if len(gaps) else 1.0
    if med <= 3:
        return 252, "Daily"
    if med <= 10:
        return 52, "Weekly"
    return 12, "Monthly"


def rolling_window_options(index: pd.DatetimeIndex) -> list[RollingWindowSpec]:
    ppy, freq = infer_periodicity(index)
    if ppy == 252:
        specs = [
            RollingWindowSpec("1M", 21, ppy, freq),
            RollingWindowSpec("3M", 63, ppy, freq),
            RollingWindowSpec("6M", 126, ppy, freq),
            RollingWindowSpec("1Y", 252, ppy, freq),
        ]
    elif ppy == 52:
        specs = [
            RollingWindowSpec("1M", 4, ppy, freq),
            RollingWindowSpec("3M", 13, ppy, freq),
            RollingWindowSpec("6M", 26, ppy, freq),
            RollingWindowSpec("1Y", 52, ppy, freq),
        ]
    else:
        specs = [
            RollingWindowSpec("3M", 3, ppy, freq),
            RollingWindowSpec("6M", 6, ppy, freq),
            RollingWindowSpec("1Y", 12, ppy, freq),
            RollingWindowSpec("2Y", 24, ppy, freq),
        ]
    # Keep at least one valid option without manufacturing observations.
    valid = [s for s in specs if s.observations < len(index)]
    return valid or [RollingWindowSpec(f"{max(2, len(index)//3)} obs", max(2, len(index)//3), ppy, freq)]


def rolling_risk_frame(df: pd.DataFrame, window: int | None = None) -> tuple[pd.DataFrame, RollingWindowSpec]:
    """
    Return rolling risk for BOTH:
      1) the underlying asset (AdjCloseCalc),
      2) the trend strategy portfolio (Portfolio).

    No missing values are filled. Early rolling-window NaNs remain NaN by design.
    """
    if "AdjCloseCalc" not in df or "Portfolio" not in df:
        raise KeyError("AdjCloseCalc and Portfolio are required for rolling risk analytics.")

    opts = rolling_window_options(df.index)
    if window is None:
        spec = next((x for x in opts if x.label == "3M"), opts[0])
    else:
        matches = [x for x in opts if x.observations == int(window)]
        spec = matches[0] if matches else RollingWindowSpec(
            f"{int(window)} obs",
            int(window),
            infer_periodicity(df.index)[0],
            infer_periodicity(df.index)[1],
        )

    win = int(spec.observations)
    ppy = int(spec.periods_per_year)

    asset_price = pd.to_numeric(df["AdjCloseCalc"], errors="coerce")
    strategy_value = pd.to_numeric(df["Portfolio"], errors="coerce")

    # Explicit fill_method=None is intentional: no forward/back fill.
    asset_1p = asset_price.pct_change(fill_method=None)
    strategy_1p = strategy_value.pct_change(fill_method=None)

    out = pd.DataFrame(index=df.index)
    out["AssetRollingReturn"] = asset_price.pct_change(periods=win, fill_method=None)
    out["AssetAnnualizedVolatility"] = asset_1p.rolling(win, min_periods=win).std(ddof=1) * math.sqrt(ppy)
    out["StrategyRollingReturn"] = strategy_value.pct_change(periods=win, fill_method=None)
    out["StrategyAnnualizedVolatility"] = strategy_1p.rolling(win, min_periods=win).std(ddof=1) * math.sqrt(ppy)

    if "Shares" in df:
        out["Invested"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0.0) > 0
    else:
        out["Invested"] = np.nan

    return out, spec


def risk_state_snapshot(df: pd.DataFrame, rolling: pd.DataFrame, spec: RollingWindowSpec) -> dict:
    latest = rolling.iloc[-1]

    invested = bool(float(df["Shares"].iloc[-1]) > 0) if "Shares" in df else None
    shares = pd.to_numeric(df["Shares"], errors="coerce") if "Shares" in df else pd.Series(index=df.index, dtype=float)
    cash_mask = shares.fillna(0.0).le(0.0) if len(shares) else pd.Series(False, index=df.index)
    cash_exposure = float(cash_mask.mean()) if len(cash_mask) else np.nan

    recent = df.tail(spec.observations)
    recent_cash = bool((pd.to_numeric(recent["Shares"], errors="coerce").fillna(0.0) <= 0).all()) if "Shares" in recent else False

    def f(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else np.nan
        except Exception:
            return np.nan

    strategy_flat_reason = None
    if recent_cash:
        strategy_flat_reason = (
            f"The strategy has been fully in cash for the latest {spec.observations} observations. "
            "A constant cash portfolio mathematically produces 0% rolling return and 0% realized volatility. "
            "This is a strategy-exposure result, not a frozen Yahoo price series."
        )

    return {
        "window_label": spec.label,
        "window_observations": spec.observations,
        "periods_per_year": spec.periods_per_year,
        "frequency_label": spec.frequency_label,
        "asset_rolling_return": f(latest["AssetRollingReturn"]),
        "asset_annualized_volatility": f(latest["AssetAnnualizedVolatility"]),
        "strategy_rolling_return": f(latest["StrategyRollingReturn"]),
        "strategy_annualized_volatility": f(latest["StrategyAnnualizedVolatility"]),
        "current_position": "LONG" if invested else "CASH",
        "cash_exposure_ratio": cash_exposure,
        "latest_window_all_cash": recent_cash,
        "strategy_flat_reason": strategy_flat_reason,
    }
