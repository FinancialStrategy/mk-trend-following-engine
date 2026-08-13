"""Risk analytics layer for MK Trend Following Analytics Engine v0.04.

Purpose
-------
1) Underlying market risk is calculated only from AdjCloseCalc.
2) Strategy risk is calculated only from Portfolio.
3) Pure-cash rolling windows are NOT drawn as a misleading 0% horizontal
   strategy-risk line. The true values remain in the calculation frame;
   only display-series values are masked to NaN for charting.
4) Rolling market exposure is shown explicitly.
5) Underlying-price/risk dynamics are checked for impossible flatness.

No market-price filling, no fallback data, no synthetic observations.

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
    valid = [s for s in specs if s.observations < len(index)]
    if valid:
        return valid
    fallback_window = max(2, len(index) // 3)
    return [RollingWindowSpec(f"{fallback_window} obs", fallback_window, ppy, freq)]


def _strict_numeric(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.notna().sum() == 0:
        raise ValueError(f"{name} contains no usable numeric observations.")
    return out


def rolling_risk_frame(
    df: pd.DataFrame,
    window: int | None = None
) -> tuple[pd.DataFrame, RollingWindowSpec]:
    """
    Calculate underlying and strategy rolling risk separately.

    Important:
    - No market-price fill.
    - Early rolling-window NaNs remain NaN.
    - Strategy true calculations remain untouched.
    - Strategy DISPLAY series are masked only for windows that are 100% CASH.
    """
    if "AdjCloseCalc" not in df.columns:
        raise KeyError("AdjCloseCalc is required for underlying rolling risk.")
    if "Portfolio" not in df.columns:
        raise KeyError("Portfolio is required for strategy rolling risk.")
    if "Shares" not in df.columns:
        raise KeyError("Shares is required for exposure-aware strategy risk.")

    opts = rolling_window_options(df.index)
    if window is None:
        spec = next((x for x in opts if x.label == "3M"), opts[0])
    else:
        matches = [x for x in opts if x.observations == int(window)]
        if matches:
            spec = matches[0]
        else:
            ppy, freq = infer_periodicity(df.index)
            spec = RollingWindowSpec(f"{int(window)} obs", int(window), ppy, freq)

    win = int(spec.observations)
    ppy = int(spec.periods_per_year)

    asset_price = _strict_numeric(df["AdjCloseCalc"], "AdjCloseCalc")
    strategy_value = _strict_numeric(df["Portfolio"], "Portfolio")
    shares = pd.to_numeric(df["Shares"], errors="coerce")

    # Strict returns: no implicit forward fill.
    asset_1p = asset_price.pct_change(fill_method=None)
    strategy_1p = strategy_value.pct_change(fill_method=None)

    out = pd.DataFrame(index=df.index)

    # Underlying market risk.
    out["AssetRollingReturn"] = asset_price.pct_change(periods=win, fill_method=None)
    out["AssetAnnualizedVolatility"] = (
        asset_1p.rolling(win, min_periods=win).std(ddof=1) * math.sqrt(ppy)
    )

    # True strategy risk.
    out["StrategyRollingReturn"] = strategy_value.pct_change(periods=win, fill_method=None)
    out["StrategyAnnualizedVolatility"] = (
        strategy_1p.rolling(win, min_periods=win).std(ddof=1) * math.sqrt(ppy)
    )

    # Exposure state. No market-price filling occurs here.
    invested = shares.gt(0)
    out["Invested"] = invested
    out["RollingExposure"] = invested.astype(float).rolling(win, min_periods=win).mean()

    # A pure-cash rolling window means every observation in that completed
    # window has no shares. These windows genuinely produce strategy
    # return/volatility equal to zero. We keep the TRUE values above, but
    # mask only their chart-display versions so the dashboard does not
    # present a misleading continuous zero-risk line.
    cash_flag = (~invested).astype(float)
    out["PureCashWindow"] = cash_flag.rolling(win, min_periods=win).mean().eq(1.0)

    out["StrategyRollingReturnDisplay"] = out["StrategyRollingReturn"].mask(out["PureCashWindow"])
    out["StrategyAnnualizedVolatilityDisplay"] = out["StrategyAnnualizedVolatility"].mask(out["PureCashWindow"])

    return out, spec


def validate_underlying_risk_dynamics(
    df: pd.DataFrame,
    rolling: pd.DataFrame,
    tolerance: float = 1e-12,
) -> dict:
    """
    Detect an impossible presentation state:
    moving underlying prices but an effectively constant underlying
    rolling-return AND rolling-volatility history.

    This does not repair or manufacture data. It only flags the condition.
    """
    price = pd.to_numeric(df["AdjCloseCalc"], errors="coerce").dropna()
    rr = pd.to_numeric(rolling["AssetRollingReturn"], errors="coerce").dropna()
    rv = pd.to_numeric(rolling["AssetAnnualizedVolatility"], errors="coerce").dropna()

    price_unique = int(price.nunique())
    rr_unique = int(rr.nunique())
    rv_unique = int(rv.nunique())

    price_range = float(price.max() - price.min()) if len(price) else np.nan
    rr_range = float(rr.max() - rr.min()) if len(rr) else np.nan
    rv_range = float(rv.max() - rv.min()) if len(rv) else np.nan

    price_moves = bool(price_unique > 3 and np.isfinite(price_range) and price_range > tolerance)
    rr_flat = bool(len(rr) >= 10 and (not np.isfinite(rr_range) or rr_range <= tolerance))
    rv_flat = bool(len(rv) >= 10 and (not np.isfinite(rv_range) or rv_range <= tolerance))

    impossible_flatness = bool(price_moves and rr_flat and rv_flat)

    return {
        "price_unique": price_unique,
        "rolling_return_unique": rr_unique,
        "rolling_vol_unique": rv_unique,
        "price_range": price_range,
        "rolling_return_range": rr_range,
        "rolling_vol_range": rv_range,
        "impossible_flatness": impossible_flatness,
    }


def cash_regimes(rolling: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous pure-cash rolling-window intervals for chart shading."""
    mask = rolling["PureCashWindow"].fillna(False).astype(bool)
    if not mask.any():
        return []

    intervals = []
    start = None
    previous = None
    for dt, is_cash in mask.items():
        if is_cash and start is None:
            start = dt
        elif not is_cash and start is not None:
            intervals.append((start, previous))
            start = None
        previous = dt

    if start is not None:
        intervals.append((start, previous))
    return intervals


def risk_state_snapshot(
    df: pd.DataFrame,
    rolling: pd.DataFrame,
    spec: RollingWindowSpec
) -> dict:
    latest = rolling.iloc[-1]

    shares = pd.to_numeric(df["Shares"], errors="coerce")
    current_invested = bool(shares.iloc[-1] > 0) if len(shares) and pd.notna(shares.iloc[-1]) else False

    valid_position_rows = shares.notna()
    if valid_position_rows.any():
        cash_exposure = float((shares.loc[valid_position_rows] <= 0).mean())
    else:
        cash_exposure = np.nan

    latest_window_all_cash = bool(latest.get("PureCashWindow", False))

    def finite_or_nan(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else np.nan
        except Exception:
            return np.nan

    flat_reason = None
    if latest_window_all_cash:
        flat_reason = (
            f"The strategy was 100% CASH throughout the latest {spec.label} "
            f"({spec.observations}-observation) rolling window. Its true strategy "
            "rolling return and realized volatility may therefore equal 0%. "
            "v0.04 does not draw that cash-only interval as a continuous zero-risk "
            "line; it is marked as a CASH REGIME instead."
        )

    return {
        "window_label": spec.label,
        "window_observations": spec.observations,
        "periods_per_year": spec.periods_per_year,
        "frequency_label": spec.frequency_label,
        "asset_rolling_return": finite_or_nan(latest["AssetRollingReturn"]),
        "asset_annualized_volatility": finite_or_nan(latest["AssetAnnualizedVolatility"]),
        "strategy_rolling_return": finite_or_nan(latest["StrategyRollingReturn"]),
        "strategy_annualized_volatility": finite_or_nan(latest["StrategyAnnualizedVolatility"]),
        "rolling_exposure": finite_or_nan(latest["RollingExposure"]),
        "current_position": "LONG" if current_invested else "CASH",
        "cash_exposure_ratio": cash_exposure,
        "latest_window_all_cash": latest_window_all_cash,
        "strategy_flat_reason": flat_reason,
    }
