"""
MK Nadaraya-Watson Trend Module v0.08.7
By Murat Konuklar

Independent Python reimplementation of the public methodology described by
QuantAlgo's open-source TradingView indicator:
"Nadaraya-Watson Trend [QuantAlgo]".

Important attribution / governance
----------------------------------
- Methodology reference: QuantAlgo, TradingView, Nadaraya-Watson Trend.
- This file is NOT a verbatim redistribution of the Pine Script source.
- It independently implements the publicly described causal endpoint estimator,
  six kernel functions, kernel-weighted absolute-residual bands, slope colouring,
  reversal markers and the five publicly documented QuantAlgo alert conditions.
- MK Momentum Upward / Momentum Downward warnings are an additive causal extension
  based on a turning point in normalized NW slope acceleration; they are explicitly
  not represented as original QuantAlgo alert conditions.
- Market data remains Yahoo Finance only through the parent engine.
- No synthetic market data, alternate provider, price forward-fill/back-fill,
  or centered/future-looking regression is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
import pandas as pd

KernelName = Literal[
    "Gaussian",
    "Rational Quadratic",
    "Epanechnikov",
    "Triangular",
    "Quartic",
    "Cosine",
]
SourceName = Literal["Adjusted Close", "HLC3", "OHLC4"]
StrategyMode = Literal["QUANTALGO_REVERSAL_TRANSLATION", "MK_CONFIRMED_TREND"]

KERNELS: tuple[str, ...] = (
    "Gaussian",
    "Rational Quadratic",
    "Epanechnikov",
    "Triangular",
    "Quartic",
    "Cosine",
)


@dataclass(frozen=True)
class NWConfig:
    """Indicator settings.

    Defaults below are MK Engine research defaults and are NOT represented as
    the creator's exact TradingView preset values.
    """
    lookback: int = 100
    bandwidth: float = 8.0
    bandwidth_multiplier: float = 1.0
    kernel: KernelName = "Gaussian"
    relative_weight: float = 1.0
    band_multiplier: float = 2.0
    source: SourceName = "Adjusted Close"
    minimum_observations: int = 30
    require_full_lookback: bool = True

    def validate(self) -> None:
        if self.lookback < 2:
            raise ValueError("NW lookback must be >= 2")
        if self.bandwidth <= 0:
            raise ValueError("NW bandwidth must be > 0")
        if self.bandwidth_multiplier <= 0:
            raise ValueError("NW bandwidth_multiplier must be > 0")
        if self.kernel not in KERNELS:
            raise ValueError(f"Unsupported NW kernel: {self.kernel}")
        if self.relative_weight <= 0:
            raise ValueError("NW relative_weight must be > 0")
        if self.band_multiplier <= 0:
            raise ValueError("NW band_multiplier must be > 0")
        if self.source not in {"Adjusted Close", "HLC3", "OHLC4"}:
            raise ValueError(f"Unsupported NW source: {self.source}")
        if self.minimum_observations < 3:
            raise ValueError("minimum_observations must be >= 3")

    @property
    def effective_bandwidth(self) -> float:
        return float(self.bandwidth * self.bandwidth_multiplier)


@dataclass(frozen=True)
class NWStrategyConfig:
    """Long/cash causal strategy built on the NW indicator.

    QUANTALGO_REVERSAL_TRANSLATION:
      - Entry: bullish NW slope reversal + source above NW path.
      - Exit: bearish NW slope reversal OR source loses NW path.

    MK_CONFIRMED_TREND:
      - Entry: NW direction has been bullish for confirmation_bars and source
        is above the path; optional upper-band chase filter.
      - Exit: source loses the NW path OR bearish direction persists for
        exit_confirmation_bars.

    All signals are evaluated on the PRIOR completed bar and executed at the
    current adjusted open to prevent look-ahead.
    """
    mode: StrategyMode = "MK_CONFIRMED_TREND"
    confirmation_bars: int = 2
    exit_confirmation_bars: int = 1
    avoid_upper_band_chase: bool = True
    initial_capital: float = 100_000.0

    def validate(self) -> None:
        if self.mode not in {"QUANTALGO_REVERSAL_TRANSLATION", "MK_CONFIRMED_TREND"}:
            raise ValueError(f"Unsupported NW strategy mode: {self.mode}")
        if self.confirmation_bars < 1:
            raise ValueError("confirmation_bars must be >= 1")
        if self.exit_confirmation_bars < 1:
            raise ValueError("exit_confirmation_bars must be >= 1")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")


def kernel_weight(distance: np.ndarray | float, h: float, kernel: str, relative_weight: float = 1.0):
    """Public-methodology kernel weight function.

    h is the effective bandwidth. Compact kernels return zero outside |u|<=1.
    """
    if h <= 0:
        raise ValueError("h must be > 0")
    if relative_weight <= 0:
        raise ValueError("relative_weight must be > 0")

    d = np.asarray(distance, dtype=float)
    u = d / float(h)

    if kernel == "Gaussian":
        w = np.exp(-(d * d) / (2.0 * h * h))
    elif kernel == "Rational Quadratic":
        w = np.power(1.0 + (d * d) / (2.0 * relative_weight * h * h), -relative_weight)
    else:
        inside = np.abs(u) <= 1.0
        w = np.zeros_like(d, dtype=float)
        if kernel == "Epanechnikov":
            w[inside] = 0.75 * (1.0 - u[inside] ** 2)
        elif kernel == "Triangular":
            w[inside] = 1.0 - np.abs(u[inside])
        elif kernel == "Quartic":
            w[inside] = (15.0 / 16.0) * (1.0 - u[inside] ** 2) ** 2
        elif kernel == "Cosine":
            w[inside] = (math.pi / 4.0) * np.cos(math.pi * u[inside] / 2.0)
        else:
            raise ValueError(f"Unsupported NW kernel: {kernel}")

    if np.isscalar(distance):
        return float(np.asarray(w))
    return w


def kernel_weight_profile(config: NWConfig) -> pd.DataFrame:
    """Lag-weight profile used by the selected estimator."""
    config.validate()
    # Pine loop semantics described publicly are inclusive: 0..lookback.
    lags = np.arange(config.lookback + 1, dtype=float)
    raw = kernel_weight(lags, config.effective_bandwidth, config.kernel, config.relative_weight)
    total = float(np.sum(raw))
    norm = raw / total if total > 0 else np.full_like(raw, np.nan)
    return pd.DataFrame({"Lag": lags.astype(int), "RawWeight": raw, "NormalizedWeight": norm})


def _source_series(df: pd.DataFrame, source: str) -> pd.Series:
    required = {"AdjOpen", "AdjHigh", "AdjLow", "AdjCloseCalc"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"NW requires adjusted OHLC columns; missing: {sorted(missing)}")
    if source == "Adjusted Close":
        s = df["AdjCloseCalc"]
    elif source == "HLC3":
        s = (df["AdjHigh"] + df["AdjLow"] + df["AdjCloseCalc"]) / 3.0
    elif source == "OHLC4":
        s = (df["AdjOpen"] + df["AdjHigh"] + df["AdjLow"] + df["AdjCloseCalc"]) / 4.0
    else:
        raise ValueError(f"Unsupported source: {source}")
    return pd.to_numeric(s, errors="raise").astype(float)


def compute_nadaraya_watson(df: pd.DataFrame, config: NWConfig = NWConfig()) -> pd.DataFrame:
    """Causal one-sided endpoint Nadaraya-Watson estimator.

    Each timestamp t uses only source[t], source[t-1], ..., source[t-lookback].
    No future bars can affect historical values.
    """
    config.validate()
    if len(df) < config.minimum_observations:
        raise ValueError(
            f"Only {len(df)} observations; NW minimum is {config.minimum_observations}."
        )
    if config.require_full_lookback and len(df) <= config.lookback:
        raise ValueError(
            f"NW lookback={config.lookback} requires at least {config.lookback + 1} observations; "
            f"only {len(df)} are available. Strict mode will not shorten the requested lookback."
        )

    src = _source_series(df, config.source).to_numpy(float)
    n = len(src)
    full_profile = kernel_weight_profile(config)
    full_w = full_profile["RawWeight"].to_numpy(float)

    trend = np.full(n, np.nan, dtype=float)
    residual = np.full(n, np.nan, dtype=float)

    for t in range(n):
        if config.require_full_lookback and t < config.lookback:
            # Pine-style inclusive 0..lookback loop has unavailable history
            # before the full window exists; preserve that warm-up as NaN.
            continue
        m = min(t, config.lookback) + 1
        # lag=0 is current bar; lag=m-1 is oldest available bar.
        values = src[t - np.arange(m)]
        weights = full_w[:m]
        sw = float(np.sum(weights))
        if sw <= 0:
            continue
        yhat = float(np.dot(weights, values) / sw)
        trend[t] = yhat
        residual[t] = float(np.dot(weights, np.abs(values - yhat)) / sw)

    slope = np.full(n, np.nan, dtype=float)
    slope[1:] = trend[1:] - trend[:-1]
    direction = np.zeros(n, dtype=int)
    direction[np.isfinite(slope) & (slope > 0)] = 1
    direction[np.isfinite(slope) & (slope < 0)] = -1

    bull_rev = np.zeros(n, dtype=bool)
    bear_rev = np.zeros(n, dtype=bool)
    if n > 1:
        bull_rev[1:] = (direction[1:] > 0) & (direction[:-1] <= 0)
        bear_rev[1:] = (direction[1:] < 0) & (direction[:-1] >= 0)

    upper = trend + residual * config.band_multiplier
    lower = trend - residual * config.band_multiplier

    # Publicly documented QuantAlgo band-cross alert conditions.
    cross_above_upper = np.zeros(n, dtype=bool)
    cross_below_lower = np.zeros(n, dtype=bool)
    if n > 1:
        valid_up = np.isfinite(src[1:]) & np.isfinite(upper[1:]) & np.isfinite(src[:-1]) & np.isfinite(upper[:-1])
        valid_dn = np.isfinite(src[1:]) & np.isfinite(lower[1:]) & np.isfinite(src[:-1]) & np.isfinite(lower[:-1])
        cross_above_upper[1:] = valid_up & (src[1:] > upper[1:]) & (src[:-1] <= upper[:-1])
        cross_below_lower[1:] = valid_dn & (src[1:] < lower[1:]) & (src[:-1] >= lower[:-1])

    # MK causal early-momentum warning layer.
    # Normalized slope acceleration changes sign BEFORE the main NW slope itself
    # necessarily reverses. This is a warning, not a replacement for the
    # QuantAlgo bullish/bearish kernel reversal condition.
    normalized_slope = np.full(n, np.nan, dtype=float)
    valid_den = np.isfinite(trend[:-1]) & (np.abs(trend[:-1]) > np.finfo(float).tiny)
    # Explicit positional assignment: avoid chained-index temporary writes.
    valid_pos = np.flatnonzero(valid_den) + 1
    normalized_slope[valid_pos] = slope[valid_pos] / trend[valid_pos - 1]
    slope_accel = np.full(n, np.nan, dtype=float)
    slope_accel[1:] = normalized_slope[1:] - normalized_slope[:-1]
    momentum_up = np.zeros(n, dtype=bool)
    momentum_down = np.zeros(n, dtype=bool)
    if n > 2:
        momentum_up[2:] = (
            np.isfinite(slope_accel[2:]) & np.isfinite(slope_accel[1:-1]) &
            (slope_accel[2:] > 0) & (slope_accel[1:-1] <= 0) & (direction[2:] <= 0)
        )
        momentum_down[2:] = (
            np.isfinite(slope_accel[2:]) & np.isfinite(slope_accel[1:-1]) &
            (slope_accel[2:] < 0) & (slope_accel[1:-1] >= 0) & (direction[2:] >= 0)
        )

    marker_gap = np.maximum(np.nan_to_num(residual, nan=0.0) * 0.35, np.abs(np.nan_to_num(trend, nan=0.0)) * 0.002)

    out = pd.DataFrame(index=df.index)
    out["NWSource"] = src
    out["NWTrend"] = trend
    out["NWResidual"] = residual
    out["NWUpper"] = upper
    out["NWLower"] = lower
    out["NWSlope"] = slope
    out["NWDirection"] = direction
    out["NWBullishReversal"] = bull_rev
    out["NWBearishReversal"] = bear_rev
    out["NWAnyReversal"] = bull_rev | bear_rev
    out["NWCrossAboveUpper"] = cross_above_upper
    out["NWCrossBelowLower"] = cross_below_lower
    out["NWNormalizedSlope"] = normalized_slope
    out["NWSlopeAcceleration"] = slope_accel
    out["NWMomentumUpwardWarning"] = momentum_up
    out["NWMomentumDownwardWarning"] = momentum_down
    out["NWBullishMarkerY"] = trend - marker_gap
    out["NWBearishMarkerY"] = trend + marker_gap
    out["NWMomentumUpMarkerY"] = trend - marker_gap * 1.8
    out["NWMomentumDownMarkerY"] = trend + marker_gap * 1.8
    out["NWPriceGap"] = src / trend - 1.0
    band_width = upper - lower
    out["NWBandWidthPct"] = np.where(trend != 0, band_width / trend, np.nan)
    out["NWAbovePath"] = src > trend
    out["NWBelowPath"] = src < trend
    out["NWAboveUpperBand"] = src > upper
    out["NWBelowLowerBand"] = src < lower
    return out



def nw_alert_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Return a chronological alert tape from already-computed causal NW columns.

    The first five alert names mirror the publicly documented QuantAlgo alert
    families. The two momentum warnings are marked as MK extensions.
    """
    specs = [
        ("NWBullishReversal", "Bullish Kernel Reversal", "QuantAlgo Public Alert", "BULLISH"),
        ("NWBearishReversal", "Bearish Kernel Reversal", "QuantAlgo Public Alert", "BEARISH"),
        ("NWAnyReversal", "Any Kernel Reversal", "QuantAlgo Public Alert", "REVERSAL"),
        ("NWCrossAboveUpper", "Source Cross Above Upper Band", "QuantAlgo Public Alert", "OVEREXTENSION"),
        ("NWCrossBelowLower", "Source Cross Below Lower Band", "QuantAlgo Public Alert", "BREAKDOWN"),
        ("NWMomentumUpwardWarning", "Momentum Upward", "MK Causal Warning", "EARLY BULLISH MOMENTUM"),
        ("NWMomentumDownwardWarning", "Momentum Downward", "MK Causal Warning", "EARLY BEARISH MOMENTUM"),
    ]
    rows = []
    for col, alert, origin, state in specs:
        if col not in df.columns:
            continue
        mask = df[col].fillna(False).astype(bool)
        for ts in df.index[mask]:
            r = df.loc[ts]
            rows.append({
                "Date": ts,
                "Alert": alert,
                "Origin": origin,
                "State": state,
                "Source": float(r["NWSource"]) if pd.notna(r.get("NWSource")) else np.nan,
                "NW Trend": float(r["NWTrend"]) if pd.notna(r.get("NWTrend")) else np.nan,
                "Upper": float(r["NWUpper"]) if pd.notna(r.get("NWUpper")) else np.nan,
                "Lower": float(r["NWLower"]) if pd.notna(r.get("NWLower")) else np.nan,
                "Normalized Slope": float(r["NWNormalizedSlope"]) if pd.notna(r.get("NWNormalizedSlope")) else np.nan,
                "Slope Acceleration": float(r["NWSlopeAcceleration"]) if pd.notna(r.get("NWSlopeAcceleration")) else np.nan,
            })
    if not rows:
        return pd.DataFrame(columns=["Date","Alert","Origin","State","Source","NW Trend","Upper","Lower","Normalized Slope","Slope Acceleration"])
    out = pd.DataFrame(rows).sort_values(["Date","Origin","Alert"]).reset_index(drop=True)
    return out

def _all_true(arr: np.ndarray, end_idx: int, length: int, value: int) -> bool:
    start = end_idx - length + 1
    if start < 0:
        return False
    return bool(np.all(arr[start:end_idx + 1] == value))


def _entry_condition(ind: pd.DataFrame, idx: int, cfg: NWStrategyConfig) -> tuple[bool, str]:
    if idx < 1 or not np.isfinite(float(ind["NWTrend"].iloc[idx])):
        return False, "NW path unavailable"

    src = float(ind["NWSource"].iloc[idx])
    trend = float(ind["NWTrend"].iloc[idx])
    upper = float(ind["NWUpper"].iloc[idx])
    direction = ind["NWDirection"].to_numpy(int)

    if cfg.mode == "QUANTALGO_REVERSAL_TRANSLATION":
        reversal = bool(ind["NWBullishReversal"].iloc[idx])
        above_path = src > trend
        band_ok = (not cfg.avoid_upper_band_chase) or (src <= upper)
        ok = reversal and above_path and band_ok
        return ok, (
            f"bullish slope reversal={reversal}; source>NW={above_path}; "
            f"upper-band chase filter={band_ok}"
        )

    confirmed = _all_true(direction, idx, cfg.confirmation_bars, 1)
    above_path = src > trend
    band_ok = (not cfg.avoid_upper_band_chase) or (src <= upper)
    ok = confirmed and above_path and band_ok
    return ok, (
        f"bullish direction confirmed {cfg.confirmation_bars} bar(s)={confirmed}; "
        f"source>NW={above_path}; upper-band chase filter={band_ok}"
    )


def _exit_condition(ind: pd.DataFrame, idx: int, cfg: NWStrategyConfig) -> tuple[bool, str]:
    if idx < 1 or not np.isfinite(float(ind["NWTrend"].iloc[idx])):
        return False, "NW path unavailable"

    src = float(ind["NWSource"].iloc[idx])
    trend = float(ind["NWTrend"].iloc[idx])
    direction = ind["NWDirection"].to_numpy(int)
    path_lost = src < trend

    if cfg.mode == "QUANTALGO_REVERSAL_TRANSLATION":
        reversal = bool(ind["NWBearishReversal"].iloc[idx])
        ok = reversal or path_lost
        return ok, f"bearish slope reversal={reversal}; source<NW={path_lost}"

    bearish_confirm = _all_true(direction, idx, cfg.exit_confirmation_bars, -1)
    ok = path_lost or bearish_confirm
    return ok, (
        f"source<NW={path_lost}; bearish direction confirmed "
        f"{cfg.exit_confirmation_bars} bar(s)={bearish_confirm}"
    )


def run_nw_strategy(
    base_df: pd.DataFrame,
    indicator: pd.DataFrame,
    config: NWStrategyConfig = NWStrategyConfig(),
) -> pd.DataFrame:
    """Apply a causal long/cash strategy to the NW indicator.

    Signal on prior completed bar; execution at current adjusted open.
    The returned DataFrame follows the parent engine's portfolio column schema,
    so existing risk/performance/trade-ledger functions remain compatible.
    """
    config.validate()
    if not base_df.index.equals(indicator.index):
        raise ValueError("base_df and NW indicator indexes must match exactly")

    out = base_df.copy()
    for c in indicator.columns:
        out[c] = indicator[c]

    n = len(out)
    adj_open = pd.to_numeric(out["AdjOpen"], errors="raise").to_numpy(float)
    adj_close = pd.to_numeric(out["AdjCloseCalc"], errors="raise").to_numpy(float)

    shares = np.zeros(n, dtype=float)
    cash = np.zeros(n, dtype=float)
    portfolio = np.zeros(n, dtype=float)
    buyhold = np.zeros(n, dtype=float)
    signal = np.full(n, "", dtype=object)
    first_buy = np.zeros(n, dtype=float)
    first_sell = np.zeros(n, dtype=float)
    buy_marker = np.zeros(n, dtype=float)
    sell_marker = np.zeros(n, dtype=float)
    entry_reason = np.full(n, "", dtype=object)
    exit_reason = np.full(n, "", dtype=object)

    cash[0] = config.initial_capital
    portfolio[0] = config.initial_capital
    buyhold[0] = config.initial_capital

    for i in range(1, n):
        j = i - 1  # only the prior completed bar is allowed to trigger execution
        prev_shares = shares[i - 1]
        prev_cash = cash[i - 1]

        enter, enter_why = _entry_condition(indicator, j, config)
        exit_, exit_why = _exit_condition(indicator, j, config)
        entry_reason[i] = enter_why
        exit_reason[i] = exit_why

        if prev_shares > 0 and exit_:
            signal[i] = "SELL"
            shares[i] = 0.0
            cash[i] = prev_cash + prev_shares * adj_open[i]
            first_sell[i] = 1.0
        elif prev_shares == 0 and enter:
            signal[i] = "BUY"
            shares[i] = prev_cash / adj_open[i]
            cash[i] = 0.0
            first_buy[i] = 1.0
        else:
            shares[i] = prev_shares
            cash[i] = prev_cash

        portfolio[i] = shares[i] * adj_close[i] + cash[i]
        buyhold[i] = config.initial_capital * adj_close[i] / adj_close[0]
        buy_marker[i] = adj_open[i] if first_buy[i] else np.nan
        sell_marker[i] = adj_open[i] if first_sell[i] else np.nan

    out["Signal"] = signal
    out["Shares"] = shares
    out["Cash"] = cash
    out["Portfolio"] = portfolio
    out["BuyHold"] = buyhold
    out["FirstBuy"] = first_buy
    out["FirstSell"] = first_sell
    out["BuyMarker"] = buy_marker
    out["SellMarker"] = sell_marker
    out["NWEntryReason"] = entry_reason
    out["NWExitReason"] = exit_reason
    out["NWExposure"] = (shares > 0).astype(float)
    return out


def nw_decision_snapshot(df: pd.DataFrame, strategy_config: NWStrategyConfig) -> dict:
    """Explain the next-action state from the latest completed NW bar."""
    strategy_config.validate()
    if len(df) < 2:
        raise ValueError("At least two observations required")

    i = len(df) - 1
    invested = bool(float(df["Shares"].iloc[i]) > 0)
    ind_cols = [
        "NWSource", "NWTrend", "NWUpper", "NWLower", "NWSlope", "NWDirection",
        "NWBullishReversal", "NWBearishReversal",
        "NWCrossAboveUpper", "NWCrossBelowLower",
        "NWMomentumUpwardWarning", "NWMomentumDownwardWarning"
    ]
    ind = df[ind_cols]
    enter, enter_why = _entry_condition(ind, i, strategy_config)
    exit_, exit_why = _exit_condition(ind, i, strategy_config)

    if invested and exit_:
        decision = "SELL"
        rationale = "A confirmed NW exit condition is active on the latest completed bar. Execution is scheduled for the next available adjusted open."
    elif invested:
        decision = "HOLD"
        rationale = "The portfolio is long and the latest completed bar has not triggered an NW exit condition."
    elif enter:
        decision = "BUY"
        rationale = "A confirmed NW entry condition is active while the portfolio is in cash. Execution is scheduled for the next available adjusted open."
    else:
        decision = "WAIT / CASH"
        rationale = "The portfolio is in cash and the latest completed bar has not satisfied the NW entry gate."

    latest = df.iloc[-1]
    src = float(latest["NWSource"])
    trend = float(latest["NWTrend"])
    upper = float(latest["NWUpper"])
    lower = float(latest["NWLower"])

    gates = pd.DataFrame([
        {
            "Gate": "NW Slope Regime",
            "Rule": "NW slope > 0 bullish / < 0 bearish",
            "Status": "BULLISH" if int(latest["NWDirection"]) > 0 else "BEARISH" if int(latest["NWDirection"]) < 0 else "FLAT",
            "Value": float(latest["NWSlope"]) if pd.notna(latest["NWSlope"]) else np.nan,
        },
        {
            "Gate": "Price vs NW Path",
            "Rule": "Source must hold above NW path for long bias",
            "Status": "ABOVE" if src > trend else "BELOW / EQUAL",
            "Value": src / trend - 1.0 if trend else np.nan,
        },
        {
            "Gate": "Bullish Reversal",
            "Rule": "Slope direction flips non-positive → positive",
            "Status": "TRIGGERED" if bool(latest["NWBullishReversal"]) else "NOT TRIGGERED",
            "Value": np.nan,
        },
        {
            "Gate": "Bearish Reversal",
            "Rule": "Slope direction flips non-negative → negative",
            "Status": "TRIGGERED" if bool(latest["NWBearishReversal"]) else "NOT TRIGGERED",
            "Value": np.nan,
        },
        {
            "Gate": "Upper Band Alert",
            "Rule": "Selected source crosses above NW upper residual band",
            "Status": "TRIGGERED" if bool(latest.get("NWCrossAboveUpper", False)) else "NOT TRIGGERED",
            "Value": np.nan,
        },
        {
            "Gate": "Lower Band Alert",
            "Rule": "Selected source crosses below NW lower residual band",
            "Status": "TRIGGERED" if bool(latest.get("NWCrossBelowLower", False)) else "NOT TRIGGERED",
            "Value": np.nan,
        },
        {
            "Gate": "Momentum Upward Warning",
            "Rule": "MK extension: normalized NW slope acceleration turns positive while NW slope is not yet bullish",
            "Status": "TRIGGERED" if bool(latest.get("NWMomentumUpwardWarning", False)) else "NOT TRIGGERED",
            "Value": float(latest.get("NWSlopeAcceleration", np.nan)),
        },
        {
            "Gate": "Momentum Downward Warning",
            "Rule": "MK extension: normalized NW slope acceleration turns negative while NW slope is not yet bearish",
            "Status": "TRIGGERED" if bool(latest.get("NWMomentumDownwardWarning", False)) else "NOT TRIGGERED",
            "Value": float(latest.get("NWSlopeAcceleration", np.nan)),
        },
        {
            "Gate": "Residual Extension",
            "Rule": "Source location vs kernel-weighted residual envelope",
            "Status": "ABOVE UPPER" if src > upper else "BELOW LOWER" if src < lower else "INSIDE BAND",
            "Value": (src - trend) / (upper - trend) if upper != trend else np.nan,
        },
        {
            "Gate": "Portfolio State",
            "Rule": "Long/cash all-in/all-out position gate",
            "Status": "LONG" if invested else "CASH",
            "Value": float(df["NWExposure"].iloc[-1]),
        },
    ])

    return {
        "decision": decision,
        "position": "LONG" if invested else "CASH",
        "rationale": rationale,
        "entry_condition": enter,
        "exit_condition": exit_,
        "entry_reason": enter_why,
        "exit_reason": exit_why,
        "trend_direction": "BULLISH" if int(latest["NWDirection"]) > 0 else "BEARISH" if int(latest["NWDirection"]) < 0 else "FLAT",
        "source": src,
        "trend": trend,
        "upper": upper,
        "lower": lower,
        "price_trend_gap": src / trend - 1.0 if trend else np.nan,
        "band_width_pct": float(latest["NWBandWidthPct"]),
        "bullish_reversal": bool(latest["NWBullishReversal"]),
        "bearish_reversal": bool(latest["NWBearishReversal"]),
        "cross_above_upper": bool(latest.get("NWCrossAboveUpper", False)),
        "cross_below_lower": bool(latest.get("NWCrossBelowLower", False)),
        "momentum_upward_warning": bool(latest.get("NWMomentumUpwardWarning", False)),
        "momentum_downward_warning": bool(latest.get("NWMomentumDownwardWarning", False)),
        "normalized_slope": float(latest.get("NWNormalizedSlope", np.nan)),
        "slope_acceleration": float(latest.get("NWSlopeAcceleration", np.nan)),
        "gates": gates,
        "timing_note": "Latest completed bar determines the signal; any trade executes at the next adjusted open.",
    }


def strategy_mode_label(mode: str) -> str:
    return {
        "QUANTALGO_REVERSAL_TRANSLATION": "Public-Methodology Reversal Translation",
        "MK_CONFIRMED_TREND": "MK Confirmed NW Trend",
    }.get(mode, str(mode))
