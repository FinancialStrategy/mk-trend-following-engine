"""
MK DEMA-MACD Confirmation & Calibration Engine v0.07
By Murat Konuklar
MK FinTECH LabGEN @2026 ATELIER ISTANBUL

Purpose
-------
Independent, auditable Python research implementation inspired by the public
methodology and description of TradingView's open-source indicator:
"DEMA MACD BUY signal confirmation" by bilguut (2026-01-09).

Governance
----------
- This is NOT a verbatim redistribution of Pine Script source.
- The public reference describes a DEMA-MACD continuation confirmation tool,
  a BUY confirmation, and SELL as an exit from a prior BUY. The MK layer below
  explicitly extends that idea into a separately specified institutional
  lifecycle: BUY WATCH -> BUY -> HOLD -> SELL WATCH -> REDUCE -> SELL/RISK EXIT.
- No market data is downloaded here. The parent engine supplies Yahoo Finance
  data under its strict no-fallback / no-synthetic-data policy.
- All executable strategy actions are decided from the PRIOR COMPLETED BAR and
  executed at the next adjusted open. No centered filters or future bars are
  used.
- Calibration never selects parameters on the final out-of-sample holdout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Literal, Optional, Iterable
import math
import numpy as np
import pandas as pd


SourceName = Literal["Adjusted Close", "HLC3", "OHLC4"]
SignalSmoothing = Literal["EMA", "DEMA"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DEMAMACDConfig:
    source: SourceName = "Adjusted Close"
    fast_length: int = 12
    slow_length: int = 26
    signal_length: int = 9
    signal_smoothing: SignalSmoothing = "DEMA"
    standard_macd_fast: int = 12
    standard_macd_slow: int = 26
    standard_macd_signal: int = 9
    atr_length: int = 14
    adx_length: int = 14
    trend_length: int = 50
    swing_lookback: int = 20
    cross_valid_bars: int = 3
    slope_confirmation_bars: int = 2
    histogram_confirmation_bars: int = 2
    adx_threshold: float = 18.0
    max_extension_atr: float = 3.0
    use_standard_macd_filter: bool = True
    use_adx_filter: bool = True
    use_nw_filter: bool = True
    avoid_chase: bool = True
    minimum_observations: int = 80

    def validate(self) -> None:
        ints = {
            "fast_length": self.fast_length,
            "slow_length": self.slow_length,
            "signal_length": self.signal_length,
            "atr_length": self.atr_length,
            "adx_length": self.adx_length,
            "trend_length": self.trend_length,
            "swing_lookback": self.swing_lookback,
            "cross_valid_bars": self.cross_valid_bars,
            "slope_confirmation_bars": self.slope_confirmation_bars,
            "histogram_confirmation_bars": self.histogram_confirmation_bars,
        }
        if any(int(v) < 1 for v in ints.values()):
            raise ValueError(f"All DEMA-MACD integer lengths must be >= 1: {ints}")
        if self.fast_length >= self.slow_length:
            raise ValueError("DEMA-MACD fast_length must be smaller than slow_length")
        if self.standard_macd_fast >= self.standard_macd_slow:
            raise ValueError("standard_macd_fast must be smaller than standard_macd_slow")
        if self.signal_smoothing not in {"EMA", "DEMA"}:
            raise ValueError(f"Unsupported signal smoothing: {self.signal_smoothing}")
        if self.source not in {"Adjusted Close", "HLC3", "OHLC4"}:
            raise ValueError(f"Unsupported DEMA-MACD source: {self.source}")
        if self.adx_threshold < 0:
            raise ValueError("adx_threshold must be >= 0")
        if self.max_extension_atr <= 0:
            raise ValueError("max_extension_atr must be > 0")
        if self.minimum_observations < 30:
            raise ValueError("minimum_observations must be >= 30")


@dataclass(frozen=True)
class DEMAStrategyConfig:
    initial_capital: float = 100_000.0
    buy_threshold: float = 65.0
    sell_watch_threshold: float = 35.0
    reduce_threshold: float = 50.0
    sell_threshold: float = 65.0
    buy_persistence_bars: int = 1
    sell_persistence_bars: int = 2
    cooldown_bars: int = 3
    atr_trailing_multiplier: float = 3.0
    hard_stop_pct: float = 0.12
    require_recent_bull_cross: bool = True
    require_recent_bear_cross_for_sell: bool = False
    execute_reduce: bool = False
    reduce_fraction: float = 0.50

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        for name in ["buy_threshold", "sell_watch_threshold", "reduce_threshold", "sell_threshold"]:
            val = float(getattr(self, name))
            if not (0 <= val <= 100):
                raise ValueError(f"{name} must be between 0 and 100")
        if not (self.sell_watch_threshold <= self.reduce_threshold <= self.sell_threshold):
            raise ValueError("SELL thresholds must satisfy watch <= reduce <= sell")
        if self.buy_persistence_bars < 1 or self.sell_persistence_bars < 1:
            raise ValueError("Persistence bars must be >= 1")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        if self.atr_trailing_multiplier <= 0:
            raise ValueError("atr_trailing_multiplier must be > 0")
        if not (0 < self.hard_stop_pct < 1):
            raise ValueError("hard_stop_pct must be between 0 and 1")
        if not (0 < self.reduce_fraction < 1):
            raise ValueError("reduce_fraction must be between 0 and 1")


@dataclass(frozen=True)
class DEMACalibrationConfig:
    grid_depth: Literal["Focused", "Balanced", "Deep"] = "Balanced"
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    walk_forward_folds: int = 3
    minimum_closed_trades: int = 2
    whipsaw_holding_bars: int = 10

    def validate(self) -> None:
        if self.grid_depth not in {"Focused", "Balanced", "Deep"}:
            raise ValueError(f"Unsupported calibration grid: {self.grid_depth}")
        if not (0.40 <= self.train_fraction <= 0.80):
            raise ValueError("train_fraction must be between 0.40 and 0.80")
        if not (0.10 <= self.validation_fraction <= 0.30):
            raise ValueError("validation_fraction must be between 0.10 and 0.30")
        if self.train_fraction + self.validation_fraction >= 0.90:
            raise ValueError("Calibration must reserve at least 10% final OOS data")
        if self.walk_forward_folds < 1:
            raise ValueError("walk_forward_folds must be >= 1")
        if self.minimum_closed_trades < 0:
            raise ValueError("minimum_closed_trades must be >= 0")


# ---------------------------------------------------------------------------
# Numeric helpers — causal only
# ---------------------------------------------------------------------------
def _source_series(df: pd.DataFrame, source: str) -> pd.Series:
    required = {"AdjOpen", "AdjHigh", "AdjLow", "AdjCloseCalc"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"DEMA-MACD requires adjusted OHLC columns; missing: {sorted(missing)}")
    if source == "Adjusted Close":
        s = df["AdjCloseCalc"]
    elif source == "HLC3":
        s = (df["AdjHigh"] + df["AdjLow"] + df["AdjCloseCalc"]) / 3.0
    elif source == "OHLC4":
        s = (df["AdjOpen"] + df["AdjHigh"] + df["AdjLow"] + df["AdjCloseCalc"]) / 4.0
    else:
        raise ValueError(f"Unsupported source: {source}")
    return pd.to_numeric(s, errors="raise").astype(float)


def ema(series: pd.Series, length: int) -> pd.Series:
    if length < 1:
        raise ValueError("EMA length must be >= 1")
    return pd.to_numeric(series, errors="raise").astype(float).ewm(
        span=int(length), adjust=False, min_periods=int(length)
    ).mean()


def dema(series: pd.Series, length: int) -> pd.Series:
    """Double Exponential Moving Average: 2*EMA(x,n) - EMA(EMA(x,n),n)."""
    e1 = ema(series, length)
    e2 = ema(e1, length)
    return 2.0 * e1 - e2


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder-style recursive moving average, causal."""
    return pd.to_numeric(series, errors="coerce").astype(float).ewm(
        alpha=1.0 / float(length), adjust=False, min_periods=int(length)
    ).mean()


def _atr_adx(df: pd.DataFrame, length: int) -> pd.DataFrame:
    high = pd.to_numeric(df["AdjHigh"], errors="raise").astype(float)
    low = pd.to_numeric(df["AdjLow"], errors="raise").astype(float)
    close = pd.to_numeric(df["AdjCloseCalc"], errors="raise").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = _rma(tr, length)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index, dtype=float)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index, dtype=float)
    plus_di = 100.0 * _rma(plus_dm, length) / atr.replace(0, np.nan)
    minus_di = 100.0 * _rma(minus_dm, length) / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, length)
    return pd.DataFrame({
        "ATR": atr,
        "PlusDI": plus_di,
        "MinusDI": minus_di,
        "ADX": adx,
    }, index=df.index)


def _rolling_all_true(series: pd.Series, length: int) -> pd.Series:
    x = series.fillna(False).astype(int)
    return x.rolling(int(length), min_periods=int(length)).sum().eq(int(length))


def _recent_true(series: pd.Series, bars: int) -> pd.Series:
    x = series.fillna(False).astype(int)
    return x.rolling(int(bars), min_periods=1).max().gt(0)


def _bool_score(components: dict[str, tuple[pd.Series, float]]) -> tuple[pd.Series, pd.DataFrame]:
    if not components:
        raise ValueError("At least one score component is required")
    index = next(iter(components.values()))[0].index
    weighted = pd.Series(0.0, index=index)
    available_weight = pd.Series(0.0, index=index)
    detail = pd.DataFrame(index=index)
    for name, (condition, weight) in components.items():
        cond = condition.astype("boolean")
        available = cond.notna()
        passed = cond.fillna(False).astype(float)
        weighted = weighted + passed * float(weight)
        available_weight = available_weight + available.astype(float) * float(weight)
        detail[name] = cond
    score = 100.0 * weighted / available_weight.replace(0.0, np.nan)
    return score.clip(0.0, 100.0), detail


# ---------------------------------------------------------------------------
# Indicator + scoring layer
# ---------------------------------------------------------------------------
def compute_dema_macd(
    base_df: pd.DataFrame,
    config: DEMAMACDConfig = DEMAMACDConfig(),
    nw_indicator: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute causal DEMA-MACD, classic MACD controls, ADX/ATR and MK scores."""
    config.validate()
    if len(base_df) < config.minimum_observations:
        raise ValueError(
            f"Only {len(base_df)} observations; DEMA-MACD minimum is {config.minimum_observations}."
        )
    if not base_df.index.is_monotonic_increasing or base_df.index.has_duplicates:
        raise ValueError("DEMA-MACD requires a unique, ascending time index")

    src = _source_series(base_df, config.source)
    out = pd.DataFrame(index=base_df.index)
    out["DEMASource"] = src

    fast = dema(src, config.fast_length)
    slow = dema(src, config.slow_length)
    line = fast - slow
    signal = dema(line, config.signal_length) if config.signal_smoothing == "DEMA" else ema(line, config.signal_length)
    hist = line - signal

    out["DEMAFast"] = fast
    out["DEMASlow"] = slow
    out["DEMAMACD"] = line
    out["DEMASignal"] = signal
    out["DEMAHistogram"] = hist
    out["DEMASignalSlope"] = signal.diff()
    out["DEMAHistogramSlope"] = hist.diff()
    out["DEMABullCross"] = (line > signal) & (line.shift(1) <= signal.shift(1))
    out["DEMABearCross"] = (line < signal) & (line.shift(1) >= signal.shift(1))
    out["DEMARecentBullCross"] = _recent_true(out["DEMABullCross"], config.cross_valid_bars)
    out["DEMARecentBearCross"] = _recent_true(out["DEMABearCross"], config.cross_valid_bars)

    std_fast = ema(src, config.standard_macd_fast)
    std_slow = ema(src, config.standard_macd_slow)
    std_line = std_fast - std_slow
    std_signal = ema(std_line, config.standard_macd_signal)
    out["StdMACD"] = std_line
    out["StdMACDSignal"] = std_signal
    out["StdMACDHistogram"] = std_line - std_signal

    atr_adx = _atr_adx(base_df, config.adx_length)
    out = out.join(atr_adx)
    # ATR length is independently configurable; recompute only when different.
    if config.atr_length != config.adx_length:
        out["ATR"] = _atr_adx(base_df, config.atr_length)["ATR"]

    out["DEMAMACD_ATR"] = out["DEMAMACD"] / out["ATR"].replace(0, np.nan)
    out["DEMAHistogram_ATR"] = out["DEMAHistogram"] / out["ATR"].replace(0, np.nan)

    trend = dema(src, config.trend_length)
    out["DEMATrend"] = trend
    out["DEMATrendSlope"] = trend.diff()
    out["PriceAboveTrend"] = src > trend
    out["PriceBelowTrend"] = src < trend

    out["SignalRisingConfirmed"] = _rolling_all_true(out["DEMASignalSlope"] > 0, config.slope_confirmation_bars)
    out["SignalFallingConfirmed"] = _rolling_all_true(out["DEMASignalSlope"] < 0, config.slope_confirmation_bars)
    out["HistogramExpandingBull"] = _rolling_all_true(out["DEMAHistogramSlope"] > 0, config.histogram_confirmation_bars)
    out["HistogramDeteriorating"] = _rolling_all_true(out["DEMAHistogramSlope"] < 0, config.histogram_confirmation_bars)

    extension_atr = (src - trend) / out["ATR"].replace(0, np.nan)
    out["ExtensionATR"] = extension_atr
    out["NoChase"] = extension_atr <= float(config.max_extension_atr)

    # Prior-window swing level. shift(1) guarantees today's close cannot create its own stop level.
    out["PriorSwingLow"] = pd.to_numeric(base_df["AdjLow"], errors="raise").astype(float).shift(1).rolling(
        config.swing_lookback, min_periods=config.swing_lookback
    ).min()
    out["PriorSwingHigh"] = pd.to_numeric(base_df["AdjHigh"], errors="raise").astype(float).shift(1).rolling(
        config.swing_lookback, min_periods=config.swing_lookback
    ).max()
    out["SwingLowBreach"] = src < out["PriorSwingLow"]

    nw_available = False
    if nw_indicator is not None:
        if not base_df.index.equals(nw_indicator.index):
            raise ValueError("NW indicator index must exactly match base data for DEMA-MACD integration")
        for c in ["NWTrend", "NWUpper", "NWLower", "NWDirection", "NWAbovePath", "NWBelowPath"]:
            if c in nw_indicator.columns:
                out[c] = nw_indicator[c]
        nw_available = {"NWTrend", "NWDirection"}.issubset(out.columns)

    # Tri-state Series make optional components disappear from score denominator when disabled/unavailable.
    def opt(condition: pd.Series, enabled: bool) -> pd.Series:
        if enabled:
            return condition.astype("boolean")
        return pd.Series(pd.NA, index=out.index, dtype="boolean")

    std_bull = (out["StdMACD"] > out["StdMACDSignal"]) & (out["StdMACDHistogram"] >= 0)
    std_bear = (out["StdMACD"] < out["StdMACDSignal"]) & (out["StdMACDHistogram"] <= 0)
    adx_bull = (out["ADX"] >= config.adx_threshold) & (out["PlusDI"] > out["MinusDI"])
    adx_bear = (out["ADX"] >= config.adx_threshold) & (out["MinusDI"] > out["PlusDI"])
    nw_bull = ((out.get("NWDirection", pd.Series(index=out.index, dtype=float)) > 0) &
               (src > out.get("NWTrend", pd.Series(index=out.index, dtype=float))))
    nw_bear = ((out.get("NWDirection", pd.Series(index=out.index, dtype=float)) < 0) |
               (src < out.get("NWTrend", pd.Series(index=out.index, dtype=float))))

    buy_components = {
        "Recent Bullish DEMA Cross": (out["DEMARecentBullCross"].astype("boolean"), 16.0),
        "DEMA MACD Above Signal": ((out["DEMAMACD"] > out["DEMASignal"]).astype("boolean"), 10.0),
        "Signal Line Rising": (out["SignalRisingConfirmed"].astype("boolean"), 10.0),
        "Histogram Positive": ((out["DEMAHistogram"] > 0).astype("boolean"), 8.0),
        "Histogram Expanding": (out["HistogramExpandingBull"].astype("boolean"), 8.0),
        "Price Above DEMA Trend": (out["PriceAboveTrend"].astype("boolean"), 10.0),
        "DEMA Trend Rising": ((out["DEMATrendSlope"] > 0).astype("boolean"), 8.0),
        "Classic MACD Confirmation": (opt(std_bull, config.use_standard_macd_filter), 10.0),
        "ADX Directional Confirmation": (opt(adx_bull, config.use_adx_filter), 8.0),
        "NW Bullish Regime": (opt(nw_bull, config.use_nw_filter and nw_available), 8.0),
        "No-Chase Extension": (opt(out["NoChase"], config.avoid_chase), 4.0),
    }
    sell_components = {
        "Recent Bearish DEMA Cross": (out["DEMARecentBearCross"].astype("boolean"), 16.0),
        "DEMA MACD Below Signal": ((out["DEMAMACD"] < out["DEMASignal"]).astype("boolean"), 10.0),
        "Signal Line Falling": (out["SignalFallingConfirmed"].astype("boolean"), 10.0),
        "Histogram Negative": ((out["DEMAHistogram"] < 0).astype("boolean"), 8.0),
        "Histogram Deteriorating": (out["HistogramDeteriorating"].astype("boolean"), 10.0),
        "Price Below DEMA Trend": (out["PriceBelowTrend"].astype("boolean"), 10.0),
        "DEMA Trend Falling": ((out["DEMATrendSlope"] < 0).astype("boolean"), 8.0),
        "Classic MACD Bearish": (opt(std_bear, config.use_standard_macd_filter), 8.0),
        "ADX Bearish Direction": (opt(adx_bear, config.use_adx_filter), 6.0),
        "NW Bearish / Path Lost": (opt(nw_bear, config.use_nw_filter and nw_available), 8.0),
        "Swing-Low Breach": (out["SwingLowBreach"].astype("boolean"), 6.0),
    }
    buy_score, buy_detail = _bool_score(buy_components)
    sell_score, sell_detail = _bool_score(sell_components)
    out["BuyScore"] = buy_score
    out["SellScore"] = sell_score
    out["ScoreSpread"] = buy_score - sell_score

    for c in buy_detail.columns:
        out[f"BUY::{c}"] = buy_detail[c]
    for c in sell_detail.columns:
        out[f"SELL::{c}"] = sell_detail[c]

    out.attrs["nw_filter_available"] = bool(nw_available)
    out.attrs["methodology"] = (
        "Independent MK implementation inspired by bilguut's public DEMA-MACD continuation description; "
        "not a verbatim Pine Script redistribution."
    )
    return out


# ---------------------------------------------------------------------------
# Lifecycle state machine and portfolio simulation
# ---------------------------------------------------------------------------
def _consecutive_true(arr: np.ndarray, idx: int, length: int) -> bool:
    start = idx - int(length) + 1
    if start < 0:
        return False
    window = arr[start:idx + 1]
    return bool(np.all(window))


def _signal_bar_action(
    indicator: pd.DataFrame,
    idx: int,
    invested: bool,
    cfg: DEMAStrategyConfig,
    *,
    entry_price: float | None = None,
    peak_close: float | None = None,
    last_exit_signal_idx: int | None = None,
) -> tuple[str, str, dict]:
    """Classify the completed signal bar. No execution occurs inside this function."""
    cfg.validate()
    if idx < 0:
        return "WAIT / CASH", "No completed bar is available.", {}

    r = indicator.iloc[idx]
    buy_score = float(r["BuyScore"]) if pd.notna(r["BuyScore"]) else np.nan
    sell_score = float(r["SellScore"]) if pd.notna(r["SellScore"]) else np.nan
    recent_bull = bool(r.get("DEMARecentBullCross", False))
    recent_bear = bool(r.get("DEMARecentBearCross", False))

    buy_candidate = bool(np.isfinite(buy_score) and buy_score >= cfg.buy_threshold)
    if cfg.require_recent_bull_cross:
        buy_candidate = buy_candidate and recent_bull

    sell_candidate = bool(np.isfinite(sell_score) and sell_score >= cfg.sell_threshold)
    if cfg.require_recent_bear_cross_for_sell:
        sell_candidate = sell_candidate and recent_bear

    buy_arr = (indicator["BuyScore"].ge(cfg.buy_threshold) &
               ((indicator["DEMARecentBullCross"]) if cfg.require_recent_bull_cross else True)).fillna(False).to_numpy(bool)
    sell_arr = (indicator["SellScore"].ge(cfg.sell_threshold) &
                ((indicator["DEMARecentBearCross"]) if cfg.require_recent_bear_cross_for_sell else True)).fillna(False).to_numpy(bool)
    buy_confirmed = buy_candidate and _consecutive_true(buy_arr, idx, cfg.buy_persistence_bars)
    sell_confirmed = sell_candidate and _consecutive_true(sell_arr, idx, cfg.sell_persistence_bars)

    cooldown_ok = True
    if last_exit_signal_idx is not None:
        cooldown_ok = (idx - int(last_exit_signal_idx)) >= int(cfg.cooldown_bars)

    risk_flags = {
        "Hard Stop": False,
        "ATR Trailing Stop": False,
        "Swing-Low Break": bool(r.get("SwingLowBreach", False)),
    }
    risk_levels = {"hard_stop": np.nan, "atr_stop": np.nan, "swing_stop": float(r.get("PriorSwingLow", np.nan))}
    if invested and entry_price is not None and np.isfinite(entry_price):
        close = float(r["DEMASource"])
        hard_stop = float(entry_price) * (1.0 - cfg.hard_stop_pct)
        risk_levels["hard_stop"] = hard_stop
        risk_flags["Hard Stop"] = close <= hard_stop
        atr = float(r.get("ATR", np.nan))
        if peak_close is not None and np.isfinite(peak_close) and np.isfinite(atr):
            atr_stop = float(peak_close) - cfg.atr_trailing_multiplier * atr
            risk_levels["atr_stop"] = atr_stop
            risk_flags["ATR Trailing Stop"] = close <= atr_stop

    risk_exit = invested and any(risk_flags.values())
    context = {
        "buy_score": buy_score,
        "sell_score": sell_score,
        "recent_bull_cross": recent_bull,
        "recent_bear_cross": recent_bear,
        "buy_confirmed": buy_confirmed,
        "sell_confirmed": sell_confirmed,
        "cooldown_ok": cooldown_ok,
        "risk_flags": risk_flags,
        "risk_levels": risk_levels,
    }

    if invested:
        if risk_exit:
            names = [k for k, v in risk_flags.items() if v]
            return "RISK EXIT", "Immediate capital-protection exit: " + ", ".join(names), context
        if sell_confirmed:
            return "SELL", f"Sell score {sell_score:.1f}/100 confirmed for {cfg.sell_persistence_bars} bar(s).", context
        if np.isfinite(sell_score) and sell_score >= cfg.reduce_threshold:
            return "REDUCE", f"Sell deterioration score {sell_score:.1f}/100 reached the reduce tier but not full SELL confirmation.", context
        if np.isfinite(sell_score) and sell_score >= cfg.sell_watch_threshold:
            return "SELL WATCH", f"Sell deterioration score {sell_score:.1f}/100 reached the watch tier.", context
        return "HOLD", f"Long position retained; sell score {sell_score:.1f}/100 remains below actionable thresholds." if np.isfinite(sell_score) else "Long position retained; sell diagnostics are warming up.", context

    if buy_confirmed and cooldown_ok:
        return "BUY", f"Buy score {buy_score:.1f}/100 confirmed with recent bullish DEMA-MACD trigger and lifecycle gates satisfied.", context
    if buy_candidate and not cooldown_ok:
        return "BUY WATCH", f"Bullish candidate exists ({buy_score:.1f}/100) but re-entry cooldown is still active.", context
    if np.isfinite(buy_score) and buy_score >= max(45.0, cfg.buy_threshold - 15.0):
        return "BUY WATCH", f"Bullish quality is improving ({buy_score:.1f}/100) but full BUY confirmation is not complete.", context
    return "WAIT / CASH", f"No confirmed entry; buy score {buy_score:.1f}/100." if np.isfinite(buy_score) else "No confirmed entry; indicators are warming up.", context


def run_dema_macd_strategy(
    base_df: pd.DataFrame,
    indicator: pd.DataFrame,
    config: DEMAStrategyConfig = DEMAStrategyConfig(),
) -> pd.DataFrame:
    """Causal long/cash DEMA-MACD strategy with optional advisory/partial REDUCE.

    Every trade executed at row i is based only on indicator row i-1.
    """
    config.validate()
    if not base_df.index.equals(indicator.index):
        raise ValueError("base_df and DEMA indicator indexes must match exactly")
    required = {"AdjOpen", "AdjCloseCalc"}
    missing = required.difference(base_df.columns)
    if missing:
        raise KeyError(f"Missing required adjusted price columns: {sorted(missing)}")

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
    executed_action = np.full(n, "", dtype=object)
    lifecycle = np.full(n, "WAIT / CASH", dtype=object)
    reason = np.full(n, "", dtype=object)
    first_buy = np.zeros(n, dtype=float)
    first_sell = np.zeros(n, dtype=float)
    reduce_exec = np.zeros(n, dtype=float)
    buy_marker = np.full(n, np.nan, dtype=float)
    sell_marker = np.full(n, np.nan, dtype=float)
    reduce_marker = np.full(n, np.nan, dtype=float)
    atr_stop_arr = np.full(n, np.nan, dtype=float)
    hard_stop_arr = np.full(n, np.nan, dtype=float)
    position_peak_arr = np.full(n, np.nan, dtype=float)

    cash[0] = config.initial_capital
    portfolio[0] = config.initial_capital
    buyhold[0] = config.initial_capital
    entry_price: float | None = None
    peak_close: float | None = None
    last_exit_signal_idx: int | None = None
    reduce_done = False

    # Classification on row 0 is diagnostic only.
    lifecycle[0], reason[0], _ = _signal_bar_action(indicator, 0, False, config)

    for i in range(1, n):
        j = i - 1
        prev_shares = float(shares[i - 1])
        prev_cash = float(cash[i - 1])
        invested = prev_shares > 0

        if invested:
            prev_close = float(adj_close[j])
            peak_close = max(float(peak_close if peak_close is not None else prev_close), prev_close)

        action, why, ctx = _signal_bar_action(
            indicator, j, invested, config,
            entry_price=entry_price,
            peak_close=peak_close,
            last_exit_signal_idx=last_exit_signal_idx,
        )
        lifecycle[j] = action
        reason[j] = why
        levels = ctx.get("risk_levels", {})
        if invested:
            atr_stop_arr[j] = float(levels.get("atr_stop", np.nan))
            hard_stop_arr[j] = float(levels.get("hard_stop", np.nan))
            position_peak_arr[j] = float(peak_close) if peak_close is not None else np.nan

        shares[i] = prev_shares
        cash[i] = prev_cash

        if invested and action in {"SELL", "RISK EXIT"}:
            cash[i] = prev_cash + prev_shares * adj_open[i]
            shares[i] = 0.0
            signal[i] = "SELL"
            executed_action[i] = action
            first_sell[i] = 1.0
            sell_marker[i] = adj_open[i]
            last_exit_signal_idx = j
            entry_price = None
            peak_close = None
            reduce_done = False
        elif invested and action == "REDUCE" and config.execute_reduce and not reduce_done:
            qty = prev_shares * config.reduce_fraction
            shares[i] = prev_shares - qty
            cash[i] = prev_cash + qty * adj_open[i]
            signal[i] = "REDUCE"
            executed_action[i] = "REDUCE"
            reduce_exec[i] = 1.0
            reduce_marker[i] = adj_open[i]
            reduce_done = True
        elif (not invested) and action == "BUY":
            shares[i] = prev_cash / adj_open[i]
            cash[i] = 0.0
            signal[i] = "BUY"
            executed_action[i] = "BUY"
            first_buy[i] = 1.0
            buy_marker[i] = adj_open[i]
            entry_price = float(adj_open[i])
            peak_close = max(float(adj_open[i]), float(adj_close[i]))
            reduce_done = False

        portfolio[i] = shares[i] * adj_close[i] + cash[i]
        buyhold[i] = config.initial_capital * adj_close[i] / adj_close[0]

    # Latest completed bar lifecycle snapshot (no next bar exists for execution in backtest frame).
    invested = bool(shares[-1] > 0)
    if invested:
        peak_close = max(float(peak_close if peak_close is not None else adj_close[-1]), float(adj_close[-1]))
    lifecycle[-1], reason[-1], latest_ctx = _signal_bar_action(
        indicator, n - 1, invested, config,
        entry_price=entry_price,
        peak_close=peak_close,
        last_exit_signal_idx=last_exit_signal_idx,
    )
    if invested:
        levels = latest_ctx.get("risk_levels", {})
        atr_stop_arr[-1] = float(levels.get("atr_stop", np.nan))
        hard_stop_arr[-1] = float(levels.get("hard_stop", np.nan))
        position_peak_arr[-1] = float(peak_close) if peak_close is not None else np.nan

    out["Signal"] = signal
    out["DEMAExecutedAction"] = executed_action
    out["DEMALifecycle"] = lifecycle
    out["DEMAActionReason"] = reason
    out["Shares"] = shares
    out["Cash"] = cash
    out["Portfolio"] = portfolio
    out["BuyHold"] = buyhold
    out["FirstBuy"] = first_buy
    out["FirstSell"] = first_sell
    out["DEMAReduceExecution"] = reduce_exec
    out["BuyMarker"] = buy_marker
    out["SellMarker"] = sell_marker
    out["DEMAReduceMarker"] = reduce_marker
    out["DEMAATRStop"] = atr_stop_arr
    out["DEMAHardStop"] = hard_stop_arr
    out["DEMAPositionPeak"] = position_peak_arr
    out["DEMAExposure"] = (shares > 0).astype(float)
    return out


# ---------------------------------------------------------------------------
# Decision snapshot and event diagnostics
# ---------------------------------------------------------------------------
def _latest_open_trade_context(df: pd.DataFrame) -> tuple[float | None, float | None, int | None]:
    buys = np.where(pd.to_numeric(df["FirstBuy"], errors="coerce").fillna(0).to_numpy(float) > 0)[0]
    sells = np.where(pd.to_numeric(df["FirstSell"], errors="coerce").fillna(0).to_numpy(float) > 0)[0]
    if not len(buys):
        return None, None, int(sells[-1] - 1) if len(sells) else None
    last_buy = int(buys[-1])
    last_sell = int(sells[-1]) if len(sells) else -1
    last_exit_signal_idx = last_sell - 1 if last_sell >= 1 else None
    if last_sell > last_buy:
        return None, None, last_exit_signal_idx
    entry_price = float(df["AdjOpen"].iloc[last_buy])
    peak = float(pd.to_numeric(df["AdjCloseCalc"].iloc[last_buy:], errors="raise").max())
    return entry_price, peak, last_exit_signal_idx


def dema_decision_snapshot(df: pd.DataFrame, config: DEMAStrategyConfig) -> dict:
    config.validate()
    if len(df) < 2:
        raise ValueError("At least two observations required for DEMA-MACD decision snapshot")
    invested = bool(float(df["Shares"].iloc[-1]) > 0)
    entry_price, peak, last_exit_signal_idx = _latest_open_trade_context(df)
    indicator_cols = [
        c for c in df.columns if c.startswith("DEMA") or c.startswith("StdMACD") or
        c.startswith("BUY::") or c.startswith("SELL::") or c in {
            "ATR", "ADX", "PlusDI", "MinusDI", "PriorSwingLow", "SwingLowBreach",
            "BuyScore", "SellScore", "ScoreSpread", "NWTrend", "NWUpper", "NWLower",
            "NWDirection", "NWAbovePath", "NWBelowPath", "NoChase", "ExtensionATR",
            "PriceAboveTrend", "PriceBelowTrend", "SignalRisingConfirmed", "SignalFallingConfirmed",
            "HistogramExpandingBull", "HistogramDeteriorating",
        }
    ]
    ind = df[indicator_cols].copy()
    action, why, ctx = _signal_bar_action(
        ind, len(ind) - 1, invested, config,
        entry_price=entry_price,
        peak_close=peak,
        last_exit_signal_idx=last_exit_signal_idx,
    )
    r = df.iloc[-1]
    buy_score = float(r["BuyScore"]) if pd.notna(r["BuyScore"]) else np.nan
    sell_score = float(r["SellScore"]) if pd.notna(r["SellScore"]) else np.nan

    gates = pd.DataFrame([
        {
            "Gate": "DEMA-MACD Trigger",
            "Rule": "Recent bullish/bearish line-signal crossover",
            "Status": "BULL CROSS" if bool(r.get("DEMARecentBullCross", False)) else "BEAR CROSS" if bool(r.get("DEMARecentBearCross", False)) else "NO RECENT CROSS",
            "Observed": f"MACD {float(r['DEMAMACD']):.4f} | Signal {float(r['DEMASignal']):.4f}" if pd.notna(r.get("DEMAMACD")) and pd.notna(r.get("DEMASignal")) else "warming up",
        },
        {
            "Gate": "Momentum Quality",
            "Rule": "Signal slope + histogram direction",
            "Status": "BULLISH" if bool(r.get("SignalRisingConfirmed", False)) and float(r.get("DEMAHistogram", 0)) > 0 else "BEARISH" if bool(r.get("SignalFallingConfirmed", False)) and float(r.get("DEMAHistogram", 0)) < 0 else "MIXED",
            "Observed": f"Histogram {float(r['DEMAHistogram']):.4f}" if pd.notna(r.get("DEMAHistogram")) else "warming up",
        },
        {
            "Gate": "Trend Regime",
            "Rule": "Price vs DEMA trend and trend slope",
            "Status": "BULLISH" if bool(r.get("PriceAboveTrend", False)) and float(r.get("DEMATrendSlope", 0)) > 0 else "BEARISH" if bool(r.get("PriceBelowTrend", False)) and float(r.get("DEMATrendSlope", 0)) < 0 else "MIXED",
            "Observed": f"Extension {float(r.get('ExtensionATR', np.nan)):.2f} ATR" if pd.notna(r.get("ExtensionATR")) else "warming up",
        },
        {
            "Gate": "Trend Strength",
            "Rule": "ADX and directional movement",
            "Status": "BULLISH" if float(r.get("PlusDI", 0)) > float(r.get("MinusDI", 0)) else "BEARISH" if float(r.get("MinusDI", 0)) > float(r.get("PlusDI", 0)) else "NEUTRAL",
            "Observed": f"ADX {float(r.get('ADX', np.nan)):.1f}" if pd.notna(r.get("ADX")) else "warming up",
        },
        {
            "Gate": "NW Confirmation",
            "Rule": "Optional price/path + NW slope regime",
            "Status": "BULLISH" if ("NWDirection" in df.columns and pd.notna(r.get("NWDirection")) and int(r.get("NWDirection")) > 0) else "BEARISH" if ("NWDirection" in df.columns and pd.notna(r.get("NWDirection")) and int(r.get("NWDirection")) < 0) else "NOT ACTIVE / FLAT",
            "Observed": "Integrated" if "NWDirection" in df.columns else "NW not supplied",
        },
        {
            "Gate": "Capital Protection",
            "Rule": "Hard stop / ATR trailing stop / prior swing-low breach",
            "Status": "RISK EXIT" if any(ctx.get("risk_flags", {}).values()) else "CLEAR",
            "Observed": ", ".join(k for k, v in ctx.get("risk_flags", {}).items() if v) or "No immediate breach",
        },
        {
            "Gate": "Lifecycle State",
            "Rule": "Position-aware action hierarchy",
            "Status": action,
            "Observed": "LONG" if invested else "CASH",
        },
    ])

    return {
        "decision": action,
        "position": "LONG" if invested else "CASH",
        "rationale": why,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "score_spread": buy_score - sell_score if np.isfinite(buy_score) and np.isfinite(sell_score) else np.nan,
        "recent_bull_cross": bool(r.get("DEMARecentBullCross", False)),
        "recent_bear_cross": bool(r.get("DEMARecentBearCross", False)),
        "adx": float(r.get("ADX", np.nan)),
        "extension_atr": float(r.get("ExtensionATR", np.nan)),
        "risk_flags": ctx.get("risk_flags", {}),
        "risk_levels": ctx.get("risk_levels", {}),
        "gates": gates,
        "timing_note": "Latest completed bar determines the next lifecycle action; executable BUY/SELL/RISK EXIT occurs at the next adjusted open.",
    }


def dema_event_ledger(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_actions = {"BUY WATCH", "BUY", "SELL WATCH", "REDUCE", "SELL", "RISK EXIT"}
    prev_action = None
    for i, (dt, r) in enumerate(df.iterrows()):
        action = str(r.get("DEMALifecycle", ""))
        executed = str(r.get("DEMAExecutedAction", ""))
        changed = action != prev_action
        if (action in key_actions and changed) or executed:
            rows.append({
                "Date": pd.Timestamp(dt),
                "Lifecycle Action": action,
                "Executed Action": executed or "—",
                "Buy Score": float(r.get("BuyScore", np.nan)),
                "Sell Score": float(r.get("SellScore", np.nan)),
                "DEMA MACD": float(r.get("DEMAMACD", np.nan)),
                "Signal": float(r.get("DEMASignal", np.nan)),
                "Histogram": float(r.get("DEMAHistogram", np.nan)),
                "ADX": float(r.get("ADX", np.nan)),
                "Reason": str(r.get("DEMAActionReason", "")),
            })
        prev_action = action
    return pd.DataFrame(rows)


def dema_trade_ledger(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    open_trade = None
    for bar_no, (dt, r) in enumerate(df.iterrows()):
        if float(r.get("FirstBuy", 0) or 0) > 0:
            open_trade = {
                "Entry Date": pd.Timestamp(dt),
                "Entry Price": float(r["AdjOpen"]),
                "Entry Bar": bar_no,
                "Peak Close": float(r["AdjCloseCalc"]),
            }
        elif open_trade is not None:
            open_trade["Peak Close"] = max(open_trade["Peak Close"], float(r["AdjCloseCalc"]))

        if float(r.get("FirstSell", 0) or 0) > 0 and open_trade is not None:
            exit_price = float(r["AdjOpen"])
            rows.append({
                "Entry Date": open_trade["Entry Date"],
                "Entry Price": open_trade["Entry Price"],
                "Exit Date": pd.Timestamp(dt),
                "Exit Price": exit_price,
                "Holding Days": int((pd.Timestamp(dt) - open_trade["Entry Date"]).days),
                "Holding Bars": int(bar_no - open_trade["Entry Bar"]),
                "Trade Return": exit_price / open_trade["Entry Price"] - 1.0,
                "Peak-to-Exit Giveback": exit_price / open_trade["Peak Close"] - 1.0,
                "Exit Type": str(r.get("DEMAExecutedAction", "SELL")) or "SELL",
                "Status": "CLOSED",
            })
            open_trade = None

    if open_trade is not None:
        last_dt = pd.Timestamp(df.index[-1])
        last_close = float(df["AdjCloseCalc"].iloc[-1])
        rows.append({
            "Entry Date": open_trade["Entry Date"],
            "Entry Price": open_trade["Entry Price"],
            "Exit Date": pd.NaT,
            "Exit Price": np.nan,
            "Holding Days": int((last_dt - open_trade["Entry Date"]).days),
            "Holding Bars": int(len(df) - 1 - open_trade["Entry Bar"]),
            "Trade Return": last_close / open_trade["Entry Price"] - 1.0,
            "Peak-to-Exit Giveback": last_close / open_trade["Peak Close"] - 1.0,
            "Exit Type": "OPEN",
            "Status": "OPEN / MARK-TO-MARKET",
        })
    return pd.DataFrame(rows)


def exit_quality_metrics(df: pd.DataFrame, forward_bars: int = 20) -> tuple[pd.DataFrame, dict]:
    """Measure downside avoided vs upside foregone after each full exit."""
    exits = np.where(pd.to_numeric(df["FirstSell"], errors="coerce").fillna(0).to_numpy(float) > 0)[0]
    rows = []
    for idx in exits:
        if idx >= len(df) - 1:
            continue
        exit_price = float(df["AdjOpen"].iloc[idx])
        fwd = pd.to_numeric(df["AdjCloseCalc"].iloc[idx + 1: idx + 1 + int(forward_bars)], errors="raise").astype(float)
        if fwd.empty:
            continue
        rel = fwd / exit_price - 1.0
        downside = float(rel.min())
        upside = float(rel.max())
        avoided = max(0.0, -downside)
        foregone = max(0.0, upside)
        rows.append({
            "Exit Date": pd.Timestamp(df.index[idx]),
            "Exit Type": str(df["DEMAExecutedAction"].iloc[idx]) or "SELL",
            "Exit Price": exit_price,
            "Forward Bars": int(len(fwd)),
            "Worst Forward Return": downside,
            "Best Forward Return": upside,
            "Downside Avoided Proxy": avoided,
            "Upside Foregone Proxy": foregone,
            "Net Exit Utility": avoided - foregone,
            "False Exit Proxy": bool(foregone > avoided + 0.03),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        summary = {
            "exits": 0, "avg_downside_avoided": np.nan, "avg_upside_foregone": np.nan,
            "avg_net_exit_utility": np.nan, "false_exit_rate": np.nan,
        }
    else:
        summary = {
            "exits": int(len(table)),
            "avg_downside_avoided": float(table["Downside Avoided Proxy"].mean()),
            "avg_upside_foregone": float(table["Upside Foregone Proxy"].mean()),
            "avg_net_exit_utility": float(table["Net Exit Utility"].mean()),
            "false_exit_rate": float(table["False Exit Proxy"].mean()),
        }
    return table, summary


# ---------------------------------------------------------------------------
# Calibration / OOS / walk-forward
# ---------------------------------------------------------------------------
def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 3:
        return 252.0
    days = np.diff(pd.DatetimeIndex(index).view("i8")) / 86_400_000_000_000.0
    med = float(np.nanmedian(days)) if len(days) else 1.0
    if med <= 2.0:
        return 252.0
    if med <= 10.0:
        return 52.0
    return 12.0


def _segment_metrics(df: pd.DataFrame, start: int, end: int, whipsaw_bars: int = 10) -> dict:
    seg = df.iloc[int(start):int(end)].copy()
    if len(seg) < 3:
        return {k: np.nan for k in ["CAGR", "MaxDD", "Vol", "Sharpe", "Sortino", "Calmar", "WinRate", "WhipsawRate", "Exposure"]} | {"ClosedTrades": 0}
    p = pd.to_numeric(seg["Portfolio"], errors="raise").astype(float)
    rets = p.pct_change().dropna()
    ppy = _periods_per_year(seg.index)
    years = max((pd.Timestamp(seg.index[-1]) - pd.Timestamp(seg.index[0])).days / 365.25, 1.0 / ppy)
    cagr = (float(p.iloc[-1]) / float(p.iloc[0])) ** (1.0 / years) - 1.0 if p.iloc[0] > 0 and p.iloc[-1] > 0 else np.nan
    dd = p / p.cummax() - 1.0
    maxdd = float(dd.min())
    vol = float(rets.std(ddof=1) * math.sqrt(ppy)) if len(rets) > 1 else np.nan
    ann_ret = float(rets.mean() * ppy) if len(rets) else np.nan
    sharpe = ann_ret / vol if np.isfinite(vol) and vol > 0 else np.nan
    neg = rets[rets < 0]
    downside = float(neg.std(ddof=1) * math.sqrt(ppy)) if len(neg) > 1 else np.nan
    sortino = ann_ret / downside if np.isfinite(downside) and downside > 0 else np.nan
    calmar = cagr / abs(maxdd) if np.isfinite(cagr) and maxdd < 0 else np.nan

    trades = dema_trade_ledger(seg)
    closed = trades[trades["Status"] == "CLOSED"] if not trades.empty else trades
    if closed is None or closed.empty:
        winrate = np.nan
        whipsaw = np.nan
        closed_n = 0
    else:
        winrate = float((closed["Trade Return"] > 0).mean())
        whipsaw = float((closed["Holding Bars"] <= int(whipsaw_bars)).mean())
        closed_n = int(len(closed))
    exposure = float(pd.to_numeric(seg["DEMAExposure"], errors="coerce").fillna(0).mean())
    return {
        "CAGR": cagr, "MaxDD": maxdd, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
        "Calmar": calmar, "WinRate": winrate, "WhipsawRate": whipsaw, "Exposure": exposure,
        "ClosedTrades": closed_n,
    }


def _rank_score(frame: pd.DataFrame, prefix: str, minimum_closed_trades: int) -> pd.Series:
    def pct(col: str, ascending: bool = True) -> pd.Series:
        x = pd.to_numeric(frame[f"{prefix}{col}"], errors="coerce")
        # pct=True means larger values rank higher when ascending=True.
        return x.rank(pct=True, ascending=ascending).fillna(0.0)

    score = (
        0.25 * pct("CAGR", True) +
        0.20 * pct("Calmar", True) +
        0.15 * pct("Sortino", True) +
        0.10 * pct("Sharpe", True) +
        0.10 * pct("MaxDD", True) +  # less negative max DD is better
        0.08 * pct("WinRate", True) +
        0.07 * pct("WhipsawRate", False) +
        0.05 * pct("Exposure", True)
    )
    trades = pd.to_numeric(frame[f"{prefix}ClosedTrades"], errors="coerce").fillna(0)
    penalty = np.where(trades < int(minimum_closed_trades), 0.55, 1.0)
    return score * penalty


def _parameter_grid(depth: str) -> list[tuple[int, int, int, float, float]]:
    if depth == "Focused":
        fasts, slows, signals, buys, sells = [10, 12], [24, 26, 30], [7, 9], [60, 65], [60, 65]
    elif depth == "Balanced":
        fasts, slows, signals, buys, sells = [8, 10, 12, 14], [21, 26, 30, 35], [5, 7, 9], [60, 65], [60, 65]
    elif depth == "Deep":
        fasts, slows, signals, buys, sells = [6, 8, 10, 12, 14, 16], [20, 24, 26, 30, 35, 40], [4, 5, 7, 9, 11], [55, 60, 65, 70], [55, 60, 65, 70]
    else:
        raise ValueError(depth)
    return [x for x in product(fasts, slows, signals, buys, sells) if x[0] < x[1]]


def calibrate_dema_macd(
    base_df: pd.DataFrame,
    base_indicator_config: DEMAMACDConfig,
    base_strategy_config: DEMAStrategyConfig,
    nw_indicator: Optional[pd.DataFrame] = None,
    calibration_config: DEMACalibrationConfig = DEMACalibrationConfig(),
) -> dict:
    """Parameter robustness research with untouched final OOS and expanding walk-forward folds.

    Selection protocol:
    1) Candidate systems are run causally across the history.
    2) Train score ranks all candidates.
    3) Only the top 25% by train score are eligible for validation selection.
    4) Final selection score = 70% train + 30% validation.
    5) Final OOS is reported but never used for selection.
    6) Expanding walk-forward folds independently re-select parameters using only data
       available before each fold's test interval.
    """
    base_indicator_config.validate()
    base_strategy_config.validate()
    calibration_config.validate()
    n = len(base_df)
    if n < max(180, base_indicator_config.minimum_observations * 2):
        raise ValueError("Calibration requires at least 180 observations and enough warm-up history")

    train_end = int(n * calibration_config.train_fraction)
    val_end = int(n * (calibration_config.train_fraction + calibration_config.validation_fraction))
    train_end = max(train_end, base_indicator_config.minimum_observations + 20)
    val_end = max(val_end, train_end + 20)
    val_end = min(val_end, n - 20)

    candidates = _parameter_grid(calibration_config.grid_depth)
    cache: dict[int, pd.DataFrame] = {}
    indicator_cache: dict[tuple[int,int,int], pd.DataFrame] = {}
    rows = []

    for cid, (fast, slow, sig, buy_thr, sell_thr) in enumerate(candidates):
        icfg = replace(base_indicator_config, fast_length=int(fast), slow_length=int(slow), signal_length=int(sig))
        scfg = replace(base_strategy_config, buy_threshold=float(buy_thr), sell_threshold=float(sell_thr))
        # Keep tiers coherent when sell threshold is moved by calibration.
        scfg = replace(
            scfg,
            sell_watch_threshold=min(scfg.sell_watch_threshold, float(sell_thr) - 20.0),
            reduce_threshold=min(max(scfg.reduce_threshold, scfg.sell_watch_threshold), float(sell_thr) - 10.0),
        )
        ikey = (int(fast), int(slow), int(sig))
        if ikey not in indicator_cache:
            indicator_cache[ikey] = compute_dema_macd(base_df, icfg, nw_indicator)
        ind = indicator_cache[ikey]
        strat = run_dema_macd_strategy(base_df, ind, scfg)
        cache[cid] = strat
        tr = _segment_metrics(strat, 0, train_end, calibration_config.whipsaw_holding_bars)
        va = _segment_metrics(strat, train_end, val_end, calibration_config.whipsaw_holding_bars)
        oo = _segment_metrics(strat, val_end, n, calibration_config.whipsaw_holding_bars)
        row = {
            "CandidateID": cid,
            "Fast": fast, "Slow": slow, "Signal": sig,
            "BuyThreshold": buy_thr, "SellThreshold": sell_thr,
        }
        row.update({f"Train_{k}": v for k, v in tr.items()})
        row.update({f"Validation_{k}": v for k, v in va.items()})
        row.update({f"OOS_{k}": v for k, v in oo.items()})
        rows.append(row)

    ranking = pd.DataFrame(rows)
    ranking["TrainScore"] = _rank_score(ranking, "Train_", calibration_config.minimum_closed_trades)
    ranking["ValidationScore"] = _rank_score(ranking, "Validation_", calibration_config.minimum_closed_trades)
    cutoff = float(ranking["TrainScore"].quantile(0.75))
    ranking["TrainShortlist"] = ranking["TrainScore"] >= cutoff
    ranking["RobustScore"] = np.where(
        ranking["TrainShortlist"],
        0.70 * ranking["TrainScore"] + 0.30 * ranking["ValidationScore"],
        0.0,
    )

    # Plateau score: number of near-neighbor candidates within 5% of this robust score.
    plateau = []
    for _, r in ranking.iterrows():
        neighbors = ranking[
            (ranking["Fast"].sub(r["Fast"]).abs() <= 2) &
            (ranking["Slow"].sub(r["Slow"]).abs() <= 5) &
            (ranking["Signal"].sub(r["Signal"]).abs() <= 2) &
            (ranking["BuyThreshold"].sub(r["BuyThreshold"]).abs() <= 5) &
            (ranking["SellThreshold"].sub(r["SellThreshold"]).abs() <= 5)
        ]
        target = float(r["RobustScore"])
        if target <= 0 or neighbors.empty:
            plateau.append(0.0)
        else:
            near = neighbors["RobustScore"] >= 0.95 * target
            plateau.append(float(near.mean()))
    ranking["PlateauScore"] = plateau
    ranking["InstitutionalScore"] = ranking["RobustScore"] * (0.85 + 0.15 * ranking["PlateauScore"])
    # IMPORTANT: final OOS columns are never used for selection or tie-breaking.
    ranking = ranking.sort_values(
        ["InstitutionalScore", "ValidationScore", "TrainScore", "PlateauScore"],
        ascending=False,
    ).reset_index(drop=True)
    best = ranking.iloc[0].to_dict()
    best_id = int(best["CandidateID"])

    # Expanding walk-forward: choose by training score only at each fold; never inspect fold test to choose.
    wf_rows = []
    initial_train = max(int(n * 0.45), base_indicator_config.minimum_observations + 20)
    remaining = n - initial_train
    fold_size = max(10, remaining // calibration_config.walk_forward_folds)
    for fold in range(calibration_config.walk_forward_folds):
        test_start = initial_train + fold * fold_size
        test_end = n if fold == calibration_config.walk_forward_folds - 1 else min(n, test_start + fold_size)
        if test_start >= n - 2 or test_end - test_start < 3:
            continue
        fold_eval = []
        for _, r in ranking.iterrows():
            cid = int(r["CandidateID"])
            m = _segment_metrics(cache[cid], 0, test_start, calibration_config.whipsaw_holding_bars)
            fold_eval.append({"CandidateID": cid, **m})
        fe = pd.DataFrame(fold_eval)
        # Build compatible prefix columns for the same cross-sectional ranking function.
        temp = fe.rename(columns={c: f"Train_{c}" for c in fe.columns if c != "CandidateID"})
        temp["FoldTrainScore"] = _rank_score(temp, "Train_", calibration_config.minimum_closed_trades)
        chosen_id = int(temp.sort_values("FoldTrainScore", ascending=False).iloc[0]["CandidateID"])
        test_m = _segment_metrics(cache[chosen_id], test_start, test_end, calibration_config.whipsaw_holding_bars)
        param = ranking[ranking["CandidateID"] == chosen_id].iloc[0]
        wf_rows.append({
            "Fold": fold + 1,
            "Train End": pd.Timestamp(base_df.index[test_start - 1]),
            "Test Start": pd.Timestamp(base_df.index[test_start]),
            "Test End": pd.Timestamp(base_df.index[test_end - 1]),
            "Fast": int(param["Fast"]), "Slow": int(param["Slow"]), "Signal": int(param["Signal"]),
            "BuyThreshold": float(param["BuyThreshold"]), "SellThreshold": float(param["SellThreshold"]),
            **{f"Test {k}": v for k, v in test_m.items()},
        })

    wf = pd.DataFrame(wf_rows)
    best_params = {
        "fast_length": int(best["Fast"]),
        "slow_length": int(best["Slow"]),
        "signal_length": int(best["Signal"]),
        "buy_threshold": float(best["BuyThreshold"]),
        "sell_threshold": float(best["SellThreshold"]),
    }
    return {
        "ranking": ranking,
        "best_params": best_params,
        "best_strategy": cache[best_id],
        "walk_forward": wf,
        "splits": {
            "train_start": pd.Timestamp(base_df.index[0]),
            "train_end": pd.Timestamp(base_df.index[train_end - 1]),
            "validation_start": pd.Timestamp(base_df.index[train_end]),
            "validation_end": pd.Timestamp(base_df.index[val_end - 1]),
            "oos_start": pd.Timestamp(base_df.index[val_end]),
            "oos_end": pd.Timestamp(base_df.index[-1]),
        },
        "selection_note": (
            "Final OOS metrics are untouched by selection. Candidate shortlist is train-only; "
            "validation chooses among the train shortlist; expanding walk-forward folds reselect from prior data only."
        ),
    }


# ---------------------------------------------------------------------------
# Presets / labels
# ---------------------------------------------------------------------------
def dema_preset(name: str) -> tuple[DEMAMACDConfig, DEMAStrategyConfig]:
    if name == "Reference-Style Continuation":
        return (
            DEMAMACDConfig(
                fast_length=12, slow_length=26, signal_length=9, signal_smoothing="DEMA",
                trend_length=50, cross_valid_bars=2, slope_confirmation_bars=1,
                histogram_confirmation_bars=1, use_standard_macd_filter=True,
                use_adx_filter=False, use_nw_filter=False, avoid_chase=False,
            ),
            DEMAStrategyConfig(
                buy_threshold=60, sell_watch_threshold=35, reduce_threshold=50, sell_threshold=60,
                buy_persistence_bars=1, sell_persistence_bars=1, cooldown_bars=1,
                require_recent_bull_cross=True, require_recent_bear_cross_for_sell=True,
            ),
        )
    if name == "MK Fast Confirmation":
        return (
            DEMAMACDConfig(
                fast_length=8, slow_length=21, signal_length=5, trend_length=34,
                cross_valid_bars=2, slope_confirmation_bars=1, histogram_confirmation_bars=1,
                adx_threshold=16, max_extension_atr=2.8,
            ),
            DEMAStrategyConfig(
                buy_threshold=60, sell_watch_threshold=30, reduce_threshold=45, sell_threshold=60,
                buy_persistence_bars=1, sell_persistence_bars=1, cooldown_bars=2,
                atr_trailing_multiplier=2.7, hard_stop_pct=0.10,
            ),
        )
    if name == "MK Smooth Position":
        return (
            DEMAMACDConfig(
                fast_length=14, slow_length=35, signal_length=11, trend_length=75,
                cross_valid_bars=4, slope_confirmation_bars=3, histogram_confirmation_bars=2,
                adx_threshold=20, max_extension_atr=3.5,
            ),
            DEMAStrategyConfig(
                buy_threshold=70, sell_watch_threshold=40, reduce_threshold=55, sell_threshold=70,
                buy_persistence_bars=2, sell_persistence_bars=3, cooldown_bars=5,
                atr_trailing_multiplier=3.5, hard_stop_pct=0.15,
            ),
        )
    if name in {"MK Institutional Balanced", "Custom"}:
        return DEMAMACDConfig(), DEMAStrategyConfig()
    raise ValueError(f"Unknown DEMA-MACD preset: {name}")


def lifecycle_explanation() -> pd.DataFrame:
    return pd.DataFrame([
        {"State": "WAIT / CASH", "Meaning": "No qualified entry; capital remains in cash.", "Execution": "None"},
        {"State": "BUY WATCH", "Meaning": "Bullish setup is developing but confirmation/cooldown gates are incomplete.", "Execution": "None"},
        {"State": "BUY", "Meaning": "Confirmed DEMA-MACD continuation entry with regime filters.", "Execution": "Next adjusted open"},
        {"State": "HOLD", "Meaning": "Long position remains valid; sell deterioration is below watch threshold.", "Execution": "None"},
        {"State": "SELL WATCH", "Meaning": "Momentum deterioration has become material; monitor exit risk.", "Execution": "None"},
        {"State": "REDUCE", "Meaning": "Deterioration reached the reduce tier; advisory by default, optional partial execution.", "Execution": "Optional next-open partial sale"},
        {"State": "SELL", "Meaning": "Multi-factor bearish exit score confirmed for the required persistence.", "Execution": "Next adjusted open"},
        {"State": "RISK EXIT", "Meaning": "Capital-protection override from hard stop, ATR trail, or swing-low break.", "Execution": "Next adjusted open"},
    ])
