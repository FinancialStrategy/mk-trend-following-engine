
"""
MK Institutional Tactical Exit / Exposure Engine v0.08.6
By Murat Konuklar

Primary purpose:
- react to Nadaraya-Watson envelope excursions quickly
- separate tactical de-risking from legacy hard-stop logic
- incorporate benchmark-relative weakness / overextension
- use staged target exposure: 100%, 75%, 50%, 25%, 0%
- keep every signal causal: completed bar -> next adjusted open

No synthetic market data, no fallback provider, no forward/back fill.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TacticalConfig:
    weak_z: float = 1.5
    strong_z: float = 2.0
    extreme_z: float = 3.0
    rvol_window: int = 20
    volume_climax: float = 1.5
    initial_capital: float = 100_000.0
    immediate_upper_band_reduce: float = 0.75
    reentry_reduce: float = 0.50
    strong_risk_reduce: float = 0.25
    initial_target_exposure: float = 1.0
    rebalance_only_on_target_change: bool = True
    cash_annual_rate: float = 0.0

    def validate(self):
        if not (0 < self.weak_z < self.strong_z < self.extreme_z):
            raise ValueError("Require 0 < weak_z < strong_z < extreme_z")
        if self.rvol_window < 5: raise ValueError("rvol_window must be >=5")
        if self.volume_climax <= 0: raise ValueError("volume_climax must be >0")
        if self.initial_capital <= 0: raise ValueError("initial_capital must be >0")
        if not (0.0 <= self.initial_target_exposure <= 1.0): raise ValueError("initial_target_exposure must be in [0,1]")
        if self.cash_annual_rate < 0: raise ValueError("cash_annual_rate must be >=0")


def add_tactical_features(df: pd.DataFrame, rel: pd.DataFrame, cfg: TacticalConfig) -> pd.DataFrame:
    cfg.validate()
    required = ["NWSource","NWTrend","NWUpper","NWLower","NWSlope","NWDirection",
                "NWBullishReversal","NWBearishReversal","AdjOpen","AdjCloseCalc","Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"Tactical engine missing columns: {missing}")

    x = df.copy()
    for c in rel.columns:
        x[c] = rel[c]

    src = pd.to_numeric(x["NWSource"], errors="raise")
    upper = pd.to_numeric(x["NWUpper"], errors="raise")
    lower = pd.to_numeric(x["NWLower"], errors="raise")
    trend = pd.to_numeric(x["NWTrend"], errors="raise")

    x["NWCrossAboveUpper"] = (src > upper) & (src.shift(1) <= upper.shift(1))
    x["NWReenterBelowUpper"] = (src <= upper) & (src.shift(1) > upper.shift(1))
    x["NWCrossBelowLower"] = (src < lower) & (src.shift(1) >= lower.shift(1))
    x["NWReenterAboveLower"] = (src >= lower) & (src.shift(1) < lower.shift(1))
    x["NWCrossBelowPath"] = (src < trend) & (src.shift(1) >= trend.shift(1))
    x["NWCrossAbovePath"] = (src > trend) & (src.shift(1) <= trend.shift(1))

    half_band = upper - trend
    x["NWEnvelopeZ"] = np.where(half_band != 0, (src-trend)/half_band, np.nan)
    normalized_slope = pd.to_numeric(x["NWSlope"], errors="coerce") / trend.shift(1)
    x["NWNormalizedSlope"] = normalized_slope
    x["NWSlopeDeceleration"] = normalized_slope.diff()

    vol = pd.to_numeric(x["Volume"], errors="coerce")
    vol_med = vol.rolling(cfg.rvol_window, min_periods=cfg.rvol_window).median().shift(1)
    x["RelativeVolume"] = vol / vol_med

    dz = pd.to_numeric(x["ResidualDriftZ"], errors="coerce")
    x["RelativeWeak"] = dz <= -cfg.weak_z
    x["RelativeStrongWeak"] = dz <= -cfg.strong_z
    x["RelativeExtremeWeak"] = dz <= -cfg.extreme_z
    x["RelativeStrongOutperformance"] = dz >= cfg.strong_z
    x["RelativeExtremeOutperformance"] = dz >= cfg.extreme_z
    return x


def _decision_for_bar(x: pd.DataFrame, i: int, current_target: float, cfg: TacticalConfig) -> tuple[float,str,str]:
    """Decision from completed bar i. Returns target exposure, action, rationale."""
    row = x.iloc[i]
    if pd.isna(row.get("NWTrend")) or pd.isna(row.get("ResidualDriftZ")):
        return current_target, "HOLD", "Tactical/relative warm-up is incomplete."

    upper_cross = bool(row["NWCrossAboveUpper"])
    upper_reentry = bool(row["NWReenterBelowUpper"])
    lower_cross = bool(row["NWCrossBelowLower"])
    lower_reentry = bool(row["NWReenterAboveLower"])
    path_below = bool(row["NWSource"] < row["NWTrend"])
    path_above = bool(row["NWSource"] > row["NWTrend"])
    bear_rev = bool(row["NWBearishReversal"])
    bull_rev = bool(row["NWBullishReversal"])
    slope_bear = int(row["NWDirection"]) < 0
    slope_bull = int(row["NWDirection"]) > 0
    rel_z = float(row["ResidualDriftZ"])
    rvol = float(row["RelativeVolume"]) if pd.notna(row["RelativeVolume"]) else np.nan
    climax = bool(np.isfinite(rvol) and rvol >= cfg.volume_climax)

    def reduce_to(level: float, reason: str):
        new_target = min(current_target, level)
        if current_target <= 0:
            return 0.0, "WAIT / CASH", "Risk gate is active but the tactical portfolio is already in cash. " + reason
        if new_target < current_target - 1e-12:
            return new_target, f"REDUCE TO {new_target:.0%}", reason
        return current_target, f"HOLD {current_target:.0%}", "Risk gate remains active; exposure is already at or below this reduction tier. " + reason

    def restore_to(level: float, reason: str):
        new_target = max(current_target, level)
        if new_target > current_target + 1e-12:
            return new_target, f"BUY / RESTORE TO {new_target:.0%}", reason
        return current_target, f"HOLD {current_target:.0%}", reason

    # -------- Hard / severe de-risking --------
    if lower_cross and slope_bear:
        return reduce_to(0.0, "Price broke below the lower NW residual band while NW slope is bearish.")
    if rel_z <= -cfg.extreme_z:
        return reduce_to(0.0, f"Extreme benchmark-relative breakdown: residual drift z={rel_z:.2f}.")
    if path_below and bear_rev and rel_z <= -cfg.strong_z:
        return reduce_to(0.0, f"NW path loss + bearish reversal + strong relative weakness (z={rel_z:.2f}).")

    # -------- Sensitive envelope exhaustion: act BEFORE a hard trend stop --------
    if upper_reentry and (slope_bear or bear_rev) and rel_z >= cfg.weak_z:
        return reduce_to(cfg.strong_risk_reduce,
            f"Upper-envelope excursion failed back inside the band with bearish confirmation; relative overextension z={rel_z:.2f}.")
    if upper_reentry:
        return reduce_to(cfg.reentry_reduce,
            "Price re-entered below the NW upper residual band after an upside excursion.")
    if upper_cross and (rel_z >= cfg.strong_z or climax):
        return reduce_to(cfg.reentry_reduce,
            f"Immediate upper-band breakout with {'strong benchmark overextension' if rel_z >= cfg.strong_z else 'volume climax'}; relative z={rel_z:.2f}, RVOL={rvol:.2f}.")
    if upper_cross:
        return reduce_to(cfg.immediate_upper_band_reduce,
            "Source crossed above the NW upper residual band: immediate tactical overextension trim.")

    if rel_z >= cfg.extreme_z and float(row["NWSlopeDeceleration"]) < 0:
        return reduce_to(cfg.immediate_upper_band_reduce,
            f"Extreme benchmark-relative overextension z={rel_z:.2f} with decelerating NW slope.")

    # -------- Relative weakness can force de-risking without waiting for a remote ATR stop --------
    if rel_z <= -cfg.strong_z:
        return reduce_to(cfg.strong_risk_reduce, f"Strong benchmark-relative weakness: residual drift z={rel_z:.2f}.")
    if rel_z <= -cfg.weak_z and (path_below or slope_bear):
        return reduce_to(cfg.reentry_reduce,
            f"Benchmark-relative weakness z={rel_z:.2f} combined with weak NW trend structure.")
    if bear_rev:
        return reduce_to(cfg.immediate_upper_band_reduce, "Bearish NW slope reversal detected.")

    # -------- Build / restore exposure only with confirmation --------
    if lower_reentry and bull_rev and rel_z <= -cfg.weak_z:
        return restore_to(0.50,
            f"Price re-entered above the lower band after relative underperformance z={rel_z:.2f} and NW slope reversed bullish.")
    if bull_rev and path_above and rel_z > -cfg.weak_z:
        return restore_to(0.75, "Bullish NW reversal with price above the NW path.")
    if path_above and slope_bull and rel_z >= 0:
        return restore_to(1.0, f"Bullish NW structure with non-negative benchmark-relative drift z={rel_z:.2f}.")

    return current_target, "HOLD", (
        f"No tactical exposure change. NW={'bullish' if slope_bull else 'bearish' if slope_bear else 'flat'}, "
        f"relative drift z={rel_z:.2f}."
    )


def _cash_growth_factor(index: pd.DatetimeIndex, i: int, annual_rate: float) -> float:
    """Accrue cash from prior close to current close using ACT/365 simple day fraction.

    Default annual_rate is 0.0, preserving the historical project assumption.
    The rate is user/config supplied; no market series is invented.
    """
    if annual_rate <= 0 or i <= 0:
        return 1.0
    days = max((pd.Timestamp(index[i]) - pd.Timestamp(index[i-1])).total_seconds() / 86400.0, 0.0)
    return float((1.0 + annual_rate) ** (days / 365.0))


def _execute_target_path(
    x: pd.DataFrame,
    desired_target: np.ndarray,
    action: np.ndarray,
    rationale: np.ndarray,
    cfg: TacticalConfig,
) -> pd.DataFrame:
    """Self-financing staged-exposure execution engine.

    Critical accounting rules:
    - portfolio begins at cfg.initial_target_exposure, not implicitly at 0% cash;
    - a HOLD does not generate hidden daily rebalancing;
    - only a target change causes an execution at the next adjusted open;
    - target exposure and actual close exposure are reported separately;
    - Buy & Hold is recomputed independently from adjusted closes.
    """
    n = len(x)
    opens = pd.to_numeric(x["AdjOpen"], errors="raise").to_numpy(float)
    closes = pd.to_numeric(x["AdjCloseCalc"], errors="raise").to_numpy(float)
    if n == 0:
        raise ValueError("Tactical execution requires at least one observation")
    if np.any(opens <= 0) or np.any(closes <= 0):
        raise ValueError("Adjusted open/close must be positive")

    shares = np.zeros(n, float)
    cash = np.zeros(n, float)
    portfolio = np.zeros(n, float)
    target = np.asarray(desired_target, dtype=float).copy()
    traded_value = np.zeros(n, float)
    target_change = np.zeros(n, float)
    rebalance_flag = np.zeros(n, bool)
    turnover = np.zeros(n, float)

    initial_target = float(cfg.initial_target_exposure)
    target[0] = initial_target
    shares[0] = initial_target * cfg.initial_capital / closes[0]
    cash[0] = (1.0 - initial_target) * cfg.initial_capital
    portfolio[0] = cfg.initial_capital
    action[0] = f"INITIALIZE {initial_target:.0%} EXPOSURE"
    rationale[0] = (
        "Institutional Tactical is a de-risking overlay. The comparison therefore initializes "
        f"at {initial_target:.0%} exposure; subsequent completed-bar signals alter exposure at the next adjusted open."
    )

    for i in range(1, n):
        # cash carry from prior close to current open/close. With the default 0% rate this is 1.0.
        growth = _cash_growth_factor(x.index, i, cfg.cash_annual_rate)
        carried_cash = cash[i-1] * growth
        pre_open_equity = carried_cash + shares[i-1] * opens[i]

        desired = float(np.clip(target[i], 0.0, 1.0))
        previous_target = float(target[i-1])
        changed = abs(desired - previous_target) > 1e-12
        target_change[i] = desired - previous_target

        if changed or not cfg.rebalance_only_on_target_change:
            desired_stock_value = desired * pre_open_equity
            desired_shares = desired_stock_value / opens[i]
            trade_shares = desired_shares - shares[i-1]
            shares[i] = desired_shares
            cash[i] = pre_open_equity - desired_stock_value
            traded_value[i] = abs(trade_shares) * opens[i]
            rebalance_flag[i] = changed
            turnover[i] = traded_value[i] / pre_open_equity if pre_open_equity > 0 else 0.0
        else:
            # TRUE HOLD: preserve shares and cash; do not secretly constant-mix rebalance.
            shares[i] = shares[i-1]
            cash[i] = carried_cash

        portfolio[i] = shares[i] * closes[i] + cash[i]

    buyhold = cfg.initial_capital * closes / closes[0]
    actual_exposure = np.where(portfolio != 0, shares * closes / portfolio, 0.0)
    wealth_ratio = np.where(buyhold != 0, portfolio / buyhold, np.nan)

    x["TacticalShares"] = shares
    x["TacticalCash"] = cash
    x["TacticalPortfolio"] = portfolio
    x["TacticalTargetExposure"] = target
    x["TacticalActualExposure"] = actual_exposure
    x["TacticalExposure"] = actual_exposure  # compatibility alias
    x["TacticalTargetChange"] = target_change
    x["TacticalRebalanceFlag"] = rebalance_flag
    x["TacticalTradedValue"] = traded_value
    x["TacticalTurnover"] = turnover
    x["TacticalAction"] = action
    x["TacticalRationale"] = rationale
    x["BuyHold"] = buyhold
    x["TacticalVsBuyHoldRatio"] = wealth_ratio
    x["TacticalReturn"] = pd.Series(portfolio, index=x.index).pct_change()
    x["BuyHoldReturn"] = pd.Series(buyhold, index=x.index).pct_change()
    x["TacticalActiveReturn"] = x["TacticalReturn"] - x["BuyHoldReturn"]
    return x


def run_tactical_strategy(df: pd.DataFrame, rel: pd.DataFrame, cfg: TacticalConfig = TacticalConfig()) -> pd.DataFrame:
    x = add_tactical_features(df, rel, cfg)
    n = len(x)

    target = np.full(n, np.nan, float)
    action = np.full(n, "", object)
    rationale = np.full(n, "", object)
    target[0] = float(cfg.initial_target_exposure)

    for i in range(1, n):
        j = i - 1
        desired, act, why = _decision_for_bar(x, j, float(target[i-1]), cfg)
        target[i] = desired
        action[i] = act
        rationale[i] = why

    return _execute_target_path(x, target, action, rationale, cfg)

def tactical_snapshot(x: pd.DataFrame, cfg: TacticalConfig) -> dict:
    if len(x) < 2: raise ValueError("At least two observations required")
    i = len(x)-1
    current_target = float(x["TacticalTargetExposure"].iloc[-1])
    desired, action, rationale = _decision_for_bar(x, i, current_target, cfg)
    row = x.iloc[-1]

    gates = pd.DataFrame([
        {"Gate":"NW Upper Band Cross","Status":"TRIGGERED" if bool(row["NWCrossAboveUpper"]) else "NO",
         "Value":float(row["NWEnvelopeZ"]) if pd.notna(row["NWEnvelopeZ"]) else np.nan},
        {"Gate":"Upper Band Re-entry","Status":"TRIGGERED" if bool(row["NWReenterBelowUpper"]) else "NO","Value":np.nan},
        {"Gate":"Lower Band Break","Status":"TRIGGERED" if bool(row["NWCrossBelowLower"]) else "NO","Value":np.nan},
        {"Gate":"NW Slope","Status":"BULLISH" if int(row["NWDirection"])>0 else "BEARISH" if int(row["NWDirection"])<0 else "FLAT",
         "Value":float(row["NWNormalizedSlope"]) if pd.notna(row["NWNormalizedSlope"]) else np.nan},
        {"Gate":"Benchmark Residual Drift","Status":
            "EXTREME WEAK" if row["ResidualDriftZ"] <= -cfg.extreme_z else
            "STRONG WEAK" if row["ResidualDriftZ"] <= -cfg.strong_z else
            "WEAK" if row["ResidualDriftZ"] <= -cfg.weak_z else
            "EXTREME OUTPERFORMANCE" if row["ResidualDriftZ"] >= cfg.extreme_z else
            "STRONG OUTPERFORMANCE" if row["ResidualDriftZ"] >= cfg.strong_z else "NORMAL",
         "Value":float(row["ResidualDriftZ"]) if pd.notna(row["ResidualDriftZ"]) else np.nan},
        {"Gate":"Relative Volume","Status":"CLIMAX" if pd.notna(row["RelativeVolume"]) and row["RelativeVolume"]>=cfg.volume_climax else "NORMAL / NA",
         "Value":float(row["RelativeVolume"]) if pd.notna(row["RelativeVolume"]) else np.nan},
        {"Gate":"Current Target Exposure","Status":f"{current_target:.0%}","Value":current_target},
        {"Gate":"Current Actual Exposure","Status":f"{float(row.get('TacticalActualExposure', current_target)):.0%}",
         "Value":float(row.get("TacticalActualExposure", current_target))},
    ])
    return {
        "decision":action,
        "target_exposure":desired,
        "current_target_exposure":current_target,
        "current_actual_exposure":float(row.get("TacticalActualExposure", current_target)),
        "cash_annual_rate":float(cfg.cash_annual_rate),
        "rebalance_policy":"TARGET_CHANGE_ONLY" if cfg.rebalance_only_on_target_change else "DAILY_TARGET_REBALANCE",
        "rationale":rationale,
        "relative_z":float(row["ResidualDriftZ"]) if pd.notna(row["ResidualDriftZ"]) else np.nan,
        "beta":float(row["RollingBeta"]) if pd.notna(row["RollingBeta"]) else np.nan,
        "envelope_z":float(row["NWEnvelopeZ"]) if pd.notna(row["NWEnvelopeZ"]) else np.nan,
        "relative_volume":float(row["RelativeVolume"]) if pd.notna(row["RelativeVolume"]) else np.nan,
        "gates":gates,
        "timing_note":"Latest completed bar sets the NEXT target exposure. The current executed target remains in force until the next adjusted-open rebalance.",
    }
