"""
MK Intraday Tactical Lab v0.08.8
By Murat Konuklar

15-minute-only research diagnostics built exclusively from the Yahoo OHLCV bars
already fetched by the application. No additional market-data provider is used.

Design principles
-----------------
- completed-bar only: an in-progress 15m bar is explicitly withheld before models run;
- no forward/back filling of market observations;
- no synthetic market observations;
- no additional Yahoo request is required for these features;
- session VWAP, opening range, same-slot relative volume, ATR, realized volatility,
  session drawdown and gap are causal transformations of observed bars;
- the Intraday Confirmation Score is an explainable research diagnostic and does not
  silently override the primary Institutional Tactical target exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionSpec:
    code: str
    label: str
    timezone: str
    open_minute: int
    close_minute: int
    rollover: bool
    continuous_7d: bool
    sessions_per_year: int
    nominal_bars_per_session: int


@dataclass(frozen=True)
class IntradayConfig:
    opening_range_bars: int = 4
    slot_rvol_sessions: int = 10
    atr_window: int = 14
    realized_vol_window: int = 32
    rvol_climax: float = 1.50
    score_up_threshold: float = 20.0
    score_strong_threshold: float = 50.0
    session_override: str = "Auto"

    def validate(self) -> None:
        if self.opening_range_bars < 2:
            raise ValueError("opening_range_bars must be >= 2")
        if self.slot_rvol_sessions < 3:
            raise ValueError("slot_rvol_sessions must be >= 3")
        if self.atr_window < 5:
            raise ValueError("atr_window must be >= 5")
        if self.realized_vol_window < 8:
            raise ValueError("realized_vol_window must be >= 8")
        if self.rvol_climax <= 0:
            raise ValueError("rvol_climax must be > 0")
        if not (0 < self.score_up_threshold < self.score_strong_threshold < 100):
            raise ValueError("Require 0 < score_up_threshold < score_strong_threshold < 100")


SESSION_SPECS = {
    "Crypto 24/7": SessionSpec(
        code="CRYPTO_UTC", label="Crypto 24/7 — UTC day", timezone="UTC",
        open_minute=0, close_minute=24 * 60, rollover=False, continuous_7d=True,
        sessions_per_year=365, nominal_bars_per_session=96,
    ),
    "BIST Cash": SessionSpec(
        code="BIST_CASH", label="BIST Cash — 10:00–18:00 Istanbul", timezone="Europe/Istanbul",
        open_minute=10 * 60, close_minute=18 * 60, rollover=False, continuous_7d=False,
        sessions_per_year=252, nominal_bars_per_session=32,
    ),
    "US Cash": SessionSpec(
        code="US_CASH", label="US Cash — 09:30–16:00 New York", timezone="America/New_York",
        open_minute=9 * 60 + 30, close_minute=16 * 60, rollover=False, continuous_7d=False,
        sessions_per_year=252, nominal_bars_per_session=26,
    ),
    "CME Metals": SessionSpec(
        code="CME_METALS", label="CME/COMEX Metals — 18:00–17:00 New York", timezone="America/New_York",
        open_minute=18 * 60, close_minute=17 * 60, rollover=True, continuous_7d=False,
        sessions_per_year=260, nominal_bars_per_session=92,
    ),
}

SESSION_OVERRIDE_OPTIONS = ["Auto", "BIST Cash", "US Cash", "Crypto 24/7", "CME Metals"]


def infer_session_spec(ticker: str, override: str = "Auto") -> SessionSpec:
    if override != "Auto":
        if override not in SESSION_SPECS:
            raise ValueError(f"Unsupported intraday session override: {override}")
        return SESSION_SPECS[override]

    t = str(ticker).upper().strip()
    if t.endswith("-USD"):
        return SESSION_SPECS["Crypto 24/7"]
    if t.endswith(".IS"):
        return SESSION_SPECS["BIST Cash"]
    if t.endswith("=F"):
        return SESSION_SPECS["CME Metals"]
    return SESSION_SPECS["US Cash"]


def withhold_incomplete_intraday_bar(
    df: pd.DataFrame,
    interval: str,
    *,
    now_utc: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, dict]:
    """Explicitly withhold an in-progress 15-minute bar.

    Yahoo intraday timestamps represent bar starts. If the last timestamp is timezone-aware,
    the bar is considered complete only when start+15 minutes <= current UTC time.
    A naive timestamp cannot be verified safely; in that case the frame is left unchanged and
    the audit explicitly reports that completion could not be verified.
    """
    x = df.copy()
    audit = {
        "incomplete_intraday_bar_withheld": "NO",
        "withheld_intraday_timestamp": "",
        "intraday_completion_check": "NOT APPLICABLE" if interval != "15m" else "PENDING",
    }
    if interval != "15m" or len(x) == 0:
        return x, audit

    idx = pd.DatetimeIndex(x.index)
    if idx.tz is None:
        audit["intraday_completion_check"] = "UNVERIFIED — timezone-naive Yahoo index"
        return x, audit

    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    last_start_utc = pd.Timestamp(idx[-1]).tz_convert("UTC")
    last_end_utc = last_start_utc + pd.Timedelta(minutes=15)

    if last_end_utc > now:
        audit["incomplete_intraday_bar_withheld"] = "YES"
        audit["withheld_intraday_timestamp"] = str(idx[-1])
        audit["intraday_completion_check"] = "WITHHELD — last 15m bar still in progress"
        x = x.iloc[:-1].copy()
    else:
        audit["intraday_completion_check"] = "PASS — latest 15m bar completed"
    return x, audit


def _local_index(index: pd.DatetimeIndex, spec: SessionSpec) -> tuple[pd.DatetimeIndex, str]:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        # We do not modify the market timestamps. For session classification only, interpret
        # the existing wall-clock labels as exchange-local and disclose the assumption.
        return idx, f"NAIVE WALL-CLOCK ASSUMED {spec.timezone} FOR SESSION CLASSIFICATION ONLY"
    return idx.tz_convert(spec.timezone), f"TIMEZONE-AWARE → {spec.timezone}"


def _session_labels(local_idx: pd.DatetimeIndex, spec: SessionSpec) -> tuple[pd.Series, pd.Series, pd.Series]:
    minutes = pd.Series(local_idx.hour * 60 + local_idx.minute, index=local_idx)
    dates = pd.Series(pd.DatetimeIndex(local_idx).tz_localize(None).normalize(), index=local_idx)

    if spec.continuous_7d:
        in_session = pd.Series(True, index=local_idx)
        session_date = dates.copy()
    elif spec.rollover:
        in_session = (minutes >= spec.open_minute) | (minutes < spec.close_minute)
        session_date = dates.copy()
        evening = minutes >= spec.open_minute
        session_date.loc[evening] = session_date.loc[evening] + pd.Timedelta(days=1)
        session_date.loc[~in_session] = pd.NaT
    else:
        in_session = (minutes >= spec.open_minute) & (minutes < spec.close_minute)
        session_date = dates.where(in_session, pd.NaT)

    return session_date, in_session.astype(bool), minutes


def compute_intraday_features(
    df: pd.DataFrame,
    ticker: str,
    config: IntradayConfig = IntradayConfig(),
) -> tuple[pd.DataFrame, dict]:
    """Compute causal 15-minute session diagnostics from an already-observed market frame."""
    config.validate()
    required = ["AdjOpen", "AdjHigh", "AdjLow", "AdjCloseCalc", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Intraday Tactical Lab missing required columns: {missing}")
    if len(df) < max(config.realized_vol_window + 2, config.atr_window + 2):
        raise ValueError("Not enough 15-minute observations for the selected intraday windows.")

    spec = infer_session_spec(ticker, config.session_override)
    x = df.copy()
    idx = pd.DatetimeIndex(x.index)
    local_idx, tz_audit = _local_index(idx, spec)
    session_date_local, in_session_local, minutes_local = _session_labels(local_idx, spec)

    # Re-key the session metadata back to the original market index without altering prices.
    session_dates = pd.Series(session_date_local.to_numpy(), index=x.index, dtype="datetime64[ns]")
    in_session = pd.Series(in_session_local.to_numpy(bool), index=x.index)
    local_minutes = pd.Series(minutes_local.to_numpy(int), index=x.index)

    x["IntradaySessionDate"] = session_dates
    x["IntradayInPrimarySession"] = in_session
    x["IntradayLocalMinute"] = local_minutes
    x["IntradaySessionBar"] = np.nan

    valid_idx = x.index[in_session & session_dates.notna()]
    if len(valid_idx) == 0:
        raise ValueError(f"No bars fell inside the selected session model: {spec.label}")

    valid = x.loc[valid_idx].copy()
    valid["_SessionDate"] = session_dates.loc[valid_idx].to_numpy()
    valid["_SessionBar"] = valid.groupby("_SessionDate", sort=True).cumcount() + 1
    x.loc[valid_idx, "IntradaySessionBar"] = valid["_SessionBar"].to_numpy(float)

    # Price/volume source series.
    high = pd.to_numeric(x["AdjHigh"], errors="raise").astype(float)
    low = pd.to_numeric(x["AdjLow"], errors="raise").astype(float)
    close = pd.to_numeric(x["AdjCloseCalc"], errors="raise").astype(float)
    open_ = pd.to_numeric(x["AdjOpen"], errors="raise").astype(float)
    volume = pd.to_numeric(x["Volume"], errors="raise").astype(float)
    typical = (high + low + close) / 3.0

    # Initialise all outputs so outside-session bars remain explicit NaN/False rather than filled.
    for col in [
        "SessionVWAP", "SessionOpen", "PriorSessionClose", "SessionGapPct", "SessionReturnPct",
        "SessionHigh", "SessionLow", "SessionDrawdownPct", "OpeningRangeHigh", "OpeningRangeLow",
        "OpeningRangeWidthPct", "SlotExpectedVolume", "SlotRelativeVolume", "IntradayATR",
        "IntradayRealizedVol", "VWAPGapPct", "VWAPGapATR", "IntradayConfirmationScore",
    ]:
        x[col] = np.nan
    for col in [
        "OpeningRangeFinalized", "OpeningRangeBreakoutUp", "OpeningRangeBreakoutDown",
        "VWAPCrossUp", "VWAPCrossDown", "IntradayRVOLClimax",
    ]:
        x[col] = False
    x["OpeningRangePosition"] = 0
    x["IntradayConfirmationState"] = "OUTSIDE PRIMARY SESSION"

    # Session-causal VWAP / opening range / drawdown / gap.
    unique_sessions = [s for s in pd.Series(session_dates.loc[valid_idx]).dropna().drop_duplicates().sort_values()]
    prior_close = np.nan
    for sess in unique_sessions:
        mask = in_session & session_dates.eq(sess)
        pos = x.index[mask]
        if len(pos) == 0:
            continue
        sess_typical = typical.loc[pos]
        sess_volume = volume.loc[pos]
        cum_vol = sess_volume.cumsum()
        cum_pv = (sess_typical * sess_volume).cumsum()
        sess_vwap = cum_pv / cum_vol.replace(0.0, np.nan)
        x.loc[pos, "SessionVWAP"] = sess_vwap.to_numpy(float)

        sess_open = float(open_.loc[pos[0]])
        x.loc[pos, "SessionOpen"] = sess_open
        x.loc[pos, "PriorSessionClose"] = prior_close
        if np.isfinite(prior_close) and prior_close > 0:
            x.loc[pos, "SessionGapPct"] = sess_open / prior_close - 1.0
        x.loc[pos, "SessionReturnPct"] = close.loc[pos].to_numpy(float) / sess_open - 1.0
        x.loc[pos, "SessionHigh"] = high.loc[pos].cummax().to_numpy(float)
        x.loc[pos, "SessionLow"] = low.loc[pos].cummin().to_numpy(float)
        x.loc[pos, "SessionDrawdownPct"] = close.loc[pos].to_numpy(float) / close.loc[pos].cummax().to_numpy(float) - 1.0

        # Causal opening range: evolves during first N bars, then freezes.
        n_or = min(config.opening_range_bars, len(pos))
        highs = high.loc[pos].to_numpy(float)
        lows = low.loc[pos].to_numpy(float)
        or_high = np.empty(len(pos), float)
        or_low = np.empty(len(pos), float)
        for j in range(len(pos)):
            k = min(j + 1, n_or)
            or_high[j] = float(np.max(highs[:k]))
            or_low[j] = float(np.min(lows[:k]))
        x.loc[pos, "OpeningRangeHigh"] = or_high
        x.loc[pos, "OpeningRangeLow"] = or_low
        finalized = np.arange(1, len(pos) + 1) >= config.opening_range_bars
        x.loc[pos, "OpeningRangeFinalized"] = finalized
        width = np.where(or_low > 0, or_high / or_low - 1.0, np.nan)
        x.loc[pos, "OpeningRangeWidthPct"] = width
        prior_close = float(close.loc[pos[-1]])

    # Same-slot relative volume: only PRIOR sessions of the same bar number enter the baseline.
    slot_df = x.loc[valid_idx, ["IntradaySessionDate", "IntradaySessionBar", "Volume"]].copy()
    slot_df = slot_df.sort_values(["IntradaySessionBar", "IntradaySessionDate"])
    slot_df["SlotExpectedVolume"] = np.nan
    for slot, g in slot_df.groupby("IntradaySessionBar", sort=True):
        s = pd.to_numeric(g["Volume"], errors="raise").astype(float)
        minp = min(3, config.slot_rvol_sessions)
        baseline = s.shift(1).rolling(config.slot_rvol_sessions, min_periods=minp).median()
        slot_df.loc[g.index, "SlotExpectedVolume"] = baseline.to_numpy(float)
    expected = slot_df["SlotExpectedVolume"].reindex(valid_idx)
    x.loc[valid_idx, "SlotExpectedVolume"] = expected.to_numpy(float)
    rvol = volume.loc[valid_idx] / expected.replace(0.0, np.nan)
    x.loc[valid_idx, "SlotRelativeVolume"] = rvol.to_numpy(float)
    x.loc[valid_idx, "IntradayRVOLClimax"] = (rvol >= config.rvol_climax).fillna(False).to_numpy(bool)

    # Rolling ATR and realized volatility from observed bars only.
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    x["IntradayATR"] = tr.rolling(config.atr_window, min_periods=config.atr_window).mean()
    logret = np.log(close / close.shift(1))
    bars_per_year = spec.sessions_per_year * spec.nominal_bars_per_session
    x["IntradayRealizedVol"] = logret.rolling(
        config.realized_vol_window, min_periods=config.realized_vol_window
    ).std(ddof=1) * np.sqrt(float(bars_per_year))

    x["VWAPGapPct"] = close / x["SessionVWAP"] - 1.0
    x["VWAPGapATR"] = (close - x["SessionVWAP"]) / x["IntradayATR"]

    # Causal crosses require the previous bar to be in the same session.
    same_session_prev = session_dates.eq(session_dates.shift(1)) & session_dates.notna()
    or_final = x["OpeningRangeFinalized"].fillna(False).astype(bool)
    x["OpeningRangeBreakoutUp"] = (
        same_session_prev & or_final & (close > x["OpeningRangeHigh"]) &
        (close.shift(1) <= x["OpeningRangeHigh"].shift(1))
    ).fillna(False)
    x["OpeningRangeBreakoutDown"] = (
        same_session_prev & or_final & (close < x["OpeningRangeLow"]) &
        (close.shift(1) >= x["OpeningRangeLow"].shift(1))
    ).fillna(False)
    x["VWAPCrossUp"] = (
        same_session_prev & (close > x["SessionVWAP"]) & (close.shift(1) <= x["SessionVWAP"].shift(1))
    ).fillna(False)
    x["VWAPCrossDown"] = (
        same_session_prev & (close < x["SessionVWAP"]) & (close.shift(1) >= x["SessionVWAP"].shift(1))
    ).fillna(False)

    x.loc[or_final & (close > x["OpeningRangeHigh"]), "OpeningRangePosition"] = 1
    x.loc[or_final & (close < x["OpeningRangeLow"]), "OpeningRangePosition"] = -1

    # Explainable confirmation score. It is a diagnostic, not a hidden exposure override.
    score = pd.Series(0.0, index=x.index)
    available = pd.Series(0, index=x.index, dtype=int)

    if "NWDirection" in x.columns:
        nd = pd.to_numeric(x["NWDirection"], errors="coerce")
        score += np.where(nd > 0, 25.0, np.where(nd < 0, -25.0, 0.0))
        available += nd.notna().astype(int)
    if "NWMomentumUpwardWarning" in x.columns:
        _mup = x["NWMomentumUpwardWarning"].fillna(False).astype(bool)
        _mdn = x["NWMomentumDownwardWarning"].fillna(False).astype(bool) if "NWMomentumDownwardWarning" in x.columns else pd.Series(False, index=x.index)
        score += _mup.astype(float) * 10.0
        score -= _mdn.astype(float) * 10.0
        available += 1

    vwap_valid = x["SessionVWAP"].notna()
    score += np.where(vwap_valid & (close > x["SessionVWAP"]), 15.0, np.where(vwap_valid & (close < x["SessionVWAP"]), -15.0, 0.0))
    available += vwap_valid.astype(int)

    or_valid = x["OpeningRangeFinalized"].fillna(False).astype(bool)
    score += np.where(or_valid & (x["OpeningRangePosition"] > 0), 15.0, np.where(or_valid & (x["OpeningRangePosition"] < 0), -15.0, 0.0))
    available += or_valid.astype(int)

    if "ResidualDriftZ" in x.columns:
        rz = pd.to_numeric(x["ResidualDriftZ"], errors="coerce")
        score += rz.clip(-3.0, 3.0).fillna(0.0) / 3.0 * 20.0
        available += rz.notna().astype(int)

    bar_ret = close.pct_change()
    rvol_valid = x["SlotRelativeVolume"].notna()
    directional_climax = x["IntradayRVOLClimax"].fillna(False).astype(bool)
    score += np.where(directional_climax & (bar_ret > 0), 10.0, np.where(directional_climax & (bar_ret < 0), -10.0, 0.0))
    available += rvol_valid.astype(int)

    score = score.clip(-100.0, 100.0)
    score = score.where(in_session, np.nan)
    x["IntradayConfirmationScore"] = score
    x["IntradayScoreComponentsAvailable"] = available.where(in_session, 0)

    state = pd.Series("NEUTRAL", index=x.index, dtype=object)
    state.loc[score >= config.score_up_threshold] = "UP CONFIRMATION"
    state.loc[score >= config.score_strong_threshold] = "STRONG UP CONFIRMATION"
    state.loc[score <= -config.score_up_threshold] = "DOWN CONFIRMATION"
    state.loc[score <= -config.score_strong_threshold] = "STRONG DOWN CONFIRMATION"
    state.loc[~in_session] = "OUTSIDE PRIMARY SESSION"
    x["IntradayConfirmationState"] = state

    audit = {
        "session_code": spec.code,
        "session_label": spec.label,
        "session_timezone": spec.timezone,
        "timestamp_handling": tz_audit,
        "opening_range_bars": config.opening_range_bars,
        "slot_rvol_sessions": config.slot_rvol_sessions,
        "atr_window": config.atr_window,
        "realized_vol_window": config.realized_vol_window,
        "bars_per_year": bars_per_year,
        "bars_total": int(len(x)),
        "bars_in_primary_session": int(in_session.sum()),
        "bars_outside_primary_session": int((~in_session).sum()),
        "sessions_observed": int(pd.Series(session_dates).dropna().nunique()),
    }
    return x, audit


def intraday_snapshot(x: pd.DataFrame, config: IntradayConfig, audit: dict) -> dict:
    if len(x) == 0:
        raise ValueError("Intraday snapshot requires observations.")
    eligible = x[x["IntradayInPrimarySession"].fillna(False).astype(bool)]
    if len(eligible) == 0:
        raise ValueError("No primary-session observations available for intraday snapshot.")
    r = eligible.iloc[-1]

    or_status = "FINAL" if bool(r.get("OpeningRangeFinalized", False)) else "FORMING"
    or_pos = int(r.get("OpeningRangePosition", 0))
    or_state = "ABOVE OR" if or_pos > 0 else "BELOW OR" if or_pos < 0 else "INSIDE OR"

    gates = pd.DataFrame([
        {"Gate": "NW Trend", "Status": "BULLISH" if float(r.get("NWDirection", 0)) > 0 else "BEARISH" if float(r.get("NWDirection", 0)) < 0 else "FLAT/NA", "Value": r.get("NWNormalizedSlope", np.nan)},
        {"Gate": "Session VWAP", "Status": "ABOVE" if r["AdjCloseCalc"] > r["SessionVWAP"] else "BELOW" if pd.notna(r["SessionVWAP"]) else "NA", "Value": r.get("VWAPGapPct", np.nan)},
        {"Gate": "Opening Range", "Status": f"{or_status} / {or_state}", "Value": r.get("OpeningRangeWidthPct", np.nan)},
        {"Gate": "Same-Slot Relative Volume", "Status": "CLIMAX" if bool(r.get("IntradayRVOLClimax", False)) else "NORMAL / WARM-UP", "Value": r.get("SlotRelativeVolume", np.nan)},
        {"Gate": "Benchmark Residual Drift", "Status": "AVAILABLE" if pd.notna(r.get("ResidualDriftZ", np.nan)) else "NA / WARM-UP", "Value": r.get("ResidualDriftZ", np.nan)},
        {"Gate": "Session Drawdown", "Status": "CURRENT", "Value": r.get("SessionDrawdownPct", np.nan)},
        {"Gate": "Intraday Confirmation", "Status": str(r.get("IntradayConfirmationState", "NEUTRAL")), "Value": r.get("IntradayConfirmationScore", np.nan)},
    ])

    return {
        "timestamp": eligible.index[-1],
        "session_date": r.get("IntradaySessionDate"),
        "session_label": audit.get("session_label", ""),
        "confirmation_state": str(r.get("IntradayConfirmationState", "NEUTRAL")),
        "confirmation_score": float(r.get("IntradayConfirmationScore", np.nan)),
        "vwap": float(r.get("SessionVWAP", np.nan)),
        "vwap_gap_pct": float(r.get("VWAPGapPct", np.nan)),
        "opening_range_high": float(r.get("OpeningRangeHigh", np.nan)),
        "opening_range_low": float(r.get("OpeningRangeLow", np.nan)),
        "opening_range_finalized": bool(r.get("OpeningRangeFinalized", False)),
        "slot_rvol": float(r.get("SlotRelativeVolume", np.nan)),
        "realized_vol": float(r.get("IntradayRealizedVol", np.nan)),
        "intraday_atr": float(r.get("IntradayATR", np.nan)),
        "session_return_pct": float(r.get("SessionReturnPct", np.nan)),
        "session_drawdown_pct": float(r.get("SessionDrawdownPct", np.nan)),
        "session_gap_pct": float(r.get("SessionGapPct", np.nan)),
        "gates": gates,
        "timing_note": (
            "Intraday metrics are computed from completed 15-minute bars only. The confirmation score is a research diagnostic; "
            "it does not silently overwrite Institutional Tactical target exposure. Any portfolio action remains completed-bar → next-open."
        ),
    }
