"""
MK Yahoo Strict Reconciliation Adapter v0.08.1
By Murat Konuklar

Governance:
- Yahoo Finance is the ONLY live market-data source.
- No alternate provider.
- No synthetic market observations.
- No forward-fill / back-fill.
- No Close -> Adj Close substitution.
- No silent row deletion.
- If the first Yahoo retrieval is incomplete, a second Yahoo retrieval path may be attempted.
- The second payload is accepted only if complete and common observed values reconcile.
- Any conflict or persistent incompleteness hard-stops the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import (
    REQUIRED_COLUMNS,
    validate_market_data,
    DataIntegrityError,
    MarketDataError,
)


@dataclass(frozen=True)
class YahooAudit:
    ticker: str
    interval: str
    requested_start: str
    requested_end: str
    accepted_route: str
    primary_status: str
    secondary_status: str
    reconciled: bool
    observations: int
    primary_missing: dict[str, list[str]]
    secondary_missing: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "interval": self.interval,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "accepted_route": self.accepted_route,
            "primary_status": self.primary_status,
            "secondary_status": self.secondary_status,
            "reconciled": self.reconciled,
            "observations": self.observations,
            "primary_missing": self.primary_missing,
            "secondary_missing": self.secondary_missing,
        }


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        if x.columns.nlevels != 2:
            raise DataIntegrityError("Unexpected Yahoo MultiIndex depth; strict mode refuses to guess column mapping.")
        lvl0 = set(x.columns.get_level_values(0))
        lvl1 = set(x.columns.get_level_values(1))
        if set(REQUIRED_COLUMNS).issubset(lvl0):
            x.columns = x.columns.get_level_values(0)
        elif set(REQUIRED_COLUMNS).issubset(lvl1):
            x.columns = x.columns.get_level_values(1)
        else:
            raise DataIntegrityError("Unexpected Yahoo MultiIndex columns; strict mode refuses to guess column mapping.")
    return x


def _canonicalize_index(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="raise")
    if x.index.tz is not None:
        # Preserve exchange-local wall-clock labels while removing tz metadata so
        # Yahoo Route A and Route B can reconcile deterministically.
        x.index = x.index.tz_localize(None)
    if interval in {"1d", "1wk", "1mo"}:
        x.index = x.index.normalize()
    return x.sort_index()


def _normalize_yahoo_frame(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    return _canonicalize_index(_canonicalize_columns(df), interval)


def missing_detail(df: pd.DataFrame) -> dict[str, list[str]]:
    """Exact missing Yahoo fields by timestamp. No values are inferred or created."""
    if df is None or len(df) == 0:
        return {"<EMPTY>": list(REQUIRED_COLUMNS)}
    x = df.copy()
    detail: dict[str, list[str]] = {}
    absent = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if absent:
        detail["<COLUMN_MISSING>"] = absent
    present = [c for c in REQUIRED_COLUMNS if c in x.columns]
    if present:
        mask = x[present].isna()
        for idx in x.index[mask.any(axis=1)]:
            detail[str(idx)] = [c for c in present if pd.isna(x.at[idx, c])]
    return detail


def _strict_validate_candidate(raw: pd.DataFrame, interval: str, minimum_observations: int):
    try:
        normalized = _normalize_yahoo_frame(raw, interval)
    except Exception as e:
        return None, f"NORMALIZATION FAILED: {e}", {"<NORMALIZATION>": [str(e)]}
    detail = missing_detail(normalized)
    try:
        validated = validate_market_data(normalized, minimum_observations)
        return validated, "PASS", detail
    except Exception as e:
        return None, f"FAIL: {e}", detail


def _cross_check_common_observations(primary_raw: pd.DataFrame, secondary_valid: pd.DataFrame, interval: str) -> None:
    """Compare only values actually present in Yahoo Route A against Route B."""
    p = _normalize_yahoo_frame(primary_raw, interval)
    s = secondary_valid.copy()
    common = p.index.intersection(s.index)
    if len(common) == 0:
        raise DataIntegrityError("No common timestamps exist between the two Yahoo retrieval paths.")

    mismatches: list[str] = []
    for c in REQUIRED_COLUMNS:
        if c not in p.columns or c not in s.columns:
            continue
        pv = pd.to_numeric(p.loc[common, c], errors="coerce")
        sv = pd.to_numeric(s.loc[common, c], errors="coerce")
        mask = pv.notna() & sv.notna()
        if not mask.any():
            continue
        if c == "Volume":
            ok = np.isclose(pv[mask].to_numpy(float), sv[mask].to_numpy(float), rtol=0.0, atol=0.5)
        else:
            ok = np.isclose(pv[mask].to_numpy(float), sv[mask].to_numpy(float), rtol=1e-7, atol=1e-8)
        if not np.all(ok):
            idxs = pv[mask].index
            for pos in np.where(~ok)[0][:5]:
                dt = idxs[pos]
                mismatches.append(f"{dt} {c}: routeA={pv.loc[dt]}, routeB={sv.loc[dt]}")
    if mismatches:
        raise DataIntegrityError(
            "Conflicting observed values across Yahoo retrieval paths. Strict mode refuses to choose one. "
            + " | ".join(mismatches[:5])
        )


class YahooFinanceAdapter:
    """
    Strict Yahoo-only adapter with same-source reconciliation.

    Route A: yfinance.download() -> Yahoo Finance
    Route B: yfinance.Ticker(...).history() -> Yahoo Finance
    """

    @staticmethod
    def fetch(ticker: str, start: str, end: str, interval: str = "1d", minimum_observations: int = 30) -> pd.DataFrame:
        if not ticker or not str(ticker).strip():
            raise ValueError("ticker is required")
        try:
            import yfinance as yf
        except Exception as e:
            raise MarketDataError("yfinance is not installed. No alternate provider is permitted.") from e

        ticker = str(ticker).strip()
        end_exclusive = (pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).date().isoformat()

        # Route A
        try:
            primary_raw = yf.download(
                tickers=ticker,
                start=str(start),
                end=end_exclusive,
                interval=interval,
                auto_adjust=False,
                repair=False,
                actions=False,
                keepna=True,
                progress=False,
                threads=False,
                group_by="column",
                multi_level_index=False,
                timeout=20,
            )
            primary_transport_error = ""
        except Exception as e:
            primary_raw = pd.DataFrame()
            primary_transport_error = str(e)

        primary_valid, primary_status, primary_missing = _strict_validate_candidate(
            primary_raw, interval, minimum_observations
        )
        if primary_valid is not None:
            audit = YahooAudit(
                ticker=ticker, interval=interval, requested_start=str(start), requested_end=str(end),
                accepted_route="Yahoo Route A / yfinance.download", primary_status="PASS",
                secondary_status="NOT REQUIRED", reconciled=False, observations=len(primary_valid),
                primary_missing=primary_missing, secondary_missing={},
            )
            primary_valid.attrs["yahoo_audit"] = audit.as_dict()
            return primary_valid

        # Route B: second Yahoo retrieval path. This is NOT an alternate provider.
        time.sleep(0.35)
        try:
            secondary_raw = yf.Ticker(ticker).history(
                start=str(start),
                end=end_exclusive,
                interval=interval,
                auto_adjust=False,
                actions=False,
                repair=False,
                keepna=True,
                raise_errors=True,
            )
            secondary_transport_error = ""
        except Exception as e:
            secondary_raw = pd.DataFrame()
            secondary_transport_error = str(e)

        secondary_valid, secondary_status, secondary_missing = _strict_validate_candidate(
            secondary_raw, interval, minimum_observations
        )
        if secondary_valid is None:
            raise MarketDataError(
                f"STRICT YAHOO DATA STOP — {ticker}. "
                f"Route A failed/incomplete: {primary_transport_error or primary_status}. "
                f"Route A missing fields: {primary_missing}. "
                f"Route B failed/incomplete: {secondary_transport_error or secondary_status}. "
                f"Route B missing fields: {secondary_missing}. "
                "No alternate provider, fill, row deletion, or substitution was used."
            )

        reconciled = False
        if primary_raw is not None and len(primary_raw) > 0:
            try:
                _cross_check_common_observations(primary_raw, secondary_valid, interval)
                reconciled = True
            except Exception as e:
                raise MarketDataError(
                    f"STRICT YAHOO RECONCILIATION STOP — {ticker}. "
                    f"Route B was complete, but it conflicted with values already observed in Route A: {e}. "
                    "No payload was selected."
                ) from e

        audit = YahooAudit(
            ticker=ticker, interval=interval, requested_start=str(start), requested_end=str(end),
            accepted_route="Yahoo Route B / Ticker.history after Route A failed or incomplete",
            primary_status=primary_transport_error or primary_status, secondary_status="PASS",
            reconciled=reconciled, observations=len(secondary_valid),
            primary_missing=primary_missing, secondary_missing=secondary_missing,
        )
        secondary_valid.attrs["yahoo_audit"] = audit.as_dict()
        return secondary_valid
