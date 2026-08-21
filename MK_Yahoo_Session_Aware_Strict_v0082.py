"""
MK Yahoo Session-Aware Strict Adapter v0.08.2
By Murat Konuklar

Yahoo Finance is the only live market-data source.

Core governance:
- no alternate market-data provider
- no synthetic market observations
- no forward-fill / back-fill
- no temporal interpolation
- no Close -> Adj Close substitution
- no silent omission of an expected completed trading session
- exchange-calendar non-session placeholders are quarantined and audited
- an unfinished current daily session is withheld until completed
- for BIST daily data every expected completed XIST session must be present
- multiple Yahoo retrieval paths can reconcile only if their observed values agree
- unresolved completed-session gaps hard-stop the analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import (
    REQUIRED_COLUMNS,
    validate_market_data,
    DataIntegrityError,
    MarketDataError,
)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
FINALIZATION_BUFFER_MINUTES = 20


@dataclass
class RouteResult:
    name: str
    raw: pd.DataFrame
    transport_error: str = ""
    normalized: pd.DataFrame = field(default_factory=pd.DataFrame)
    non_session_placeholders: list[str] = field(default_factory=list)
    unfinished_session_rows: list[str] = field(default_factory=list)
    partial_rows: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class YahooAudit:
    ticker: str
    interval: str
    requested_start: str
    requested_end: str
    effective_completed_cutoff: str
    accepted_mode: str
    routes_attempted: list[str]
    routes_with_data: list[str]
    reconciled: bool
    observations: int
    non_session_placeholders: list[str]
    unfinished_session_rows: list[str]
    recovered_sessions: list[str]
    unresolved_completed_sessions: list[str]
    partial_rows: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "interval": self.interval,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "effective_completed_cutoff": self.effective_completed_cutoff,
            "accepted_mode": self.accepted_mode,
            "routes_attempted": self.routes_attempted,
            "routes_with_data": self.routes_with_data,
            "reconciled": self.reconciled,
            "observations": self.observations,
            "non_session_placeholders": self.non_session_placeholders,
            "unfinished_session_rows": self.unfinished_session_rows,
            "recovered_sessions": self.recovered_sessions,
            "unresolved_completed_sessions": self.unresolved_completed_sessions,
            "partial_rows": self.partial_rows,
        }


def _is_bist(ticker: str) -> bool:
    return str(ticker).upper().endswith(".IS")


def _get_xist_calendar():
    try:
        import exchange_calendars as xcals
    except Exception as e:
        raise MarketDataError(
            "exchange-calendars is required for BIST session validation; strict mode will not guess exchange holidays."
        ) from e
    return xcals.get_calendar("XIST")


def _completed_xist_sessions(
    start: str,
    end: str,
    *,
    now_utc: pd.Timestamp | None = None,
    finalization_buffer_minutes: int = FINALIZATION_BUFFER_MINUTES,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, str]:
    cal = _get_xist_calendar()
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    sessions = pd.DatetimeIndex(cal.sessions_in_range(s, e)).normalize()

    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    complete = []
    for sess in sessions:
        close_utc = cal.session_close(sess)
        if close_utc + pd.Timedelta(minutes=finalization_buffer_minutes) <= now:
            complete.append(pd.Timestamp(sess).normalize())

    completed = pd.DatetimeIndex(complete)
    cutoff = completed[-1].date().isoformat() if len(completed) else ""
    return sessions, completed, cutoff


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        if x.columns.nlevels != 2:
            raise DataIntegrityError("Unexpected Yahoo MultiIndex depth.")
        lvl0 = set(x.columns.get_level_values(0))
        lvl1 = set(x.columns.get_level_values(1))
        if set(REQUIRED_COLUMNS).issubset(lvl0):
            x.columns = x.columns.get_level_values(0)
        elif set(REQUIRED_COLUMNS).issubset(lvl1):
            x.columns = x.columns.get_level_values(1)
        else:
            raise DataIntegrityError("Unexpected Yahoo MultiIndex column mapping.")
    if "AdjClose" in x.columns and "Adj Close" not in x.columns:
        x = x.rename(columns={"AdjClose": "Adj Close"})
    return x


def _canonicalize_index(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="raise")
    if x.index.tz is not None:
        x.index = x.index.tz_localize(None)
    if interval in {"1d", "1wk", "1mo"}:
        x.index = x.index.normalize()
    return x[~x.index.duplicated(keep="last")].sort_index()


def _normalize_yahoo_frame(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    return _canonicalize_index(_canonicalize_columns(df), interval)


def missing_detail(df: pd.DataFrame) -> dict[str, list[str]]:
    if df is None or len(df) == 0:
        return {"<EMPTY>": list(REQUIRED_COLUMNS)}
    detail: dict[str, list[str]] = {}
    absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absent:
        detail["<COLUMN_MISSING>"] = absent
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if present:
        bad = df[present].isna()
        for idx in df.index[bad.any(axis=1)]:
            detail[str(idx)] = [c for c in present if pd.isna(df.at[idx, c])]
    return detail


def _row_is_empty_placeholder(row: pd.Series) -> bool:
    prices_missing = all(pd.isna(row.get(c, np.nan)) for c in PRICE_COLUMNS)
    volume = row.get("Volume", np.nan)
    volume_empty = pd.isna(volume) or float(volume) == 0.0
    return bool(prices_missing and volume_empty)


def _prepare_route(
    route: RouteResult,
    *,
    ticker: str,
    interval: str,
    start: str,
    end: str,
    now_utc: pd.Timestamp | None = None,
) -> RouteResult:
    if route.raw is None or len(route.raw) == 0:
        route.normalized = pd.DataFrame()
        return route

    x = _normalize_yahoo_frame(route.raw, interval)

    if _is_bist(ticker) and interval == "1d":
        all_sessions, completed_sessions, _ = _completed_xist_sessions(
            start, end, now_utc=now_utc
        )
        all_set = set(all_sessions)
        completed_set = set(completed_sessions)
        keep = np.ones(len(x), dtype=bool)

        for pos, (idx, row) in enumerate(x.iterrows()):
            d = pd.Timestamp(idx).normalize()

            # Officially closed calendar date + empty Yahoo placeholder:
            # not a market observation; quarantine it explicitly.
            if d not in all_set and _row_is_empty_placeholder(row):
                route.non_session_placeholders.append(str(d))
                keep[pos] = False
                continue

            # A valid exchange session that has not yet been finalized:
            # do not allow a partial daily bar into a completed-bar strategy.
            if d in all_set and d not in completed_set:
                route.unfinished_session_rows.append(str(d))
                keep[pos] = False
                continue

            missing = [
                c for c in REQUIRED_COLUMNS
                if c not in x.columns or pd.isna(row.get(c, np.nan))
            ]
            if missing:
                route.partial_rows[str(d)] = missing

        x = x.loc[keep]

    route.normalized = x
    return route


def _values_agree(values: list[float], column: str) -> bool:
    if len(values) <= 1:
        return True
    arr = np.asarray(values, dtype=float)
    if column == "Volume":
        return bool(np.all(np.isclose(arr, arr[0], rtol=0.0, atol=0.5)))
    return bool(np.all(np.isclose(arr, arr[0], rtol=1e-7, atol=1e-8)))


def _consensus_frame(
    routes: list[RouteResult],
    *,
    ticker: str,
    interval: str,
    start: str,
    end: str,
    now_utc: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], str]:
    usable = [r.normalized for r in routes if r.normalized is not None and len(r.normalized)]
    if not usable:
        return pd.DataFrame(), [], {"<EMPTY>": list(REQUIRED_COLUMNS)}, ""

    if _is_bist(ticker) and interval == "1d":
        _, expected, cutoff = _completed_xist_sessions(start, end, now_utc=now_utc)
        target_index = expected
    else:
        target_index = usable[0].index
        for u in usable[1:]:
            target_index = target_index.union(u.index)
        target_index = target_index.sort_values()
        cutoff = str(target_index[-1]) if len(target_index) else ""

    out = pd.DataFrame(index=target_index, columns=REQUIRED_COLUMNS, dtype=float)
    conflicts: list[str] = []

    for idx in target_index:
        for c in REQUIRED_COLUMNS:
            observed: list[float] = []
            for r in routes:
                rf = r.normalized
                if rf is None or len(rf) == 0 or idx not in rf.index or c not in rf.columns:
                    continue
                v = rf.at[idx, c]
                if pd.notna(v):
                    observed.append(float(v))
            if observed:
                if not _values_agree(observed, c):
                    conflicts.append(
                        f"{idx} {c}: Yahoo routes disagree {observed[:4]}"
                    )
                else:
                    out.at[idx, c] = observed[0]

    if conflicts:
        raise DataIntegrityError(
            "Yahoo same-source routes disagree on observed values. "
            + " | ".join(conflicts[:8])
        )

    detail = missing_detail(out)
    unresolved = [
        str(idx) for idx in out.index
        if any(pd.isna(out.at[idx, c]) for c in REQUIRED_COLUMNS)
    ]
    return out, unresolved, detail, cutoff


def _direct_chart_request(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    host: str,
) -> pd.DataFrame:
    try:
        import requests
    except Exception as e:
        raise MarketDataError("requests is required for Yahoo Chart reconciliation.") from e

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())

    url = f"https://{host}/v8/finance/chart/{quote(ticker, safe='')}"
    params = {
        "period1": p1,
        "period2": p2,
        "interval": interval,
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MKTrendFollowing/0.08.2)"}
    response = requests.get(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise MarketDataError(f"Yahoo Chart API {host}: {chart['error']}")
    result_list = chart.get("result") or []
    if not result_list:
        return pd.DataFrame()

    result = result_list[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame()

    indicators = result.get("indicators") or {}
    q = (indicators.get("quote") or [{}])[0]
    adj_blocks = indicators.get("adjclose") or [{}]
    adj = adj_blocks[0] if adj_blocks else {}
    n = len(timestamps)

    def _arr(block, key):
        values = block.get(key)
        if values is None:
            return [np.nan] * n
        if len(values) != n:
            raise DataIntegrityError(f"Yahoo {host} length mismatch for {key}.")
        return values

    meta = result.get("meta") or {}
    timezone = meta.get("exchangeTimezoneName") or "UTC"
    idx = pd.to_datetime(timestamps, unit="s", utc=True)
    try:
        idx = idx.tz_convert(timezone)
    except Exception:
        pass
    idx = idx.tz_localize(None)
    if interval in {"1d", "1wk", "1mo"}:
        idx = idx.normalize()

    return pd.DataFrame(
        {
            "Open": _arr(q, "open"),
            "High": _arr(q, "high"),
            "Low": _arr(q, "low"),
            "Close": _arr(q, "close"),
            "Volume": _arr(q, "volume"),
            "Adj Close": _arr(adj, "adjclose"),
        },
        index=idx,
    )


def _route_yf_download(ticker: str, start: str, end_exclusive: str, interval: str):
    import yfinance as yf
    return yf.download(
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


def _route_yf_history(ticker: str, start: str, end_exclusive: str, interval: str):
    import yfinance as yf
    return yf.Ticker(ticker).history(
        start=str(start),
        end=end_exclusive,
        interval=interval,
        auto_adjust=False,
        actions=False,
        repair=False,
        keepna=True,
        raise_errors=True,
    )


def _fetch_route(name: str, fn) -> RouteResult:
    try:
        return RouteResult(name=name, raw=fn(), transport_error="")
    except Exception as e:
        return RouteResult(name=name, raw=pd.DataFrame(), transport_error=str(e))


def _targeted_recovery_routes(
    ticker: str,
    unresolved_dates: list[str],
    interval: str,
) -> list[RouteResult]:
    routes: list[RouteResult] = []
    if interval != "1d":
        return routes

    for dstr in unresolved_dates:
        d = pd.Timestamp(dstr).normalize()
        w_start = (d - pd.Timedelta(days=2)).date().isoformat()
        w_end = (d + pd.Timedelta(days=2)).date().isoformat()
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            routes.append(
                _fetch_route(
                    f"Targeted {host} / {d.date()}",
                    lambda h=host, s=w_start, e=w_end: _direct_chart_request(
                        ticker, s, e, interval, h
                    ),
                )
            )
    return routes


class YahooFinanceAdapter:
    @staticmethod
    def fetch(
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        minimum_observations: int = 30,
    ) -> pd.DataFrame:
        if not ticker or not str(ticker).strip():
            raise ValueError("ticker is required")

        ticker = str(ticker).strip()
        end_exclusive = (
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        ).date().isoformat()

        routes = [
            _fetch_route(
                "Yahoo Route A / yfinance.download",
                lambda: _route_yf_download(ticker, str(start), end_exclusive, interval),
            ),
            _fetch_route(
                "Yahoo Route B / Ticker.history",
                lambda: _route_yf_history(ticker, str(start), end_exclusive, interval),
            ),
            _fetch_route(
                "Yahoo Route C / query1 Chart API",
                lambda: _direct_chart_request(
                    ticker, str(start), str(end), interval, "query1.finance.yahoo.com"
                ),
            ),
            _fetch_route(
                "Yahoo Route D / query2 Chart API",
                lambda: _direct_chart_request(
                    ticker, str(start), str(end), interval, "query2.finance.yahoo.com"
                ),
            ),
        ]

        for r in routes:
            _prepare_route(r, ticker=ticker, interval=interval, start=str(start), end=str(end))

        try:
            consensus, unresolved, detail, cutoff = _consensus_frame(
                routes, ticker=ticker, interval=interval, start=str(start), end=str(end)
            )
        except Exception as e:
            raise MarketDataError(f"STRICT YAHOO CONSENSUS STOP — {ticker}. {e}") from e

        recovered_sessions: list[str] = []
        if unresolved and _is_bist(ticker) and interval == "1d":
            targeted = _targeted_recovery_routes(ticker, unresolved, interval)
            for r in targeted:
                _prepare_route(r, ticker=ticker, interval=interval, start=str(start), end=str(end))
            routes.extend(targeted)

            consensus2, unresolved2, detail2, cutoff2 = _consensus_frame(
                routes, ticker=ticker, interval=interval, start=str(start), end=str(end)
            )
            recovered_sessions = sorted(set(unresolved) - set(unresolved2))
            consensus, unresolved, detail, cutoff = consensus2, unresolved2, detail2, cutoff2

        non_session_placeholders = sorted(
            set(v for r in routes for v in r.non_session_placeholders)
        )
        unfinished_session_rows = sorted(
            set(v for r in routes for v in r.unfinished_session_rows)
        )
        partial_rows: dict[str, list[str]] = {}
        for r in routes:
            for k, vals in r.partial_rows.items():
                partial_rows.setdefault(k, [])
                partial_rows[k] = sorted(set(partial_rows[k]).union(vals))

        if unresolved:
            transport_errors = {r.name: r.transport_error for r in routes if r.transport_error}
            raise MarketDataError(
                f"STRICT COMPLETED-SESSION DATA STOP — {ticker}. "
                f"Non-session Yahoo placeholders quarantined: {non_session_placeholders}. "
                f"Unfinished current/future sessions withheld: {unfinished_session_rows}. "
                f"Expected COMPLETED trading sessions still unresolved after Yahoo multi-route + targeted recovery: "
                f"{unresolved}. Missing fields: {detail}. Yahoo transport errors: {transport_errors}. "
                "No alternate provider, forward/back fill, temporal interpolation, Close→Adj Close substitution, "
                "or silent omission of a completed trading session was used."
            )

        try:
            validated = validate_market_data(consensus, minimum_observations)
        except Exception as e:
            raise MarketDataError(
                f"STRICT YAHOO DATA STOP — {ticker}. Final Yahoo consensus failed validation: {e}. "
                f"Missing detail: {missing_detail(consensus)}."
            ) from e

        data_routes = [r.name for r in routes if r.normalized is not None and len(r.normalized)]
        audit = YahooAudit(
            ticker=ticker,
            interval=interval,
            requested_start=str(start),
            requested_end=str(end),
            effective_completed_cutoff=cutoff,
            accepted_mode="Yahoo same-source multi-route consensus",
            routes_attempted=[r.name for r in routes],
            routes_with_data=data_routes,
            reconciled=len(data_routes) > 1,
            observations=len(validated),
            non_session_placeholders=non_session_placeholders,
            unfinished_session_rows=unfinished_session_rows,
            recovered_sessions=recovered_sessions,
            unresolved_completed_sessions=[],
            partial_rows=partial_rows,
        )
        validated.attrs["yahoo_audit"] = audit.as_dict()
        return validated
