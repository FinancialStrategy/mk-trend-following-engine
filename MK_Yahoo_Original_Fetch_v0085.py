"""
MK Yahoo Original Fetch Adapter v0.08.5.2
By Murat Konuklar

Purpose
-------
Keep the original, previously stable Yahoo acquisition contract while adding
ONLY transport resilience for temporary Yahoo rate limiting.

Live market-data source:
    Yahoo Finance via yfinance.download ONLY.

Explicitly NOT used:
- Ticker.history
- query1/query2 direct history routes
- alternate providers
- multi-route consensus
- exchange-calendar enforcement
- synthetic observations
- forward-fill / back-fill
- interpolation
- Close -> Adj Close substitution

Rate-limit resilience
---------------------
The exact same yf.download request can be retried after bounded backoff if the
transport raises or returns an empty payload. A valid non-empty Yahoo payload is
never altered by the retry layer.

The Streamlit application also caches successful requests, so normal UI reruns
do not repeatedly hit Yahoo for identical ticker/date/interval requests.
"""
from __future__ import annotations

import time
import pandas as pd

from MK_Trend_Following_Engine_v001 import (
    validate_market_data,
    MarketDataError,
)


RETRY_DELAYS_SECONDS = (0, 5, 20)
MAX_ATTEMPTS = len(RETRY_DELAYS_SECONDS)


def _last_yfinance_error(ticker: str) -> str:
    """
    Best-effort diagnostic only.
    yfinance sometimes logs a failed ticker and returns an empty DataFrame
    rather than raising the underlying exception.
    """
    try:
        import yfinance.shared as shared
        errors = getattr(shared, "_ERRORS", {}) or {}
        value = errors.get(str(ticker))
        return "" if value is None else str(value)
    except Exception:
        return ""


def _looks_rate_limited(text: str) -> bool:
    x = str(text).lower()
    return (
        "ratelimit" in x
        or "rate limit" in x
        or "too many requests" in x
        or "429" in x
    )


def _download_once(yf, ticker: str, start: str, end_exclusive: str, interval: str):
    # IMPORTANT: this call signature is intentionally the original working one.
    return yf.download(
        tickers=ticker,
        start=str(start),
        end=end_exclusive,
        interval=interval,
        auto_adjust=False,
        repair=False,
        actions=False,
        keepna=False,
        progress=False,
        threads=False,
        group_by="column",
        multi_level_index=False,
        timeout=20,
    )


class YahooFinanceAdapter:
    """Original Yahoo/yfinance fetch with bounded same-call retry only."""

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

        try:
            import yfinance as yf
        except Exception as e:
            raise MarketDataError(
                "yfinance is not installed. No alternate source is used."
            ) from e

        ticker = str(ticker).strip()

        # yfinance end is EXCLUSIVE; user-facing end is inclusive.
        end_exclusive = (
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        ).date().isoformat()

        last_error = ""
        rate_limited = False
        raw = None
        attempts_used = 0

        for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
            attempts_used = attempt
            if delay:
                time.sleep(delay)

            try:
                raw = _download_once(
                    yf=yf,
                    ticker=ticker,
                    start=str(start),
                    end_exclusive=end_exclusive,
                    interval=interval,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                rate_limited = rate_limited or _looks_rate_limited(last_error)
                raw = None
                continue

            # yfinance can log YFRateLimitError and return an empty DataFrame.
            if raw is None or len(raw) == 0:
                diag = _last_yfinance_error(ticker)
                last_error = diag or "Yahoo returned an empty payload."
                rate_limited = rate_limited or _looks_rate_limited(last_error)
                continue

            # IMPORTANT: validation errors are NOT retried or masked.
            # We have a non-empty Yahoo payload; strict validation applies once.
            try:
                validated = validate_market_data(raw, minimum_observations)
            except Exception as exc:
                raise MarketDataError(
                    f"Yahoo Finance data failed strict validation for {ticker}. "
                    f"No fallback will be used: {exc}"
                ) from exc

            validated.attrs["yahoo_audit"] = {
                "ticker": ticker,
                "interval": interval,
                "requested_start": str(start),
                "requested_end": str(end),
                "accepted_mode": "Original Yahoo / yfinance.download",
                "observations": len(validated),
                "effective_completed_cutoff": (
                    validated.index[-1].date().isoformat()
                    if len(validated) and hasattr(validated.index[-1], "date")
                    else str(validated.index[-1]) if len(validated) else ""
                ),
                "primary_status": "PASS",
                "retry_status": (
                    "NOT USED" if attempts_used == 1
                    else f"SUCCESS ON ATTEMPT {attempts_used}"
                ),
                "attempts_used": attempts_used,
                "historical_missing_sessions": [],
                "critical_missing_sessions": [],
                "recovered_sessions": [],
                "unfinished_sessions_withheld": [],
                "coverage_ratio": 1.0,
                # Compatibility keys used by current UI.
                "routes_with_data": ["Original Yahoo / yfinance.download"],
                "routes_attempted": [
                    f"yfinance.download attempt {i}"
                    for i in range(1, attempts_used + 1)
                ],
                "reconciled": False,
                "non_session_placeholders": [],
                "unfinished_session_rows": [],
                "unresolved_completed_sessions": [],
                "partial_rows": {},
                "precision_normalized_matches": 0,
                "max_precision_spread_bps": 0.0,
                "precision_examples": [],
            }
            return validated

        if rate_limited:
            raise MarketDataError(
                f"YAHOO RATE LIMIT STOP — {ticker}. Yahoo rejected or returned empty data "
                f"after {MAX_ATTEMPTS} attempts using the original yfinance.download path. "
                f"Last Yahoo diagnostic: {last_error}. "
                "The application did not switch provider or manufacture data. "
                "Successful identical requests are cached to reduce further Yahoo traffic."
            )

        raise MarketDataError(
            f"Yahoo Finance download failed for {ticker} after {MAX_ATTEMPTS} attempts "
            f"using the original yfinance.download path. Last diagnostic: {last_error}. "
            "No fallback provider will be used."
        )
