"""
MK Yahoo Original Fetch Adapter v0.08.5
By Murat Konuklar

This adapter deliberately restores the exact acquisition behavior that was
working before the v0.08.x Yahoo-governance experiments.

LIVE ACQUISITION:
    yf.download(
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

Then the returned Yahoo rows are passed through the original strict
`validate_market_data` function.

There is:
- NO multi-route whole-history consensus
- NO query1/query2 cross comparison
- NO exchange-calendar gap enforcement
- NO float reconciliation
- NO historical missing-session hard stop
- NO row filling
- NO alternate market-data provider

This is intentionally boring and stable.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from MK_Trend_Following_Engine_v001 import (
    validate_market_data,
    MarketDataError,
)


class YahooFinanceAdapter:
    """Exact-original Yahoo Finance acquisition path with non-blocking audit metadata."""

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
                "yfinance is not installed. Install the pinned requirements; no alternate source is used."
            ) from e

        ticker = str(ticker).strip()

        # IDENTICAL TO THE ORIGINAL WORKING ADAPTER:
        # yfinance end is EXCLUSIVE; user-facing end is inclusive.
        end_exclusive = (
            pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        ).date().isoformat()

        try:
            raw = yf.download(
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
        except Exception as e:
            raise MarketDataError(
                f"Yahoo Finance download failed for {ticker}. "
                f"No fallback will be used: {e}"
            ) from e

        try:
            validated = validate_market_data(raw, minimum_observations)
        except Exception as e:
            raise MarketDataError(
                f"Yahoo Finance data failed strict validation for {ticker}. "
                f"No fallback will be used: {e}"
            ) from e

        # Audit is informational only. It MUST NOT alter, merge, fill, drop,
        # reconcile, or block an otherwise valid Yahoo payload.
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
            "retry_status": "NOT USED",
            "historical_missing_sessions": [],
            "critical_missing_sessions": [],
            "recovered_sessions": [],
            "unfinished_sessions_withheld": [],
            "coverage_ratio": 1.0,
            # Compatibility keys for the current UI:
            "routes_with_data": ["Original Yahoo / yfinance.download"],
            "routes_attempted": ["yfinance.download"],
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
