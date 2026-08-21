"""Institutional visual governance for Tactical Envelope v0.08.8.2.

Purpose: keep the Tactical Envelope legend, range selector and title in a
reserved top rail outside the market-data plotting domain.
"""
from __future__ import annotations

from copy import deepcopy

TACTICAL_ENVELOPE_HEIGHT = 900
TACTICAL_ENVELOPE_TOP_MARGIN = 265
TACTICAL_ENVELOPE_LEGEND_Y = 1.105
TACTICAL_ENVELOPE_RANGE_SELECTOR_Y = 1.335
TACTICAL_ENVELOPE_LEGEND_ENTRY_WIDTH = 178

# Concise institutional legend labels. Full original labels are preserved in
# trace.meta['full_series_name'] for audit/debugging.
_TRACE_GOVERNANCE = {
    "Adjusted OHLC": (10, "Adjusted OHLC", "price"),
    "NW Upper Band": (20, "NW Upper", "nw"),
    "NW Lower Band": (30, "NW Lower", "nw"),
    "NW Bullish Path": (40, "NW Bullish", "nw"),
    "NW Bearish Path": (50, "NW Bearish", "nw"),
    "Bullish Kernel Reversal": (60, "Bullish Reversal", "reversal"),
    "Bearish Kernel Reversal": (70, "Bearish Reversal", "reversal"),
    "Momentum Upward — MK Warning": (80, "Momentum Up", "momentum"),
    "Momentum Downward — MK Warning": (90, "Momentum Down", "momentum"),
    "Upper Band Cross — Early De-risk": (100, "Upper Cross · De-risk", "event"),
    "Upper Band Re-entry — Confirmed Exhaustion": (110, "Upper Re-entry · Exhaustion", "event"),
    "Lower Band Break": (120, "Lower Band Break", "event"),
    "Relative Volume": (140, "Relative Volume", "diagnostic"),
}


def _trace_rule(name: str, benchmark_ticker: str):
    if name.startswith("Beta-Adjusted Relative Drift Z vs "):
        return 130, f"Relative Drift Z · {benchmark_ticker}", "diagnostic"
    return _TRACE_GOVERNANCE.get(name)


def tactical_range_selector(base_selector: dict) -> dict:
    selector = deepcopy(base_selector)
    selector.update({
        "x": 0.0,
        "xanchor": "left",
        "y": TACTICAL_ENVELOPE_RANGE_SELECTOR_Y,
        "yanchor": "bottom",
        "bgcolor": "#FFFFFF",
        "activecolor": "#E2E8F0",
        "bordercolor": "#CBD5E1",
        "borderwidth": 1,
        "font": {"size": 10, "color": "#334155"},
    })
    return selector


def apply_tactical_envelope_visual_governance(fig, benchmark_ticker: str, base_range_selector: dict):
    """Apply a dedicated non-overlapping legend/control rail to the Tactical chart."""
    for trace in fig.data:
        original_name = str(getattr(trace, "name", "") or "")
        rule = _trace_rule(original_name, benchmark_ticker)
        if rule is None:
            continue
        rank, short_name, group = rule
        # Preserve the full semantic label for audit/debugging while keeping the
        # visible legend compact enough for a two-row institutional rail.
        existing_meta = getattr(trace, "meta", None)
        if isinstance(existing_meta, dict):
            meta = dict(existing_meta)
        else:
            meta = {}
        meta["full_series_name"] = original_name
        trace.meta = meta
        trace.name = short_name
        trace.legendrank = rank
        trace.legendgroup = group
        trace.showlegend = True

    fig.update_layout(
        title=dict(
            text="Institutional Tactical Envelope + Benchmark Relative Deviation",
            x=0.0,
            xanchor="left",
            y=0.992,
            yanchor="top",
            font=dict(size=16, color="#0F172A"),
            pad=dict(t=2, b=8),
        ),
        template="plotly_white",
        height=TACTICAL_ENVELOPE_HEIGHT,
        hovermode="x unified",
        margin=dict(l=62, r=42, t=TACTICAL_ENVELOPE_TOP_MARGIN, b=46),
        legend=dict(
            title=dict(text="Series & Tactical Signals", font=dict(size=10, color="#475569")),
            orientation="h",
            x=0.0,
            xanchor="left",
            y=TACTICAL_ENVELOPE_LEGEND_Y,
            yanchor="bottom",
            bgcolor="rgba(248,250,252,0.98)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(size=10, color="#334155"),
            itemsizing="constant",
            entrywidth=TACTICAL_ENVELOPE_LEGEND_ENTRY_WIDTH,
            entrywidthmode="pixels",
            traceorder="normal",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="toggleitem",
            tracegroupgap=4,
        ),
        font=dict(family="Arial Narrow, Helvetica Neue, Arial, sans-serif", size=11, color="#334155"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        rangeselector=tactical_range_selector(base_range_selector),
        rangeslider=dict(visible=False),
        row=1,
        col=1,
    )
    return fig
