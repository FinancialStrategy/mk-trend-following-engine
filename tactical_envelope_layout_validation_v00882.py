from pathlib import Path
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from MK_Tactical_Visual_Governance_v00882 import (
    apply_tactical_envelope_visual_governance,
    TACTICAL_ENVELOPE_HEIGHT,
    TACTICAL_ENVELOPE_TOP_MARGIN,
    TACTICAL_ENVELOPE_LEGEND_Y,
    TACTICAL_ENVELOPE_RANGE_SELECTOR_Y,
)

ROOT = Path(__file__).resolve().parent
app = (ROOT / "app.py").read_text(encoding="utf-8")
ast.parse(app)

assert 'APP_VERSION = "v0.08.8.2"' in app
assert 'apply_tactical_envelope_visual_governance(fig, benchmark_ticker, RANGE_SELECTOR)' in app
tactical_block = app[app.index('def make_tactical_envelope_chart'):app.index('def make_tactical_portfolio_chart')]
assert 'legend=dict(orientation="h",y=1.035' not in tactical_block
assert 'margin=dict(l=50,r=35,t=125,b=35)' not in tactical_block

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{}],[{"secondary_y": True}]])
for name in [
    "Adjusted OHLC", "NW Upper Band", "NW Lower Band", "NW Bullish Path", "NW Bearish Path",
    "Bullish Kernel Reversal", "Bearish Kernel Reversal", "Momentum Upward — MK Warning",
    "Momentum Downward — MK Warning", "Upper Band Cross — Early De-risk",
    "Upper Band Re-entry — Confirmed Exhaustion", "Lower Band Break",
    "Beta-Adjusted Relative Drift Z vs XBANK.IS", "Relative Volume",
]:
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name=name), row=1, col=1)

base_selector = dict(
    buttons=[dict(count=1, label="1M", step="month", stepmode="backward")],
    x=0, xanchor="left", y=1.22, yanchor="top",
)
apply_tactical_envelope_visual_governance(fig, "XBANK.IS", base_selector)

assert fig.layout.height == TACTICAL_ENVELOPE_HEIGHT
assert fig.layout.margin.t == TACTICAL_ENVELOPE_TOP_MARGIN
assert fig.layout.margin.t >= 240
assert fig.layout.legend.orientation == "h"
assert fig.layout.legend.x == 0
assert fig.layout.legend.xanchor == "left"
assert fig.layout.legend.yanchor == "bottom"
assert float(fig.layout.legend.y) == TACTICAL_ENVELOPE_LEGEND_Y
assert float(fig.layout.legend.y) > 1.0
assert fig.layout.legend.entrywidthmode == "pixels"
assert fig.layout.legend.entrywidth <= 180
assert fig.layout.legend.bgcolor.startswith("rgba(")
assert fig.layout.legend.borderwidth == 1
assert float(fig.layout.xaxis.rangeselector.y) == TACTICAL_ENVELOPE_RANGE_SELECTOR_Y
assert float(fig.layout.xaxis.rangeselector.y) > float(fig.layout.legend.y)

names = [t.name for t in fig.data]
assert "Relative Drift Z · XBANK.IS" in names
assert "Upper Re-entry · Exhaustion" in names
assert "Momentum Up" in names and "Momentum Down" in names
assert all(len(str(n)) <= 29 for n in names), names

ranks = [int(t.legendrank) for t in fig.data]
assert ranks == sorted(ranks)
assert len(ranks) == len(set(ranks))
assert all(isinstance(t.meta, dict) and t.meta.get("full_series_name") for t in fig.data)

print("PASS — Tactical Envelope legend/control rail v0.08.8.2")
print("Legend outside plotting domain: PASS")
print("Range selector above legend rail: PASS")
print("Reserved top margin >= 240 px: PASS")
print("Concise institutional legend labels: PASS")
print("Full semantic labels preserved in trace metadata: PASS")
print("Legend rank/order governance: PASS")
