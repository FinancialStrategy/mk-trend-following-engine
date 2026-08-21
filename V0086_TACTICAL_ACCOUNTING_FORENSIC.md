# v0.08.6 — Institutional Tactical Accounting & Strategy-vs-Buy/Hold Forensic

## Finding
The user's concern was valid. The Strategy vs Buy & Hold panel contained a mixture of correct chart wiring and problematic portfolio-accounting semantics inherited from the v0.07 tactical research layer.

### 1. Target Exposure was technically plotted, but could be visually absent
The v0.08.5.3 chart already referenced `TacticalTargetExposure` in a small lower subplot. However, the tactical engine initialized `target[0] = 0.0`. Long zero/low-exposure regimes therefore pushed the series onto the bottom axis, where a filled-to-zero trace was difficult to distinguish.

v0.08.6 renders Target Exposure as an explicit `hv` staircase, increases the panel height, adds 0/25/50/75/100% guide levels, shows rebalance markers, and overlays Actual Close Exposure.

### 2. The Tactical overlay incorrectly initialized in cash
The v0.07 execution engine began with:
- `cash[0] = initial_capital`
- `target[0] = 0.0`

That is appropriate for a standalone entry system, but inconsistent with an **institutional de-risking overlay** whose purpose is to trim an existing long position from 100% to 75/50/25/0%.

v0.08.6 initializes the Tactical overlay at 100% exposure. The NW standalone strategy remains the separate cash/entry strategy.

### 3. HOLD was secretly rebalancing every bar
The v0.07 engine recomputed desired shares at every next open even when the target had not changed. At a 75% or 50% target this created an undocumented daily constant-mix rebalance.

Consequences:
- hidden turnover;
- `TacticalTradedValue` was positive even on many HOLD bars;
- the portfolio path was not a pure staged-exposure path;
- trade/event counts were inflated.

v0.08.6 carries shares and cash unchanged on HOLD. A trade occurs only when Target Exposure changes.

### 4. Target and actual exposure were conflated
A 75% target set at the execution open will drift away from 75% at the close as the stock price moves. The previous UI did not distinguish the decision target from actual close exposure.

v0.08.6 reports:
- `TacticalTargetExposure`
- `TacticalActualExposure`
- `TacticalTargetChange`
- `TacticalRebalanceFlag`
- `TacticalTurnover`

### 5. Buy & Hold was inherited from the NW input frame
The inherited `BuyHold` formula itself was sensible, but the Tactical comparison did not need to depend on the NW strategy's portfolio-accounting columns.

v0.08.6 independently recomputes Buy & Hold from adjusted close:

`Initial Capital × AdjClose_t / AdjClose_0`

This removes an unnecessary cross-layer dependency.

### 6. A Tactical NAV does not mechanically converge toward Buy & Hold when exposure rises
Target Exposure controls **future participation**, not a target NAV level.

If Tactical/B&H wealth ratio has fallen to 0.70x and Tactical subsequently returns to 100% exposure, then—while both portfolios remain 100% long—their percentage returns become similar and the 0.70x wealth ratio tends to remain approximately flat. The Tactical NAV only closes the historical gap if future de-risk/re-entry decisions create positive active return.

v0.08.6 therefore adds a third chart panel: `Tactical / Buy & Hold Wealth Ratio`.

### 7. Cash carry materially affects a de-risking strategy
The historical Tactical implementation treated uninvested cash as 0%-yield cash. This is conservative but can materially depress a strategy that spends substantial time at 0–75% exposure, especially in a high-rate market.

v0.08.6 adds an explicit user-supplied **Uninvested Cash Annual Yield (%)** control. Default remains 0%. No cash-rate market series is fabricated or fetched.

### 8. Next target vs current executed target was semantically mixed in the UI
`tactical_snapshot()` computes the next target from the latest completed bar, while the plotted exposure series is the currently executed target. Both were labeled “Target Exposure.”

v0.08.6 labels the decision KPI **Next Target Exposure** and keeps the chart as the executed Target Exposure path. The timing note now states that the new target executes at the next adjusted open.

## What was NOT causing the black line
The Strategy vs Buy & Hold panel was not driven by the old Legacy ATR Trailing Stop. The black line uses `TacticalPortfolio`. The principal defects were in the v0.07 Tactical portfolio-accounting semantics, not the Golden Master legacy signal columns.

## Validation
Using real GE Golden Master market observations:
- portfolio identity: PASS
- independent Buy & Hold construction: PASS
- initial 100% Tactical = Buy & Hold before first target change: PASS
- no trade when target unchanged: PASS
- full-exposure Tactical/B&H wealth-ratio invariance: PASS
- 0% cash-state flatness at 0% carry: PASS
- target tiers restricted to 0/25/50/75/100%: PASS
- synthetic market prices: NO
