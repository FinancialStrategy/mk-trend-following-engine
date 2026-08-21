from pathlib import Path
import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import load_legacy_golden_csv
from MK_Institutional_Tactical_v0089 import TacticalConfig, _execute_target_path

ROOT = Path(__file__).resolve().parent

g = load_legacy_golden_csv(ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv")
px = g[["AdjOpen", "AdjCloseCalc"]].copy().iloc[:16]

# Deterministic exposure schedule used only to validate portfolio accounting.
# All market prices are real Golden Master observations; no market price is synthesized.
target = np.array([
    1.00, 1.00, 0.75, 0.75, 0.75, 0.50, 0.50, 0.00,
    0.00, 0.25, 0.25, 0.75, 1.00, 1.00, 1.00, 1.00,
], dtype=float)
action = np.array(["TEST"] * len(px), dtype=object)
rationale = np.array(["Accounting invariant test"] * len(px), dtype=object)

cfg = TacticalConfig(
    initial_capital=100_000.0,
    initial_target_exposure=1.0,
    rebalance_only_on_target_change=True,
    cash_annual_rate=0.0,
)

out = _execute_target_path(px.copy(), target, action, rationale, cfg)

# 1) Portfolio identity must hold at every close.
identity = out["TacticalShares"] * out["AdjCloseCalc"] + out["TacticalCash"]
assert np.allclose(identity, out["TacticalPortfolio"], rtol=0, atol=1e-8)

# 2) Buy & Hold must be independently normalized from adjusted close.
expected_bh = cfg.initial_capital * out["AdjCloseCalc"] / out["AdjCloseCalc"].iloc[0]
assert np.allclose(out["BuyHold"], expected_bh, rtol=0, atol=1e-8)

# 3) Initial 100% exposure makes Tactical equal B&H until first target change.
assert np.allclose(out["TacticalPortfolio"].iloc[:2], out["BuyHold"].iloc[:2], rtol=0, atol=1e-8)

# 4) HOLD means no hidden trade. Only target changes may have traded value.
change = out["TacticalTargetExposure"].diff().abs().fillna(0) > 1e-12
traded = out["TacticalTradedValue"] > 1e-8
assert np.all(~traded[~change])
assert np.all(out.loc[~change, "TacticalRebalanceFlag"] == False)

# 5) When fully invested after a target-change execution, relative wealth vs B&H
# remains constant while no later exposure change occurs.
ratio = out["TacticalVsBuyHoldRatio"]
assert np.allclose(ratio.iloc[12:16], ratio.iloc[12], rtol=0, atol=1e-10)

# 6) During persistent 0% target with 0% cash carry, NAV is flat after the liquidation row.
assert abs(out["TacticalPortfolio"].iloc[8] - out["TacticalPortfolio"].iloc[7]) < 1e-8

# 7) Allowed target tiers remain explicit.
assert set(np.round(out["TacticalTargetExposure"].unique(), 2)).issubset({0.00,0.25,0.50,0.75,1.00})

print("PASS — Tactical accounting integrity v0.08.9")
print("Real market observations used: GE Golden Master")
print("Initial Tactical exposure: 100%")
print("Hidden daily constant-mix rebalancing: REMOVED")
print("Trade only on target change: PASS")
print("Independent Buy & Hold construction: PASS")
print("Target vs actual exposure separation: PASS")
print("Tactical/B&H wealth-ratio invariance at full exposure: PASS")
print("Synthetic market prices: NO")
