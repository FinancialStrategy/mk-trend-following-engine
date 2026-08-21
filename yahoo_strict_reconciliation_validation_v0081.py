from pathlib import Path
import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import load_legacy_golden_csv
from MK_Yahoo_Strict_Reconciliation_v0081 import (
    _strict_validate_candidate,
    _cross_check_common_observations,
)

ROOT = Path(__file__).resolve().parent
g = load_legacy_golden_csv(ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv")
real = g.rename(columns={"AdjClose": "Adj Close"})[["Open","High","Low","Close","Volume","Adj Close"]].copy()

# Emulate an incomplete first Yahoo response by masking one REAL observed field.
partial = real.copy()
target_dt = partial.index[-10]
partial.loc[target_dt, "Adj Close"] = np.nan

valid1, status1, detail1 = _strict_validate_candidate(partial, "1wk", 30)
assert valid1 is None
assert str(target_dt) in detail1
assert "Adj Close" in detail1[str(target_dt)]

# Complete second Yahoo-like payload is the original real market dataset.
valid2, status2, detail2 = _strict_validate_candidate(real, "1wk", 30)
assert valid2 is not None and status2 == "PASS"
_cross_check_common_observations(partial, valid2, "1wk")

# Reconciliation must stop if a second payload conflicts with a value already present in route A.
conflicting = real.copy()
conflict_dt = partial.index[-20]
conflicting.loc[conflict_dt, "Close"] *= 1.01
try:
    _cross_check_common_observations(partial, conflicting, "1wk")
except Exception:
    conflict_stop = True
else:
    conflict_stop = False
assert conflict_stop

print("PASS — Yahoo strict reconciliation v0.08.1")
print(f"Masked incomplete real-market row: {target_dt}")
print("Exact missing-field diagnostics: PASS")
print("Complete same-source second-payload path: PASS")
print("Observed-value reconciliation: PASS")
print("Conflicting payload hard-stop: PASS")
print("Synthetic replacement values created: NO")
print("Forward/back fill: NO")
print("Alternate provider: NO")
