from pathlib import Path
import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import load_legacy_golden_csv
from MK_Yahoo_Session_Aware_Strict_v0082 import (
    _completed_xist_sessions,
    _prepare_route,
    _consensus_frame,
    RouteResult,
)

ROOT = Path(__file__).resolve().parent

# Exact incident classification at 2026-08-21 09:36 TRT (06:36 UTC).
all_sessions, completed, cutoff = _completed_xist_sessions(
    "2024-04-09",
    "2026-08-21",
    now_utc=pd.Timestamp("2026-08-21 06:36:00+00:00"),
)
all_set = set(all_sessions)
completed_set = set(completed)

assert pd.Timestamp("2024-04-09") in completed_set
for d in ["2026-04-23","2026-05-01","2026-05-19","2026-07-15"]:
    assert pd.Timestamp(d) not in all_set
assert pd.Timestamp("2026-08-20") in completed_set
assert pd.Timestamp("2026-08-21") in all_set
assert pd.Timestamp("2026-08-21") not in completed_set
assert cutoff == "2026-08-20"

# Same-timestamp Yahoo-route recovery using only real Golden Master observations.
g = load_legacy_golden_csv(ROOT/"MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv")
real = g.rename(columns={"AdjClose":"Adj Close"})[
    ["Open","High","Low","Close","Volume","Adj Close"]
].copy()
masked = real.copy()
target = masked.index[-20]
masked.loc[target, "Adj Close"] = np.nan

r1 = RouteResult(name="route-with-masked-real-field", raw=masked)
r2 = RouteResult(name="route-with-complete-real-field", raw=real)

for r in [r1,r2]:
    _prepare_route(
        r, ticker="GE", interval="1wk",
        start=str(real.index[0].date()), end=str(real.index[-1].date())
    )

consensus, unresolved, detail, _ = _consensus_frame(
    [r1,r2], ticker="GE", interval="1wk",
    start=str(real.index[0].date()), end=str(real.index[-1].date())
)
assert not unresolved
assert abs(float(consensus.loc[target,"Adj Close"]) - float(real.loc[target,"Adj Close"])) < 1e-12

print("PASS — Yahoo Session-Aware Strict Adapter v0.08.2")
print("2024-04-09: EXPECTED COMPLETED XIST SESSION")
print("2026-04-23: NON-SESSION")
print("2026-05-01: NON-SESSION")
print("2026-05-19: NON-SESSION")
print("2026-07-15: NON-SESSION")
print("2026-08-20: EXPECTED COMPLETED XIST SESSION")
print("2026-08-21 at 09:36 TRT: UNFINISHED SESSION — WITHHELD")
print("Effective completed cutoff:", cutoff)
print("Same-timestamp Yahoo-route recovery: PASS")
print("Synthetic market observation created: NO")
print("Forward/back fill: NO")
print("Alternate market-data provider: NO")
