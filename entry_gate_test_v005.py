from pathlib import Path
import sys

from MK_Trend_Following_Engine_v001 import (
    EngineConfig, load_legacy_golden_csv, run_legacy_engine
)
from MK_Trend_Following_Entry_Gate_v005 import (
    resolve_entry_lookback,
    effective_gate_state,
    longest_cash_regime,
)

ROOT = Path(__file__).resolve().parent
golden = ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv"

g = load_legacy_golden_csv(golden)
market = g.rename(columns={"AdjClose":"Adj Close"})[
    ["Open","High","Low","Close","Volume","Adj Close"]
]

# Legacy exact must remain unchanged.
legacy = run_legacy_engine(market, EngineConfig(max_buy_weeks=2000), validate=False)
assert abs(float(legacy["Portfolio"].iloc[-1]) - 1449095.2235074965) < 1e-6
assert int(legacy["FirstBuy"].sum()) == 1
assert int(legacy["FirstSell"].sum()) == 1

# Frequency-aware weekly 12M = 52 observations.
obs, label = resolve_entry_lookback("1wk", "Frequency-Aware", horizon="12M")
assert obs == 52
adaptive = run_legacy_engine(market, EngineConfig(max_buy_weeks=obs), validate=False)

legacy_cash = longest_cash_regime(legacy)
adaptive_cash = longest_cash_regime(adaptive)

assert legacy_cash is not None
assert adaptive_cash is not None
assert adaptive_cash["Observations"] < legacy_cash["Observations"]
assert int(adaptive["FirstBuy"].sum()) > int(legacy["FirstBuy"].sum())

gate = effective_gate_state(legacy, 2000)
assert gate["effective_all_history"] is True

print("PASS — Entry gate governance v0.05")
print(f"Legacy exact final portfolio: {float(legacy['Portfolio'].iloc[-1]):,.6f}")
print(f"Legacy buys/sells: {int(legacy['FirstBuy'].sum())}/{int(legacy['FirstSell'].sum())}")
print(f"Legacy longest cash regime: {legacy_cash['Observations']} observations")
print(f"12M weekly adaptive lookback: {obs} observations")
print(f"Adaptive buys/sells: {int(adaptive['FirstBuy'].sum())}/{int(adaptive['FirstSell'].sum())}")
print(f"Adaptive longest cash regime: {adaptive_cash['Observations']} observations")
print("No synthetic market data, fallback source, forward fill or backfill was used.")
