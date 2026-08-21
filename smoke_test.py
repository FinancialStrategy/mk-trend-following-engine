from pathlib import Path
import sys

from MK_Trend_Following_Engine_v001 import (
    EngineConfig,
    golden_master_validation,
    load_legacy_golden_csv,
    run_legacy_engine,
    performance_summary,
)

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv"

cfg = EngineConfig()
validation = golden_master_validation(GOLDEN, cfg, atol=1e-8)

if not bool(validation["Pass"].all()):
    failed = validation.loc[~validation["Pass"]]
    print(failed.to_string(index=False))
    raise SystemExit("Golden Master replication FAILED")

g = load_legacy_golden_csv(GOLDEN)
market = g.rename(columns={"AdjClose": "Adj Close"})[
    ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
]
result = run_legacy_engine(market, cfg, validate=False)
summary = performance_summary(result, cfg.initial_capital)

assert len(result) == 1043
assert abs(summary["portfolio_final"] - 1449095.2235074965) < 1e-6
assert abs(summary["buyhold_final"] - 747161.5720524008) < 1e-6

print("PASS — Golden Master replication")
print(f"Observations: {len(result):,}")
print(f"Final Strategy: {summary['portfolio_final']:,.6f}")
print(f"Final Buy & Hold: {summary['buyhold_final']:,.6f}")
print("No live market data was requested by this test.")
