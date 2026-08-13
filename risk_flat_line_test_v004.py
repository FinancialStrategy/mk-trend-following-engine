from pathlib import Path
import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import (
    EngineConfig, load_legacy_golden_csv, run_legacy_engine
)
from MK_Trend_Following_Risk_Analytics_v004 import (
    rolling_risk_frame,
    validate_underlying_risk_dynamics,
)

ROOT = Path(__file__).resolve().parent
golden = ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv"

g = load_legacy_golden_csv(golden)
market = g.rename(columns={"AdjClose":"Adj Close"})[
    ["Open","High","Low","Close","Volume","Adj Close"]
]
df = run_legacy_engine(market, EngineConfig(), validate=False)
rolling, spec = rolling_risk_frame(df)
diag = validate_underlying_risk_dynamics(df, rolling)

assert diag["impossible_flatness"] is False
assert diag["rolling_return_unique"] > 10
assert diag["rolling_vol_unique"] > 10

# Confirm that cash-only true strategy calculations are preserved,
# while display series are masked (no synthetic replacement).
pure_cash = rolling["PureCashWindow"].fillna(False)
if pure_cash.any():
    assert rolling.loc[pure_cash, "StrategyRollingReturnDisplay"].isna().all()
    assert rolling.loc[pure_cash, "StrategyAnnualizedVolatilityDisplay"].isna().all()

# Construct a controlled cash regime from the real Golden Master engine output.
# This does NOT alter market prices; it only tests the chart-display rule.
test_df = df.copy()
tail_n = min(80, len(test_df))
test_df.loc[test_df.index[-tail_n:], "Shares"] = 0.0
test_df.loc[test_df.index[-tail_n:], "Portfolio"] = float(test_df["Portfolio"].iloc[-tail_n - 1])
rolling2, spec2 = rolling_risk_frame(test_df)
cash2 = rolling2["PureCashWindow"].fillna(False)
assert cash2.any()
assert rolling2.loc[cash2, "StrategyRollingReturnDisplay"].isna().all()
assert rolling2.loc[cash2, "StrategyAnnualizedVolatilityDisplay"].isna().all()

print("PASS — v0.04 flat-line presentation fix")
print(f"Window: {spec.label} / {spec.observations} observations")
print(f"Underlying rolling-return unique values: {diag['rolling_return_unique']:,}")
print(f"Underlying rolling-vol unique values: {diag['rolling_vol_unique']:,}")
print(f"Impossible underlying flatness: {diag['impossible_flatness']}")
print(f"Cash-only display masking test: PASS")
print("No price fill, synthetic market data, or fallback source used.")
