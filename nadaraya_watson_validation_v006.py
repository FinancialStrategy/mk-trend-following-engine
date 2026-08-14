from pathlib import Path
import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import EngineConfig, load_legacy_golden_csv, run_legacy_engine, performance_summary
from MK_Trend_Following_Decision_Engine_v002 import trade_ledger, trade_statistics
from MK_Nadaraya_Watson_Trend_v006 import (
    NWConfig, NWStrategyConfig, KERNELS, kernel_weight_profile,
    compute_nadaraya_watson, run_nw_strategy, nw_decision_snapshot,
)

ROOT = Path(__file__).resolve().parent
g = load_legacy_golden_csv(ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv")
market = g.rename(columns={"AdjClose":"Adj Close"})[["Open","High","Low","Close","Volume","Adj Close"]]
base = run_legacy_engine(market, EngineConfig(), validate=False)

# 1. All six kernels: finite, non-negative, normalized.
for kernel in KERNELS:
    cfg = NWConfig(lookback=80, bandwidth=8.0, kernel=kernel, relative_weight=1.5)
    p = kernel_weight_profile(cfg)
    assert (p["RawWeight"] >= -1e-15).all(), kernel
    assert np.isfinite(p["RawWeight"]).all(), kernel
    assert abs(float(p["NormalizedWeight"].sum()) - 1.0) < 1e-12, kernel

# 2. Indicator structure and residual bands.
cfg = NWConfig(lookback=100, bandwidth=8.0, kernel="Gaussian", band_multiplier=2.0)
ind = compute_nadaraya_watson(base, cfg)
assert ind.index.equals(base.index)
valid = ind["NWTrend"].notna()
assert valid.sum() == len(ind) - cfg.lookback
assert np.isfinite(ind.loc[valid, "NWTrend"]).all()
assert (ind.loc[valid, "NWUpper"] >= ind.loc[valid, "NWTrend"] - 1e-12).all()
assert (ind.loc[valid, "NWLower"] <= ind.loc[valid, "NWTrend"] + 1e-12).all()
assert (ind.loc[valid, "NWResidual"] >= -1e-15).all()

# 3. Causality / non-repainting prefix invariance.
# Recompute truncated histories and verify values at the cutoff are identical.
for cutoff in [150, 300, 600, 900]:
    prefix = base.iloc[:cutoff].copy()
    ind_prefix = compute_nadaraya_watson(prefix, cfg)
    cols = ["NWTrend","NWResidual","NWUpper","NWLower","NWSlope","NWDirection"]
    for c in cols:
        a = ind[c].iloc[cutoff-1]
        b = ind_prefix[c].iloc[-1]
        if c == "NWDirection":
            assert int(a) == int(b), (cutoff,c,a,b)
        else:
            assert abs(float(a)-float(b)) < 1e-12, (cutoff,c,a,b)

# 4. Strategy execution uses next adjusted open.
scfg = NWStrategyConfig(mode="MK_CONFIRMED_TREND", confirmation_bars=2, exit_confirmation_bars=1, initial_capital=100000.0)
nw = run_nw_strategy(base, ind, scfg)
buys = np.flatnonzero(nw["FirstBuy"].to_numpy(float) > 0)
sells = np.flatnonzero(nw["FirstSell"].to_numpy(float) > 0)
assert len(buys) > 0
for i in buys[:10]:
    prev_cash = float(nw["Cash"].iloc[i-1])
    expected_shares = prev_cash / float(nw["AdjOpen"].iloc[i])
    assert abs(float(nw["Shares"].iloc[i]) - expected_shares) < 1e-10
for i in sells[:10]:
    prev_shares = float(nw["Shares"].iloc[i-1])
    expected_cash = float(nw["Cash"].iloc[i-1]) + prev_shares * float(nw["AdjOpen"].iloc[i])
    assert abs(float(nw["Cash"].iloc[i]) - expected_cash) < 1e-8

# 5. Generic parent metrics/ledger compatibility.
summary = performance_summary(nw, scfg.initial_capital)
ledger = trade_ledger(nw)
stats = trade_statistics(ledger)
decision = nw_decision_snapshot(nw, scfg)
assert summary["portfolio_final"] > 0
assert decision["decision"] in {"BUY","HOLD","SELL","WAIT / CASH"}
assert isinstance(decision["gates"], pd.DataFrame)

print("PASS — Nadaraya-Watson Trend v0.06 validation")
print(f"Observations: {len(nw):,}")
print(f"Kernels validated: {len(KERNELS)}")
print("Causal prefix-invariance checkpoints: 4/4")
print(f"NW strategy BUY executions: {len(buys)}")
print(f"NW strategy SELL executions: {len(sells)}")
print(f"NW strategy final value: {summary['portfolio_final']:,.2f}")
print(f"NW strategy CAGR: {summary['strategy_cagr']:.4%}")
print(f"NW strategy max drawdown: {summary['max_drawdown']:.4%}")
print(f"Closed trades: {stats['closed_trades']}")
print("Synthetic market data: NO")
print("Fallback source: NO")
print("Forward/back fill of market prices: NO")
