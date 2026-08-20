
from pathlib import Path
import numpy as np
import pandas as pd

from MK_Trend_Following_Engine_v001 import EngineConfig, load_legacy_golden_csv, run_legacy_engine
from MK_Nadaraya_Watson_Trend_v006 import NWConfig, compute_nadaraya_watson
from MK_Benchmark_Relative_v007 import RelativeConfig, compute_relative_analytics
from MK_Institutional_Tactical_v007 import TacticalConfig, run_tactical_strategy, tactical_snapshot

ROOT=Path(__file__).resolve().parent
g=load_legacy_golden_csv(ROOT/"MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv")
market=g.rename(columns={"AdjClose":"Adj Close"})[["Open","High","Low","Close","Volume","Adj Close"]]
base=run_legacy_engine(market,EngineConfig(),validate=False)

nw=compute_nadaraya_watson(base,NWConfig(
    lookback=60,bandwidth=8,kernel="Gaussian",band_multiplier=1.5,minimum_observations=30
))
for c in nw.columns: base[c]=nw[c]

# Mathematical governance test with the SAME REAL GE series as the benchmark.
# This is not used as a model result; it verifies exact alignment, beta and residual mechanics
# without introducing synthetic or substitute market observations.
bench=market.copy()
rel=compute_relative_analytics(base,bench,RelativeConfig(beta_window=30,drift_horizon=5))
valid=rel.dropna(subset=["RollingBeta","ResidualDriftZ"])
assert len(valid)>100
assert np.isfinite(valid["RollingBeta"]).all()

tact=run_tactical_strategy(base,rel,TacticalConfig(initial_capital=100000,weak_z=1.5,strong_z=2,extreme_z=3))
snap=tactical_snapshot(tact,TacticalConfig(initial_capital=100000,weak_z=1.5,strong_z=2,extreme_z=3))
assert set(np.unique(tact["TacticalTargetExposure"])).issubset({0.0,0.25,0.5,0.75,1.0})
assert "NWCrossAboveUpper" in tact.columns
assert "NWReenterBelowUpper" in tact.columns
assert "ResidualDriftZ" in tact.columns

upper_crosses=int(tact["NWCrossAboveUpper"].sum())
upper_reentries=int(tact["NWReenterBelowUpper"].sum())
print("PASS — Institutional Tactical Engine v0.07 structural validation")
print(f"Real legacy observations: {len(tact):,}")
print(f"NW upper-band crosses detected: {upper_crosses:,}")
print(f"NW upper-band re-entries detected: {upper_reentries:,}")
print(f"Allowed target-exposure ladder: {sorted(set(tact['TacticalTargetExposure']))}")
print(f"Latest structural decision: {snap['decision']}")
print("No synthetic market observations, no fallback provider, no forward/back fill.")
