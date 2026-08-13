from pathlib import Path
import numpy as np
from MK_Trend_Following_Engine_v001 import EngineConfig, load_legacy_golden_csv, run_legacy_engine
from MK_Trend_Following_Risk_Analytics_v003 import rolling_risk_frame, risk_state_snapshot

ROOT = Path(__file__).resolve().parent
golden = ROOT / "MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv"
g = load_legacy_golden_csv(golden)
market = g.rename(columns={"AdjClose":"Adj Close"})[["Open","High","Low","Close","Volume","Adj Close"]]
df = run_legacy_engine(market, EngineConfig(), validate=False)
rolling, spec = rolling_risk_frame(df)
state = risk_state_snapshot(df, rolling, spec)

asset = rolling["AssetRollingReturn"].dropna()
asset_vol = rolling["AssetAnnualizedVolatility"].dropna()
strategy = rolling["StrategyRollingReturn"].dropna()

assert len(asset) > 0
assert len(asset_vol) > 0
assert float(asset.abs().max()) > 0
assert float(asset_vol.max()) > 0
assert not np.allclose(asset.to_numpy(), 0.0)
assert "cash_exposure_ratio" in state

print("PASS — Rolling Risk source separation")
print(f"Window: {spec.label} / {spec.observations} observations")
print(f"Underlying rolling-return nonzero max abs: {float(asset.abs().max()):.8f}")
print(f"Underlying rolling-vol max: {float(asset_vol.max()):.8f}")
print(f"Strategy rolling observations: {len(strategy):,}")
print(f"Cash exposure ratio: {state['cash_exposure_ratio']:.4%}")
