from pathlib import Path
import numpy as np,pandas as pd
from MK_Trend_Following_Engine_v001 import EngineConfig,load_legacy_golden_csv,run_legacy_engine
from MK_Institutional_Risk_Analytics_v008 import rolling_window_options,VaRConfig,build_var_table
ROOT=Path(__file__).resolve().parent
g=load_legacy_golden_csv(ROOT/'MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv')
market=g.rename(columns={'AdjClose':'Adj Close'})[['Open','High','Low','Close','Volume','Adj Close']]
df=run_legacy_engine(market,EngineConfig(),validate=False)
r=df['AdjCloseCalc'].pct_change(fill_method=None)
# Use same real series for a deterministic method-coverage test only; no synthetic market observations.
t=build_var_table({'Observed GE':r},calibration_observations=252,config=VaRConfig(horizon_bars=5,mc_scenarios=10000,min_observations=30))
assert set(t['Method'])=={'Historical','Parametric Normal','Monte Carlo Bootstrap'}
assert set(t['Confidence'])=={0.95,0.99}
assert np.isfinite(t['VaR']).all()
assert (t['VaR']>=0).all()
print('PASS — v0.08 VaR engine method coverage')
print(t[['Method','Confidence','VaR']].to_string(index=False))
print('Monte Carlo scenarios are risk bootstrap scenarios only; no market history was altered.')
