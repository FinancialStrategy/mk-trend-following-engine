from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from MK_Trend_Following_Universe_v0087 import UNIVERSE
from MK_Benchmark_Relative_v0087 import default_benchmark
from MK_Trend_Following_Engine_v001 import load_legacy_golden_csv, EngineConfig, run_legacy_engine, performance_summary
from MK_Nadaraya_Watson_Trend_v0087 import NWConfig, compute_nadaraya_watson, nw_alert_ledger
from MK_Nadaraya_Watson_Visuals_v0087 import NWVisualConfig, build_nw_price_figure, regime_path_series
from MK_Institutional_Risk_Analytics_v0087 import infer_periodicity

# Universe integrity
assert ("Copper Futures","HG=F") in UNIVERSE["Industrial Metals"]["Futures"]
crypto=UNIVERSE["Crypto Assets"]["Major Non-Stable Crypto — USD"]
assert len(crypto)==12
symbols=[x[1] for x in crypto]
expected_crypto=["BTC-USD","ETH-USD","BNB-USD","XRP-USD","SOL-USD","TRX-USD","HYPE32196-USD","DOGE-USD","ZEC-USD","LEO-USD","LINK-USD","ADA-USD"]
assert symbols==expected_crypto
assert len(symbols)==len(set(symbols))
assert all(default_benchmark(x) for x in ["HG=F"]+symbols)
assert all(default_benchmark(x)!=x for x in ["HG=F"]+symbols)

# Real market observations from immutable Golden Master; no synthetic prices.
g=load_legacy_golden_csv(ROOT/"MK_Trend_Following_Legacy_Golden_Master_GE_1988_2008.csv")
cfg=EngineConfig(strategy="ATR_TRAILING_STOP",atr_weeks=8,atr_multiplier=10.0,bollinger_weeks=40,bollinger_sd=2.5,max_buy_weeks=2000,initial_capital=100000.0,legacy_inclusive_windows=True)
market=g.rename(columns={"AdjClose":"Adj Close"})[["Open","High","Low","Close","Volume","Adj Close"]]
base=run_legacy_engine(market,cfg,validate=False)
nwcfg=NWConfig(lookback=40,bandwidth=8.0,kernel="Gaussian",band_multiplier=2.0,minimum_observations=30)
ind=compute_nadaraya_watson(base,nwcfg)

# Exact public alert identities.
dirn=ind["NWDirection"].to_numpy(int)
expected_bull=np.zeros(len(ind),dtype=bool); expected_bear=np.zeros(len(ind),dtype=bool)
expected_bull[1:]=(dirn[1:]>0)&(dirn[:-1]<=0)
expected_bear[1:]=(dirn[1:]<0)&(dirn[:-1]>=0)
assert np.array_equal(ind["NWBullishReversal"].to_numpy(bool),expected_bull)
assert np.array_equal(ind["NWBearishReversal"].to_numpy(bool),expected_bear)
assert np.array_equal(ind["NWAnyReversal"].to_numpy(bool),expected_bull|expected_bear)

src=ind["NWSource"].to_numpy(float); up=ind["NWUpper"].to_numpy(float); lo=ind["NWLower"].to_numpy(float)
exp_up=np.zeros(len(ind),dtype=bool); exp_dn=np.zeros(len(ind),dtype=bool)
valid=np.isfinite(src[1:])&np.isfinite(src[:-1])&np.isfinite(up[1:])&np.isfinite(up[:-1])
exp_up[1:]=valid&(src[1:]>up[1:])&(src[:-1]<=up[:-1])
valid=np.isfinite(src[1:])&np.isfinite(src[:-1])&np.isfinite(lo[1:])&np.isfinite(lo[:-1])
exp_dn[1:]=valid&(src[1:]<lo[1:])&(src[:-1]>=lo[:-1])
assert np.array_equal(ind["NWCrossAboveUpper"].to_numpy(bool),exp_up)
assert np.array_equal(ind["NWCrossBelowLower"].to_numpy(bool),exp_dn)

# Causal prefix invariance for all core/public/MK-warning outputs.
cut=700
ind_prefix=compute_nadaraya_watson(base.iloc[:cut].copy(),nwcfg)
cols=["NWTrend","NWUpper","NWLower","NWDirection","NWBullishReversal","NWBearishReversal","NWCrossAboveUpper","NWCrossBelowLower","NWMomentumUpwardWarning","NWMomentumDownwardWarning"]
for c in cols:
    a=ind[c].iloc[:cut]
    b=ind_prefix[c]
    if a.dtype==bool or b.dtype==bool:
        assert np.array_equal(a.to_numpy(),b.to_numpy()),c
    else:
        assert np.allclose(a.to_numpy(float),b.to_numpy(float),equal_nan=True,rtol=0,atol=1e-12),c

# Visual figure contains red/green paths + reversal and momentum alert traces.
visdf=base.join(ind)
bull_path,bear_path,flat_path=regime_path_series(visdf)
# Segment t-1 -> t must be coloured by current slope state, matching the
# public Pine visual semantics. Reversal segments therefore carry both endpoints.
for i in np.flatnonzero(ind["NWBullishReversal"].to_numpy(bool))[:10]:
    if i > 0 and np.isfinite(ind["NWTrend"].iloc[i-1]):
        assert np.isfinite(bull_path.iloc[i-1]) and np.isfinite(bull_path.iloc[i])
for i in np.flatnonzero(ind["NWBearishReversal"].to_numpy(bool))[:10]:
    if i > 0 and np.isfinite(ind["NWTrend"].iloc[i-1]):
        assert np.isfinite(bear_path.iloc[i-1]) and np.isfinite(bear_path.iloc[i])
fig=build_nw_price_figure(visdf,nwcfg,NWVisualConfig())
names=[t.name for t in fig.data]
for required in ["NW Bullish Path","NW Bearish Path","Bullish Kernel Reversal","Bearish Kernel Reversal","Momentum Upward — MK Warning","Momentum Downward — MK Warning","Source Cross Above Upper Band","Source Cross Below Lower Band"]:
    assert required in names,required

# Alert ledger origins are explicit.
ledger=nw_alert_ledger(visdf)
if len(ledger):
    assert set(ledger["Origin"]).issubset({"QuantAlgo Public Alert","MK Causal Warning"})

# Periodicity regression: Golden Master weekly remains 52.
assert infer_periodicity(base.index)[0]==52

# 7D periodicity test uses only timestamps, not market prices.
crypto_idx=pd.date_range("2026-01-01",periods=120,freq="D")
assert infer_periodicity(crypto_idx)==(365,"Daily 7D")

print("PASS — v0.08.7 Crypto/Copper/QuantAlgo integration")
print("Copper HG=F in universe: PASS")
print("12 major non-stable crypto Yahoo tickers: PASS")
print("All new assets have explicit Yahoo benchmark mapping: PASS")
print("QuantAlgo public reversal alert identities: PASS")
print("QuantAlgo public band-cross alert identities: PASS")
print("NW prefix invariance / causality: PASS")
print("Red/green Pine-style segment path + reversal markers + momentum warnings: PASS")
print("Crypto 7D annualization detection: PASS")
print("Synthetic market observations used in validation: NO")
