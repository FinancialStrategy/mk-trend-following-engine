"""MK Institutional Risk Analytics v0.08.7.

Client-facing risk layer for MK Trend Following Analytics Engine.

Design principles
-----------------
- Yahoo-derived market observations only.
- No alternate market-data provider.
- No forward fill / back fill.
- Exact timestamp alignment for asset / benchmark comparisons.
- Historical VaR uses empirical observed returns.
- Parametric VaR uses moments estimated from observed returns.
- Monte Carlo VaR uses an empirical bootstrap of observed returns to create
  in-memory risk scenarios. These scenarios are NEVER appended to or used as
  substitutes for the Yahoo market-data history.

By Murat Konuklar
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from statistics import NormalDist
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingWindowSpec:
    label: str
    observations: int
    periods_per_year: int
    frequency_label: str


@dataclass(frozen=True)
class VaRConfig:
    horizon_bars: int = 1
    mc_scenarios: int = 25000
    seed: int = 20260820
    min_observations: int = 30

    def validate(self):
        if self.horizon_bars < 1:
            raise ValueError('VaR horizon_bars must be >= 1')
        if self.mc_scenarios < 1000:
            raise ValueError('Monte Carlo scenarios must be >= 1,000')
        if self.min_observations < 20:
            raise ValueError('min_observations must be >= 20')


def _has_material_weekend_sessions(index: pd.DatetimeIndex) -> bool:
    idx = pd.DatetimeIndex(index)
    if len(idx) < 14:
        return False
    days = pd.DatetimeIndex(pd.Series(idx.normalize()).drop_duplicates().sort_values())
    if len(days) < 10:
        return False
    weekend_share = float((days.dayofweek >= 5).mean())
    return weekend_share >= 0.10


def infer_periodicity(index: pd.DatetimeIndex) -> tuple[int, str]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252, 'Daily'
    idx=pd.DatetimeIndex(index).sort_values()
    sec=pd.Series(idx).diff().dt.total_seconds().dropna()
    med=float(sec.median()) if len(sec) else 86400.0
    continuous_7d = _has_material_weekend_sessions(idx)
    if med < 18*3600:
        minutes=max(1,int(round(med/60)))
        counts=pd.Series(1,index=idx).groupby(idx.date).sum()
        counts=counts[counts>1]
        default_bpd = int(round((24*60 if continuous_7d else 6.5*60)/minutes))
        bars_day=max(1,int(round(float(counts.median())))) if len(counts) else max(1,default_bpd)
        days_year = 365 if continuous_7d else 252
        label = f'Intraday 7D ({minutes}m)' if continuous_7d else f'Intraday ({minutes}m)'
        return int(days_year*bars_day), label
    if med <= 3*86400:
        return (365,'Daily 7D') if continuous_7d else (252,'Daily')
    if med <= 10*86400:
        return 52,'Weekly'
    return 12,'Monthly'


def rolling_window_options(index: pd.DatetimeIndex) -> list[RollingWindowSpec]:
    ppy,freq=infer_periodicity(index)
    if freq.startswith('Intraday'):
        idx=pd.DatetimeIndex(index)
        counts=pd.Series(1,index=idx).groupby(idx.date).sum()
        counts=counts[counts>1]
        bpd=max(1,int(round(float(counts.median())))) if len(counts) else max(1,ppy//252)
        seven_day = '7D' in freq
        specs=[
            RollingWindowSpec('1D',bpd,ppy,freq),
            RollingWindowSpec('1W',bpd*(7 if seven_day else 5),ppy,freq),
            RollingWindowSpec('1M',bpd*(30 if seven_day else 21),ppy,freq),
            RollingWindowSpec('2M',bpd*(60 if seven_day else 42),ppy,freq),
        ]
    elif ppy==365:
        specs=[RollingWindowSpec('1M',30,ppy,freq),RollingWindowSpec('3M',91,ppy,freq),RollingWindowSpec('6M',182,ppy,freq),RollingWindowSpec('1Y',365,ppy,freq)]
    elif ppy==252:
        specs=[RollingWindowSpec('1M',21,ppy,freq),RollingWindowSpec('3M',63,ppy,freq),RollingWindowSpec('6M',126,ppy,freq),RollingWindowSpec('1Y',252,ppy,freq)]
    elif ppy==52:
        specs=[RollingWindowSpec('1M',4,ppy,freq),RollingWindowSpec('3M',13,ppy,freq),RollingWindowSpec('6M',26,ppy,freq),RollingWindowSpec('1Y',52,ppy,freq)]
    else:
        specs=[RollingWindowSpec('3M',3,ppy,freq),RollingWindowSpec('6M',6,ppy,freq),RollingWindowSpec('1Y',12,ppy,freq),RollingWindowSpec('2Y',24,ppy,freq)]
    valid=[s for s in specs if s.observations < len(index)]
    if valid: return valid
    w=max(2,len(index)//3)
    return [RollingWindowSpec(f'{w} obs',w,ppy,freq)]


def _strict_numeric(series: pd.Series,name: str)->pd.Series:
    out=pd.to_numeric(series,errors='coerce')
    if out.notna().sum()==0: raise ValueError(f'{name} contains no usable numeric observations.')
    return out


def rolling_risk_frame(df: pd.DataFrame, window: int|None=None):
    for c in ['AdjCloseCalc','Portfolio','Shares']:
        if c not in df.columns: raise KeyError(f'{c} is required for rolling risk.')
    opts=rolling_window_options(df.index)
    if window is None:
        spec=next((x for x in opts if x.label in {'3M','1M'}),opts[0])
    else:
        matches=[x for x in opts if x.observations==int(window)]
        if matches: spec=matches[0]
        else:
            ppy,freq=infer_periodicity(df.index)
            spec=RollingWindowSpec(f'{int(window)} obs',int(window),ppy,freq)
    win=int(spec.observations); ppy=int(spec.periods_per_year)
    asset=_strict_numeric(df['AdjCloseCalc'],'AdjCloseCalc')
    strat=_strict_numeric(df['Portfolio'],'Portfolio')
    shares=pd.to_numeric(df['Shares'],errors='coerce')
    ar=asset.pct_change(fill_method=None); sr=strat.pct_change(fill_method=None)
    out=pd.DataFrame(index=df.index)
    out['AssetRollingReturn']=asset.pct_change(periods=win,fill_method=None)
    out['AssetAnnualizedVolatility']=ar.rolling(win,min_periods=win).std(ddof=1)*math.sqrt(ppy)
    out['StrategyRollingReturn']=strat.pct_change(periods=win,fill_method=None)
    out['StrategyAnnualizedVolatility']=sr.rolling(win,min_periods=win).std(ddof=1)*math.sqrt(ppy)
    invested=shares.gt(0)
    out['Invested']=invested
    out['RollingExposure']=invested.astype(float).rolling(win,min_periods=win).mean()
    out['PureCashWindow']=(~invested).astype(float).rolling(win,min_periods=win).mean().eq(1.0)
    out['StrategyRollingReturnDisplay']=out['StrategyRollingReturn'].mask(out['PureCashWindow'])
    out['StrategyAnnualizedVolatilityDisplay']=out['StrategyAnnualizedVolatility'].mask(out['PureCashWindow'])
    return out,spec


def validate_underlying_risk_dynamics(df,rolling,tolerance=1e-12):
    price=pd.to_numeric(df['AdjCloseCalc'],errors='coerce').dropna()
    rr=pd.to_numeric(rolling['AssetRollingReturn'],errors='coerce').dropna()
    rv=pd.to_numeric(rolling['AssetAnnualizedVolatility'],errors='coerce').dropna()
    pu,ru,vu=int(price.nunique()),int(rr.nunique()),int(rv.nunique())
    pr=float(price.max()-price.min()) if len(price) else np.nan
    rrng=float(rr.max()-rr.min()) if len(rr) else np.nan
    vrng=float(rv.max()-rv.min()) if len(rv) else np.nan
    moving=bool(pu>3 and np.isfinite(pr) and pr>tolerance)
    flat_r=bool(len(rr)>=10 and (not np.isfinite(rrng) or rrng<=tolerance))
    flat_v=bool(len(rv)>=10 and (not np.isfinite(vrng) or vrng<=tolerance))
    return {'price_unique':pu,'rolling_return_unique':ru,'rolling_vol_unique':vu,'price_range':pr,'rolling_return_range':rrng,'rolling_vol_range':vrng,'impossible_flatness':bool(moving and flat_r and flat_v)}


def risk_state_snapshot(df,rolling,spec):
    latest=rolling.iloc[-1]
    shares=pd.to_numeric(df['Shares'],errors='coerce')
    return {
        'window_label':spec.label,
        'asset_rolling_return':latest['AssetRollingReturn'],
        'asset_annualized_volatility':latest['AssetAnnualizedVolatility'],
        'strategy_rolling_return':latest['StrategyRollingReturn'],
        'strategy_annualized_volatility':latest['StrategyAnnualizedVolatility'],
        'rolling_exposure':latest['RollingExposure'],
        'current_position':'INVESTED' if float(shares.iloc[-1])>0 else 'CASH',
        'cash_exposure_ratio':float(shares.le(0).mean()),
        'strategy_flat_reason':(
            'The selected rolling window is 100% cash. True strategy return/volatility may therefore be 0%; the chart masks that interval to avoid presenting cash as underlying market risk.'
            if bool(latest.get('PureCashWindow',False)) else ''
        ),
    }


def cash_regimes(rolling):
    mask=rolling['PureCashWindow'].fillna(False).astype(bool)
    out=[]; start=None; prev=None
    for dt,is_cash in mask.items():
        if is_cash and start is None: start=dt
        elif not is_cash and start is not None:
            out.append((start,prev)); start=None
        prev=dt
    if start is not None: out.append((start,prev))
    return out


def _clean_return_series(series: pd.Series, tail_observations: int|None=None)->pd.Series:
    r=pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    r=r[r>-0.999999999]
    if tail_observations is not None: r=r.tail(int(tail_observations))
    return r


def _compound_horizon(r: pd.Series,h: int)->pd.Series:
    if h==1: return r.copy()
    return (1.0+r).rolling(h,min_periods=h).apply(np.prod,raw=True)-1.0


def _historical_var(r: pd.Series, confidence: float, h: int)->float:
    agg=_compound_horizon(r,h).dropna()
    if len(agg)==0: return np.nan
    q=float(agg.quantile(1.0-confidence))
    return max(0.0,-q)


def _parametric_var(r: pd.Series, confidence: float, h: int)->float:
    lr=np.log1p(r.to_numpy(float))
    if len(lr)<2: return np.nan
    mu=float(np.mean(lr)); sd=float(np.std(lr,ddof=1))
    if not np.isfinite(sd): return np.nan
    z=NormalDist().inv_cdf(1.0-confidence)
    qlog=mu*h + z*sd*math.sqrt(h)
    qret=math.expm1(qlog)
    return max(0.0,-qret)


def _mc_bootstrap_var(r: pd.Series, confidence: float, h: int, scenarios: int, seed: int)->float:
    lr=np.log1p(r.to_numpy(float))
    if len(lr)==0: return np.nan
    rng=np.random.default_rng(seed)
    # Empirical Monte Carlo bootstrap: scenarios are constructed only from observed returns.
    idx=rng.integers(0,len(lr),size=(scenarios,h),endpoint=False)
    scen_log=lr[idx].sum(axis=1)
    scen=np.expm1(scen_log)
    q=float(np.quantile(scen,1.0-confidence))
    return max(0.0,-q)


def build_var_table(series_map: dict[str,pd.Series], calibration_observations: int, config: VaRConfig=VaRConfig())->pd.DataFrame:
    config.validate()
    rows=[]
    for label,series in series_map.items():
        r=_clean_return_series(series,calibration_observations)
        if len(r)<config.min_observations:
            rows.append({'Series':label,'Method':'DATA STOP','Confidence':np.nan,'VaR':np.nan,'Calibration Obs':len(r),'Horizon Bars':config.horizon_bars,'Status':f'Insufficient observations (<{config.min_observations})'})
            continue
        for conf in (0.95,0.99):
            vals={
                'Historical':_historical_var(r,conf,config.horizon_bars),
                'Parametric Normal':_parametric_var(r,conf,config.horizon_bars),
                'Monte Carlo Bootstrap':_mc_bootstrap_var(r,conf,config.horizon_bars,config.mc_scenarios,config.seed+int(conf*1000)),
            }
            for method,var in vals.items():
                rows.append({'Series':label,'Method':method,'Confidence':conf,'VaR':var,'Calibration Obs':len(r),'Horizon Bars':config.horizon_bars,'Status':'OK'})
    return pd.DataFrame(rows)


def exact_aligned_return_series(asset_price: pd.Series, benchmark_price: pd.Series):
    a=pd.to_numeric(asset_price,errors='coerce').rename('Asset')
    b=pd.to_numeric(benchmark_price,errors='coerce').rename('Benchmark')
    x=pd.concat([a,b],axis=1,join='inner').dropna()
    return x['Asset'].pct_change(fill_method=None),x['Benchmark'].pct_change(fill_method=None),x.index
