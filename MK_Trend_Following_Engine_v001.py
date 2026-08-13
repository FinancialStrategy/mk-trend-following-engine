"""
MK Trend Following Analytics Engine v0.01
Legacy Fidelity + Modern Yahoo Finance Data Layer
By Murat Konuklar

Governance:
- NO synthetic data.
- NO fallback market data source.
- NO forward-fill/back-fill of market prices.
- Yahoo download failures are hard failures.
- yfinance repair=False; auto_adjust=False.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import math
import numpy as np
import pandas as pd

Strategy = Literal["ATR", "BOLLINGER", "ATR_TRAILING_STOP"]

@dataclass(frozen=True)
class EngineConfig:
    initial_capital: float = 100_000.0
    atr_weeks: int = 8
    atr_multiplier: float = 10.0
    bollinger_weeks: int = 40
    bollinger_sd: float = 2.5
    max_buy_weeks: int = 2000
    strategy: Strategy = "ATR_TRAILING_STOP"
    legacy_inclusive_windows: bool = True
    minimum_observations: int = 30

    def validate(self) -> None:
        if self.initial_capital <= 0: raise ValueError("initial_capital must be > 0")
        if self.atr_weeks < 1: raise ValueError("atr_weeks must be >= 1")
        if self.atr_multiplier <= 0: raise ValueError("atr_multiplier must be > 0")
        if self.bollinger_weeks < 2: raise ValueError("bollinger_weeks must be >= 2")
        if self.bollinger_sd <= 0: raise ValueError("bollinger_sd must be > 0")
        if self.max_buy_weeks < 1: raise ValueError("max_buy_weeks must be >= 1")
        if self.strategy not in {"ATR","BOLLINGER","ATR_TRAILING_STOP"}:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

class DataIntegrityError(RuntimeError): pass
class MarketDataError(RuntimeError): pass

REQUIRED_COLUMNS = ["Open","High","Low","Close","Volume","Adj Close"]


def validate_market_data(df: pd.DataFrame, minimum_observations: int = 30) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise DataIntegrityError("Market data is empty. No fallback will be used.")
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        if x.columns.nlevels == 2 and len(set(x.columns.get_level_values(-1))) == 1:
            x.columns = x.columns.get_level_values(0)
        else:
            raise DataIntegrityError("Unexpected MultiIndex columns; strict mode refuses to guess column mapping.")
    missing = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if missing:
        raise DataIntegrityError(f"Missing required Yahoo fields: {missing}. No fallback will be used.")
    x = x[REQUIRED_COLUMNS].copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        try: x.index = pd.to_datetime(x.index, errors="raise")
        except Exception as e: raise DataIntegrityError("Date index cannot be parsed.") from e
    x = x.sort_index()
    if x.index.has_duplicates:
        dup = x.index[x.index.duplicated()].unique().tolist()[:5]
        raise DataIntegrityError(f"Duplicate market dates detected: {dup}")
    bad = x[REQUIRED_COLUMNS].isna().any(axis=1)
    if bad.any():
        dates = [str(v) for v in x.index[bad][:5]]
        raise DataIntegrityError(f"Missing OHLCV/Adj Close values on {dates}; strict mode will not fill them.")
    for c in REQUIRED_COLUMNS:
        x[c] = pd.to_numeric(x[c], errors="raise")
    if (x[["Open","High","Low","Close","Adj Close"]] <= 0).any().any():
        raise DataIntegrityError("Non-positive price detected; strict mode rejects the dataset.")
    if (x["Volume"] < 0).any(): raise DataIntegrityError("Negative volume detected.")
    if len(x) < minimum_observations:
        raise DataIntegrityError(f"Only {len(x)} observations; minimum is {minimum_observations}.")
    return x


class YahooFinanceAdapter:
    """Strict single-source Yahoo Finance adapter. No alternate source/fallback."""
    @staticmethod
    def fetch(ticker: str, start: str, end: str, interval: str = "1d", minimum_observations: int = 30) -> pd.DataFrame:
        if not ticker or not str(ticker).strip(): raise ValueError("ticker is required")
        try:
            import yfinance as yf
        except Exception as e:
            raise MarketDataError("yfinance is not installed. Install the pinned requirements; no alternate source is used.") from e
        # yfinance end is EXCLUSIVE. User-facing end is inclusive, so request +1 calendar day.
        end_exclusive = (pd.Timestamp(end).normalize() + pd.Timedelta(days=1)).date().isoformat()
        try:
            raw = yf.download(
                tickers=str(ticker).strip(), start=str(start), end=end_exclusive,
                interval=interval, auto_adjust=False, repair=False, actions=False,
                keepna=False, progress=False, threads=False, group_by="column",
                multi_level_index=False, timeout=20,
            )
        except Exception as e:
            raise MarketDataError(f"Yahoo Finance download failed for {ticker}. No fallback will be used: {e}") from e
        try:
            return validate_market_data(raw, minimum_observations)
        except Exception as e:
            raise MarketDataError(f"Yahoo Finance data failed strict validation for {ticker}. No fallback will be used: {e}") from e


def _window_start(i: int, lookback: int, legacy_inclusive: bool) -> int:
    if legacy_inclusive:
        # Exact legacy OFFSET(...,-MIN(counter,lookback),0):current behavior.
        # Once saturated this is lookback+1 observations (legacy off-by-one retained intentionally).
        return max(0, i - min(i, lookback))
    return max(0, i - lookback + 1)


def run_legacy_engine(market: pd.DataFrame, config: EngineConfig = EngineConfig(), *, validate: bool = True) -> pd.DataFrame:
    config.validate()
    m = validate_market_data(market, config.minimum_observations) if validate else market.copy()
    n=len(m); idx=m.index
    out=m.copy()
    # adjusted OHLC using Yahoo Adj Close / Close factor; exact legacy methodology.
    scale = out["Adj Close"].to_numpy(float) / out["Close"].to_numpy(float)
    adj_open=out["Open"].to_numpy(float)*scale
    adj_high=out["High"].to_numpy(float)*scale
    adj_low=out["Low"].to_numpy(float)*scale
    adj_close=out["Close"].to_numpy(float)*scale

    ret=np.full(n,np.nan); volume_k=out["Volume"].to_numpy(float)/1000.0
    counter=np.full(n,np.nan); hl=np.full(n,np.nan); hp=np.full(n,np.nan); lp=np.full(n,np.nan); tr=np.full(n,np.nan)
    atr_stop=np.full(n,np.nan); max_price=np.full(n,np.nan); signal=np.full(n,"",dtype=object)
    shares=np.zeros(n); cash=np.zeros(n); portfolio=np.zeros(n); buyhold=np.zeros(n)
    buy_marker=np.zeros(n); sell_marker=np.zeros(n); first_buy=np.zeros(n); first_sell=np.zeros(n)
    std_pop=np.full(n,np.nan); lower_bolli=np.full(n,np.nan); atr_trailing=np.full(n,np.nan)

    ret[0]=out["Close"].iloc[0]/out["Open"].iloc[0]-1.0
    max_price[0]=adj_close[0]
    shares[0]=0.0; cash[0]=config.initial_capital; portfolio[0]=config.initial_capital; buyhold[0]=config.initial_capital

    for i in range(1,n):
        ret[i]=adj_close[i]/adj_close[i-1]-1.0
        counter[i]=float(i)
        hl[i]=adj_high[i]-adj_low[i]
        hp[i]=abs(adj_high[i]-adj_close[i-1])
        lp[i]=abs(adj_low[i]-adj_close[i-1])
        tr[i]=max(hl[i],hp[i],lp[i])

        a0=_window_start(i, config.atr_weeks, config.legacy_inclusive_windows)
        valid_tr=tr[a0:i+1]
        valid_tr=valid_tr[np.isfinite(valid_tr)]
        atr_stop[i]=config.atr_multiplier*float(valid_tr.mean()) if len(valid_tr) else np.nan

        p0=_window_start(i, config.max_buy_weeks, config.legacy_inclusive_windows)
        max_price[i]=float(np.max(adj_close[p0:i+1]))

        b0=_window_start(i, config.bollinger_weeks, config.legacy_inclusive_windows)
        boll_slice=adj_close[b0:i+1]
        std_pop[i]=float(np.std(boll_slice,ddof=0))
        lower_bolli[i]=float(np.mean(boll_slice)-config.bollinger_sd*std_pop[i])

        # Signal is based only on PRIOR completed bar; execution occurs at current adjusted open.
        if adj_close[i-1] >= max_price[i-1]:
            signal[i]="BUY"
        else:
            if config.strategy=="ATR": selected=atr_stop[i-1]
            elif config.strategy=="BOLLINGER": selected=lower_bolli[i-1]
            else: selected=atr_trailing[i-1]
            if np.isfinite(selected) and adj_close[i-1] <= selected:
                signal[i]="SELL"

        # Exact legacy ATR Trailing Stop behavior: monotonic max of ATR_Stop only on BUY-signal rows.
        if i==1:
            atr_trailing[i]=atr_stop[i]
        else:
            prev=atr_trailing[i-1]
            if signal[i]=="BUY" and np.isfinite(atr_stop[i]) and (not np.isfinite(prev) or atr_stop[i]>prev):
                atr_trailing[i]=atr_stop[i]
            else:
                atr_trailing[i]=prev

        prev_shares,prev_cash=shares[i-1],cash[i-1]
        if signal[i]=="SELL":
            shares[i]=0.0
            cash[i]=prev_cash + prev_shares*adj_open[i]
        elif signal[i]=="BUY" and prev_shares==0.0:
            shares[i]=prev_shares + prev_cash/adj_open[i]
            cash[i]=0.0
        else:
            shares[i]=prev_shares; cash[i]=prev_cash
        portfolio[i]=shares[i]*adj_close[i]+cash[i]
        buyhold[i]=config.initial_capital*adj_close[i]/adj_close[0]
        first_buy[i]=1.0 if signal[i]=="BUY" and prev_shares==0.0 else 0.0
        first_sell[i]=1.0 if signal[i]=="SELL" and prev_shares>0.0 else 0.0
        buy_marker[i]=buyhold[i]*first_buy[i]
        sell_marker[i]=buyhold[i]*first_sell[i]

    out["Return"]=ret;out["VolumeK"]=volume_k;out["Counter"]=counter
    out["HighLow"]=hl;out["HighPrevCloseAbs"]=hp;out["LowPrevCloseAbs"]=lp;out["TrueRange"]=tr
    out["ATR_Stop"]=atr_stop;out["MaxPrice"]=max_price;out["Scale"]=scale;out["Signal"]=signal
    out["Shares"]=shares;out["Cash"]=cash;out["Portfolio"]=portfolio
    out["AdjOpen"]=adj_open;out["AdjHigh"]=adj_high;out["AdjLow"]=adj_low;out["AdjCloseCalc"]=adj_close
    out["BuyHold"]=buyhold;out["BuyMarker"]=buy_marker;out["SellMarker"]=sell_marker
    out["FirstBuy"]=first_buy;out["FirstSell"]=first_sell;out["StdDevPop"]=std_pop
    out["LowerBollinger"]=lower_bolli;out["ATRTrailingStop"]=atr_trailing
    return out


def performance_summary(df: pd.DataFrame, initial_capital: float, periods_per_year: Optional[float]=None) -> dict:
    if len(df)<2: raise ValueError("At least two observations required")
    if periods_per_year is None:
        gaps=pd.Series(df.index).diff().dt.days.dropna()
        median_days=float(gaps.median()) if len(gaps) else 1
        periods_per_year=252.0 if median_days<=3 else 52.0 if median_days<=10 else 12.0
    years=(df.index[-1]-df.index[0]).days/365.25
    port_final=float(df["Portfolio"].iloc[-1]); bh_final=float(df["BuyHold"].iloc[-1])
    p_ret=df["Portfolio"].pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    running=df["Portfolio"]/df["Portfolio"].cummax()-1
    bh_running=df["BuyHold"]/df["BuyHold"].cummax()-1
    vol=float(p_ret.std(ddof=1)*math.sqrt(periods_per_year)) if len(p_ret)>1 else np.nan
    cagr=(port_final/initial_capital)**(1/years)-1 if years>0 else np.nan
    bh_cagr=(bh_final/initial_capital)**(1/years)-1 if years>0 else np.nan
    return {
        "start":df.index[0],"end":df.index[-1],"observations":len(df),
        "portfolio_final":port_final,"buyhold_final":bh_final,"strategy_cagr":cagr,"buyhold_cagr":bh_cagr,
        "annualized_volatility":vol,"max_drawdown":float(running.min()),"buyhold_max_drawdown":float(bh_running.min()),
        "first_buys":int(df["FirstBuy"].sum()),"first_sells":int(df["FirstSell"].sum()),
        "current_signal":str(df["Signal"].iloc[-1] or "HOLD"),
    }


def load_legacy_golden_csv(path: str | Path) -> pd.DataFrame:
    g=pd.read_csv(path,parse_dates=["Date"]).set_index("Date")
    return g


def golden_master_validation(golden_csv: str | Path, config: EngineConfig=EngineConfig(), atol: float=1e-8) -> pd.DataFrame:
    g=load_legacy_golden_csv(golden_csv)
    market=g.rename(columns={"AdjClose":"Adj Close"})[["Open","High","Low","Close","Volume","Adj Close"]]
    calc=run_legacy_engine(market,config,validate=False)
    mapping=["Return","VolumeK","Counter","HighLow","HighPrevCloseAbs","LowPrevCloseAbs","TrueRange","ATR_Stop","MaxPrice","Scale",
             "Shares","Cash","Portfolio","AdjOpen","AdjHigh","AdjLow","AdjCloseCalc","BuyHold","BuyMarker","SellMarker","FirstBuy","FirstSell",
             "StdDevPop","LowerBollinger"]
    rows=[]
    for c in mapping:
        a=pd.to_numeric(g[c],errors="coerce").to_numpy(float); b=pd.to_numeric(calc[c],errors="coerce").to_numpy(float)
        mask=np.isfinite(a)&np.isfinite(b)
        max_abs=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.0
        mismatch=int(np.sum(np.abs(a[mask]-b[mask])>atol))
        rows.append({"Field":c,"Compared":int(mask.sum()),"MaxAbsError":max_abs,"Mismatches":mismatch,"Pass":mismatch==0})
    # ATRTrailingStop legacy first row is a text label; compare from row 2 onward.
    a=pd.to_numeric(g["ATRTrailingStop"],errors="coerce").to_numpy(float);b=calc["ATRTrailingStop"].to_numpy(float)
    mask=np.isfinite(a)&np.isfinite(b);err=np.abs(a[mask]-b[mask]);rows.append({"Field":"ATRTrailingStop","Compared":int(mask.sum()),"MaxAbsError":float(err.max()) if len(err) else 0.0,"Mismatches":int((err>atol).sum()),"Pass":bool((err<=atol).all()) if len(err) else True})
    signal_equal=(g["Signal"].fillna("").astype(str).to_numpy()==calc["Signal"].fillna("").astype(str).to_numpy())
    rows.append({"Field":"Signal","Compared":len(signal_equal),"MaxAbsError":np.nan,"Mismatches":int((~signal_equal).sum()),"Pass":bool(signal_equal.all())})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ticker");ap.add_argument("--start");ap.add_argument("--end");ap.add_argument("--interval",default="1d")
    ap.add_argument("--strategy",default="ATR_TRAILING_STOP",choices=["ATR","BOLLINGER","ATR_TRAILING_STOP"])
    ap.add_argument("--golden-test")
    args=ap.parse_args()
    cfg=EngineConfig(strategy=args.strategy)
    if args.golden_test:
        v=golden_master_validation(args.golden_test,cfg);print(v.to_string(index=False));
        if not bool(v["Pass"].all()): raise SystemExit(2)
    elif args.ticker and args.start and args.end:
        m=YahooFinanceAdapter.fetch(args.ticker,args.start,args.end,args.interval,cfg.minimum_observations)
        r=run_legacy_engine(m,cfg); print(performance_summary(r,cfg.initial_capital))
    else:
        ap.error("Provide --golden-test OR --ticker --start --end")
