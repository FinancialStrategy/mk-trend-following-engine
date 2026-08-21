
"""
MK Benchmark Relative Analytics v0.08.7
By Murat Konuklar

Strict Yahoo-only benchmark layer.
No alternate provider, no price filling, no synthetic observations.

Purpose:
- map each instrument to an explicit benchmark
- align asset and benchmark on exact timestamps only
- estimate lagged rolling beta
- calculate beta-adjusted residual returns
- calculate cumulative residual drift and standardized relative deviation
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

# Explicit research benchmark map. User can override in the UI.
TICKER_BENCHMARKS = {
    # BIST Banks
    "AKBNK.IS":"XBANK.IS","GARAN.IS":"XBANK.IS","HALKB.IS":"XBANK.IS",
    "ISCTR.IS":"XBANK.IS","SKBNK.IS":"XBANK.IS","TSKB.IS":"XBANK.IS",
    "VAKBN.IS":"XBANK.IS","YKBNK.IS":"XBANK.IS",
    # BIST Insurance / finance
    "ANSGR.IS":"XSGRT.IS","TURSG.IS":"XSGRT.IS",
    "VAKFA.IS":"XUMAL.IS","GARFA.IS":"XUMAL.IS",
    # BIST industrial / manufacturing
    "ASELS.IS":"XUSIN.IS","ARCLK.IS":"XUSIN.IS","EREGL.IS":"XUSIN.IS",
    "FROTO.IS":"XUSIN.IS","KRDMD.IS":"XUSIN.IS","SISE.IS":"XUSIN.IS",
    "TOASO.IS":"XUSIN.IS","TUPRS.IS":"XKMYA.IS","PETKM.IS":"XKMYA.IS",
    "CIMSA.IS":"XUSIN.IS","KARTN.IS":"XUSIN.IS","ADEL.IS":"XUSIN.IS",
    "ASTOR.IS":"XUSIN.IS","SMRTG.IS":"XELKT.IS","CWENE.IS":"XELKT.IS",
    "AKSEN.IS":"XELKT.IS","ENJSA.IS":"XELKT.IS","ODAS.IS":"XELKT.IS",
    "ZOREN.IS":"XELKT.IS","CANTE.IS":"XELKT.IS","ESEN.IS":"XELKT.IS",
    "GWIND.IS":"XELKT.IS",
    # BIST retail
    "BIMAS.IS":"XTCRT.IS","MGROS.IS":"XTCRT.IS","SOKM.IS":"XTCRT.IS","CRFSA.IS":"XTCRT.IS",
    # BIST transport / holdings / telecom
    "THYAO.IS":"XULAS.IS","PGSUS.IS":"XULAS.IS","TAVHL.IS":"XULAS.IS",
    "KCHOL.IS":"XHOLD.IS","SAHOL.IS":"XHOLD.IS",
    "TCELL.IS":"XU100.IS","TTKOM.IS":"XU100.IS",

    # US: liquid sector / industry ETFs as explicit relative benchmarks
    "AAPL":"XLK","MSFT":"XLK","NVDA":"SMH","GOOGL":"QQQ","AMZN":"QQQ","META":"QQQ","AVGO":"SMH","ORCL":"XLK",
    "AMD":"SMH","AMAT":"SMH","MU":"SMH","QCOM":"SMH","INTC":"SMH","TXN":"SMH","MRVL":"SMH","LRCX":"SMH",
    "LLY":"XLV","BMY":"XLV","JNJ":"XLV","ABBV":"XLV","UNH":"XLV","MRK":"XLV",
    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","V":"XLF","MA":"XLF",
    "CAT":"XLI","GE":"XLI","RTX":"ITA","BA":"ITA","HON":"XLI","DE":"XLI",
    "XOM":"XLE","CVX":"XLE","COP":"XLE","SLB":"XLE",
    "WMT":"XLP","COST":"XLP","HD":"XLY","MCD":"XLY","TSLA":"XLY","NKE":"XLY",

    # Precious metals: explicit cross-vehicle benchmark pairs
    "GC=F":"GLD","GLD":"GC=F","IAU":"GC=F",
    "SI=F":"SLV","SLV":"SI=F","SIVR":"SI=F",
    "PL=F":"PPLT","PPLT":"PL=F",
    "PA=F":"PALL","PALL":"PA=F",

    # Industrial metals
    "HG=F":"CPER","CPER":"HG=F",

    # Crypto — stablecoins intentionally excluded from the investable universe.
    # BTC uses ETH as a liquid peer benchmark; ETH and major altcoins use BTC.
    "BTC-USD":"ETH-USD",
    "ETH-USD":"BTC-USD",
    "BNB-USD":"BTC-USD","XRP-USD":"BTC-USD","SOL-USD":"BTC-USD",
    "TRX-USD":"BTC-USD","HYPE32196-USD":"BTC-USD","DOGE-USD":"BTC-USD",
    "ZEC-USD":"BTC-USD","LEO-USD":"BTC-USD","LINK-USD":"BTC-USD","ADA-USD":"BTC-USD",
}

BENCHMARK_NAMES = {
    "XBANK.IS":"BIST BANKA",
    "XSGRT.IS":"BIST SIGORTA",
    "XUMAL.IS":"BIST MALI",
    "XUSIN.IS":"BIST SINAI",
    "XKMYA.IS":"BIST KIMYA PETROL PLASTIK",
    "XELKT.IS":"BIST ELEKTRIK",
    "XTCRT.IS":"BIST TICARET",
    "XULAS.IS":"BIST ULASTIRMA",
    "XHOLD.IS":"BIST HOLDING VE YATIRIM",
    "XU100.IS":"BIST 100",
    "XLK":"Technology Select Sector SPDR",
    "QQQ":"Invesco QQQ",
    "SMH":"VanEck Semiconductor ETF",
    "XLV":"Health Care Select Sector SPDR",
    "XLF":"Financial Select Sector SPDR",
    "XLI":"Industrial Select Sector SPDR",
    "ITA":"iShares U.S. Aerospace & Defense ETF",
    "XLE":"Energy Select Sector SPDR",
    "XLP":"Consumer Staples Select Sector SPDR",
    "XLY":"Consumer Discretionary Select Sector SPDR",
    "GLD":"SPDR Gold Shares","SLV":"iShares Silver Trust",
    "PPLT":"abrdn Physical Platinum Shares ETF","PALL":"abrdn Physical Palladium Shares ETF",
    "GC=F":"Gold Futures","SI=F":"Silver Futures","PL=F":"Platinum Futures","PA=F":"Palladium Futures",
    "HG=F":"Copper Futures","CPER":"United States Copper Index Fund",
    "BTC-USD":"Bitcoin USD","ETH-USD":"Ethereum USD","BNB-USD":"BNB USD","XRP-USD":"XRP USD",
    "SOL-USD":"Solana USD","TRX-USD":"TRON USD","HYPE32196-USD":"Hyperliquid USD",
    "DOGE-USD":"Dogecoin USD","ZEC-USD":"Zcash USD","LEO-USD":"UNUS SED LEO USD",
    "LINK-USD":"Chainlink USD","ADA-USD":"Cardano USD",
}

@dataclass(frozen=True)
class RelativeConfig:
    beta_window: int = 40
    drift_horizon: int = 8
    minimum_alignment_ratio: float = 0.80
    weak_z: float = 1.5
    strong_z: float = 2.0
    extreme_z: float = 3.0

    def validate(self):
        if self.beta_window < 10: raise ValueError("beta_window must be >= 10")
        if self.drift_horizon < 1: raise ValueError("drift_horizon must be >= 1")
        if not 0 < self.minimum_alignment_ratio <= 1: raise ValueError("minimum_alignment_ratio must be in (0,1]")
        if not (0 < self.weak_z < self.strong_z < self.extreme_z):
            raise ValueError("Require 0 < weak_z < strong_z < extreme_z")


def default_benchmark(ticker: str) -> str | None:
    return TICKER_BENCHMARKS.get(str(ticker).upper())


def benchmark_name(ticker: str) -> str:
    return BENCHMARK_NAMES.get(str(ticker).upper(), str(ticker).upper())


def compute_relative_analytics(
    asset_df: pd.DataFrame,
    benchmark_market: pd.DataFrame,
    config: RelativeConfig = RelativeConfig(),
) -> pd.DataFrame:
    config.validate()
    if "AdjCloseCalc" not in asset_df.columns:
        raise KeyError("asset_df must contain AdjCloseCalc")
    if "Adj Close" not in benchmark_market.columns:
        raise KeyError("benchmark_market must contain Adj Close")

    asset_px = pd.to_numeric(asset_df["AdjCloseCalc"], errors="raise").rename("AssetPrice")
    bench_px = pd.to_numeric(benchmark_market["Adj Close"], errors="raise").rename("BenchmarkPrice")

    aligned = pd.concat([asset_px, bench_px], axis=1, join="inner").dropna()
    denom = min(len(asset_px), len(bench_px))
    ratio = len(aligned) / denom if denom else 0.0
    if ratio < config.minimum_alignment_ratio:
        raise ValueError(
            f"Asset/benchmark exact-timestamp alignment is only {ratio:.1%}; "
            f"minimum is {config.minimum_alignment_ratio:.0%}. No fill/substitution is permitted."
        )
    if len(aligned) <= config.beta_window + config.drift_horizon + 2:
        raise ValueError(
            f"Only {len(aligned)} aligned observations; relative model requires more than "
            f"{config.beta_window + config.drift_horizon + 2}."
        )

    a = np.log(aligned["AssetPrice"]).diff()
    b = np.log(aligned["BenchmarkPrice"]).diff()

    beta_raw = a.rolling(config.beta_window, min_periods=config.beta_window).cov(b) / \
               b.rolling(config.beta_window, min_periods=config.beta_window).var()
    beta = beta_raw.shift(1)

    mean_a = a.rolling(config.beta_window, min_periods=config.beta_window).mean().shift(1)
    mean_b = b.rolling(config.beta_window, min_periods=config.beta_window).mean().shift(1)
    alpha = mean_a - beta * mean_b

    residual = a - (alpha + beta * b)
    resid_sigma = residual.rolling(config.beta_window, min_periods=config.beta_window).std(ddof=1).shift(1)

    residual_z = residual / resid_sigma
    drift = residual.rolling(config.drift_horizon, min_periods=config.drift_horizon).sum()
    drift_z = drift / (resid_sigma * math.sqrt(config.drift_horizon))

    log_ratio = np.log(aligned["AssetPrice"] / aligned["BenchmarkPrice"])
    ratio_mean = log_ratio.rolling(config.beta_window, min_periods=config.beta_window).mean().shift(1)
    ratio_std = log_ratio.rolling(config.beta_window, min_periods=config.beta_window).std(ddof=1).shift(1)
    ratio_z = (log_ratio - ratio_mean) / ratio_std

    h = config.drift_horizon
    asset_h = aligned["AssetPrice"].pct_change(h, fill_method=None)
    bench_h = aligned["BenchmarkPrice"].pct_change(h, fill_method=None)

    rel = pd.DataFrame(index=aligned.index)
    rel["BenchmarkPrice"] = aligned["BenchmarkPrice"]
    rel["RollingBeta"] = beta
    rel["RollingAlpha"] = alpha
    rel["ResidualReturn"] = residual
    rel["ResidualZ"] = residual_z
    rel["ResidualDrift"] = drift
    rel["ResidualDriftZ"] = drift_z
    rel["PriceRatioZ"] = ratio_z
    rel["AssetReturnH"] = asset_h
    rel["BenchmarkReturnH"] = bench_h
    rel["RelativeReturnH"] = asset_h - bench_h
    rel["AlignmentRatio"] = ratio

    # Map only exact timestamps back to the asset index. Missing benchmark bars stay NaN.
    return rel.reindex(asset_df.index)


def relative_snapshot(rel: pd.DataFrame, config: RelativeConfig) -> dict:
    valid = rel.dropna(subset=["ResidualDriftZ"])
    if valid.empty:
        return {
            "status":"UNAVAILABLE","drift_z":np.nan,"residual_z":np.nan,"beta":np.nan,
            "ratio_z":np.nan,"relative_return_h":np.nan,
        }
    x = valid.iloc[-1]
    dz = float(x["ResidualDriftZ"])
    if dz <= -config.extreme_z:
        status = "EXTREME RELATIVE BREAKDOWN"
    elif dz <= -config.strong_z:
        status = "STRONG RELATIVE WEAKNESS"
    elif dz <= -config.weak_z:
        status = "RELATIVE WEAKNESS"
    elif dz >= config.extreme_z:
        status = "EXTREME RELATIVE OVEREXTENSION"
    elif dz >= config.strong_z:
        status = "STRONG RELATIVE OUTPERFORMANCE"
    elif dz >= config.weak_z:
        status = "RELATIVE OUTPERFORMANCE"
    else:
        status = "NORMAL RELATIVE RANGE"
    return {
        "status":status,
        "drift_z":dz,
        "residual_z":float(x["ResidualZ"]) if pd.notna(x["ResidualZ"]) else np.nan,
        "beta":float(x["RollingBeta"]) if pd.notna(x["RollingBeta"]) else np.nan,
        "ratio_z":float(x["PriceRatioZ"]) if pd.notna(x["PriceRatioZ"]) else np.nan,
        "relative_return_h":float(x["RelativeReturnH"]) if pd.notna(x["RelativeReturnH"]) else np.nan,
    }
