"""Curated Yahoo Finance instrument universe for MK Trend Following Analytics Engine v0.02.
By Murat Konuklar

This module contains identifiers only. It does not download, fabricate, or substitute market data.
Live price history remains Yahoo Finance-only through the strict adapter in the core engine.
"""
from __future__ import annotations

UNIVERSE = {
    "BIST": {
        "Banks": [
            ("Akbank", "AKBNK.IS"), ("Garanti BBVA", "GARAN.IS"), ("Halkbank", "HALKB.IS"),
            ("İş Bankası C", "ISCTR.IS"), ("Şekerbank", "SKBNK.IS"), ("TSKB", "TSKB.IS"),
            ("VakıfBank", "VAKBN.IS"), ("Yapı Kredi", "YKBNK.IS"),
        ],
        "Insurance & Financials": [
            ("Anadolu Sigorta", "ANSGR.IS"), ("Türkiye Sigorta", "TURSG.IS"),
            ("Vakıf Faktoring", "VAKFA.IS"), ("Garanti Faktoring", "GARFA.IS"),
        ],
        "Industrials & Materials": [
            ("ASELSAN", "ASELS.IS"), ("Arçelik", "ARCLK.IS"), ("Ereğli Demir Çelik", "EREGL.IS"),
            ("Ford Otosan", "FROTO.IS"), ("Kardemir D", "KRDMD.IS"), ("Şişecam", "SISE.IS"),
            ("Tofaş", "TOASO.IS"), ("Tüpraş", "TUPRS.IS"), ("Petkim", "PETKM.IS"),
            ("Çimsa", "CIMSA.IS"), ("Kartonsan", "KARTN.IS"), ("Adel Kalemcilik", "ADEL.IS"),
        ],
        "Energy & Utilities": [
            ("Aksa Enerji", "AKSEN.IS"), ("Enerjisa Enerji", "ENJSA.IS"), ("Odaş Elektrik", "ODAS.IS"),
            ("Zorlu Enerji", "ZOREN.IS"), ("Çan2 Termik", "CANTE.IS"), ("Esenboğa Elektrik", "ESEN.IS"),
            ("Astor Enerji", "ASTOR.IS"), ("Smart Güneş", "SMRTG.IS"), ("CW Enerji", "CWENE.IS"),
            ("Galata Wind", "GWIND.IS"),
        ],
        "Retail & Consumer": [
            ("BİM", "BIMAS.IS"), ("Migros", "MGROS.IS"), ("Şok Marketler", "SOKM.IS"),
            ("CarrefourSA", "CRFSA.IS"),
        ],
        "Transport, Holdings & Telecom": [
            ("Türk Hava Yolları", "THYAO.IS"), ("Pegasus", "PGSUS.IS"), ("TAV Havalimanları", "TAVHL.IS"),
            ("Koç Holding", "KCHOL.IS"), ("Sabancı Holding", "SAHOL.IS"),
            ("Turkcell", "TCELL.IS"), ("Türk Telekom", "TTKOM.IS"),
        ],
    },
    "US Stocks": {
        "Mega Cap Technology & AI": [
            ("Apple", "AAPL"), ("Microsoft", "MSFT"), ("NVIDIA", "NVDA"), ("Alphabet Class A", "GOOGL"),
            ("Amazon", "AMZN"), ("Meta Platforms", "META"), ("Broadcom", "AVGO"), ("Oracle", "ORCL"),
        ],
        "Semiconductors": [
            ("AMD", "AMD"), ("Applied Materials", "AMAT"), ("Micron Technology", "MU"),
            ("Qualcomm", "QCOM"), ("Intel", "INTC"), ("Texas Instruments", "TXN"),
            ("Marvell Technology", "MRVL"), ("Lam Research", "LRCX"),
        ],
        "Healthcare": [
            ("Eli Lilly", "LLY"), ("Bristol Myers Squibb", "BMY"), ("Johnson & Johnson", "JNJ"),
            ("AbbVie", "ABBV"), ("UnitedHealth Group", "UNH"), ("Merck", "MRK"),
        ],
        "Financials": [
            ("JPMorgan Chase", "JPM"), ("Bank of America", "BAC"), ("Goldman Sachs", "GS"),
            ("Morgan Stanley", "MS"), ("Visa", "V"), ("Mastercard", "MA"),
        ],
        "Industrials & Aerospace": [
            ("Caterpillar", "CAT"), ("GE Aerospace", "GE"), ("RTX", "RTX"),
            ("Boeing", "BA"), ("Honeywell", "HON"), ("Deere", "DE"),
        ],
        "Energy": [
            ("Exxon Mobil", "XOM"), ("Chevron", "CVX"), ("ConocoPhillips", "COP"),
            ("Schlumberger", "SLB"),
        ],
        "Consumer & Retail": [
            ("Walmart", "WMT"), ("Costco", "COST"), ("Home Depot", "HD"),
            ("McDonald's", "MCD"), ("Tesla", "TSLA"), ("Nike", "NKE"),
        ],
    },
    "Precious Metals": {
        "Futures": [
            ("Gold Futures", "GC=F"), ("Silver Futures", "SI=F"),
            ("Platinum Futures", "PL=F"), ("Palladium Futures", "PA=F"),
        ],
        "Exchange-Traded Products": [
            ("SPDR Gold Shares", "GLD"), ("iShares Gold Trust", "IAU"),
            ("iShares Silver Trust", "SLV"), ("abrdn Physical Silver Shares ETF", "SIVR"),
            ("abrdn Physical Platinum Shares ETF", "PPLT"), ("abrdn Physical Palladium Shares ETF", "PALL"),
        ],
    },
}


def market_names() -> list[str]:
    return list(UNIVERSE.keys()) + ["Manual Yahoo Ticker"]


def groups_for(market: str) -> list[str]:
    return list(UNIVERSE.get(market, {}).keys())


def instruments_for(market: str, group: str) -> list[tuple[str, str]]:
    return list(UNIVERSE.get(market, {}).get(group, []))


def flat_universe_rows() -> list[dict]:
    rows=[]
    for market, groups in UNIVERSE.items():
        for group, items in groups.items():
            for name, ticker in items:
                rows.append({"Market": market, "Group": group, "Instrument": name, "Yahoo Ticker": ticker})
    return rows
