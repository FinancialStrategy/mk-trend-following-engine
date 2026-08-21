# v0.08.7 — Crypto + Copper + QuantAlgo Visual/Alert Integration

## Investment universe additions

### Industrial Metals
- Copper Futures — `HG=F`

### Major non-stable crypto assets — Yahoo USD pairs
Selection is the current top 12 non-stablecoin market-cap set at build time, with Yahoo-compatible USD symbols:
1. Bitcoin — `BTC-USD`
2. Ethereum — `ETH-USD`
3. BNB — `BNB-USD`
4. XRP — `XRP-USD`
5. Solana — `SOL-USD`
6. TRON — `TRX-USD`
7. Hyperliquid — `HYPE32196-USD`
8. Dogecoin — `DOGE-USD`
9. Zcash — `ZEC-USD`
10. UNUS SED LEO — `LEO-USD`
11. Chainlink — `LINK-USD`
12. Cardano — `ADA-USD`

Stablecoins are deliberately excluded because a USD-pegged asset is not an appropriate primary trend-following target.

## Benchmark mapping
- Copper `HG=F` → `CPER`
- BTC → ETH as a liquid crypto peer benchmark
- ETH and the other 10 crypto assets → BTC

All price data remain Yahoo Finance only.

## QuantAlgo public-methodology parity
The implementation preserves the publicly described one-sided causal endpoint estimator, six kernels, effective bandwidth, kernel-weighted absolute residual bands, slope-based bullish/bearish path, reversal markers, and five public alert families:
- Bullish Kernel Reversal
- Bearish Kernel Reversal
- Any Kernel Reversal
- Source Cross Above Upper Band
- Source Cross Below Lower Band

The Pine source code itself is not redistributed.

## Visual parity layer
- green bullish NW path / red bearish NW path; segment `t-1 → t` is coloured by the current NW slope, matching Pine-style turn timing
- layered glow around the main trend path
- residual envelope fill
- green triangle below the NW path on bullish reversal
- red triangle above the NW path on bearish reversal
- optional trend-coloured candles
- optional trend background tint
- public band-cross markers
- alert tape
- Classic / Aqua / Cosmic / Cyber / Neon / Institutional Light / Custom themes

## MK Momentum warning extension
`Momentum Upward` and `Momentum Downward` are explicitly marked as an MK extension, not a QuantAlgo built-in alert.

They detect a causal sign change in normalized NW slope acceleration while the main NW slope has not yet completed the corresponding trend reversal. Their purpose is early warning only; they do not replace the confirmed kernel reversal signal.

## Crypto annualization fix
Risk analytics now detects material weekend sessions. Daily 7-day assets annualize at 365 observations; 24/7 intraday series use 365 × observed bars/day. Traditional exchange-traded daily data remains at 252.
