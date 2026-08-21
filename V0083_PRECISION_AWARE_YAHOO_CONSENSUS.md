# v0.08.3 — Precision-Aware Yahoo Consensus

## AKBNK.IS incident
The reported Yahoo values differ only around the sixth decimal place, e.g. 4.547914028 vs 4.547913074. This is approximately 0.0021 basis points and is consistent with binary/JSON float serialization differences across Yahoo routes, not a market-price conflict.

## New rule
Price-route reconciliation uses a bounded machine-precision envelope:

`max(0.000002, 16 × float32_epsilon × max(abs(price), 1))`

This is about 0.019 basis points relative and remains far below an economically tradable BIST price tick.

If Yahoo values are inside the envelope, the engine preserves the highest-priority ACTUAL Yahoo observation. It never averages, takes a median, interpolates, or synthesizes a value.

If the route spread exceeds the envelope, strict consensus still hard-stops. Volume remains integer-strict (±0.5 unit).
