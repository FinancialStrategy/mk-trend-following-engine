# v0.08.1 — Yahoo Strict Reconciliation

## Incident
GARAN.IS returned an incomplete Yahoo payload containing at least one mandatory missing field on 2026-08-20.
The previous adapter correctly hard-stopped, but it treated the first Yahoo response as final.

## New protocol
1. Yahoo Route A: `yfinance.download`
2. Strict completeness validation
3. If incomplete, Yahoo Route B: `Ticker.history`
4. Strict completeness validation
5. Cross-check all values actually supplied by Route A against Route B
6. Accept Route B only if it is complete and all common observed values reconcile
7. Otherwise hard-stop

Both acquisition routes are Yahoo Finance. There is no alternate market-data provider in this path.

## Explicit prohibitions
- no forward fill
- no backfill
- no Close -> Adj Close substitution
- no deletion of an incomplete Yahoo row to make the test pass
- no Reuters / Investing / other price-data fallback
- no preference between conflicting Yahoo payloads: conflict means STOP

## Auditability
The Executive & Primary Decision page includes a collapsed **Yahoo Data Quality & Reconciliation Audit** showing:
- asset / benchmark ticker
- accepted Yahoo route
- primary / secondary status
- whether same-source reconciliation was needed
- accepted observation count

## Enhanced failure detail
If both Yahoo routes remain incomplete, the STOP message reports the exact missing fields by timestamp.
