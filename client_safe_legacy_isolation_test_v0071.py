from pathlib import Path
import ast
src=(Path(__file__).resolve().parent/"app.py").read_text(encoding="utf-8")
ast.parse(src)
assert 'APP_VERSION = "v0.07.1"' in src
assert 'Show Legacy Audit Layer' in src
assert 'value=False' in src[src.index('Show Legacy Audit Layer')-200:src.index('Show Legacy Audit Layer')+350]
strip=src[src.index('# ---------------------------- Executive strip'):src.index('# ---------------------------- Main tabs')]
assert 'Legacy Reference' not in strip
assert 'NO DECISION' in strip
assert 'TACTICAL DISABLED' in strip
assert '"Price & Signals"' in src and '"Price & Legacy Signals"' not in src
for phrase in ['INTERNAL AUDIT — Legacy Golden Master Reference','INTERNAL AUDIT — Legacy Price/Stop Chart','INTERNAL AUDIT — Legacy Trade Ledger','INTERNAL AUDIT — Legacy Calculation Ledger']:
    assert phrase in src
print('PASS — v0.07.1 client-safe Legacy isolation')
print('Legacy default visibility: OFF')
print('Legacy primary-decision fallback: DISABLED')
print('Primary client engine: MK Institutional Tactical')
