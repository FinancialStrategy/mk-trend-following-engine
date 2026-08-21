from pathlib import Path
import ast,re
ROOT=Path(__file__).resolve().parent
src=(ROOT/'app.py').read_text(encoding='utf-8')
ast.parse(src)
assert 'APP_VERSION = "v0.08"' in src
# Forbidden client headings / text
for forbidden in [
    'Price & Legacy Signals',
    'Legacy Reference / Audit Layer',
    'Trend Diagnostics — All Legacy Thresholds',
    'INTERNAL AUDIT — Legacy Golden Master Reference',
    'INTERNAL AUDIT — Legacy Price/Stop Chart',
    'INTERNAL AUDIT — Legacy Threshold Diagnostics',
    'Show Legacy Audit Layer',
]:
    assert forbidden not in src, forbidden
assert 'Nadaraya-Watson + Tactical Rulebook' in src
assert 'Value at Risk — Asset, Benchmark, Active Residual & Strategy' in src
assert 'Historical' in src and 'Parametric Normal' in src and 'Monte Carlo Bootstrap' in src
assert 'Institutional Trend Diagnostics — NW Structure, Relative Regime & Exposure' in src
print('PASS — v0.08 institutional client surface regression')
print('Client Legacy panels: REMOVED')
print('NW rulebook: PRESENT')
print('VaR Historical / Parametric / Monte Carlo: PRESENT')
print('Trend diagnostics: NW + benchmark-relative only')
