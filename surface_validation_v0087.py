from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent
app=(ROOT/'app.py').read_text(encoding='utf-8')
ast.parse(app)
assert 'APP_VERSION = "v0.08.7"' in app
assert 'from MK_Trend_Following_Universe_v0087 import (' in app
assert 'from MK_Nadaraya_Watson_Trend_v0087 import (' in app
assert 'from MK_Nadaraya_Watson_Visuals_v0087 import (' in app
assert 'from MK_Benchmark_Relative_v0087 import (' in app
assert 'from MK_Institutional_Risk_Analytics_v0087 import (' in app
assert 'Momentum Up / Down Warnings' in app
assert 'NW Alert Tape — QuantAlgo Public Alerts + MK Momentum Warnings' in app
assert 'STATE_SCHEMA_VERSION = 6' in app
print('PASS — v0.08.7 application surface wiring')
