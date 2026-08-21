from pathlib import Path
import ast
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
app = (ROOT / "app.py").read_text(encoding="utf-8")
vis = (ROOT / "MK_Intraday_Visuals_v0088.py").read_text(encoding="utf-8")
ast.parse(app)
ast.parse(vis)

assert 'APP_VERSION = "v0.08.8.1"' in app
assert 'Frequency / Bar Size' in app
assert '15-MINUTE MODE SELECTED' in app
assert 'Intraday 15-Minute Mode Active' in app
assert '("15m Intraday Tactical Lab · ACTIVE" if interval_used == "15m"' in app
assert 'with tabs[1]:' in app
assert '## 15-Minute Intraday Tactical Lab' in app
assert 'def _plotly_chart(' in app
assert '"15-Minute" not in str(_title)' in app

# Direct st.plotly_chart is allowed exactly once inside the unified renderer.
assert app.count('st.plotly_chart(') == 1, app.count('st.plotly_chart(')
assert app.count('_plotly_chart(') >= 16

assert 'rows=5' in vis
assert 'TACTICAL TARGET EXPOSURE' in vis
assert 'line_shape="hv"' in vis
assert 'tickvals=[0,25,50,75,100]' in vis
assert 'row=5, col=1' in vis

from MK_Intraday_Visuals_v0088 import build_intraday_tactical_figure
idx = pd.date_range('2026-08-20 10:00', periods=12, freq='15min')
df = pd.DataFrame(index=idx)
df['AdjOpen']=100+np.arange(12)*0.1
df['AdjHigh']=df['AdjOpen']+0.3
df['AdjLow']=df['AdjOpen']-0.3
df['AdjCloseCalc']=df['AdjOpen']+0.05
df['Volume']=1000+np.arange(12)*25
df['SessionVWAP']=df['AdjCloseCalc'].expanding().mean()
df['OpeningRangeHigh']=101.0
df['OpeningRangeLow']=99.5
df['NWTrend']=df['AdjCloseCalc'].rolling(2,min_periods=1).mean()
df['NWDirection']=np.where(np.arange(12)%4<2,1,-1)
df['SlotRelativeVolume']=1.0
df['IntradayRealizedVol']=0.2
df['SessionDrawdownPct']=-0.01
df['IntradayConfirmationScore']=np.linspace(-40,60,12)
df['TacticalTargetExposure']=[1,1,.75,.75,.5,.5,.25,.25,.5,.75,1,1]
fig=build_intraday_tactical_figure(df,'TEST','Validation Session')
assert len(fig._grid_ref)==5
exp=[t for t in fig.data if t.name=='TACTICAL TARGET EXPOSURE']
assert len(exp)==1
assert exp[0].line.shape=='hv'
assert list(exp[0].y)==[100,100,75,75,50,50,25,25,50,75,100,100]
assert fig.layout.title.text.startswith('15-Minute Intraday Tactical Lab')
print('PASS — v0.08.8.1 intraday visibility validation')
print('15m sidebar discoverability: PASS')
print('15m selected + active banners: PASS')
print('15m tab promoted to position 2: PASS')
print('All Plotly charts route through 15-minute title wrapper: PASS')
print('Dedicated Target Exposure staircase panel: PASS')
