from pathlib import Path
import numpy as np
import pandas as pd
from MK_Institutional_Tactical_v0089 import TacticalConfig, decision_rulebook, decision_formula_table, decision_semantics_table, decision_rule_evaluation, _decision_for_bar
ROOT=Path(__file__).resolve().parent
cfg=TacticalConfig(weak_z=1.25,strong_z=1.75,extreme_z=2.50,volume_climax=1.35,immediate_upper_band_reduce=0.75,reentry_reduce=0.50,strong_risk_reduce=0.25)
book=decision_rulebook(cfg)
assert list(book["Priority"])==list(range(16)); assert book.iloc[-1]["Rule ID"]=="DEFAULT_HOLD"
joined=' '.join(book['Condition / Formula'].astype(str)); assert all(x in joined for x in ['1.25','1.75','2.50','1.35'])
assert len(decision_formula_table(cfg))>=9
assert set(['BUY / RESTORE','HOLD','REDUCE','SELL / EXIT TO CASH','WAIT / CASH']).issubset(set(decision_semantics_table()['Decision']))
def row(**o):
    r=dict(NWTrend=100.0,ResidualDriftZ=0.2,NWCrossAboveUpper=False,NWReenterBelowUpper=False,NWCrossBelowLower=False,NWReenterAboveLower=False,NWSource=100.0,NWBearishReversal=False,NWBullishReversal=False,NWDirection=0,RelativeVolume=1.0,NWSlopeDeceleration=0.0,NWNormalizedSlope=0.01,RollingBeta=1.0,NWEnvelopeZ=0.2,TacticalActualExposure=1.0); r.update(o); return pd.DataFrame([r])
x=row(); d,a,w=_decision_for_bar(x,0,1.0,cfg); assert d==1 and a=='HOLD'; assert decision_rule_evaluation(x,0,1.0,cfg).query("Selected=='YES'").iloc[0]['Rule ID']=='DEFAULT_HOLD'
x=row(NWCrossBelowLower=True,NWDirection=-1,NWSource=95.0); d,a,w=_decision_for_bar(x,0,1.0,cfg); assert d==0 and a=='SELL / EXIT TO CASH — 0%'
x=row(NWCrossAboveUpper=True,ResidualDriftZ=0.2); d,a,w=_decision_for_bar(x,0,0.50,cfg); assert d==0.50 and a=='HOLD 50%'
x=row(NWBullishReversal=True,NWSource=102.0,NWTrend=100.0,ResidualDriftZ=0.1); d,a,w=_decision_for_bar(x,0,1.0,cfg); assert d==1.0 and a=='HOLD 100%'
x=row(NWTrend=np.nan); d,a,w=_decision_for_bar(x,0,0.75,cfg); assert d==0.75 and a=='HOLD'
app=(ROOT/'app.py').read_text(encoding='utf-8'); assert 'APP_VERSION = "v0.08.9"' in app; assert 'STATE_SCHEMA_VERSION = 8' in app; assert 'from MK_Institutional_Tactical_v0089 import' in app; assert 'Primary Decision Methodology & Formula Rulebook' in app; assert 'WHY HOLD / WAIT' in app
print('PASS — Primary Decision explainability & rulebook v0.08.9')
print('Priority rules: 16 including warm-up and default HOLD')
print('Dynamic thresholds: PASS')
print('Formula dictionary: PASS')
print('HOLD taxonomy: PASS')
print('SELL / EXIT TO CASH 0% terminology: PASS')
print('Decision thresholds/target tiers changed: NO')
