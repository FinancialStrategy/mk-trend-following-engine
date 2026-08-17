from __future__ import annotations
from pathlib import Path
import ast
import importlib
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

PY_FILES=[
    "app.py",
    "MK_Trend_Following_Engine_v001.py",
    "MK_Trend_Following_Decision_Engine_v002.py",
    "MK_Trend_Following_Entry_Gate_v005.py",
    "MK_Trend_Following_Risk_Analytics_v003.py",
    "MK_Trend_Following_Risk_Analytics_v004.py",
    "MK_Trend_Following_Universe_v002.py",
    "MK_Trend_Following_HTML_Report_v003.py",
    "MK_Nadaraya_Watson_Trend_v006.py",
    "MK_Nadaraya_Watson_HTML_Report_v006.py",
    "MK_DEMA_MACD_Confirmation_v007.py",
    "MK_DEMA_MACD_HTML_Report_v007.py",
    "dema_macd_validation_v007.py",
]
for name in PY_FILES:
    ast.parse((ROOT/name).read_text(encoding="utf-8"),filename=name)
    print("AST PASS",name)

IMPORT_MODULES=[
    "MK_Trend_Following_Engine_v001","MK_Trend_Following_Decision_Engine_v002",
    "MK_Trend_Following_Entry_Gate_v005","MK_Trend_Following_Risk_Analytics_v003",
    "MK_Trend_Following_Risk_Analytics_v004","MK_Trend_Following_Universe_v002",
    "MK_Trend_Following_HTML_Report_v003","MK_Nadaraya_Watson_Trend_v006",
    "MK_Nadaraya_Watson_HTML_Report_v006","MK_DEMA_MACD_Confirmation_v007",
    "MK_DEMA_MACD_HTML_Report_v007",
]
for name in IMPORT_MODULES:
    importlib.import_module(name)
    print("IMPORT PASS",name)

src=(ROOT/"app.py").read_text(encoding="utf-8")
required=["APP_VERSION = \"v0.07\"","DEMA-MACD Confirmation","SELL WATCH","RISK EXIT","calibrate_dema_macd"]
for token in required:
    assert token in src, token
print("APP v0.07 INTEGRATION TOKENS PASS")
print("SMOKE TEST PASS")
