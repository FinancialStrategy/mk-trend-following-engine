from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent
src = (ROOT / "app.py").read_text(encoding="utf-8")
ast.parse(src)

assert 'st.session_state.get("entry_lookback", cfg.max_buy_weeks)' not in src
assert 'STATE_SCHEMA_VERSION = 3' in src
assert 'getattr(cfg, "max_buy_weeks", 2000)' in src

result_guard = src.index('if result is None:')
cfg_read = src.index('cfg = st.session_state.get("config")')
assert result_guard < cfg_read

print("PASS — v0.05.1 cold-start regression")
print("Result guard precedes config dereference: YES")
print("Eager cfg.max_buy_weeks default evaluation: REMOVED")
print("Redeploy stale-state reset: ENABLED")
