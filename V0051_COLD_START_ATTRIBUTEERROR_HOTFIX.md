# v0.05.1 — Cold-Start AttributeError Hotfix

Reported failing line:

`entry_lookback_used = int(st.session_state.get("entry_lookback", cfg.max_buy_weeks))`

Two root causes were corrected:

1. Python evaluates a function argument before the call. Therefore the default
   expression `cfg.max_buy_weeks` was evaluated even if `entry_lookback`
   already existed.
2. The code read `cfg` before checking whether a completed analysis existed.
   On a clean Streamlit session, `config=None` is expected.

Correction:
- Check `result` first and stop on the universe screen when no run exists.
- Read config/decision/trade objects only after a completed result.
- Use safe `getattr(cfg, "max_buy_weeks", 2000)` only as a compatibility fallback.
- Add a session-state schema reset to remove stale analysis objects after redeploy.
- Add a GitHub Actions cold-start regression and Streamlit health-check test.

No change to market-data governance or strategy mathematics.
