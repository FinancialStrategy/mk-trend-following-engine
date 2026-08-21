# Streamlit Cloud — Update to v0.07

Replace/upload the full contents of the v0.07 package over the existing GitHub repository.
Do not delete `.streamlit/` or `.github/`.

Critical new files:
- `MK_Benchmark_Relative_v007.py`
- `MK_Institutional_Tactical_v007.py`
- `institutional_tactical_validation_v007.py`

Replace:
- `app.py`
- `.github/workflows/validate.yml`
- `MK_Trend_Following_Entry_Gate_v005.py`

After GitHub commit, allow Streamlit Cloud to redeploy or use Manage app -> Reboot.

For the 15-minute workflow:
- Frequency: 15 Minutes
- Date span: <= 60 calendar days
- Nadaraya-Watson layer: ON
- Institutional Tactical Layer: ON
- Benchmark: Auto Mapped or explicit manual Yahoo benchmark
- Tactical Sensitivity: High Sensitivity for the fastest envelope response
