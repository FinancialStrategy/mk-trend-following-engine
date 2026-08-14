# v0.06.1 — Layout Hotfix for Range Selector / Title Overlap

## Root cause
Plotly range-selector buttons (1M / 3M / 6M / YTD / 1Y / 3Y / ALL) were placed too close
to the chart title area. On narrower chart widths or longer titles, the controls could
overlap the graph header.

## Correction
The hotfix applies a unified title/header spacing system across all charts:
- range selector moved upward to its own control lane (`y=1.22`)
- title anchored slightly lower and given padding
- top chart margin expanded
- same spacing logic applied in:
  - Streamlit app charts
  - Nadaraya-Watson standalone HTML report
  - Legacy trend-following standalone HTML report

## Data governance
No market-data logic was changed.
No strategy math was changed.
No synthetic observations.
No fallback data source.
No forward-fill / back-fill.
