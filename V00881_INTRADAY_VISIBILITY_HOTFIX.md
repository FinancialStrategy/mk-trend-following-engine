# v0.08.8.1 — Intraday Visibility & Target Exposure Hotfix

This release does not change the 15-minute signal mathematics. It changes discoverability and chart semantics.

- Frequency control renamed to **Frequency / Bar Size**.
- Sidebar explicitly announces whether 15m mode is selected.
- A large main-page banner appears before run when 15m is selected.
- A second banner appears after a completed 15m run and reports the latest completed bar/session.
- **15m Intraday Tactical Lab** is promoted from the last tab to the **second tab**, immediately after Executive.
- The tab label displays `ACTIVE` for completed 15m runs.
- Every Plotly chart in a completed 15m run is routed through a unified renderer that adds `— 15-Minute Bars` to titles where the title does not already say 15-Minute.
- Intraday Target Exposure is moved off the confirmation secondary axis into a dedicated fifth panel.
- Target Exposure is shown as a thick 0/25/50/75/100% step line with fill and change markers.

No Yahoo source, Tactical accounting, NW signal logic, execution causality, or data policy is changed.
