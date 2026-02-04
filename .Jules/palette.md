## 2025-02-18 - Dependency Crashes & Financial Context
**Learning:** Missing visualization dependencies (matplotlib) caused a complete runtime crash in Streamlit despite successful app startup, highlighting that lazy imports in libraries (pandas styler) can be UX landmines. Additionally, financial inputs like "bps" and "quantiles" proved opaque without concrete "10 bps = 0.10%" examples.
**Action:** proactively verify `requirements.txt` against utilized visualization features (like background gradients) and standardise on "example-first" tooltips for all domain-specific parameter inputs.
