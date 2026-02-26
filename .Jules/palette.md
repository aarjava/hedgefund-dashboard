# Palette's Journal

## 2024-05-22 - [Hidden Dependency in Streamlit]
**Learning:** `pandas.io.formats.style` methods (like `background_gradient`) fail silently or crash the app if `matplotlib` is not installed, even if `pandas` is.
**Action:** Always verify `matplotlib` is in `requirements.txt` if using pandas styling features in Streamlit.

## 2024-05-22 - [CI Formatting vs Diff Size]
**Learning:** CI pipelines that enforce global formatting (e.g., `black .`) override soft constraints on PR size. Partial formatting leads to CI failure.
**Action:** Prioritize passing CI over diff size constraints when working in repositories with strict global formatting rules.
