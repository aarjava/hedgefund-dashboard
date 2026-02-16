# Palette's Journal

## 2026-02-13 - Missing Dependencies Impacting UX
**Learning:** The application crashed with `ImportError: background_gradient requires matplotlib` during verification. Financial dashboards often rely on `pandas.io.formats.style.Styler` for visual data density, which has implicit dependencies not always caught by standard linting.
**Action:** Always verify runtime visualization components (like pandas styling) even if the code is syntactically correct. Added `matplotlib` to `requirements.txt` to prevent runtime crashes.

## 2026-02-13 - Financial Parameter Accessibility
**Learning:** Complex financial inputs (Quantiles, Beta Windows) in Streamlit sidebars are intimidating without context. Adding `help` tooltips significantly reduces cognitive load by providing definitions and example values inline.
**Action:** Standardize adding `help="..."` to all `st.slider` and `st.number_input` calls involving domain-specific parameters.
