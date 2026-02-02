## 2024-05-21 - [Initial Setup]
**Learning:** Initializing Palette Journal.
**Action:** Record critical UX/a11y learnings here.

## 2024-05-21 - [Financial Parameter Context]
**Learning:** Financial inputs (bps, quantiles) require explicit translation for non-experts. Streamlit's 'help' param is the standard, accessible way to provide this without clutter.
**Action:** Always check st.number_input/slider for complexity and add 'help' if unit is ambiguous.
## 2024-05-21 - [CI Compliance]
**Learning:** CI strictly enforces Black formatting and Ruff linting (including C408 dict literals).
**Action:** Always run 'black .' and 'ruff check .' before submitting.
