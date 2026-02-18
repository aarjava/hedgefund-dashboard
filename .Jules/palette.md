## 2025-02-17 - Streamlit Financial Input Accessibility
**Learning:** Financial parameters like 'bps' and 'Vol Quantile' are jargon-heavy and Streamlit inputs default to labels without explanation, creating a barrier for non-expert users.
**Action:** Always add `help` tooltips with concrete examples (e.g., "10 bps = 0.10%") to `st.number_input` and `st.slider` for technical financial parameters.
