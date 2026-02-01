## 2026-02-01 - [Streamlit Tooltips for Financial Data]
**Learning:** Users (even researchers) struggle with raw inputs like "basis points" or "quantiles" without immediate context.
**Action:** Always add `help="..."` tooltips to `st.number_input` and `st.slider` when dealing with financial/statistical parameters, specifically providing a conversion example (e.g., "10 bps = 0.10%").
