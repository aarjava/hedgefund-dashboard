## 2024-05-22 - [The "Jump" in Toggle UIs]
**Learning:** In Streamlit, toggling between different input types (like `selectbox` vs `text_input` based on a radio button) causes a layout "jump" that disorients users.
**Action:** Use `help` tooltips heavily in complex sidebars to ground the user, and consider consistent input containers or `st.empty` placeholders if layout stability is critical.

## 2024-05-22 - [Color-Coded Text Metrics]
**Learning:** `st.metric`'s `delta` parameter is the most accessible and standard way to show "Good/Bad" states (Green/Red), even for non-numeric states like "BULLISH/BEARISH".
**Action:** Prefer `delta` with string values (e.g., `delta="Above Trend"`) over custom CSS classes which are brittle and harder to maintain.
