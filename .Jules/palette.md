# Palette's Journal 🎨

## 2024-05-23 - [Jargon Demystification]
**Learning:** This financial dashboard relies heavily on industry-standard but non-obvious terms (e.g., "bps", "ADV", "Quantile"). Users may misinterpret scales (e.g., entering 10 for 10% instead of 0.10%).
**Action:** Mandatory `help` tooltips for all numerical inputs involving ratios, basis points, or statistical thresholds. Include concrete conversion examples (e.g., "10 bps = 0.10%").

## 2024-05-23 - [Streamlit DataFrame Styling]
**Learning:** The `st.dataframe` component using `pandas.style.background_gradient` triggers a hard crash if `matplotlib` is missing, with no fallback.
**Action:** Treat `matplotlib` as a core dependency for any Streamlit app using styled dataframes, even if no plots are drawn explicitly.
