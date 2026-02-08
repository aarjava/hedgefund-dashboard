## 2024-05-21 - Contextual Help for Domain Jargon
**Learning:** Financial and technical jargon (e.g., "bps", "quantiles") creates friction for non-expert users, requiring them to look up terms external to the app.
**Action:** Always include inline `help` tooltips for parameters with domain-specific units or concepts to provide immediate definition and context.

## 2024-05-21 - Styling Dependencies
**Learning:** Streamlit's `dataframe.style` methods (like `background_gradient`) have a silent runtime dependency on `matplotlib`. Missing this crashes the app without failing build-time checks.
**Action:** Ensure `matplotlib` is included in `requirements.txt` whenever pandas styling is used.
