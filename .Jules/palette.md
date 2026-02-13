
## 2026-02-13 - [Tooltip Clarity and Dependency Crash]
**Learning:** Streamlit's `st.tabs` executes logic immediately, causing hidden crashes if optional dependencies (e.g., `matplotlib` for pandas styling) are missing, even if the tab is inactive. Adding tooltips to inputs like 'Transaction Cost (bps)' significantly aids user comprehension of domain-specific units.
**Action:** Ensure all dependencies for tab contents are installed and use `help` parameters for technical inputs.

## 2026-02-13 - [CI Formatting Requirements]
**Learning:** CI checks in this repo enforce strict formatting (Black) and linting (Ruff) on all files, not just modified ones. Aggressive auto-formatting is required to pass checks, even if it bloats the PR.
**Action:** Run `ruff check --fix` and `black` on the entire codebase before submitting.
