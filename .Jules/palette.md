## 2025-02-20 - Streamlit Tooltip Verification

**Learning:** Streamlit tooltips are rendered as `stTooltipIcon` divs nested within or adjacent to widget labels. Verification requires locating the specific label container first (e.g., `page.locator('label').filter(has_text=...)`), then finding the scoped `[data-testid='stTooltipIcon']` to hover, rather than iterating through all global icons.

**Action:** When verifying tooltips in Streamlit, scope the search to the label's container to ensure the correct tooltip is being tested.

## 2025-02-20 - Streamlit Tabs and Dependencies

**Learning:** Streamlit's `st.tabs` executes code within the blocks immediately on load; missing dependencies (e.g., `matplotlib` for styling) will crash the app even if the tab is not active.

**Action:** Ensure all dependencies used in any tab are installed and available at app startup, even if the tab is not the default view.
