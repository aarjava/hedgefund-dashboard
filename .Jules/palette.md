## 2025-02-18 - [Testing Streamlit Tooltips with Playwright]
**Learning:** Streamlit tooltips are rendered as icons (`[data-testid='stTooltipIcon']`) often nested within or adjacent to the widget label. To verify them with Playwright, locate the label by text, find the nested icon, hover, and wait for the tooltip text.
**Action:** Use `label_locator.locator("[data-testid='stTooltipIcon']").hover()` pattern for future tooltip verifications.
