## 2025-02-18 - Verifying Streamlit Tooltips
**Learning:** Streamlit tooltips are rendered as `stTooltipIcon` divs that create a separate `stTooltipContent` overlay on hover. Verification requires locating the icon, hovering (often with force or careful mouse movement), and waiting for the content to appear in the overlay.
**Action:** Use `page.locator("[data-testid='stTooltipIcon']").hover()` and verify text in `page.locator("[data-testid='stTooltipContent']")`. Ensure viewport is large enough.
