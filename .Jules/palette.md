## 2026-02-09 - Streamlit Dependencies and Tooltips
**Learning:** `pandas.style.background_gradient` in Streamlit apps causes runtime crashes if `matplotlib` is not installed, even if not explicitly imported. This degrades UX severely.
**Action:** Always verify `matplotlib` is in `requirements.txt` when using pandas styling in Streamlit.

## 2026-02-09 - Verifying Streamlit Tooltips
**Learning:** Streamlit tooltips are rendered as `stTooltipIcon` and require hover to reveal content in `stTooltipContent`. They are robustly verifiable with Playwright by locating the icon near the label text.
**Action:** Use specific locator strategy: `page.locator("div").filter(has=text).filter(has=icon)` for verification.
