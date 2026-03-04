## 2024-05-22 - Verifying Streamlit Tooltips
**Learning:** Streamlit tooltips persist until the mouse moves away, causing automated tests to fail if they try to hover a new tooltip without clearing the old one.
**Action:** In Playwright tests, explicitly move the mouse to a neutral position (e.g., `page.mouse.move(0, 0)`) before verifying the next tooltip.
