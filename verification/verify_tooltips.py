from playwright.sync_api import sync_playwright, expect
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 1600})
        page = context.new_page()

        try:
            print("Navigating to app...")
            page.goto("http://localhost:8501")

            # Wait for main content
            print("Waiting for content...")
            # Look for "Research Config" which is in the sidebar
            page.wait_for_selector("text=Research Config", timeout=60000)
            print("App loaded.")

            # Look for the Transaction Cost (bps) input
            print("Looking for Transaction Cost (bps)...")

            # The input label should be visible
            # We use a relaxed text match because sometimes labels have extra whitespace
            label = page.get_by_text("Transaction Cost (bps)", exact=False)
            expect(label).to_be_visible()

            # Locate the widget container
            # Using data-testid="stNumberInput"
            # We filter by text to find the specific widget
            widget = page.locator('div[data-testid="stNumberInput"]').filter(has_text="Transaction Cost (bps)")

            # Wait for widget to be attached
            widget.wait_for(state="attached", timeout=10000)

            if widget.count() == 0:
                print("Could not find number input widget.")
                page.screenshot(path="/home/jules/verification/debug_no_widget.png")
                return

            print("Found widget container.")

            # Find the tooltip icon within the widget
            # Note: Depending on Streamlit version, the tooltip icon might be a sibling or nested.
            # Usually it's inside a `div` with class `stTooltipIcon` or similar.
            # Using the `data-testid` is reliable if present.
            tooltip_icon = widget.locator('[data-testid="stTooltipIcon"]')

            if tooltip_icon.count() > 0:
                print("Found tooltip icon.")
                # Hover over it
                tooltip_icon.first.hover(force=True)

                # Wait for tooltip content to appear
                # The tooltip content usually appears in a portal at the end of the body.
                # Use get_by_text for the help text content.
                expected_text = "Simulated trading cost in basis points (e.g., 10 bps = 0.10%)."

                print(f"Waiting for tooltip text: '{expected_text}'")
                # Wait for text to appear anywhere on page (since tooltip is often absolute positioned)
                tooltip_content = page.get_by_text(expected_text)
                expect(tooltip_content).to_be_visible(timeout=5000)

                print("Tooltip verified successfully!")

                # Take screenshot
                page.screenshot(path="/home/jules/verification/verification.png")
                print("Screenshot saved to /home/jules/verification/verification.png")

            else:
                print("Tooltip icon not found in widget.")
                page.screenshot(path="/home/jules/verification/debug_no_icon.png")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="/home/jules/verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
