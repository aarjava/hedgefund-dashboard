from playwright.sync_api import sync_playwright


def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1600})
    page.goto("http://localhost:8501")

    # Wait for sidebar to be visible
    page.wait_for_selector("[data-testid='stSidebar']", state="visible", timeout=60000)

    # Wait for the app to load fully
    try:
        page.wait_for_text("Research Question", timeout=60000)
    except:
        print("Timeout waiting for 'Research Question' text.")
        page.screenshot(path="verification/timeout.png")
        # Continue anyway to see if we can salvage

    # Locate "High Volatility Quantile" slider label
    print("Locating High Volatility Quantile slider...")

    # Taking a screenshot of the sidebar first
    page.screenshot(path="verification/sidebar_initial.png")

    # Find all tooltip icons
    tooltips = page.locator("[data-testid='stTooltipIcon']")
    # Wait for at least one tooltip
    tooltips.first.wait_for(timeout=10000)

    count = tooltips.count()
    print(f"Found {count} tooltips.")

    # We know the expected order from the code:
    # 1. SMA Window (slider)
    # 2. Momentum Lookback (slider)
    # 3. Out-of-Sample Mode (toggle)
    # 4. High Volatility Quantile (slider) - NEW
    # 5. Transaction Cost (number_input) - NEW
    # 6. Allow Short Selling (checkbox) - NEW

    if count >= 6:
        # Hover over "High Volatility Quantile" (Index 3, 0-based)
        print("Hovering over High Volatility Quantile tooltip...")
        tooltips.nth(3).hover()
        page.wait_for_timeout(2000) # Wait for tooltip to appear
        # Verify content if possible, but screenshot is key
        page.screenshot(path="verification/tooltip_volatility.png")

        # Hover over "Transaction Cost" (Index 4)
        print("Hovering over Transaction Cost tooltip...")
        tooltips.nth(4).hover()
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/tooltip_cost.png")

        # Hover over "Allow Short Selling" (Index 5)
        print("Hovering over Allow Short Selling tooltip...")
        tooltips.nth(5).hover()
        page.wait_for_timeout(2000)
        page.screenshot(path="verification/tooltip_short.png")
    else:
        print("Not enough tooltips found. Check if app loaded correctly.")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
