import time
from playwright.sync_api import sync_playwright

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1600})

        # Navigate to the Streamlit app
        page.goto("http://localhost:8501")

        # Wait for the main app container to load
        page.wait_for_selector(".stApp", timeout=60000)
        time.sleep(5) # Allow sidebar and inputs to fully hydrate

        # We need to verify 5 tooltips: bt_cost, factor_window, vol_window, adv_pct, vol_q_high
        # Note: some are in Single-Asset mode, some in Portfolio mode

        # 1. vol_q_high (available in both modes, Single-Asset is default)
        vol_q_high_label = page.locator("label").filter(has_text="High Volatility Quantile")
        vol_q_high_icon = vol_q_high_label.locator("[data-testid='stTooltipIcon']")
        vol_q_high_icon.hover()
        time.sleep(1)
        page.screenshot(path="vol_q_high_tooltip.png")
        page.mouse.move(0, 0)
        time.sleep(1)

        # 2. bt_cost (available in Single-Asset mode)
        bt_cost_label = page.locator("label").filter(has_text="Transaction Cost (bps)")
        bt_cost_icon = bt_cost_label.locator("[data-testid='stTooltipIcon']")
        bt_cost_icon.hover()
        time.sleep(1)
        page.screenshot(path="bt_cost_tooltip.png")
        page.mouse.move(0, 0)
        time.sleep(1)

        # Switch to Portfolio mode to verify the remaining 3
        page.get_by_text("Portfolio").click()
        time.sleep(5)

        # 3. factor_window
        factor_window_label = page.locator("label").filter(has_text="Factor Beta Window (days)")
        factor_window_icon = factor_window_label.locator("[data-testid='stTooltipIcon']")
        factor_window_icon.hover()
        time.sleep(1)
        page.screenshot(path="factor_window_tooltip.png")
        page.mouse.move(0, 0)
        time.sleep(1)

        # 4. vol_window
        vol_window_label = page.locator("label").filter(has_text="Regime Vol Window (days)")
        vol_window_icon = vol_window_label.locator("[data-testid='stTooltipIcon']")
        vol_window_icon.hover()
        time.sleep(1)
        page.screenshot(path="vol_window_tooltip.png")
        page.mouse.move(0, 0)
        time.sleep(1)

        # 5. adv_pct
        adv_pct_label = page.locator("label").filter(has_text="ADV Participation %")
        adv_pct_icon = adv_pct_label.locator("[data-testid='stTooltipIcon']")
        adv_pct_icon.hover()
        time.sleep(1)
        page.screenshot(path="adv_pct_tooltip.png")

        browser.close()

if __name__ == "__main__":
    verify()
