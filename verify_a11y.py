
from playwright.sync_api import sync_playwright


def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        page.goto("http://localhost:8501")
        # wait a bit longer for streamlit to load data
        page.wait_for_selector(".stApp", timeout=60000)
        page.wait_for_timeout(5000)

        # Take a screenshot to see the initial state
        page.screenshot(path="verify_step1.png")

        browser.close()


verify()
