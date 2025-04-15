
# "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
# start msedge --remote-debugging-port=9222
# "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9222 --user-data-dir="C:\temp\edge-debug-profile"
# Use edge as the API provider, chrome runs sillytavern

# Run method:
# 1. clear history
# 2. set params
# 3. input value
# 4. click send
# 5. get results
# 6. return value

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:9222")
    default_context = browser.contexts[0]
    page = default_context.pages[0]