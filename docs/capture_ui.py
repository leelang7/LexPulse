"""웹 UI 캡처 (Playwright)

생성:
    docs/figures/08_ui_main.png       — 메인 페이지 hero+검색
    docs/figures/09_ui_result.png     — 답변+출처 결과 화면
"""
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8000/"

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        device_scale_factor=2,                       # retina
    )
    page = context.new_page()

    # 1) 메인 페이지
    print("[1] capturing main page...")
    page.goto(URL, wait_until="networkidle")
    time.sleep(0.6)                                  # 애니메이션 끝나길 대기
    page.screenshot(path=str(OUT / "08_ui_main.png"),
                    full_page=False, animations="disabled")
    print(f"  saved 08_ui_main.png")

    # 2) 결과 페이지 (질의 입력 후 검색만 결과)
    print("[2] capturing search result...")
    page.fill("#query", "가격 담합 사건의 처분 내용은 무엇인가요?")
    page.click("#searchBtn")
    # 결과 카드가 보일 때까지 대기 (최대 30s)
    page.wait_for_selector("#sources:not(.hidden)", timeout=30000)
    time.sleep(0.5)
    # full page (결과 까지 다 캡처)
    page.screenshot(path=str(OUT / "09_ui_result.png"),
                    full_page=True, animations="disabled")
    print(f"  saved 09_ui_result.png")

    browser.close()

print("[done] UI captures complete.")
