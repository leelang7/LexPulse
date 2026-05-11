"""스모크 테스트 — /predict 다중 쿼리"""
import json, time, urllib.request

queries = [
    "허위 광고로 받은 처분은?",
    "다단계판매업자 미등록 사례",
    "하도급법 위반 과징금",
    "가맹사업법 구입강제 처벌",
    "대형마트 부당 반품 사건",
]

for q in queries:
    body = json.dumps({"query": q}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=body, method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=35) as r:
            d = json.loads(r.read())
        el = (time.perf_counter() - t0) * 1000
        ans = (d.get("answer") or "")[:200].replace("\n", " ")
        print(f"[{el:.0f}ms] q={q!r}")
        print(f"  chunks={len(d.get('chunk_ids',[]))}")
        print(f"  ans={ans}")
        print()
    except Exception as e:
        print(f"[FAIL] q={q!r}: {e}")
