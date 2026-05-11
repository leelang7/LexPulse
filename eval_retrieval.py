"""Retrieval-only 평가 (LLM 제외 → 훨씬 빠름).

Usage:
    python eval_retrieval.py qa_sample_doc.jsonl [--url http://localhost:8000/search]
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
import argparse

def call_search(url: str, query: str, top_k: int = 5, timeout: float = 30.0,
                doc_diversity: bool | None = None) -> dict:
    payload = {"query": query, "top_k": top_k}
    if doc_diversity is not None:
        payload["doc_diversity"] = doc_diversity
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def main(qa_path, url):
    qa = [json.loads(l) for l in open(qa_path, encoding="utf-8") if l.strip()]
    print(f"[load] {len(qa)} QA from {qa_path}", flush=True)
    hits=mrr=0
    t_total=0
    for i, item in enumerate(qa, 1):
        q = item["query"]
        gold = set(item.get("gold_chunk_ids", []))
        t0 = time.perf_counter()
        try:
            r = call_search(url, q, top_k=5)
            t_total += time.perf_counter() - t0
            preds = [h["chunk_id"] for h in r.get("hits", [])][:5]
            if set(preds) & gold:
                hits += 1
            for rank, p in enumerate(preds, 1):
                if p in gold:
                    mrr += 1.0/rank
                    break
        except Exception as e:
            print(f"[{i}] FAIL: {e}", flush=True)
        if i % 25 == 0 or i == len(qa):
            print(f"[{i}/{len(qa)}] R@5={hits/i:.4f} MRR={mrr/i:.4f} avg_ms={t_total/i*1000:.0f}", flush=True)
    print()
    print(f"=== Retrieval-only ===")
    print(f"  R@5  : {hits/len(qa):.4f}")
    print(f"  MRR  : {mrr/len(qa):.4f}")
    print(f"  avg  : {t_total/len(qa)*1000:.0f}ms")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qa")
    ap.add_argument("--url", default="http://localhost:8000/search")
    a = ap.parse_args()
    main(a.qa, a.url)
