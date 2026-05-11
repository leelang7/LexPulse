"""
자체 평가용 QA 테스트셋 생성

eval_full.py 가 요구하는 포맷:
    {"query": "...", "gold_chunk_ids": ["..."], "gold_answer": "..."}

전략:
    - 각 의결서당 2~3 개 질의 합성
        a) 메타데이터 기반 자연어 (위반유형 + 조치유형 조합)
        b) 사건 키워드 (제목·피심인) 기반
    - gold_chunk_ids: 해당 의결서의 모든 청크 ID (해당 doc 의 청크면 정답)
    - gold_answer: 해당 의결서의 처분/결론 섹션을 결합한 짧은 텍스트
                   (BERTScore/F1 ground truth)

실제 공모전 평가셋과 다를 수 있으나, 도메인 일관성을 가지므로 모델
fine-tune 효과·검색기 회귀 테스트에 충분.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


QUERY_TEMPLATES = [
    "{label} 에 대한 처분은 무엇인가요?",
    "{label} 사건의 위법성 판단 근거",
    "{label} 으로 받은 제재는?",
    "{피심인} 사건은 어떻게 처리되었나요?",
    "{피심인} 의 {label} 행위에 대한 결정",
    "{title} 결론 요약",
    "{label} 위반 시 어떤 처벌을 받나요?",
    "{label} 관련 의결서 사건",
]


def _gen_queries(meta: dict, doc_chunks: list[dict]) -> list[str]:
    title = meta.get("title", "")
    피심 = ", ".join(meta.get("피심인", []) or [])
    위반 = (meta.get("위반유형") or [""])[0]
    세부 = (meta.get("세부위반유형") or [""])[0]
    조치 = (meta.get("조치유형") or [""])[0]

    qs: set[str] = set()
    for tpl in QUERY_TEMPLATES:
        for label in [세부 or 위반, 위반]:
            if not label:
                continue
            try:
                q = tpl.format(label=label, title=title, 피심인=피심 or label)
                qs.add(q.strip())
            except (KeyError, IndexError):
                continue
        if len(qs) >= 4:
            break
    return list(qs)[:3]  # 의결서당 3개


def _gold_answer(doc_chunks: list[dict]) -> str:
    """의결서의 처분/결론 청크를 모아 gold answer 로 사용."""
    parts: list[str] = []
    for c in doc_chunks:
        if c.get("section") in ("처분", "결론", "주문"):
            parts.append(c.get("text", "").strip())
    if not parts:
        # fallback: 첫 두 청크
        parts = [c["text"] for c in doc_chunks[:2]]
    return "\n\n".join(parts)


def main(chunks_path: str, out_path: str, seed: int,
         gold_top_k: int = 3) -> None:
    """gold_top_k: 의결서당 정답으로 보는 chunk 수 (주문/처분/결론 우선)."""
    random.seed(seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by_doc[c["doc_id"]].append(c)

    # 정답 chunks 우선순위: 주문 > 처분 > 결론 > 이유 (앞쪽) > 기초사실
    section_priority = {"주문": 0, "처분": 1, "결론": 2, "이유": 3,
                         "위법성판단": 4, "기초사실": 5}

    n_q = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for doc_id, doc_chunks in sorted(by_doc.items()):
            meta = {
                "title":     doc_chunks[0].get("title", ""),
                "피심인":     doc_chunks[0].get("피심인", []),
                "위반유형":   doc_chunks[0].get("위반유형", []),
                "세부위반유형": doc_chunks[0].get("세부위반유형", []),
                "조치유형":   doc_chunks[0].get("조치유형", []),
            }
            # 섹션 우선순위 + chunk_idx 로 정렬 후 상위 K
            sorted_chunks = sorted(
                doc_chunks,
                key=lambda c: (section_priority.get(c.get("section", ""), 9),
                                c.get("chunk_idx", 0)))
            gold_ids = [c["chunk_id"] for c in sorted_chunks[:gold_top_k]]
            gold_ans = _gold_answer(doc_chunks)
            for q in _gen_queries(meta, doc_chunks):
                fout.write(json.dumps({
                    "query": q,
                    "gold_chunk_ids": gold_ids,
                    "gold_answer": gold_ans,
                    "doc_id": doc_id,
                }, ensure_ascii=False) + "\n")
                n_q += 1

    print(f"문서 {len(by_doc)} → 질의 {n_q} 개 (gold_top_k={gold_top_k})")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", help="chunks.jsonl")
    ap.add_argument("out", help="QA 테스트셋 jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold-top-k", type=int, default=3,
                    help="의결서당 정답으로 사용할 상위 청크 수")
    a = ap.parse_args()
    main(a.chunks, a.out, a.seed, a.gold_top_k)
