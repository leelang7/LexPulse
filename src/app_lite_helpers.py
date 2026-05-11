"""eval_full.py 의 offline 모드용 헬퍼 — app_lite 의 LiteRetriever 와 동일."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_STOPWORDS = {
    "그리고", "그러나", "그런데", "또한", "또는", "있다", "한다", "하는", "하여",
    "위해", "통해", "대해", "관한", "있는", "이를", "그러므로", "따라서", "이에",
    "있어", "보다", "보면", "관련", "기재", "다음과", "같은", "같이", "이상", "이하",
    "다음", "위와", "있어서", "로서", "으로",
}
def _tok(t: str) -> list[str]:
    return [x for x in _TOKEN_RE.findall(t) if len(x) >= 2 and x not in _STOPWORDS]


@dataclass
class _Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    bm25_rank: int
    피심인: list
    위반유형: list
    조치유형: list


class OfflineRetriever:
    def __init__(self, index_dir: Path):
        from rank_bm25 import BM25Okapi
        self.chunks: list[dict] = []
        with open(index_dir / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        bm_path = index_dir / "bm25.pkl"
        if bm_path.exists():
            with open(bm_path, "rb") as f:
                obj = pickle.load(f)
            self.bm25 = obj["bm25"]
        else:
            self.bm25 = BM25Okapi([_tok(c["text"]) for c in self.chunks])

    def search(self, query: str, top_k: int = 5) -> list[_Hit]:
        import numpy as np
        scores = self.bm25.get_scores(_tok(query))
        order = np.argsort(scores)[::-1]
        out: list[_Hit] = []
        for rank, idx in enumerate(order):
            if scores[idx] <= 0:
                break
            c = self.chunks[idx]
            out.append(_Hit(
                chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                title=c.get("title", ""), section=c.get("section", ""),
                text=c["text"], score=float(scores[idx]), bm25_rank=rank,
                피심인=c.get("피심인", []),
                위반유형=c.get("위반유형", []),
                조치유형=c.get("조치유형", []),
            ))
            if len(out) >= top_k:
                break
        return out
