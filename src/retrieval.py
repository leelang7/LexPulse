"""
검색 모듈 - BM25(Sparse) + FAISS(Dense) + RRF 결합

검색 흐름:
    1. 질의 토큰화 → BM25 점수 산출 → 상위 K개
    2. 질의 임베딩 → FAISS 유사도 검색 → 상위 K개
    3. RRF로 두 결과의 순위를 결합 → 최종 Top-K
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# sentence_transformers 를 먼저 임포트 (torch/faiss segfault 회피, sbert 5.x 이슈)
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from build_index import korean_tokenize, DEFAULT_MODEL


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float            # 최종 점수 (RRF)
    bm25_rank: Optional[int]
    dense_rank: Optional[int]
    피심인: list
    위반유형: list
    조치유형: list


class HybridRetriever:
    """BM25 + Dense + RRF 검색기 (선택적 Re-ranker 지원)"""

    def __init__(self, index_dir: str | Path,
                 model_name: str | None = None,
                 rrf_k: int = 60,
                 reranker_path: str | Path | None = None):
        index_dir = Path(index_dir)
        self.rrf_k = rrf_k

        # BM25
        with open(index_dir / "bm25.pkl", "rb") as f:
            obj = pickle.load(f)
            self.bm25: BM25Okapi = obj["bm25"]

        # FAISS
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.bin"))

        # 메타 (청크 본문·메타데이터)
        self.chunks: list[dict] = []
        with open(index_dir / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        # 임베딩 모델 — 인덱스 빌드 때 사용한 모델을 자동 로드
        # build_index.py 가 model_name.txt 를 남기므로 그걸 우선 사용
        resolved_model = model_name
        if resolved_model is None:
            mn = index_dir / "model_name.txt"
            if mn.exists():
                resolved_model = mn.read_text(encoding="utf-8").strip()
            else:
                resolved_model = DEFAULT_MODEL

        print(f"[Retriever] 임베딩 모델 로드: {resolved_model}")
        if Path(resolved_model).exists():
            self.model = SentenceTransformer(str(Path(resolved_model).resolve()))
        else:
            self.model = SentenceTransformer(resolved_model)
        self.model_name = resolved_model

        # 선택적 re-ranker (transformers AutoModelForSequenceClassification 기반)
        self.reranker = None
        self._reranker_tokenizer = None
        if reranker_path and Path(reranker_path).exists():
            try:
                from transformers import (AutoTokenizer,
                                          AutoModelForSequenceClassification)
                import torch as _torch
                print(f"[Retriever] Re-ranker 로드: {reranker_path}")
                self._reranker_tokenizer = AutoTokenizer.from_pretrained(
                    str(reranker_path))
                self.reranker = AutoModelForSequenceClassification.from_pretrained(
                    str(reranker_path))
                self._reranker_device = "cuda" if _torch.cuda.is_available() else "cpu"
                self.reranker.to(self._reranker_device)
                self.reranker.eval()
            except Exception as e:
                print(f"[Retriever] re-ranker 로드 실패 (무시): {e}")
                self.reranker = None
                self._reranker_tokenizer = None

    # ─────────── 개별 검색기 ───────────

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = korean_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        emb = self.model.encode([query], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        scores, ids = self.faiss_index.search(emb, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    # ─────────── RRF ───────────

    @staticmethod
    def _rrf_merge(rank_lists: list[list[int]], k: int) -> dict[int, float]:
        """
        Reciprocal Rank Fusion.
        각 검색기에서 i가 r번째에 나왔다면 1/(k+r)을 가산.
        """
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for r, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
        return scores

    # ─────────── 공개 검색 함수 ───────────

    def search(self, query: str, top_k: int = 5,
               candidate_k: int = 30,
               filter_조치유형: Optional[list[str]] = None,
               filter_위반유형: Optional[list[str]] = None,
               use_reranker: bool = False) -> list[SearchResult]:
        """
        하이브리드 검색.
        candidate_k 만큼 각 검색기에서 후보를 뽑아 RRF로 결합.
        use_reranker=True 면 RRF 후 cross-encoder 로 재정렬 후 top_k 반환.
        """
        bm25_hits = self._bm25_search(query, candidate_k)
        dense_hits = self._dense_search(query, candidate_k)

        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_hits)}
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits)}

        merged = self._rrf_merge(
            [[i for i, _ in bm25_hits], [i for i, _ in dense_hits]],
            k=self.rrf_k,
        )

        # 1) 필터링된 RRF 후보 정렬 리스트 (재랭킹 전 사이즈는 candidate_k 까지)
        ordered = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        candidates: list[tuple[int, float]] = []
        for idx, score in ordered:
            c = self.chunks[idx]
            if filter_조치유형 and not any(t in c["조치유형"] for t in filter_조치유형):
                continue
            if filter_위반유형 and not any(t in c["위반유형"] for t in filter_위반유형):
                continue
            candidates.append((idx, score))
            if len(candidates) >= candidate_k:
                break

        # 2) 선택적 re-ranking (transformers 기반)
        if use_reranker and self.reranker is not None and candidates:
            import torch as _torch
            queries = [query] * len(candidates)
            texts = [self.chunks[idx]["text"] for idx, _ in candidates]
            enc = self._reranker_tokenizer(
                queries, texts, truncation=True, padding=True,
                max_length=256, return_tensors="pt")
            enc = {k: v.to(self._reranker_device) for k, v in enc.items()}
            with _torch.no_grad():
                logits = self.reranker(**enc).logits.squeeze(-1)
                ce_scores = _torch.sigmoid(logits).cpu().tolist()
            reranked = sorted(zip(candidates, ce_scores),
                              key=lambda x: x[1], reverse=True)
            final = [(idx, float(ce)) for (idx, _), ce in reranked]
        else:
            final = candidates

        # 3) 결과 패키징
        results: list[SearchResult] = []
        for idx, score in final[:top_k]:
            c = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                title=c["title"],
                section=c["section"],
                text=c["text"],
                score=score,
                bm25_rank=bm25_rank.get(idx),
                dense_rank=dense_rank.get(idx),
                피심인=c["피심인"],
                위반유형=c["위반유형"],
                조치유형=c["조치유형"],
            ))
        return results


# ─────────── CLI 테스트 ───────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("사용법: python retrieval.py <index_dir> '<질의>'")
        sys.exit(1)

    retriever = HybridRetriever(sys.argv[1])
    query = sys.argv[2]
    print(f"\n질의: {query}\n")

    hits = retriever.search(query, top_k=5)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.title} / {h.section}")
        print(f"    chunk_id={h.chunk_id}  RRF={h.score:.4f}  "
              f"BM25_rank={h.bm25_rank}  Dense_rank={h.dense_rank}")
        print(f"    조치={h.조치유형}  위반={h.위반유형}")
        preview = h.text.replace("\n", " ")[:200]
        print(f"    {preview}...\n")
