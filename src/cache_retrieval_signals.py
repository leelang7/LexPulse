"""
단계 5-2 (1/2): Retrieval 신호 캐싱

각 평가 쿼리에 대해 후보 청크의 모든 검색 신호를 1회 계산해 pickle 로 저장.
이후 Optuna 가중치 최적화는 이 캐시만 읽어 CPU 에서 초고속으로 수행
(GPU/reranker 재호출 없음 → trial 당 밀리초).

캐싱 신호 (후보 = BM25∪e5∪mpnet top-K union):
    - bm25_rank, e5_rank, mpnet_rank  (없으면 None)
    - reranker_score                  (Jina cross-encoder)
    - doc_id, 위반유형, gold 여부

GPU 안전: dense 는 fastembed ONNX(가벼움), reranker 는 CPU. avatar 공존.
"""
from __future__ import annotations

import os
import importlib.util
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
for _sub in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc",
             "nvidia.cudnn", "nvidia.cufft", "nvidia.nvjitlink"):
    _s = importlib.util.find_spec(_sub)
    if _s and _s.submodule_search_locations:
        _b = Path(list(_s.submodule_search_locations)[0]) / "bin"
        if _b.exists():
            try:
                os.add_dll_directory(str(_b))
            except Exception:
                pass
            os.environ["PATH"] = str(_b) + os.pathsep + os.environ.get("PATH", "")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import pickle
import re
import time
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi


def log(m):
    print(m, flush=True)


def main():
    INDEX = "index_official"
    QA = "qa_sample_doc.jsonl"
    DENSE1 = "index_dense_e5large"   # fastembed
    DENSE2 = "index_dense_mpnet"     # fastembed
    RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
    TOPK = 100        # 각 신호별 후보 top-K
    RERANK_POOL = 60  # reranker 점수 매길 후보 상한 (쿼리당)
    OUT = "retrieval_cache.pkl"

    log("[1] 청크/BM25 로드")
    chunks = [json.loads(l) for l in open(f"{INDEX}/meta.jsonl", encoding="utf-8") if l.strip()]
    with open(f"{INDEX}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)["bm25"]

    sys.path.insert(0, "src")
    from korean_tok import korean_tokenize

    log("[2] dense 임베더 로드 (fastembed ONNX GPU)")
    import faiss
    from fastembed import TextEmbedding

    idx1 = faiss.read_index(f"{DENSE1}/faiss.bin")
    m1 = (Path(DENSE1) / "embedding_model.txt").read_text(encoding="utf-8").strip()
    idx2 = faiss.read_index(f"{DENSE2}/faiss.bin")
    m2 = (Path(DENSE2) / "embedding_model.txt").read_text(encoding="utf-8").strip()

    def mk_embedder(name):
        try:
            return TextEmbedding(name, providers=["CUDAExecutionProvider"])
        except Exception:
            return TextEmbedding(name, providers=["CPUExecutionProvider"])

    emb1 = mk_embedder(m1)
    emb2 = mk_embedder(m2)
    is_e5_1 = "e5" in m1.lower()
    is_e5_2 = "e5" in m2.lower()

    def dense_search(emb, idx, query, is_e5):
        q = f"query: {query}" if is_e5 else query
        v = next(iter(emb.embed([q])))
        v = np.asarray(v, dtype=np.float32).reshape(1, -1)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        sc, ids = idx.search(v, TOPK)
        return [int(i) for i in ids[0] if i >= 0]

    log("[3] reranker 로드 (CPU)")
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    reranker = TextCrossEncoder(RERANKER, providers=["CPUExecutionProvider"])

    qa = [json.loads(l) for l in open(QA, encoding="utf-8") if l.strip()]
    log(f"[4] {len(qa)} 쿼리 신호 캐싱 시작")

    cache = []
    t0 = time.time()
    for qi, item in enumerate(qa):
        query = item["query"]
        gold = set(item["gold_chunk_ids"])

        # 1) 신호별 rank
        bm25_scores = bm25.get_scores(korean_tokenize(query))
        bm25_order = list(np.argsort(bm25_scores)[::-1][:TOPK])
        bm25_rank = {int(idx): r for r, idx in enumerate(bm25_order) if bm25_scores[idx] > 0}
        e5_order = dense_search(emb1, idx1, query, is_e5_1)
        e5_rank = {idx: r for r, idx in enumerate(e5_order)}
        mp_order = dense_search(emb2, idx2, query, is_e5_2)
        mp_rank = {idx: r for r, idx in enumerate(mp_order)}

        # 2) 후보 union
        cand = set(bm25_rank) | set(e5_rank) | set(mp_rank)

        # 3) reranker: 임시 RRF 상위 RERANK_POOL 개만 점수화 (비용 절감)
        tmp = []
        for idx in cand:
            s = (1.0/(60+bm25_rank.get(idx, 999)+1) +
                 1.0/(60+e5_rank.get(idx, 999)+1) +
                 1.0/(60+mp_rank.get(idx, 999)+1))
            tmp.append((idx, s))
        tmp.sort(key=lambda x: x[1], reverse=True)
        rerank_cand = [idx for idx, _ in tmp[:RERANK_POOL]]
        texts = [chunks[idx]["text"][:512] for idx in rerank_cand]
        rr_scores = {}
        if texts:
            scores = list(reranker.rerank(query, texts))
            rr_scores = {idx: float(s) for idx, s in zip(rerank_cand, scores)}

        # 4) 후보별 레코드
        records = []
        for idx in cand:
            c = chunks[idx]
            records.append({
                "idx": idx,
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "위반유형": c.get("위반유형", []),
                "bm25_rank": bm25_rank.get(idx),
                "e5_rank": e5_rank.get(idx),
                "mp_rank": mp_rank.get(idx),
                "rr_score": rr_scores.get(idx),
                "is_gold": c["chunk_id"] in gold,
            })
        cache.append({
            "query": query,
            "gold_chunk_ids": list(gold),
            "gold_doc": item["gold_chunk_ids"][0].rsplit("-CH-", 1)[0] if gold else None,
            "candidates": records,
        })

        if (qi + 1) % 25 == 0:
            el = time.time() - t0
            log(f"  {qi+1}/{len(qa)} ({el:.0f}s, eta {el/(qi+1)*(len(qa)-qi-1):.0f}s)")

    with open(OUT, "wb") as f:
        pickle.dump(cache, f)
    log(f"[done] {OUT} 저장 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
