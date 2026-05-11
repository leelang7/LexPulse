"""
sentence-transformers 백엔드로 한국어 SOTA 임베딩 인덱스 빌드.

torch CUDA 가 작동하는 환경 전용. ONNX(fastembed) 비교용으로 BM25 인덱스는 손대지
않고 dense FAISS만 새로 만든다.

기본 모델: nlpai-lab/KURE-v1 (Korean SOTA, bge-m3 fine-tune, 1024d)
대안:      upskyy/bge-m3-korean (1024d), BAAI/bge-m3 (multilingual)

사용:
    python build_dense_st.py <chunks.jsonl> <out_dir> [--model <id>] [--batch <n>]
"""

from __future__ import annotations

# import 순서 — sbert 가 torch 보다 먼저
from sentence_transformers import SentenceTransformer

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_MODEL = "nlpai-lab/KURE-v1"


def main(chunks_path: str, out_dir: str, model_name: str, batch_size: int) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[load] chunks={len(chunks)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] {model_name} device={device}", flush=True)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512

    is_e5 = "e5" in model_name.lower() and "kure" not in model_name.lower()
    texts = [(f"passage: {c['text']}" if is_e5 else c["text"]) for c in chunks]

    t0 = time.time()
    print(f"[embed] {len(texts)} texts, batch={batch_size}", flush=True)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    M = np.asarray(embs, dtype=np.float32)
    print(f"[embed] done {time.time()-t0:.0f}s shape={M.shape}", flush=True)

    import faiss
    index = faiss.IndexFlatIP(M.shape[1])
    index.add(M)
    faiss.write_index(index, str(out / "faiss.bin"))
    (out / "embedding_model.txt").write_text(model_name, encoding="utf-8")
    (out / "embedding_backend.txt").write_text("sentence-transformers", encoding="utf-8")
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    main(a.chunks, a.out, a.model, a.batch)
