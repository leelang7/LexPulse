"""
fastembed (ONNX, torch 비의존) 으로 Dense FAISS 인덱스 빌드.

torch CUDA state 가 손상된 환경에서도 임베딩 가능 — ONNX runtime 별도.
sentence-transformers 의 fine-tuned 모델은 못 쓰지만, baseline 다국어 모델로
BM25-only 보다 더 나은 retrieval.

사용:
    python build_index_fastembed.py <chunks.jsonl> <out_dir> [--model <id>] [--gpu]
"""

from __future__ import annotations

# 중요: NVIDIA DLL 디렉토리 PATH에 추가 (ONNX runtime CUDA provider 가 찾도록)
import os
import importlib.util
from pathlib import Path as _Path
def _add_nvidia_paths():
    if os.name != "nt":
        return
    added = []
    for sub in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc",
                "nvidia.cudnn", "nvidia.cufft", "nvidia.nvjitlink"):
        s = importlib.util.find_spec(sub)
        if s and s.submodule_search_locations:
            binp = _Path(list(s.submodule_search_locations)[0]) / "bin"
            if binp.exists():
                added.append(str(binp))
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
_add_nvidia_paths()

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "intfloat/multilingual-e5-large"  # 1024d, 한국어 양호


def main(chunks_path: str, out_dir: str, model_name: str,
         use_gpu: bool, batch_size: int) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[load] chunks={len(chunks)}", flush=True)

    print(f"[init] fastembed model={model_name} gpu={use_gpu}", flush=True)
    from fastembed import TextEmbedding
    providers = ["CUDAExecutionProvider"] if use_gpu else None
    embedder = TextEmbedding(model_name=model_name, providers=providers)

    # e5 family 는 query/passage prefix 권장. corpus는 "passage:" prefix.
    is_e5 = "e5" in model_name.lower()

    # 메타데이터 enrichment — 위반유형/세부위반/조치/피심인을 텍스트 앞에 prepend
    # 검색 시 metadata 키워드로도 매칭됨. enrich=False 로 끄고 싶으면 환경변수 ENRICH=0
    enrich = os.environ.get("ENRICH", "1") == "1"
    def _enriched_text(c: dict) -> str:
        if not enrich:
            return c["text"]
        parts = []
        if c.get("title"):
            parts.append(f"[{c['title']}]")
        violation = c.get("위반유형") or []
        sub_violation = c.get("세부위반유형") or []
        sanction = c.get("조치유형") or []
        피심 = c.get("피심인") or []
        meta = []
        if violation: meta.append(f"위반유형: {', '.join(violation)}")
        if sub_violation: meta.append(f"세부위반: {', '.join(sub_violation)}")
        if sanction: meta.append(f"조치: {', '.join(sanction)}")
        if 피심: meta.append(f"피심인: {', '.join(피심[:3])}")
        if meta:
            parts.append(" / ".join(meta))
        parts.append(c["text"])
        return "\n".join(parts)

    texts = [(f"passage: {_enriched_text(c)}" if is_e5 else _enriched_text(c)) for c in chunks]

    t0 = time.time()
    print(f"[embed] {len(texts)} texts...", flush=True)
    embs = []
    for i, e in enumerate(embedder.embed(texts, batch_size=batch_size)):
        embs.append(e)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(texts)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[embed] done {time.time()-t0:.0f}s", flush=True)

    M = np.vstack([np.asarray(e, dtype=np.float32) for e in embs])
    # L2 normalize → cosine = inner product
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M /= norms
    print(f"[embed] matrix {M.shape}, dim={M.shape[1]}", flush=True)

    import faiss
    index = faiss.IndexFlatIP(M.shape[1])
    index.add(M)
    print(f"[faiss] vectors={index.ntotal} dim={M.shape[1]}", flush=True)

    faiss.write_index(index, str(out / "faiss.bin"))
    # 모델 정보 저장 — retrieval.py 가 쿼리 임베딩 시 동일 모델 사용
    (out / "embedding_model.txt").write_text(model_name, encoding="utf-8")
    (out / "embedding_backend.txt").write_text("fastembed", encoding="utf-8")
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu", action="store_true",
                    help="ONNX CUDA provider 사용 (torch 와 무관)")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    main(a.chunks, a.out, a.model, a.gpu, a.batch_size)
