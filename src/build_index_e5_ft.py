"""
파인튜닝된 e5 모델 (transformers AutoModel) 로 Dense FAISS 인덱스 빌드.

finetune_e5_transformers.py 로 저장한 로컬 모델 디렉토리를 사용.
e5 표준: "passage: " prefix + mean pooling.

사용:
    python build_index_e5_ft.py chunks_official.jsonl index_dense_e5ft \
        --model models/e5large_ft
"""
from __future__ import annotations

# nvidia DLL 선등록
import os
import importlib.util
from pathlib import Path as _P
for _sub in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc",
             "nvidia.cudnn", "nvidia.cufft", "nvidia.nvjitlink"):
    _s = importlib.util.find_spec(_sub)
    if _s and _s.submodule_search_locations:
        _binp = _P(list(_s.submodule_search_locations)[0]) / "bin"
        if _binp.exists():
            try:
                os.add_dll_directory(str(_binp))
            except Exception:
                pass
            os.environ["PATH"] = str(_binp) + os.pathsep + os.environ.get("PATH", "")

import argparse
import json
import time

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] model={a.model} device={device}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).to(device)
    model.eval()

    chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8") if l.strip()]
    texts = [f"passage: {c['text'][:2000]}" for c in chunks]
    print(f"[load] {len(texts)} chunks")

    t0 = time.time()
    vecs = []
    for i in range(0, len(texts), a.batch):
        enc = tok(texts[i:i+a.batch], padding=True, truncation=True,
                  max_length=a.max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        vecs.append(emb.cpu().float().numpy())
        done = i + a.batch
        if done % 8000 < a.batch:
            print(f"  {min(done,len(texts))}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)

    M = np.vstack(vecs).astype(np.float32)
    print(f"[embed] {M.shape} in {time.time()-t0:.0f}s")

    out = _P(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = faiss.IndexFlatIP(M.shape[1])
    idx.add(M)
    faiss.write_index(idx, str(out / "faiss.bin"))
    # 쿼리 인코딩 시 동일 모델/방식 사용하도록 메타 기록
    (out / "embedding_model.txt").write_text(str(a.model), encoding="utf-8")
    (out / "embedding_backend.txt").write_text("transformers-mean-e5", encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
