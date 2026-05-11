"""
Re-ranker GPU fine-tune (cross-encoder, 직접 학습 루프)

sentence_transformers 5.x 와 torch 2.6 호환성을 위해 transformers
직접 사용. BAAI/bge-reranker-base 의 sequence classifier 헤드를
(query, chunk) 페어에 대해 binary 학습 (positive=1, hard-neg=0).

Hard negatives: BM25 top-K 에서 다른 doc 청크.
"""

from __future__ import annotations

# 중요: transformers 를 torch 보다 먼저
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from rank_bm25 import BM25Okapi

# build_index 의 한글 토크나이저
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import korean_tokenize


DEFAULT_BASE = "BAAI/bge-reranker-base"


class TripletDataset(Dataset):
    """각 페어에 대해 (query, positive, label=1) + (query, hard_neg, label=0) 생성."""

    def __init__(self, pairs: list[dict], chunks: list[dict], n_neg: int,
                 seed: int = 42):
        random.seed(seed)
        # BM25 hard negative 풀
        bm25 = BM25Okapi([korean_tokenize(c["text"]) for c in chunks])

        self.examples: list[tuple[str, str, int]] = []
        for p in pairs:
            self.examples.append((p["query"], p["text"], 1))

            scores = bm25.get_scores(korean_tokenize(p["query"]))
            top_idx = np.argsort(scores)[::-1][:50]
            negs = [chunks[i] for i in top_idx if chunks[i]["doc_id"] != p["doc_id"]]
            random.shuffle(negs)
            for neg in negs[:n_neg]:
                self.examples.append((p["query"], neg["text"], 0))

        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch: list[tuple[str, str, int]], tokenizer, max_len: int):
    queries = [b[0] for b in batch]
    texts   = [b[1] for b in batch]
    labels  = torch.tensor([b[2] for b in batch], dtype=torch.float)
    enc = tokenizer(queries, texts, truncation=True, padding=True,
                    max_length=max_len, return_tensors="pt")
    return enc, labels


def main(chunks_path: str, pairs_path: str, out_dir: str, base: str,
         epochs: int, batch_size: int, lr: float, n_neg: int,
         max_pairs: int | None, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[load] chunks={chunks_path}", flush=True)
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[load] chunks={len(chunks)}", flush=True)

    pairs = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs); pairs = pairs[:max_pairs]
    print(f"[load] pairs={len(pairs)} (max_pairs={max_pairs})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  base={base}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(device)
    model.train()

    print(f"[ds] building triplets (n_neg={n_neg})...", flush=True)
    ds = TripletDataset(pairs, chunks, n_neg=n_neg, seed=seed)
    print(f"[ds] examples={len(ds)}", flush=True)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer, 256),
                        num_workers=0)
    n_steps = len(loader) * epochs
    warmup = max(1, int(n_steps * 0.1))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s/warmup) if s < warmup else
                       max(0.0, (n_steps-s)/(n_steps-warmup))
    )
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    print(f"[train] epochs={epochs} bs={batch_size} steps={n_steps}", flush=True)
    t0 = time.time(); step = 0
    for epoch in range(epochs):
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            opt.zero_grad()
            if device == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out_logits = model(**enc).logits.squeeze(-1)
                    loss = bce(out_logits, labels)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                out_logits = model(**enc).logits.squeeze(-1)
                loss = bce(out_logits, labels)
                loss.backward(); opt.step()
            sched.step(); step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  step {step}/{n_steps}  loss={loss.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  {el:.0f}s", flush=True)

    print(f"[save] → {out}", flush=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--max-pairs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.chunks, a.pairs, a.out, a.base, a.epochs, a.batch_size,
         a.lr, a.n_neg, a.max_pairs, a.seed)
