"""
간소화된 GPU 임베딩 fine-tune (sbert 5.x + torch 2.6 segfault 우회)

- sentence_transformers 의 복잡한 import 체인을 피하고
  기본 SentenceTransformer + 직접 학습 루프 사용
- Multiple Negatives Ranking Loss 를 직접 구현 (cosine + cross-entropy)

사용:
    python finetune_embedding_simple.py pairs.jsonl out_dir \
        [--base intfloat/multilingual-e5-small] \
        [--epochs 3] [--batch-size 32] [--lr 2e-5] [--max-pairs 50000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# 중요: torch 2.6 + sentence-transformers 5.x 환경에서
# `import torch` 후 `from sentence_transformers import SentenceTransformer` 가
# segfault 를 일으킨다. sbert 를 먼저 임포트하면 회피된다.
from sentence_transformers import SentenceTransformer

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        p = self.pairs[idx]
        return p["query"], p["text"]


def collate(batch: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    return [q for q, _ in batch], [t for _, t in batch]


def info_nce_loss(query_emb: torch.Tensor, pos_emb: torch.Tensor,
                   temperature: float = 20.0) -> torch.Tensor:
    """
    Multiple Negatives Ranking Loss = InfoNCE.
    배치 내 다른 샘플의 positive 가 negative 역할.
    """
    # 정규화 (코사인 유사도)
    q = F.normalize(query_emb, dim=-1)
    p = F.normalize(pos_emb, dim=-1)
    logits = (q @ p.T) * temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def main(pairs_path: str, out_dir: str, base_model: str,
         epochs: int, batch_size: int, lr: float, max_pairs: int | None,
         seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 페어 로드
    print(f"[load] {pairs_path}", flush=True)
    pairs: list[dict] = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    print(f"[load] pairs={len(pairs)}", flush=True)

    # 모델
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    print(f"[base] {base_model}", flush=True)
    model = SentenceTransformer(base_model, device=device)
    # 시퀀스 길이 제한 — 메모리 보호 + 속도 향상
    model.max_seq_length = 256
    print(f"[max_seq_length] {model.max_seq_length}", flush=True)
    model.train()

    transformer = model._first_module()  # transformer wrapper
    pooler = model[1] if len(model._modules) > 1 else None

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_steps = math.ceil(len(pairs) / batch_size) * epochs
    warmup = max(1, int(n_steps * 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, step / warmup) if step < warmup else
                          max(0.0, (n_steps - step) / (n_steps - warmup))
    )

    # DataLoader
    ds = PairDataset(pairs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
    print(f"[train] epochs={epochs} bs={batch_size} steps/epoch={len(loader)} "
          f"total={n_steps} warmup={warmup}", flush=True)

    # AMP
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    def encode_with_grad(texts: list[str]) -> torch.Tensor:
        """학습용 forward — model.encode 와 달리 gradient 추적."""
        features = model.preprocess(texts)
        # tensor 만 device로 이동 (modality 등 str 필드는 그대로)
        moved = {}
        for k, v in features.items():
            moved[k] = v.to(device) if hasattr(v, "to") else v
        out = model(moved)
        return out["sentence_embedding"]

    # 학습 루프
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        for queries, texts in loader:
            opt.zero_grad()

            if device == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    q_emb = encode_with_grad(queries)
                    p_emb = encode_with_grad(texts)
                    loss = info_nce_loss(q_emb, p_emb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                q_emb = encode_with_grad(queries)
                p_emb = encode_with_grad(texts)
                loss = info_nce_loss(q_emb, p_emb)
                loss.backward()
                opt.step()
            sched.step()
            step += 1

            if step % 50 == 0:
                el = time.time() - t0
                print(f"  step {step}/{n_steps}  loss={loss.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}  {el:.0f}s",
                      flush=True)
        print(f"[epoch {epoch+1}/{epochs}] done", flush=True)

    # 저장
    print(f"[save] → {out}", flush=True)
    model.save(str(out))
    print(f"[done] {time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", help="pairs.jsonl")
    ap.add_argument("out", help="output dir")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.pairs, a.out, a.base, a.epochs, a.batch_size, a.lr,
         a.max_pairs, a.seed)
