"""
조치유형 예측기 (단일라벨 분류)

입력 = 기초사실 + 위법성판단 본문
출력 = 조치유형 (단일 카테고리; 멀티값이면 첫 번째)
       정규화: '시정명령 및 과징금' → '과징금' 으로 우선순위 적용

처분 예측은 시연에서 임팩트가 큰 모델 — "AI가 이 사실관계로 어떤 처분이 내려질지 예측"
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


DEFAULT_BASE = "klue/roberta-base"

# 조치유형 정규화 — 강한 처분 우선
SANCTION_PRIORITY = ["과징금", "시정명령", "시정권고", "고발"]


def _normalize_sanction(values: list[str]) -> str | None:
    """여러 조치유형 중 우선순위가 높은 하나로 정규화"""
    if not values:
        return None
    # '시정명령 및 과징금' 같은 복합 표현 분해
    expanded: list[str] = []
    for v in values:
        for part in v.replace(" 및 ", "/").replace(", ", "/").split("/"):
            expanded.append(part.strip())
    for p in SANCTION_PRIORITY:
        if any(p in e for e in expanded):
            return p
    return expanded[0] if expanded else None


# ───────────────────────── 데이터 ─────────────────────────

def _load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _doc_signal_text(chunks_of_doc: list[dict]) -> str:
    """
    한 의결서의 기초사실 + 위법성판단 본문을 이어붙여 입력으로 사용.
    너무 길면 토크나이저가 자름.
    """
    texts: list[str] = []
    for c in chunks_of_doc:
        if c["section"] in ("기초사실", "위법성판단"):
            texts.append(c["text"])
    if not texts:  # fallback
        texts = [c["text"] for c in chunks_of_doc[:2]]
    return "\n".join(texts)


class DocSanctionDataset(Dataset):
    def __init__(self, doc_items: list[dict], tokenizer, label2id: dict[str, int],
                 max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []
        for d in doc_items:
            label = d["sanction"]
            if label not in label2id:
                continue
            self.items.append({
                "text":  d["text"],
                "label": label2id[label],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        enc = self.tokenizer(
            it["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels":         torch.tensor(it["label"], dtype=torch.long),
        }


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    # macro F1
    n_classes = logits.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2*p*r / (p + r + 1e-9)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": float(acc), "macro_f1": macro_f1}


# ───────────────────────── 메인 ─────────────────────────

def main(chunks_path: str, out_dir: str, base_model: str,
         epochs: int, batch_size: int, lr: float, dev_ratio: float,
         seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    chunks = _load_chunks(Path(chunks_path))

    # doc 단위 묶기
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    doc_items: list[dict] = []
    for doc_id, doc_chunks in by_doc.items():
        sanctions = []
        for c in doc_chunks:
            sanctions.extend(c.get("조치유형") or [])
        norm = _normalize_sanction(list(set(sanctions)))
        if norm is None:
            continue
        doc_items.append({
            "doc_id":   doc_id,
            "text":     _doc_signal_text(doc_chunks),
            "sanction": norm,
        })

    label_counter = Counter(d["sanction"] for d in doc_items)
    labels = sorted(label_counter.keys())
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"[load] docs={len(doc_items)}  labels={labels}  dist={dict(label_counter)}")

    # split
    random.shuffle(doc_items)
    n_dev = max(1, int(len(doc_items) * dev_ratio))
    dev = doc_items[:n_dev]
    train = doc_items[n_dev:]
    print(f"[split] train={len(train)}  dev={len(dev)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels)
    )

    train_ds = DocSanctionDataset(train, tokenizer, label2id)
    dev_ds   = DocSanctionDataset(dev,   tokenizer, label2id)

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch" if len(dev_ds) else "no",
        save_strategy="no",
        logging_steps=5,
        warmup_ratio=0.1,
        seed=seed,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if len(dev_ds) else None,
        compute_metrics=_compute_metrics if len(dev_ds) else None,
    )
    trainer.train()

    if len(dev_ds):
        m = trainer.evaluate()
        print(f"\n[eval] {m}")
        with open(out / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label2id": label2id},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dev-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.chunks, a.out, a.base, a.epochs, a.batch_size, a.lr, a.dev_ratio, a.seed)
