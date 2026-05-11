"""
위반유형 멀티라벨 분류기 학습

입력:
    chunks.jsonl  (preprocess.py 출력)

출력:
    out_dir/  (transformers 모델 + tokenizer + label_map.json)

모델:
    klue/roberta-base  (한국어 사전학습, Korean Legal/General 도메인에 무난)
    또는 klue/bert-base

작업 정의:
    입력 = 청크 본문 (특히 기초사실/위법성판단 섹션이 학습 신호 강함)
    출력 = 위반유형 멀티라벨 (예: ['공정거래법 위반', '하도급법 위반'])

평가:
    문서 단위 split, micro/macro F1, per-label P/R
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

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


# ───────────────────────── 데이터 로드 ─────────────────────────

def _load_chunks(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _build_label_set(chunks: list[dict]) -> list[str]:
    counter: Counter = Counter()
    for c in chunks:
        for v in c.get("위반유형") or []:
            counter[v] += 1
    return sorted(counter.keys())


def _multi_hot(labels: list[str], all_labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(all_labels)}
    vec = np.zeros(len(all_labels), dtype=np.float32)
    for l in labels:
        if l in idx:
            vec[idx[l]] = 1.0
    return vec


# ───────────────────────── 데이터셋 ─────────────────────────

class ChunkDataset(Dataset):
    def __init__(self, chunks: list[dict], tokenizer, all_labels: list[str],
                 max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 학습 신호가 강한 섹션만 사용 (주문/기초사실/위법성판단/처분)
        relevant = {"주문", "기초사실", "위법성판단", "처분"}
        self.items = []
        for c in chunks:
            if c.get("section") in relevant and (c.get("위반유형") or []):
                self.items.append({
                    "text": c["text"],
                    "labels": _multi_hot(c["위반유형"], all_labels),
                    "doc_id": c["doc_id"],
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
            "labels":         torch.tensor(it["labels"], dtype=torch.float),
        }


# ───────────────────────── 메트릭 ─────────────────────────

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = (preds & labels).sum()
    fp = (preds & ~labels).sum()
    fn = (~preds & labels).sum()
    micro_p = tp / (tp + fp + 1e-9)
    micro_r = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    # macro
    n_labels = labels.shape[1]
    macros = []
    for j in range(n_labels):
        tp_j = ((preds[:, j] == 1) & (labels[:, j] == 1)).sum()
        fp_j = ((preds[:, j] == 1) & (labels[:, j] == 0)).sum()
        fn_j = ((preds[:, j] == 0) & (labels[:, j] == 1)).sum()
        p_j = tp_j / (tp_j + fp_j + 1e-9)
        r_j = tp_j / (tp_j + fn_j + 1e-9)
        f1_j = 2 * p_j * r_j / (p_j + r_j + 1e-9)
        macros.append(f1_j)
    macro_f1 = float(np.mean(macros))

    # exact match (모든 라벨 정확히 일치)
    exact = (preds == labels).all(axis=1).mean()

    return {
        "micro_p":  float(micro_p),
        "micro_r":  float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
        "exact":    float(exact),
    }


# ───────────────────────── 메인 ─────────────────────────

def main(chunks_path: str, out_dir: str, base_model: str,
         epochs: int, batch_size: int, lr: float, dev_ratio: float,
         seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks = _load_chunks(Path(chunks_path))
    all_labels = _build_label_set(chunks)
    print(f"[load] chunks={len(chunks)}  labels={len(all_labels)}")
    print(f"       {all_labels}")

    # 문서 단위 split
    docs = sorted({c["doc_id"] for c in chunks})
    random.shuffle(docs)
    n_dev = max(1, int(len(docs) * dev_ratio))
    dev_docs = set(docs[:n_dev])
    train_chunks = [c for c in chunks if c["doc_id"] not in dev_docs]
    dev_chunks = [c for c in chunks if c["doc_id"] in dev_docs]
    print(f"[split] train_docs={len(docs)-n_dev}  dev_docs={n_dev}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(all_labels),
        problem_type="multi_label_classification",
    )

    train_ds = ChunkDataset(train_chunks, tokenizer, all_labels)
    dev_ds   = ChunkDataset(dev_chunks,   tokenizer, all_labels)
    print(f"[ds] train_items={len(train_ds)}  dev_items={len(dev_ds)}")

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch" if dev_ds and len(dev_ds) else "no",
        save_strategy="no",
        logging_steps=10,
        warmup_ratio=0.1,
        seed=seed,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if len(dev_ds) else None,
        compute_metrics=_compute_metrics if len(dev_ds) else None,
    )

    trainer.train()

    # 평가
    if len(dev_ds):
        metrics = trainer.evaluate()
        print(f"\n[eval] {metrics}")
        with open(out / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 저장
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels}, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", help="chunks.jsonl")
    ap.add_argument("out", help="모델 저장 디렉토리")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dev-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.chunks, args.out, args.base, args.epochs, args.batch_size,
         args.lr, args.dev_ratio, args.seed)
