"""eval_full.py 의 per-query 결과 jsonl 만으로 BERTScore 보강 계산."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# eval_full.py 의 BERTScore 함수 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_full import bertscore_batch


def main(per_query_path: str, model_name: str, num_layers: int) -> None:
    rows: list[dict] = []
    with open(per_query_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    preds = [r.get("pred_answer", "") or "" for r in rows]
    refs  = [r.get("gold_answer", "") or "" for r in rows]
    print(f"[bert] {len(rows)} pairs, model={model_name}")

    bs = bertscore_batch(preds, refs, model_name=model_name,
                          num_layers=num_layers)
    bs_avg = sum(bs) / max(len(bs), 1)

    # 기존 메트릭과 BERTScore 가중합 재계산
    r5  = sum(r.get("Recall@5", 0.0) for r in rows) / len(rows)
    mrr = sum(r.get("RR", 0.0) for r in rows) / len(rows)
    f1  = sum(r.get("F1", 0.0) for r in rows) / len(rows)
    composite = r5 * 0.35 + mrr * 0.15 + bs_avg * 0.30 + f1 * 0.25

    print()
    print("=== 트랙2 평가 결과 (BERTScore 포함) ===")
    print(f"  Recall@5    : {r5:.4f}  (가중 35%)")
    print(f"  MRR         : {mrr:.4f}  (가중 15%)")
    print(f"  BERTScore   : {bs_avg:.4f}  (가중 30%)")
    print(f"  F1          : {f1:.4f}  (가중 25%)")
    print(f"  ─────────")
    print(f"  종합 점수   : {composite:.4f}")

    # 기존 jsonl 의 BERTScore 컬럼 갱신
    out = Path(per_query_path).with_suffix(".bs.jsonl")
    with open(out, "w", encoding="utf-8") as fout:
        for r, b in zip(rows, bs):
            r["BERTScore"] = float(b)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("per_query", help="eval_full.py 의 --out-jsonl 경로")
    ap.add_argument("--model", default="bert-base-multilingual-cased")
    ap.add_argument("--layers", type=int, default=9)
    a = ap.parse_args()
    main(a.per_query, a.model, a.layers)
