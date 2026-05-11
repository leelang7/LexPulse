"""
공모전 트랙2 전용 평가기

평가 가중치 (공식):
    Retrieval 50%
        Recall@5      35%
        MRR           15%
    Generation 50%
        BERTScore     30%
        F1            25%

입력: QA 테스트셋 (jsonl)
    {"query": "...", "gold_chunk_ids": ["..."], "gold_answer": "..."}

출력:
    각 메트릭 + 가중합 종합 점수 + per-query 결과 jsonl

사용:
    python eval_full.py <qa_test.jsonl> --index <dir> --predict-url http://localhost:8000/predict
    또는
    python eval_full.py <qa_test.jsonl> --index <dir> --offline   # 서버 없이 로컬 모듈 호출

NOTE:
    BERTScore 는 transformers 가 필요 → torch 가 working state 여야 함.
    재부팅 후 또는 완전한 환경에서만 정상 동작.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


# ───────────────────────── 메트릭 ─────────────────────────

def recall_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    """1 if any gold_id in predicted[:k] else 0 (per-query binary)."""
    return float(bool(set(predicted[:k]) & set(gold)))


def reciprocal_rank(predicted: list[str], gold: list[str]) -> float:
    """1/rank of first gold hit (1-indexed); 0 if none."""
    gold_set = set(gold)
    for r, p in enumerate(predicted, start=1):
        if p in gold_set:
            return 1.0 / r
    return 0.0


# 한국어 토큰 F1 (공모전 BLEU/F1 같은 토큰 수준 평가용)
_F1_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def _tokenize_ko(text: str) -> list[str]:
    return _F1_TOKEN_RE.findall(text or "")


def token_f1(prediction: str, reference: str) -> float:
    p = _tokenize_ko(prediction)
    r = _tokenize_ko(reference)
    if not p or not r:
        return 0.0
    pc, rc = Counter(p), Counter(r)
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall    = common / len(r)
    return 2 * precision * recall / (precision + recall)


# BERTScore — transformers 가 working state 일 때만
def bertscore_batch(predictions: list[str], references: list[str],
                    model_name: str = "bert-base-multilingual-cased",
                    num_layers: int = 9) -> list[float]:
    """
    BERT 임베딩 기반 의미 유사도. 각 (pred, ref) 쌍에 대한 F-measure 반환.
    bert_score 패키지의 한국어 미지원 → 다국어 BERT 사용.
    """
    try:
        from bert_score import score as bs_score
    except ImportError as e:
        raise RuntimeError(
            "bert-score 패키지 필요. `pip install bert-score`") from e
    P, R, F = bs_score(
        cands=predictions,
        refs=references,
        model_type=model_name,
        num_layers=num_layers,           # mBERT 권장 layer
        verbose=False,
        rescale_with_baseline=False,
    )
    return F.tolist()


# ───────────────────────── 예측 호출 (서버 또는 오프라인) ─────────────────────────

def _call_server(predict_url: str, query: str, timeout: float,
                 retries: int = 2, retry_wait: float = 2.0) -> dict[str, Any]:
    body = json.dumps({"query": query}).encode()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(predict_url, method="POST",
                                         headers={"Content-Type": "application/json"},
                                         data=body)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"server {e.code}: {e.read()[:300]!r}")
        except (ConnectionResetError, ConnectionRefusedError, OSError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_wait * (attempt + 1))  # 백오프
                continue
            raise RuntimeError(f"connection failed after {retries+1} attempts: {e}")
    raise RuntimeError(f"unreachable: {last_err}")


def _call_offline(query: str, retriever, llm_fn) -> dict[str, Any]:
    """서버 거치지 않고 직접 retriever + LLM 호출."""
    t0 = time.perf_counter()
    hits = retriever.search(query, top_k=5)
    chunk_ids = [h.chunk_id for h in hits[:5]]
    answer = ""
    if llm_fn is not None and hits:
        try:
            answer = llm_fn(query, hits[:5])
        except Exception:
            answer = ""
    elapsed = (time.perf_counter() - t0) * 1000
    return {"chunk_ids": chunk_ids, "answer": answer, "elapsed_ms": elapsed}


# ───────────────────────── 메인 ─────────────────────────

def _load_qa(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(qa_path: str, predict_url: str | None,
         offline: bool, index_dir: str | None,
         skip_bertscore: bool, out_jsonl: str | None,
         timeout: float) -> None:
    qa_path_p = Path(qa_path)
    qa = _load_qa(qa_path_p)
    print(f"[load] {len(qa)} QA items from {qa_path_p}")

    retriever = None
    llm_fn = None
    if offline:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        # lite 검색기를 그대로 활용 (BM25-only)
        from app_lite_helpers import OfflineRetriever  # 런타임 호환을 위해 분리
        retriever = OfflineRetriever(Path(index_dir or "./index"))
        try:
            from local_llm import generate as _gen
            llm_fn = lambda q, h: _gen(q, h)  # noqa: E731
            print("[init] 로컬 LLM 사용")
        except Exception as e:
            print(f"[init] 로컬 LLM 사용 불가 ({e}) — 답변은 빈 문자열")

    # 1) 예측 수집
    preds: list[dict] = []
    for i, item in enumerate(qa, 1):
        q = item["query"]
        try:
            if offline:
                p = _call_offline(q, retriever, llm_fn)
            else:
                p = _call_server(predict_url, q, timeout=timeout)
        except Exception as e:
            print(f"[{i}/{len(qa)}] 실패: {e}")
            p = {"chunk_ids": [], "answer": ""}
        preds.append(p)
        if i % 10 == 0 or i == len(qa):
            print(f"[{i}/{len(qa)}] elapsed_ms={p.get('elapsed_ms', '?')}")

    # 2) Retrieval 메트릭
    r5_list = [recall_at_k(p.get("chunk_ids", []), q.get("gold_chunk_ids", []), 5)
               for p, q in zip(preds, qa)]
    mrr_list = [reciprocal_rank(p.get("chunk_ids", []), q.get("gold_chunk_ids", []))
                for p, q in zip(preds, qa)]
    recall5 = sum(r5_list) / max(len(r5_list), 1)
    mrr     = sum(mrr_list) / max(len(mrr_list), 1)

    # 3) Generation 메트릭
    f1_list = [token_f1(p.get("answer", ""), q.get("gold_answer", ""))
               for p, q in zip(preds, qa)]
    f1_avg = sum(f1_list) / max(len(f1_list), 1)

    bs_avg = float("nan")
    bs_list: list[float] = []
    if not skip_bertscore:
        try:
            print("[bert] BERTScore 계산 중...")
            bs_list = bertscore_batch(
                [p.get("answer", "") or "" for p in preds],
                [q.get("gold_answer", "") or "" for q in qa],
            )
            bs_avg = sum(bs_list) / max(len(bs_list), 1)
        except Exception as e:
            print(f"[bert] BERTScore 실패 (메트릭 NaN): {e}")
            bs_list = [float("nan")] * len(preds)
    else:
        bs_list = [float("nan")] * len(preds)

    # 4) 종합 (공식 가중치)
    composite = (recall5 * 0.35 + mrr * 0.15 +
                 (bs_avg if not (bs_avg != bs_avg) else 0.0) * 0.30 +
                 f1_avg * 0.25)

    print()
    print("=== 트랙2 평가 결과 ===")
    print(f"  Recall@5    : {recall5:.4f}  (가중 35%)")
    print(f"  MRR         : {mrr:.4f}  (가중 15%)")
    print(f"  BERTScore   : {bs_avg:.4f}  (가중 30%)")
    print(f"  F1          : {f1_avg:.4f}  (가중 25%)")
    print(f"  ─────────")
    print(f"  종합 점수   : {composite:.4f}  (BERTScore=NaN 시 그 부분 0)")

    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for q, p, r5, mr, f1, bs in zip(qa, preds, r5_list, mrr_list, f1_list, bs_list):
                f.write(json.dumps({
                    "query": q["query"],
                    "gold_chunk_ids": q.get("gold_chunk_ids", []),
                    "gold_answer": q.get("gold_answer", ""),
                    "pred_chunk_ids": p.get("chunk_ids", []),
                    "pred_answer": p.get("answer", ""),
                    "Recall@5": r5, "RR": mr, "F1": f1,
                    "BERTScore": bs,
                    "elapsed_ms": p.get("elapsed_ms"),
                }, ensure_ascii=False) + "\n")
        print(f"\n[저장] per-query 결과 → {out_jsonl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qa", help="QA 테스트셋 jsonl")
    ap.add_argument("--predict-url", default="http://localhost:8000/predict",
                    help="서버 모드용 /predict 엔드포인트")
    ap.add_argument("--offline", action="store_true",
                    help="서버 없이 직접 검색기·LLM 호출")
    ap.add_argument("--index", default="./index", help="offline 모드용 인덱스 디렉토리")
    ap.add_argument("--skip-bertscore", action="store_true",
                    help="BERTScore 스킵 (torch unavailable 시)")
    ap.add_argument("--out-jsonl", default=None, help="per-query 결과 저장")
    ap.add_argument("--timeout", type=float, default=35.0,
                    help="서버 호출 타임아웃 (30초 + 마진)")
    a = ap.parse_args()
    main(a.qa, a.predict_url, a.offline, a.index,
         a.skip_bertscore, a.out_jsonl, a.timeout)
