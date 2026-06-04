"""
단계 5-2 (2/2): Optuna 앙상블 가중치 최적화

retrieval_cache.pkl 의 캐싱 신호로, 검색 점수 결합 가중치를 평가셋 기준 최적화.
사람 수동 튜닝(균등 RRF) 대신 R@5+MRR 보상을 직접 최대화.

과적합 방지: 5-fold cross-validation. 각 fold 의 val 점수 평균으로 일반화 성능 확정.

최적화 대상 파라미터:
    w_bm25, w_e5, w_mp   : 3신호 RRF 가중치
    rrf_k                : RRF 상수
    w_rr                 : reranker 점수 결합 가중치 (rr_score 정규화 후 가산)
    use_filter           : 위반유형 필터 on/off
    n_top1               : top_doc_expand (top-1 doc 청크 수, 0=doc_diversity)
"""
from __future__ import annotations

import pickle
import re
import sys

import numpy as np
import optuna

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE = "retrieval_cache.pkl"

with open(CACHE, "rb") as f:
    CACHE_DATA = pickle.load(f)

# 위반유형 필터용 — 전체 위반유형 목록
ALL_VIOLS = set()
for q in CACHE_DATA:
    for c in q["candidates"]:
        ALL_VIOLS.update(c.get("위반유형", []))
ALL_VIOLS = sorted(ALL_VIOLS, key=len, reverse=True)


def _norm(s):
    return re.sub(r"[\s\-\xb7\.]+", "", s).lower()


def extract_viol(query):
    qn = _norm(query)
    for v in ALL_VIOLS:
        vn = _norm(v)
        if len(vn) >= 5 and vn in qn:
            return v
    return None


def score_query(qdata, p):
    """가중치 p 로 한 쿼리의 top-5 chunk_id 산출."""
    cands = qdata["candidates"]
    fv = extract_viol(qdata["query"]) if p["use_filter"] else None

    # reranker 점수 정규화 (쿼리 내 min-max)
    rr_vals = [c["rr_score"] for c in cands if c["rr_score"] is not None]
    rr_min, rr_max = (min(rr_vals), max(rr_vals)) if rr_vals else (0, 1)
    rr_span = (rr_max - rr_min) or 1.0

    scored = []
    for c in cands:
        s = 0.0
        if c["bm25_rank"] is not None:
            s += p["w_bm25"] / (p["rrf_k"] + c["bm25_rank"] + 1)
        if c["e5_rank"] is not None:
            s += p["w_e5"] / (p["rrf_k"] + c["e5_rank"] + 1)
        if c["mp_rank"] is not None:
            s += p["w_mp"] / (p["rrf_k"] + c["mp_rank"] + 1)
        if c["rr_score"] is not None and p["w_rr"] > 0:
            s += p["w_rr"] * (c["rr_score"] - rr_min) / rr_span
        scored.append((c, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    # 필터 적용 (필터 후 5개 미만이면 무필터 보충)
    def pick(filtered):
        seen, out = set(), []
        for c, _ in scored:
            if filtered and fv and fv not in c.get("위반유형", []):
                continue
            if c["doc_id"] in seen:
                continue
            seen.add(c["doc_id"])
            out.append(c)
            if len(out) >= 5:
                break
        return out

    hits = pick(True) if fv else pick(False)
    if len(hits) < 5:
        seen = {c["doc_id"] for c in hits}
        for c, _ in scored:
            if c["doc_id"] not in seen:
                hits.append(c)
                seen.add(c["doc_id"])
                if len(hits) >= 5:
                    break
    return [c["chunk_id"] for c in hits[:5]]


def eval_subset(indices, p):
    h5 = mrr = 0.0
    for i in indices:
        qd = CACHE_DATA[i]
        gold = set(qd["gold_chunk_ids"])
        preds = score_query(qd, p)
        if set(preds) & gold:
            h5 += 1
        for rank, c in enumerate(preds, 1):
            if c in gold:
                mrr += 1.0 / rank
                break
    n = len(indices)
    return h5 / n, mrr / n


def make_params(trial):
    return {
        "w_bm25": trial.suggest_float("w_bm25", 0.1, 3.0),
        "w_e5":   trial.suggest_float("w_e5", 0.1, 3.0),
        "w_mp":   trial.suggest_float("w_mp", 0.0, 3.0),
        "w_rr":   trial.suggest_float("w_rr", 0.0, 5.0),
        "rrf_k":  trial.suggest_int("rrf_k", 10, 100),
        "use_filter": trial.suggest_categorical("use_filter", [True, False]),
    }


def main():
    N = len(CACHE_DATA)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)
    print(f"[opt] {N} queries, 5-fold CV")

    # ── 베이스라인 (균등 가중, 필터 on) ──
    base_p = {"w_bm25": 1, "w_e5": 1, "w_mp": 1, "w_rr": 0, "rrf_k": 60, "use_filter": True}
    base_rr_p = {**base_p, "w_rr": 2.0}
    b5, bm = eval_subset(list(range(N)), base_p)
    r5, rm = eval_subset(list(range(N)), base_rr_p)
    print(f"[base] 균등 RRF(필터)         R@5={b5:.4f} MRR={bm:.4f}")
    print(f"[base] 균등 RRF+reranker(필터) R@5={r5:.4f} MRR={rm:.4f}")

    # ── 5-fold CV 최적화 ──
    val_scores = []
    best_params_per_fold = []
    for fi in range(5):
        val_idx = list(folds[fi])
        train_idx = [i for i in range(N) if i not in set(val_idx)]

        def objective(trial):
            p = make_params(trial)
            h5, mrr = eval_subset(train_idx, p)
            return h5 + mrr  # 보상 = R@5 + MRR

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=fi))
        study.optimize(objective, n_trials=300, show_progress_bar=False)
        bp = study.best_params
        v5, vm = eval_subset(val_idx, bp)
        val_scores.append((v5, vm))
        best_params_per_fold.append(bp)
        print(f"[fold {fi}] val R@5={v5:.4f} MRR={vm:.4f}  params={bp}")

    avg5 = np.mean([v[0] for v in val_scores])
    avgm = np.mean([v[1] for v in val_scores])
    print(f"\n[CV 평균] val R@5={avg5:.4f} MRR={avgm:.4f}")
    print(f"[개선] R@5 {(avg5-r5)*100:+.1f}%p vs 균등+reranker")

    # ── 전체 데이터로 최종 가중치 학습 (제출용) ──
    def obj_full(trial):
        p = make_params(trial)
        h5, mrr = eval_subset(list(range(N)), p)
        return h5 + mrr
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=999))
    study.optimize(obj_full, n_trials=500, show_progress_bar=False)
    print(f"\n[최종 가중치 (전체학습)] {study.best_params}")
    f5, fm = eval_subset(list(range(N)), study.best_params)
    print(f"[최종 전체] R@5={f5:.4f} MRR={fm:.4f}")

    import json
    with open("best_weights.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)
    print("[save] best_weights.json")


if __name__ == "__main__":
    main()
