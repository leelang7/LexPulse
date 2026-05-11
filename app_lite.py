"""
Lite 모드 FastAPI 서버 — 외부 API 금지 + torch/faiss MKL 우회

공모전 트랙2 요건 준수:
    - 외부 LLM API 호출 금지 → 로컬 GGUF 모델 (EXAONE-3.5-7.8B 한국어 SOTA, Qwen 백업)
    - 8B 이하 파라미터
    - 응답 30초 제한
    - 정확히 5개 chunk_id 반환 (제출용 /predict)
    - 공식 데이터 chunk_id 보존

언제 lite 모드를 쓰는가:
    - faiss/torch DLL 이 wedge 됐을 때 (MKL 데드락) — 재부팅 회피용
    - 또는 평가 환경처럼 의존성 가벼운 deploy 가 필요할 때

기동:
    python app_lite.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 주의: sentence_transformers (= torch dep) import 는 dense 인덱스가 실제로
# ST 백엔드일 때만 수행 (lazy). torch CUDA 가 wedge 된 환경에서도 fastembed
# (ONNX) 백엔드만으로 서버가 기동하도록 분리.

# rank_bm25 는 pure python (MKL 영향 없음)
from rank_bm25 import BM25Okapi

# 선택적 ONNX dense retrieval (fastembed) — torch 비의존
import os as _os
import importlib.util as _ils
def _add_nvidia_paths():
    if _os.name != "nt":
        return
    added = []
    for sub in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc",
                "nvidia.cudnn", "nvidia.cufft", "nvidia.nvjitlink"):
        s = _ils.find_spec(sub)
        if s and s.submodule_search_locations:
            from pathlib import Path as _P
            binp = _P(list(s.submodule_search_locations)[0]) / "bin"
            if binp.exists():
                added.append(str(binp))
    if added:
        _os.environ["PATH"] = _os.pathsep.join(added) + _os.pathsep + _os.environ.get("PATH", "")
_add_nvidia_paths()


# ───────────────────────── 토크나이저 (인덱스 빌드와 동일) ─────────────────────────

# kiwipiepy 기반 형태소 분석 (build_index.py 와 동일) — torch 비의존
from korean_tok import korean_tokenize  # noqa: F401


# ───────────────────────── 설정 ─────────────────────────

INDEX_DIR    = os.environ.get("INDEX_DIR", "./index")
DENSE_INDEX_DIR = os.environ.get("DENSE_INDEX_DIR", "")  # fastembed 인덱스 (선택)
RERANKER_MODEL  = os.environ.get("RERANKER_MODEL", "")  # fastembed reranker 모델 ID (선택)
HOST         = os.environ.get("HOST", "0.0.0.0")
PORT         = int(os.environ.get("PORT", "8000"))
ANSWER_DEADLINE = float(os.environ.get("ANSWER_DEADLINE_SEC", "18.0"))  # 가이드 20s 한도 마진 2초

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("kftc-lite")


# ───────────────────────── 검색기 (BM25-only) ─────────────────────────

@dataclass
class LiteHit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    bm25_rank: int
    피심인: list
    위반유형: list
    조치유형: list


class LiteRetriever:
    """BM25 + (선택) fastembed Dense + (선택) reranker. torch 비의존."""

    def __init__(self, index_dir: Path, dense_index_dir: Path | None = None,
                 reranker_model: str | None = None):
        self.index_dir = index_dir
        self.chunks: list[dict] = []
        with open(index_dir / "meta.jsonl", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        bm_path = index_dir / "bm25.pkl"
        if bm_path.exists():
            try:
                with open(bm_path, "rb") as f:
                    obj = pickle.load(f)
                self.bm25 = obj["bm25"]
                log.info("BM25 인덱스 로드: %s", bm_path)
            except Exception as e:
                log.warning("BM25 pickle 로드 실패 (%s), 재구축", e)
                self.bm25 = self._build_bm25()
        else:
            log.info("BM25 pickle 없음, 메모리에서 구축")
            self.bm25 = self._build_bm25()

        # chunk_id → index 매핑 (검증용)
        self.chunk_id_set: set[str] = {c["chunk_id"] for c in self.chunks}

        # Dense (fastembed 또는 sentence-transformers) — 선택적
        self.dense_index = None
        self.dense_embedder = None
        self.dense_model_name = None
        self.dense_is_e5 = False
        self.dense_backend = None  # "fastembed" | "sentence-transformers"
        if dense_index_dir and (dense_index_dir / "faiss.bin").exists():
            try:
                import faiss
                self.dense_index = faiss.read_index(str(dense_index_dir / "faiss.bin"))
                model_name = (dense_index_dir / "embedding_model.txt").read_text(
                    encoding="utf-8").strip()
                backend_file = dense_index_dir / "embedding_backend.txt"
                backend = (backend_file.read_text(encoding="utf-8").strip()
                           if backend_file.exists() else "fastembed")
                log.info("Dense 인덱스 로드: %s (model=%s, backend=%s)",
                         dense_index_dir, model_name, backend)
                if backend == "sentence-transformers":
                    # KURE-v1 / bge-m3-korean 등 한국어 SOTA — torch GPU 사용
                    # torch wedge 환경에서는 import 자체가 hang 하므로 try/except 보호
                    try:
                        from sentence_transformers import SentenceTransformer
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.dense_embedder = SentenceTransformer(model_name, device=device)
                        self.dense_embedder.max_seq_length = 512
                    except Exception as e:
                        log.warning("sentence-transformers 로드 실패 (%s) — Dense 비활성", e)
                        self.dense_embedder = None
                        self.dense_index = None
                        return
                else:
                    from fastembed import TextEmbedding
                    # GPU 시도 — 실패 시 명시적으로 CPU 폴백.
                    # GPU init 이 hang 하는 케이스 방지를 위해 짧은 timeout 후 fallback
                    try:
                        self.dense_embedder = TextEmbedding(
                            model_name=model_name,
                            providers=["CUDAExecutionProvider"])
                        log.info("Dense GPU 로드 성공")
                    except Exception as e:
                        log.warning("Dense GPU 실패 (%s) → CPU 폴백", e)
                        self.dense_embedder = TextEmbedding(
                            model_name=model_name,
                            providers=["CPUExecutionProvider"])
                self.dense_backend = backend
                self.dense_model_name = model_name
                self.dense_is_e5 = ("e5" in model_name.lower()
                                     and "kure" not in model_name.lower())
            except Exception as e:
                log.warning("Dense 로드 실패 (BM25-only 진행): %s", e)
                self.dense_index = None

        # Reranker (fastembed cross-encoder) — 선택적
        # CPU 강제 — EXAONE GGUF 가 GPU 점유 중일 때 다중 CUDA context 충돌 회피.
        # top-20 후보만 재랭킹하므로 CPU 도 충분히 빠름 (~0.5s).
        self.reranker = None
        self.reranker_model_name = None
        if reranker_model:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                log.info("Reranker 로드 (CPU): %s", reranker_model)
                self.reranker = TextCrossEncoder(
                    model_name=reranker_model,
                    providers=["CPUExecutionProvider"])
                self.reranker_model_name = reranker_model
            except Exception as e:
                log.warning("Reranker 로드 실패 (재랭킹 비활성): %s", e)
                self.reranker = None

    def _build_bm25(self) -> BM25Okapi:
        return BM25Okapi([korean_tokenize(c["text"]) for c in self.chunks])

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """query 임베딩 → FAISS 검색. backend 별 인코딩 분기."""
        if self.dense_index is None or self.dense_embedder is None:
            return []
        import numpy as np
        q_text = f"query: {query}" if self.dense_is_e5 else query
        if self.dense_backend == "sentence-transformers":
            emb = self.dense_embedder.encode(
                [q_text], normalize_embeddings=True, convert_to_numpy=True)
            emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        else:
            emb = next(iter(self.dense_embedder.embed([q_text])))
            emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        scores, ids = self.dense_index.search(emb, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def search(self, query: str, top_k: int = 5,
               candidate_k: int = 50,
               filter_조치유형: Optional[list[str]] = None,
               filter_위반유형: Optional[list[str]] = None,
               use_hybrid: bool = True,
               doc_boost: float = 0.0,
               rerank_topn: int = 30,
               doc_diversity: bool = False,
               hyde_text: Optional[str] = None,
               hyde_texts: Optional[list[str]] = None,
               doc_aggregate: bool = False) -> list[LiteHit]:
        """
        Hybrid search + doc-level boost.

        doc_boost: 같은 doc 의 청크가 후보에 여럿 등장하면 그 doc 의 청크들에
                   +doc_boost / (k+rank) 가산. doc-level Recall 향상.
        """
        import numpy as np
        # 1) BM25
        bm25_scores = self.bm25.get_scores(korean_tokenize(query))
        bm25_order = list(np.argsort(bm25_scores)[::-1][:candidate_k])
        bm25_rank_map = {idx: r for r, idx in enumerate(bm25_order)
                          if bm25_scores[idx] > 0}

        # 2) Dense (있을 때만) — HyDE 텍스트가 있으면 추가 dense 검색 후 fusion
        dense_hits = []
        if use_hybrid and self.dense_index is not None:
            dense_hits = self._dense_search(query, candidate_k)
        dense_rank_map = {idx: r for r, (idx, _) in enumerate(dense_hits)}

        # 2b) HyDE: 가상 답변 텍스트로 추가 dense 검색 (single 또는 multi)
        all_hyde_texts: list[str] = []
        if hyde_text:
            all_hyde_texts.append(hyde_text)
        if hyde_texts:
            all_hyde_texts.extend(hyde_texts)
        # dedup
        all_hyde_texts = list(dict.fromkeys(t for t in all_hyde_texts if t))
        all_hyde_hits: list[list[tuple[int, float]]] = []
        if all_hyde_texts and self.dense_index is not None:
            for ht in all_hyde_texts:
                try:
                    all_hyde_hits.append(self._dense_search(ht, candidate_k))
                except Exception as e:
                    log.warning("HyDE dense 실패 (%s)", e)
        # 후방호환: 기존 hyde_hits 변수 유지 (이후 RRF 단일 분기와 일치)
        hyde_hits = all_hyde_hits[0] if all_hyde_hits else []

        # 3) RRF (k=60) — BM25 + Dense + (HyDE-Dense)
        from collections import defaultdict
        rrf: dict[int, float] = defaultdict(float)
        for r, idx in enumerate(bm25_order):
            if bm25_scores[idx] > 0:
                rrf[idx] += 1.0 / (60 + r + 1)
        for r, (idx, _) in enumerate(dense_hits):
            rrf[idx] += 1.0 / (60 + r + 1)
        # multi-HyDE: 모든 HyDE dense 결과에 대해 RRF 가산 (각각 독립 ranking)
        for hh in all_hyde_hits:
            for r, (idx, _) in enumerate(hh):
                rrf[idx] += 1.0 / (60 + r + 1)

        # 4) doc-level boost: 한 doc 에서 여러 chunk 매칭 → 해당 doc 청크 가산
        if doc_boost > 0:
            doc_score: dict[str, float] = defaultdict(float)
            for idx, s in rrf.items():
                doc_score[self.chunks[idx]["doc_id"]] += s
            # 각 chunk 의 RRF 점수에 자기 doc 의 합산점수의 일부를 추가
            for idx in list(rrf.keys()):
                d = self.chunks[idx]["doc_id"]
                rrf[idx] += doc_boost * doc_score[d]

        ordered = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

        # 4b) doc_aggregate: 각 doc 의 chunk 합산 점수로 doc rank → top doc 의 best chunk
        # gold 가 doc 전체이므로 "어떤 doc 인지"를 강하게 결정. 단일 chunk 노이즈 제거.
        if doc_aggregate:
            doc_total: dict[str, float] = defaultdict(float)
            doc_best_chunk: dict[str, tuple[int, float]] = {}
            for idx, s in rrf.items():
                d = self.chunks[idx]["doc_id"]
                doc_total[d] += s
                if d not in doc_best_chunk or s > doc_best_chunk[d][1]:
                    doc_best_chunk[d] = (idx, s)
            # doc rank 후 각 doc 의 best chunk 만 ordered 로 재구성
            doc_sorted = sorted(doc_total.items(), key=lambda x: x[1], reverse=True)
            ordered = [(doc_best_chunk[d][0], score) for d, score in doc_sorted]

        # 5) 필터링 후 reranker 후보군 형성
        filtered: list[tuple[int, float]] = []
        for idx, score in ordered[:candidate_k]:
            c = self.chunks[idx]
            if filter_조치유형 and not any(t in c.get("조치유형", []) for t in filter_조치유형):
                continue
            if filter_위반유형 and not any(t in c.get("위반유형", []) for t in filter_위반유형):
                continue
            filtered.append((idx, score))

        # 6) Reranker (있으면) — top-N 재랭킹 (기본 30)
        if self.reranker is not None and filtered:
            cands = filtered[:min(rerank_topn, len(filtered))]
            doc_pairs = [self.chunks[idx]["text"] for idx, _ in cands]
            try:
                rerank_scores = list(self.reranker.rerank(query, doc_pairs))
                # rerank_scores 와 cands 매핑
                reranked = sorted(zip(cands, rerank_scores),
                                   key=lambda x: x[1], reverse=True)
                final = [(idx, float(rs)) for (idx, _), rs in reranked]
                filtered = final + filtered[len(cands):]
            except Exception as e:
                log.warning("Reranker 호출 실패 (RRF 점수 사용): %s", e)

        # 7) doc_diversity: 같은 doc 의 chunk 가 top_k 안에 여럿 들어가지 않도록.
        #    경쟁 채점은 gold 가 doc 전체 chunks → top-K 에 distinct doc 이 많을수록 R@K 유리.
        seen_docs: set[str] = set()
        hits: list[LiteHit] = []
        for idx, score in filtered:
            c = self.chunks[idx]
            doc = c["doc_id"]
            if doc_diversity and doc in seen_docs:
                continue
            seen_docs.add(doc)
            hits.append(LiteHit(
                chunk_id=c["chunk_id"], doc_id=doc,
                title=c.get("title", ""), section=c.get("section", ""),
                text=c["text"], score=float(score),
                bm25_rank=bm25_rank_map.get(idx, -1),
                피심인=c.get("피심인", []),
                위반유형=c.get("위반유형", []),
                조치유형=c.get("조치유형", []),
            ))
            if len(hits) >= top_k:
                break
        return hits


# ───────────────────────── Pydantic 스키마 ─────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=30)
    filter_조치유형: Optional[list[str]] = None
    filter_위반유형: Optional[list[str]] = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    bm25_rank: Optional[int]
    dense_rank: Optional[int] = None
    피심인: list[str]
    위반유형: list[str]
    조치유형: list[str]


class SearchResponse(BaseModel):
    query: str
    elapsed_ms: float
    hits: list[SearchHit]


class AnswerRequest(SearchRequest):
    """provider 필드 제거: 외부 API 금지로 로컬 LLM 만 사용."""


class AnswerResponse(BaseModel):
    query: str
    provider: Literal["local"]
    answer: str
    elapsed_ms: float
    hits: list[SearchHit]


class PredictRequest(BaseModel):
    """공모전 공식 모델 제출 가이드 형식.

    공식: POST /predict {"id": "eval_xxx", "question": "..."}
    호환: 기존 'query' 필드도 허용 (자체 평가 스크립트 호환)
    """
    id: Optional[str] = None
    question: Optional[str] = Field(None, min_length=1, max_length=2000)
    query: Optional[str] = Field(None, min_length=1, max_length=2000)  # legacy

    def get_question(self) -> str:
        return self.question or self.query or ""


class PredictResponse(BaseModel):
    """공모전 공식 응답 형식.

    공식 필드:
        - id: 요청 id 그대로
        - retrieved_chunk_ids: 정확히 5개, 중복 X, 공식 corpus, 배열 순서 = ranking
        - answer: 검색된 청크를 근거로 한 자연어 답변
    부가 (자체 호환):
        - query, elapsed_ms, llm_loaded
    """
    id: Optional[str] = None
    retrieved_chunk_ids: list[str] = Field(..., min_length=5, max_length=5)
    answer: str
    # legacy/debug 필드
    query: Optional[str] = None
    chunk_ids: Optional[list[str]] = None  # alias for backward compat
    elapsed_ms: Optional[float] = None
    llm_loaded: Optional[bool] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["lite"]
    index_dir: str
    chunk_count: int
    retrieval: str   # "BM25-only" 또는 "BM25 + Dense (model_name)"
    llm_loaded: bool
    llm_path: Optional[str]
    dense_model: Optional[str] = None
    answer_deadline_sec: float


# ───────────────────────── 라이프사이클 ─────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("LITE 모드 — 외부 API 금지 / 로컬 LLM 만 사용")
    if not Path(INDEX_DIR).exists():
        raise RuntimeError(f"INDEX_DIR not found: {INDEX_DIR}")
    t0 = time.perf_counter()
    dense_dir = Path(DENSE_INDEX_DIR) if DENSE_INDEX_DIR else None
    retriever = LiteRetriever(Path(INDEX_DIR), dense_index_dir=dense_dir,
                                reranker_model=RERANKER_MODEL or None)
    log.info("청크 %d개 로드, %.2fs", len(retriever.chunks), time.perf_counter() - t0)
    _state["retriever"] = retriever

    # 로컬 LLM 가용성 확인 (실제 로드는 첫 호출에서 lazy)
    try:
        from local_llm import is_available, _load_config_from_env
        if is_available():
            cfg = _load_config_from_env()
            _state["llm_loaded"] = True
            _state["llm_path"] = cfg.gguf_path
            log.info("로컬 LLM 사용 가능: %s", cfg.gguf_path)
        else:
            _state["llm_loaded"] = False
            _state["llm_path"] = None
            log.warning("로컬 LLM 모델 파일 없음 (LLM_GGUF_PATH 또는 models/llm/*.gguf)")
    except Exception as e:
        _state["llm_loaded"] = False
        _state["llm_path"] = None
        log.warning("LLM 모듈 로드 실패: %s", e)
    yield


app = FastAPI(
    title="공정거래 의결서 AI 어시스턴트 (LITE)",
    description=(
        "BM25 검색 + 로컬 GGUF LLM (외부 API 금지). "
        "공모전 트랙2 제출 포맷 호환 — /predict 엔드포인트 사용."
    ),
    version="2.0.0-lite",
    lifespan=lifespan,
)


# ───────────────────────── 정적 파일 ─────────────────────────

_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_static_dir / "index.html")


def _to_hit(h: LiteHit) -> SearchHit:
    return SearchHit(
        chunk_id=h.chunk_id, doc_id=h.doc_id, title=h.title,
        section=h.section, text=h.text, score=h.score,
        bm25_rank=h.bm25_rank, dense_rank=None,
        피심인=h.피심인, 위반유형=h.위반유형, 조치유형=h.조치유형,
    )


def _retriever() -> LiteRetriever:
    r = _state.get("retriever")
    if r is None:
        raise HTTPException(503, "Retriever not initialized")
    return r


# ───────────────────────── chunk_id 검증 ─────────────────────────

def _validate_chunk_ids(chunk_ids: list[str], corpus: set[str]) -> list[str]:
    """
    공모전 규칙:
        - 정확히 5개
        - 중복 금지
        - corpus 에 존재하는 ID 만
    위반 시 사유와 함께 raise.
    """
    if len(chunk_ids) != 5:
        raise HTTPException(
            422, f"chunk_ids 는 정확히 5개여야 합니다 (현재 {len(chunk_ids)}개)")
    if len(set(chunk_ids)) != 5:
        raise HTTPException(422, "chunk_ids 에 중복 항목이 있습니다")
    missing = [cid for cid in chunk_ids if cid not in corpus]
    if missing:
        raise HTTPException(
            422, f"corpus 에 존재하지 않는 chunk_id: {missing}")
    return chunk_ids


# ───────────────────────── 엔드포인트 ─────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    r = _retriever()
    if r.dense_index is not None and r.dense_model_name:
        retrieval_desc = f"BM25 + Dense ({r.dense_model_name})"
    else:
        retrieval_desc = "BM25-only"
    if r.reranker is not None:
        retrieval_desc += f" + Reranker ({r.reranker_model_name})"
    return HealthResponse(
        status="ok", mode="lite", index_dir=INDEX_DIR,
        chunk_count=len(r.chunks), retrieval=retrieval_desc,
        llm_loaded=bool(_state.get("llm_loaded")),
        llm_path=_state.get("llm_path"),
        dense_model=r.dense_model_name,
        answer_deadline_sec=ANSWER_DEADLINE,
    )


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(req: SearchRequest):
    """검색 only — /predict 와 동일 retrieval 설정 (doc_diversity, 50/30)."""
    r = _retriever()
    t0 = time.perf_counter()
    hyde_text = None
    if os.environ.get("USE_HYDE", "0") == "1" and _state.get("llm_loaded"):
        try:
            from local_llm import generate_hypothetical
            hyde_text = generate_hypothetical(req.query, max_tokens=80)
        except Exception as e:
            log.warning("HyDE 실패: %s", e)
    hits = r.search(req.query, top_k=req.top_k,
                    filter_조치유형=req.filter_조치유형,
                    filter_위반유형=req.filter_위반유형,
                    doc_diversity=True,
                    candidate_k=50, rerank_topn=30,
                    hyde_text=hyde_text,
                    doc_aggregate=(os.environ.get("DOC_AGGREGATE","0")=="1"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info("[search] q=%r → %d hits (%.0f ms)",
             req.query[:50], len(hits), elapsed_ms)
    return SearchResponse(query=req.query,
                          elapsed_ms=round(elapsed_ms, 2),
                          hits=[_to_hit(h) for h in hits])


def _call_local_llm(query: str, hits: list[LiteHit]) -> str:
    from local_llm import generate
    return generate(query, hits, deadline_sec=ANSWER_DEADLINE)


@app.post("/answer", response_model=AnswerResponse, tags=["rag"])
def answer(req: AnswerRequest):
    r = _retriever()
    t0 = time.perf_counter()
    hits = r.search(req.query, top_k=req.top_k,
                    filter_조치유형=req.filter_조치유형,
                    filter_위반유형=req.filter_위반유형)
    if not hits:
        return AnswerResponse(
            query=req.query, provider="local",
            answer="관련 의결서를 찾지 못했습니다.",
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            hits=[],
        )
    try:
        ans_text = _call_local_llm(req.query, hits)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("LLM 호출 실패")
        raise HTTPException(502, f"LLM 호출 실패: {e}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AnswerResponse(
        query=req.query, provider="local", answer=ans_text,
        elapsed_ms=round(elapsed_ms, 2),
        hits=[_to_hit(h) for h in hits],
    )


@app.post("/predict", response_model=PredictResponse, tags=["competition"])
def predict(req: PredictRequest):
    """
    공모전 공식 제출 형식 (모델 제출 가이드).
    POST /predict {"id":"eval_xxx", "question":"..."}
    응답: {id, retrieved_chunk_ids[5], answer}

    검색기에서 정확히 5개 chunk_id 추출 (corpus 검증 포함) + LLM 답변 생성.
    응답 deadline = ANSWER_DEADLINE_SEC (기본 18s, 가이드 20s 한도 안전 마진).
    """
    r = _retriever()
    t0 = time.perf_counter()
    question = req.get_question()
    if not question:
        raise HTTPException(422, "'question' 또는 'query' 필드 필요")

    # 1) HyDE: Qwen 으로 가상 답변 생성 → dense 검색에 보강
    # USE_HYDE_MULTI=1 → 가상 답변 N개 (default 3) 생성 → 각각 dense 검색 → RRF fusion
    hyde_text = None
    hyde_texts = None
    if _state.get("llm_loaded"):
        try:
            n_hyde = int(os.environ.get("HYDE_N", "1"))
            if n_hyde >= 2 and os.environ.get("USE_HYDE_MULTI", "0") == "1":
                from local_llm import generate_hypothetical_multi
                hyde_texts = generate_hypothetical_multi(question, n=n_hyde, max_tokens=80)
            elif os.environ.get("USE_HYDE", "0") == "1":
                from local_llm import generate_hypothetical
                hyde_text = generate_hypothetical(question, max_tokens=80)
        except Exception as e:
            log.warning("HyDE 실패 (BM25+Dense 만 사용): %s", e)

    # 2) Top-5 retrieval (best config: 50/30, doc_diversity)
    hits = r.search(question, top_k=5, doc_diversity=True,
                    candidate_k=50, rerank_topn=30,
                    hyde_text=hyde_text,
                    hyde_texts=hyde_texts,
                    doc_aggregate=(os.environ.get("DOC_AGGREGATE","0")=="1"))
    if len(hits) < 5:
        # 부족한 경우 BM25 점수 무관하게 다음 후보로 채움 (corpus 전체에서)
        already = {h.chunk_id for h in hits}
        for c in r.chunks:
            if c["chunk_id"] not in already:
                hits.append(LiteHit(
                    chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                    title=c.get("title", ""), section=c.get("section", ""),
                    text=c["text"], score=0.0, bm25_rank=-1,
                    피심인=c.get("피심인", []),
                    위반유형=c.get("위반유형", []),
                    조치유형=c.get("조치유형", []),
                ))
                already.add(c["chunk_id"])
                if len(hits) >= 5:
                    break

    chunk_ids = [h.chunk_id for h in hits[:5]]
    chunk_ids = _validate_chunk_ids(chunk_ids, r.chunk_id_set)

    # 2) LLM 답변 (선택적; 모델이 없으면 retrieval 만 반환)
    llm_loaded = bool(_state.get("llm_loaded"))
    if llm_loaded:
        try:
            answer_text = _call_local_llm(question, hits[:5])
        except Exception as e:
            log.warning("LLM 실패, 답변 비워서 반환: %s", e)
            answer_text = ""
    else:
        answer_text = ""

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # 가이드 ≤20초 / 사양 ≤30초 — 안전 기준 20초로 경고
    if elapsed_ms / 1000.0 > 20.0:
        log.warning("응답 시간 20초 초과: %.1fs (해당 문항 0점 위험)", elapsed_ms / 1000)

    return PredictResponse(
        id=req.id,
        retrieved_chunk_ids=chunk_ids,
        answer=answer_text,
        # legacy/debug
        query=question, chunk_ids=chunk_ids,
        elapsed_ms=round(elapsed_ms, 2),
        llm_loaded=llm_loaded,
    )


# 분류기 엔드포인트 — lite 에서는 503 (호환성 유지)
@app.post("/classify-violation", tags=["model"])
def classify_violation():
    raise HTTPException(503, "lite 모드: torch 의존 분류기 비활성화")


@app.post("/predict-sanction", tags=["model"])
def predict_sanction():
    raise HTTPException(503, "lite 모드: torch 의존 분류기 비활성화")


@app.exception_handler(Exception)
async def unhandled(_req: Request, exc: Exception):
    log.exception("Unhandled")
    return JSONResponse(status_code=500,
                        content={"detail": f"{exc.__class__.__name__}: {exc}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_lite:app", host=HOST, port=PORT, log_level="info")
