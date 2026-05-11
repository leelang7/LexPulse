"""
공정거래 의결서 AI 어시스턴트 - FastAPI 서버 (FULL 모드)

공모전 트랙2 요건 준수:
    - 외부 API 호출 금지 → 로컬 GGUF 모델 (EXAONE-3.5-7.8B / Qwen2.5-7B / Mistral / Gemma)
    - 8B 이하 파라미터
    - 응답 30초 제한
    - 정확히 5개 chunk_id 반환 (제출용 /predict)

차이점 (vs app_lite.py):
    - Dense (FAISS) + BM25 + RRF 하이브리드 검색
    - fine-tuned 임베딩 사용
    - 위반유형 분류기 / 조치유형 예측기 호출 가능
    - 그 외 모든 엔드포인트는 동일 시그너처

엔드포인트:
    GET  /health
    POST /search           Hybrid (BM25 + Dense + RRF)
    POST /answer           검색 + 로컬 LLM 답변
    POST /predict          공모전 제출 포맷 (5 chunk_id + answer)
    POST /classify-violation
    POST /predict-sanction
    GET  /docs / /redoc

환경변수:
    INDEX_DIR             기본 ./index
    LLM_GGUF_PATH         GGUF 모델 (필수)
    RERANKER_DIR / VIOLATION_CLF_DIR / SANCTION_CLF_DIR 선택
    HOST, PORT, ANSWER_DEADLINE_SEC
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

# .env 자동 로드 — 서비스 기동 전 환경변수 확보
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# src 디렉토리를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retrieval import HybridRetriever, SearchResult
# 외부 API LLM 모듈은 사용 금지 (공모전 트랙2 요건). 로컬 LLM 만 import.
from local_llm import generate as local_llm_generate, is_available as llm_available


# ───────────────────────── 설정 ─────────────────────────

INDEX_DIR    = os.environ.get("INDEX_DIR", "./index")
HOST         = os.environ.get("HOST", "0.0.0.0")
PORT         = int(os.environ.get("PORT", "8000"))
ANSWER_DEADLINE = float(os.environ.get("ANSWER_DEADLINE_SEC", "25.0"))

# 학습된 모델 경로 (선택). 없으면 해당 엔드포인트는 503 반환.
RERANKER_DIR        = os.environ.get("RERANKER_DIR", "./models/reranker")
VIOLATION_CLF_DIR   = os.environ.get("VIOLATION_CLF_DIR", "./models/violation_clf")
SANCTION_CLF_DIR    = os.environ.get("SANCTION_CLF_DIR", "./models/sanction_clf")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kftc")


# ───────────────────────── Pydantic 스키마 ─────────────────────────

# 외부 API 금지 — provider 는 항상 "local"
Provider = Literal["local"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="자연어 질의")
    top_k: int = Field(5, ge=1, le=30, description="반환할 청크 수")
    candidate_k: int = Field(30, ge=5, le=200, description="각 검색기에서 후보 풀 크기")
    filter_조치유형: Optional[list[str]] = Field(None, description="조치유형 필터 (예: ['과징금'])")
    filter_위반유형: Optional[list[str]] = Field(None, description="위반유형 필터")


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float
    bm25_rank: Optional[int]
    dense_rank: Optional[int]
    피심인: list[str]
    위반유형: list[str]
    조치유형: list[str]


class SearchResponse(BaseModel):
    query: str
    elapsed_ms: float
    hits: list[SearchHit]


class AnswerRequest(SearchRequest):
    """provider 필드는 외부 API 금지로 제거됨. 항상 로컬 LLM."""


# Re-ranker 사용 여부를 SearchRequest 에 추가
class SearchRequestRerank(SearchRequest):
    use_reranker: bool = Field(False, description="cross-encoder re-ranker 사용 여부")


class AnswerResponse(BaseModel):
    query: str
    provider: Provider
    answer: str
    elapsed_ms: float
    hits: list[SearchHit]


class PredictRequest(BaseModel):
    """공모전 제출 포맷."""
    query: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    """트랙2 제출 포맷: 정확히 5 chunk_id + answer."""
    query: str
    chunk_ids: list[str] = Field(..., min_length=5, max_length=5)
    answer: str
    elapsed_ms: float
    llm_loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["full"] = "full"
    index_dir: str
    chunk_count: int
    embedding_model: str
    default_provider: Provider
    llm_loaded: bool = False
    reranker_loaded: bool
    violation_clf_loaded: bool
    sanction_clf_loaded: bool
    answer_deadline_sec: float = 25.0


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000,
                      description="기초사실/위법성판단 등 자유 텍스트")


class ViolationResponse(BaseModel):
    labels: list[str]
    scores: dict[str, float]


class SanctionResponse(BaseModel):
    label: str
    score: float
    distribution: dict[str, float]


# ───────────────────────── 라이프사이클 ─────────────────────────

# 앱 인스턴스 외부에 둬서 모듈 import 시점이 아닌 startup에서만 로드
_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("인덱스 로드 시작: %s", INDEX_DIR)
    if not Path(INDEX_DIR).exists():
        log.error("인덱스 디렉토리를 찾을 수 없습니다: %s", INDEX_DIR)
        log.error("먼저 `python src/build_index.py chunks.jsonl index` 를 실행하세요.")
        raise RuntimeError(f"INDEX_DIR not found: {INDEX_DIR}")

    t0 = time.perf_counter()
    reranker_path = RERANKER_DIR if Path(RERANKER_DIR).exists() else None
    retriever = HybridRetriever(INDEX_DIR, reranker_path=reranker_path)
    elapsed = time.perf_counter() - t0
    log.info("인덱스 로드 완료: 청크 %d개, %.2fs", len(retriever.chunks), elapsed)

    _state["retriever"] = retriever
    _state["embedding_model"] = retriever.model_name
    _state["chunk_id_set"] = {c["chunk_id"] for c in retriever.chunks}

    # 로컬 LLM (외부 API 금지) — startup 시 warmup 까지 수행해
    # 첫 /predict 호출이 30초 deadline 안에 들어오도록 한다.
    _state["llm_loaded"] = bool(llm_available())
    if _state["llm_loaded"]:
        log.info("로컬 LLM 사용 가능 — warmup 시작 (모델 로드 + 더미 추론 1회)")
        try:
            t_w = time.perf_counter()
            # 더미 hit 1 개로 generate 호출 — 모델 메모리 매핑 + KV 캐시 워밍업
            class _DummyHit:
                chunk_id = "warmup"; doc_id = "warmup"
                title = "warmup"; section = "warmup"; text = "공정거래위원회"
                피심인 = []; 위반유형 = []; 조치유형 = []
            _ = local_llm_generate("warmup", [_DummyHit()],
                                    deadline_sec=ANSWER_DEADLINE)
            log.info("LLM warmup 완료, %.1fs", time.perf_counter() - t_w)
        except Exception as e:
            log.warning("LLM warmup 실패 (런타임 문제 가능): %s", e)
    else:
        log.warning("로컬 LLM 미로드 — LLM_GGUF_PATH 또는 models/llm/*.gguf 확인")

    # 분류기 로드 (선택)
    _state["violation_clf"] = None
    _state["sanction_clf"] = None
    try:
        if Path(VIOLATION_CLF_DIR).exists():
            from classifier_infer import ViolationClassifier
            log.info("위반유형 분류기 로드: %s", VIOLATION_CLF_DIR)
            _state["violation_clf"] = ViolationClassifier(VIOLATION_CLF_DIR)
    except Exception as e:
        log.warning("위반유형 분류기 로드 실패: %s", e)
    try:
        if Path(SANCTION_CLF_DIR).exists():
            from classifier_infer import SanctionPredictor
            log.info("조치유형 예측기 로드: %s", SANCTION_CLF_DIR)
            _state["sanction_clf"] = SanctionPredictor(SANCTION_CLF_DIR)
    except Exception as e:
        log.warning("조치유형 예측기 로드 실패: %s", e)

    yield
    log.info("종료")


app = FastAPI(
    title="공정거래 의결서 AI 어시스턴트",
    description=(
        "공정거래위원회 의결서를 BM25 + FAISS 하이브리드 검색으로 찾고, "
        "검색된 의결서를 근거로 LLM 답변을 생성하는 RAG REST API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ───────────────────────── 정적 파일 / 프론트엔드 ─────────────────────────

_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_static_dir / "index.html")


# ───────────────────────── 변환 헬퍼 ─────────────────────────

def _to_hit(r: SearchResult) -> SearchHit:
    return SearchHit(
        chunk_id=r.chunk_id,
        doc_id=r.doc_id,
        title=r.title,
        section=r.section,
        text=r.text,
        score=r.score,
        bm25_rank=r.bm25_rank,
        dense_rank=r.dense_rank,
        피심인=r.피심인,
        위반유형=r.위반유형,
        조치유형=r.조치유형,
    )


def _retriever() -> HybridRetriever:
    r = _state.get("retriever")
    if r is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return r


# ───────────────────────── 엔드포인트 ─────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    r = _retriever()
    return HealthResponse(
        status="ok",
        mode="full",
        index_dir=INDEX_DIR,
        chunk_count=len(r.chunks),
        embedding_model=_state.get("embedding_model", "unknown"),
        default_provider="local",
        llm_loaded=bool(_state.get("llm_loaded")),
        reranker_loaded=r.reranker is not None,
        violation_clf_loaded=_state.get("violation_clf") is not None,
        sanction_clf_loaded=_state.get("sanction_clf") is not None,
        answer_deadline_sec=ANSWER_DEADLINE,
    )


@app.post("/classify-violation", response_model=ViolationResponse, tags=["model"])
def classify_violation(req: ClassifyRequest):
    clf = _state.get("violation_clf")
    if clf is None:
        raise HTTPException(503, "위반유형 분류기가 로드되지 않았습니다. "
                                  f"VIOLATION_CLF_DIR={VIOLATION_CLF_DIR}")
    pred = clf.predict(req.text)
    return ViolationResponse(labels=pred.labels, scores=pred.scores)


@app.post("/predict-sanction", response_model=SanctionResponse, tags=["model"])
def predict_sanction(req: ClassifyRequest):
    clf = _state.get("sanction_clf")
    if clf is None:
        raise HTTPException(503, "조치유형 예측기가 로드되지 않았습니다. "
                                  f"SANCTION_CLF_DIR={SANCTION_CLF_DIR}")
    pred = clf.predict(req.text)
    return SanctionResponse(
        label=pred.label,
        score=pred.score,
        distribution=pred.distribution,
    )


@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(req: SearchRequest):
    r = _retriever()
    t0 = time.perf_counter()
    results = r.search(
        req.query,
        top_k=req.top_k,
        candidate_k=req.candidate_k,
        filter_조치유형=req.filter_조치유형,
        filter_위반유형=req.filter_위반유형,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info("[search] q=%r top_k=%d → %d hits (%.0f ms)",
             req.query[:50], req.top_k, len(results), elapsed_ms)

    return SearchResponse(
        query=req.query,
        elapsed_ms=round(elapsed_ms, 2),
        hits=[_to_hit(h) for h in results],
    )


@app.post("/answer", response_model=AnswerResponse, tags=["rag"])
def answer(req: AnswerRequest):
    r = _retriever()
    t0 = time.perf_counter()

    hits = r.search(
        req.query,
        top_k=req.top_k,
        candidate_k=req.candidate_k,
        filter_조치유형=req.filter_조치유형,
        filter_위반유형=req.filter_위반유형,
    )

    if not hits:
        return AnswerResponse(
            query=req.query,
            provider="local",
            answer="관련 의결서를 찾지 못했습니다. 질문을 다르게 표현해 주세요.",
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            hits=[],
        )

    # LLM 컨텍스트는 top-3 만 (n_ctx 보호) — 응답 hits 는 top_k 그대로
    try:
        ans_text = local_llm_generate(req.query, hits[:3],
                                       deadline_sec=ANSWER_DEADLINE)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("LLM 호출 실패")
        raise HTTPException(status_code=502, detail=f"LLM 호출 실패: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info("[answer] q=%r → %d hits (%.0f ms)",
             req.query[:50], len(hits), elapsed_ms)

    return AnswerResponse(
        query=req.query,
        provider="local",
        answer=ans_text,
        elapsed_ms=round(elapsed_ms, 2),
        hits=[_to_hit(h) for h in hits],
    )


@app.post("/predict", response_model=PredictResponse, tags=["competition"])
def predict(req: PredictRequest):
    """공모전 트랙2 제출 포맷: 정확히 5 chunk_id + answer."""
    r = _retriever()
    t0 = time.perf_counter()

    # reranker 가 로드되어 있으면 자동 활성화 (RRF 후 재정렬)
    hits = r.search(req.query, top_k=5, candidate_k=30,
                    use_reranker=(r.reranker is not None))
    # 부족한 경우 corpus 에서 채움 (5개 보장)
    already = {h.chunk_id for h in hits}
    if len(hits) < 5:
        for c in r.chunks:
            if c["chunk_id"] not in already:
                # 빈 SearchResult 만들기엔 dataclass 라 부담 — 직접 dict 사용
                # 검색 결과가 부족한 경우는 매우 드물다 (corpus > 5)
                hits.append(SearchResult(
                    chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                    title=c.get("title", ""), section=c.get("section", ""),
                    text=c["text"], score=0.0, bm25_rank=None, dense_rank=None,
                    피심인=c.get("피심인", []),
                    위반유형=c.get("위반유형", []),
                    조치유형=c.get("조치유형", []),
                ))
                already.add(c["chunk_id"])
                if len(hits) >= 5:
                    break

    chunk_ids = [h.chunk_id for h in hits[:5]]
    if len(chunk_ids) != 5 or len(set(chunk_ids)) != 5:
        raise HTTPException(422, f"chunk_id 검증 실패: {chunk_ids}")
    corpus_set = _state.get("chunk_id_set") or set()
    missing = [c for c in chunk_ids if c not in corpus_set]
    if missing:
        raise HTTPException(422, f"corpus 외 chunk_id: {missing}")

    answer_text = ""
    if _state.get("llm_loaded"):
        # LLM 입력은 top-3 청크만 (응답 chunk_ids 는 5 그대로 유지). 30초 deadline 보호.
        try:
            answer_text = local_llm_generate(req.query, hits[:3],
                                              deadline_sec=ANSWER_DEADLINE)
        except Exception as e:
            log.warning("LLM 실패, 답변 비움: %s", e)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms / 1000 > 30:
        log.warning("응답 30초 초과: %.1fs", elapsed_ms / 1000)
    return PredictResponse(
        query=req.query, chunk_ids=chunk_ids,
        answer=answer_text,
        elapsed_ms=round(elapsed_ms, 2),
        llm_loaded=bool(_state.get("llm_loaded")),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_req: Request, exc: Exception):
    log.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {exc.__class__.__name__}: {exc}"},
    )


# ───────────────────────── 진입점 ─────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, log_level="info")
