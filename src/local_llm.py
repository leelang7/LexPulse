"""
로컬 LLM 호출 모듈 (외부 API 금지 — 공정위 트랙2 요건)

기본 백엔드: llama-cpp-python (GGUF 로더; Llama 모델이 아니라 임의 GGUF 로딩)
권장 모델 : Qwen2.5-7B-Instruct (Q4_K_M GGUF, 한국어 성능 ↑)
대안       : Mistral-7B-Instruct, Gemma-2-7B-Instruct
금지       : Llama 시리즈 (사용자 정책)

응답 시간 제약: 공모전 30초 — 그 이하로 답변 마칠 수 있도록 토큰 수·temperature
                        조정.

환경변수:
    LLM_GGUF_PATH      GGUF 모델 파일 경로 (필수)
    LLM_N_CTX          context window (기본 4096)
    LLM_N_THREADS      CPU 스레드 (기본 자동)
    LLM_MAX_TOKENS     생성 토큰 상한 (기본 800)
    LLM_TEMPERATURE    sampling temp (기본 0.2)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path


# ───────────────────────── CUDA DLL 경로 보강 (Windows) ─────────────────────────
# llama-cpp-python CUDA wheel 은 cudart64_12.dll, cublas64_12.dll 등을 시스템에서
# 찾는다. nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12 pip 패키지가 이 DLL 들을
# site-packages/nvidia/.../bin 에 둔다.
#
# llama_cpp/_ctypes_extensions.py 가 winmode=RTLD_GLOBAL 로 로드하므로
# add_dll_directory 가 의존성 DLL 검색에 적용되지 않는다.
# PATH 환경변수에 nvidia bin 들을 prepend 해야 작동.
def _add_nvidia_dll_dirs() -> None:
    if os.name != "nt":
        return
    try:
        import importlib.util
        added = []
        for sub in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc"):
            spec = importlib.util.find_spec(sub)
            if spec is None or not spec.submodule_search_locations:
                continue
            base = Path(list(spec.submodule_search_locations)[0])
            binp = base / "bin"
            if binp.exists():
                os.add_dll_directory(str(binp))   # 보조
                added.append(str(binp))
        if added:
            os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

_add_nvidia_dll_dirs()


SYSTEM_PROMPT = """당신은 공정거래위원회 의결서 "주문" 발췌기다. 반드시 한국어로만 답변한다.

규칙:
1. **답변은 오직 [참고 의결서] 본문에서만 발췌**한다. 추측·창작·요약 금지.
2. **가장 관련 있는 의결서의 "주문" 섹션 본문을 원문 그대로, 가능한 길게 인용**한다.
   - "주문" 본문이 있으면 그 문장들("- 1. 피심인은 …", "- 2. 과징금 …")을
     번호·조사까지 **원문 표현 그대로** 옮긴다. 다시 쓰거나 요약하지 않는다.
   - 주문에 없는 정보(위반행위·법조항)는 같은 의결서 본문에서 원문으로 보충.
3. "위반행위 요지:" 같은 레이블을 붙이지 말고, 의결서 문장을 그대로 서술한다.
4. 답변은 가장 관련 있는 한 의결서에만 집중. 다른 의결서로 헷갈리지 않는다.
5. 영어·중국어·일본어 절대 금지. 100% 한국어. 마지막 줄에 "[의결서: 사건명]".
6. **근거를 찾을 수 없으면** "관련 의결서에서 확인되지 않습니다" 한 줄만 출력.
"""

# 모델 우선순위
_PREFERRED_MODEL_PATTERNS = [
    "qwen2.5",
    "qwen",
    "exaone",
    "mistral",
    "gemma",
]

USER_PROMPT_TEMPLATE = """질문: {query}

[참고 의결서]
{context}

가장 관련 있는 의결서의 "주문" 섹션 본문을 원문 그대로 옮겨 적으십시오.
"- 1. 피심인은 …", "- 2. 과징금 …" 처럼 번호와 문장을 원문 표현 그대로,
빠짐없이 길게 인용하십시오. 요약·레이블·재작성 금지. 마지막 줄에 "[의결서: 사건명]".
"""


# 섹션 우선순위 (주문/처분 → 위법성판단 → 기초사실 순)
_SECTION_PRIORITY = {"주문": 0, "처분": 1, "결론": 2, "위법성판단": 3, "기초사실": 4}


def _long_context_reorder(hits: list) -> list:
    """LongContextReorder (Lost-in-the-Middle 방어).
    입력은 관련도 내림차순. 가장 관련 높은 항목을 컨텍스트 양 끝(맨 앞/맨 뒤)에
    배치해 LLM 이 중간에 묻힌 정답을 놓치지 않도록 재배열.
    [1,2,3,4,5] (관련도순) -> [1,3,5,4,2] (1등 맨앞, 2등 맨뒤)
    """
    reordered = []
    for i, h in enumerate(hits):
        if i % 2 == 0:
            reordered.append(h)       # 짝수 인덱스 → 앞쪽에 append
        else:
            reordered.insert(0, h)    # 홀수 인덱스 → 맨 앞에 insert (결과적으로 뒤로 밀림)
    # 위 방식은 [1,3,5,...] 앞 + [...,4,2] 뒤 형태가 됨
    return reordered


def format_context(hits: list, max_text_chars: int = 400,
                   reorder: bool = True) -> str:
    """
    LLM 컨텍스트 빌드.
    - 동일 doc 청크라면 섹션 우선순위(주문>처분>위법성판단>기초사실) 로 정렬.
    - 서로 다른 doc 이면 LongContextReorder 로 관련도 높은 것을 양 끝에 배치.
    - 각 청크 본문을 max_text_chars 로 잘라 컨텍스트 윈도우 보호.
    """
    doc_ids = set(getattr(h, "doc_id", "") for h in hits)
    if len(doc_ids) == 1:
        # 같은 문서 청크 → 섹션 우선순위 정렬 (주문/처분 본문 앞에 오게)
        hits_sorted = sorted(
            hits,
            key=lambda h: _SECTION_PRIORITY.get(getattr(h, "section", ""), 99)
        )
    elif reorder:
        # 서로 다른 문서 → Lost-in-the-Middle 방어 재배열
        hits_sorted = _long_context_reorder(hits)
    else:
        hits_sorted = hits

    blocks = []
    for i, h in enumerate(hits_sorted, 1):
        피심인 = ", ".join(getattr(h, "피심인", []) or [])
        위반 = ", ".join(getattr(h, "위반유형", []) or [])
        조치 = ", ".join(getattr(h, "조치유형", []) or [])
        text = getattr(h, "text", "") or ""
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "…"
        section = getattr(h, "section", "")
        blocks.append(
            f"--- [{section}] {getattr(h, 'title', '')} ---\n"
            f"피심인: {피심인} | 위반: {위반} | 조치: {조치}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)


# ───────────────────────── llama-cpp 백엔드 ─────────────────────────

@dataclass
class LlamaConfig:
    gguf_path: str
    n_ctx:    int = 4096
    n_threads: int | None = None
    max_tokens: int = 800
    temperature: float = 0.1   # 법률 해석 일관성·환각 차단 (단계3: 0.2→0.1)
    top_p: float = 0.8         # 단계3: 결정적 샘플링
    n_gpu_layers: int = -1   # -1 = 모든 레이어를 GPU 오프로드 (CUDA wheel 시)


_LLAMA_LOCK = threading.Lock()
_LLAMA_INSTANCE = None


def _get_llama(cfg: LlamaConfig):
    """프로세스당 단일 인스턴스 (모델 로드는 무겁다)."""
    global _LLAMA_INSTANCE
    if _LLAMA_INSTANCE is not None:
        return _LLAMA_INSTANCE
    with _LLAMA_LOCK:
        if _LLAMA_INSTANCE is None:
            from llama_cpp import Llama, llama_supports_gpu_offload
            gpu_ok = bool(llama_supports_gpu_offload())
            _LLAMA_INSTANCE = Llama(
                model_path=cfg.gguf_path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.n_threads or os.cpu_count() or 4,
                n_gpu_layers=cfg.n_gpu_layers if gpu_ok else 0,
                verbose=False,
            )
    return _LLAMA_INSTANCE


def _load_config_from_env() -> LlamaConfig:
    gguf = os.environ.get("LLM_GGUF_PATH", "").strip()
    if not gguf:
        # 자동 탐색 — _PREFERRED_MODEL_PATTERNS 순서로 우선 탐색 (EXAONE 한국어 SOTA)
        all_gguf = sorted(Path("models/llm").glob("*.gguf"))
        for pat in _PREFERRED_MODEL_PATTERNS:
            for cand in all_gguf:
                if pat in cand.name.lower():
                    gguf = str(cand)
                    break
            if gguf:
                break
        if not gguf and all_gguf:
            gguf = str(all_gguf[0])
    if not gguf or not Path(gguf).exists():
        raise RuntimeError(
            "GGUF 모델 파일을 찾지 못했습니다. "
            "LLM_GGUF_PATH 환경변수 또는 models/llm/*.gguf 배치.")
    return LlamaConfig(
        gguf_path=gguf,
        n_ctx=int(os.environ.get("LLM_N_CTX", "4096")),
        n_threads=(int(os.environ["LLM_N_THREADS"])
                   if os.environ.get("LLM_N_THREADS") else None),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "800")),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
        top_p=float(os.environ.get("LLM_TOP_P", "0.8")),
        n_gpu_layers=int(os.environ.get("LLM_N_GPU_LAYERS", "-1")),
    )


_HYDE_SYSTEM = """공정거래위원회 의결서 검색 전문가.
질문에 답하는 의결서 본문 핵심 문장을 60자 이내로 생성. 피심인명·위반행위·법조항·처분 포함. 한국어만."""

_HYDE_USER = "질문: {query}\n의결서 핵심 내용 (60자 이내):"

_VARIANT_SYSTEM = """공정거래법 전문가. 주어진 질문을 의결서 검색에 유리한 법률 용어로 바꿔 표현하라.
원문과 다른 표현 1개만, 20자 이내. 한국어만."""

_VARIANT_USER = "질문: {query}\n법률 검색 표현:"


def generate_hypothetical(query: str, max_tokens: int = 60) -> str:
    """HyDE: 가상의 답변을 생성하여 dense 검색 쿼리로 사용.
    검색 정확도 향상 (의미적 매칭 강화) — vague 쿼리에 효과적.
    """
    cfg = _load_config_from_env()
    llm = _get_llama(cfg)
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": _HYDE_USER.format(query=query)},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return out["choices"][0]["message"]["content"].strip()


def generate_hypothetical_multi(query: str, n: int = 3, max_tokens: int = 80) -> list[str]:
    """Multi-query HyDE: 다른 temperature/seed 로 가상 답변 N개 생성 → RRF 다양성 ↑.
    각 답변이 서로 다른 의미 측면을 담아 검색 결과 fusion 시 보완 효과.
    """
    cfg = _load_config_from_env()
    llm = _get_llama(cfg)
    results = []
    temperatures = [0.3, 0.5, 0.7][:n]
    for temp in temperatures:
        try:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM},
                    {"role": "user", "content": _HYDE_USER.format(query=query)},
                ],
                max_tokens=max_tokens,
                temperature=temp,
                seed=int(temp * 1000),
            )
            text = out["choices"][0]["message"]["content"].strip()
            if text and text not in results:
                results.append(text)
        except Exception:
            pass
    return results


def generate_query_variants(query: str, max_tokens: int = 40) -> list[str]:
    """BM25 멀티쿼리용 법률 용어 변형쿼리 생성 (1개).
    원본 쿼리와 다른 법률 표현으로 BM25 max-fusion → 추가 청크 커버.
    """
    try:
        cfg = _load_config_from_env()
        llm = _get_llama(cfg)
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _VARIANT_SYSTEM},
                {"role": "user",   "content": _VARIANT_USER.format(query=query)},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        variant = out["choices"][0]["message"]["content"].strip()
        return [variant] if variant and variant != query else []
    except Exception:
        return []


def generate(query: str, hits: list, deadline_sec: float = 18.0) -> str:
    """
    로컬 LLM 으로 답변 생성. 모든 호출은 동일 인스턴스 사용 (모델 재로드 X).
    deadline_sec: 30 초 한도에서 안전 마진 5초.

    n_ctx 초과 시 자동으로 chunk text 를 점차 줄여 재시도 (최대 3회).
    """
    cfg = _load_config_from_env()
    llm = _get_llama(cfg)

    chars_budget = 350  # format_context 기본값
    last_err: Exception | None = None
    for attempt in range(3):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": USER_PROMPT_TEMPLATE.format(
                 query=query,
                 context=format_context(hits, max_text_chars=chars_budget))},
        ]
        try:
            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            return out["choices"][0]["message"]["content"]
        except ValueError as e:
            # llama-cpp 의 "Requested tokens (X) exceed context window of Y" 처리
            msg = str(e)
            if "exceed context window" not in msg:
                raise
            last_err = e
            chars_budget = max(120, chars_budget // 2)  # 절반으로 축소
    # 마지막에도 실패 — 컨텍스트 없이 query 만으로 시도
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"질문: {query}\n\n[참고 의결서 누락 — 자료 부족]"},
            ],
            max_tokens=80,
            temperature=cfg.temperature,
        )
        return out["choices"][0]["message"]["content"]
    except Exception:
        raise last_err if last_err else RuntimeError("LLM 호출 실패 (알 수 없는 이유)")


def is_available() -> bool:
    """기동 시 LLM 로드 가능 여부 확인 (체크용, 실제 로드는 첫 generate 에서)."""
    try:
        _ = _load_config_from_env()
        return True
    except Exception:
        return False
