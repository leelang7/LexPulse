# 평가 환경 호환 Docker 이미지 (트랙2 제출용)
# - 인터넷 차단 오프라인 동작 (빌드 시 모델 사전 다운로드)
# - 외부 LLM API 호출 금지 (공모전 트랙2 요건)
# - LLM: Qwen2.5-7B-Instruct Q4_K_M GGUF (로컬 번들)
# - Dense: KURE-v1 (sentence-transformers, 빌드 시 다운로드)
# - Reranker: jina-reranker-v2-base-multilingual (fastembed ONNX, 빌드 시 다운로드)

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libgomp1 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성
COPY requirements.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-llm.txt \
 && pip install --no-cache-dir kiwipiepy fastembed sentence-transformers \
 && pip install --no-cache-dir \
        nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12

# 모델 사전 다운로드 (빌드 시 → 런타임 오프라인 동작)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nlpai-lab/KURE-v1')"
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('jinaai/jina-reranker-v2-base-multilingual')"

# 코드
COPY src/     ./src/
COPY static/  ./static/
COPY app_lite.py ./

# 인덱스 번들
COPY index_official/   ./index_official/
COPY index_dense_kure/ ./index_dense_kure/

# LLM (Qwen2.5-7B만 번들, ~4.3GB)
COPY models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf ./models/llm/

ENV INDEX_DIR=/app/index_official \
    DENSE_INDEX_DIR=/app/index_dense_kure \
    RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual \
    LLM_GGUF_PATH=/app/models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    LLM_N_CTX=4096 \
    LLM_MAX_TOKENS=512 \
    LLM_N_GPU_LAYERS=-1 \
    ANSWER_DEADLINE_SEC=22 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
CMD ["python", "app_lite.py"]
