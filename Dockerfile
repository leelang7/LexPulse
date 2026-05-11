# 평가 환경 호환 Docker 이미지 (트랙2 제출용)
# - 인터넷 차단된 채점 환경에서도 동작하도록 모델·임베딩 모두 번들
# - 외부 LLM API 호출 금지 (공모전 트랙2 요건)
# - 8B 이하 GGUF 로컬 LLM (Qwen2.5-7B-Instruct Q4_K_M)
# - Dense 임베딩: mpnet 다국어 (fastembed ONNX, GPU)
# - Reranker: jina-reranker-v2-base-multilingual (fastembed ONNX, CPU)
# - GPU 자동 사용 (llama-cpp-python CUDA + onnxruntime CUDA)

FROM python:3.12-slim

# 시스템 의존성: BLAS, libgomp, 한국어 폰트
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libgomp1 libopenblas-dev \
        fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성
COPY requirements.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-llm.txt \
 && pip install --no-cache-dir kiwipiepy fastembed \
 && pip install --no-cache-dir nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12

# 코드
COPY src/   ./src/
COPY static/ ./static/
COPY app.py app_lite.py ./

# 번들 자산
COPY index_official/    ./index_official/
COPY index_dense_kure/  ./index_dense_kure/
COPY models/            ./models/
COPY chunks_official.jsonl ./

# 환경변수 — 기본값
ENV INDEX_DIR=/app/index_official \
    DENSE_INDEX_DIR=/app/index_dense_kure \
    LLM_GGUF_PATH=/app/models/llm/EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf \
    LLM_N_CTX=4096 \
    LLM_MAX_TOKENS=200 \
    LLM_N_GPU_LAYERS=-1 \
    ANSWER_DEADLINE_SEC=28 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# 진입점 선택:
# - lite (기본): BM25 + fastembed Dense + RRF + Qwen GPU. torch 비의존, 안전.
# - full (옵션): app.py 사용. fine-tuned e5-small + reranker + Qwen.
#   docker run -e MODE=full ... 로 전환 가능 (CMD override).
CMD ["python", "app_lite.py"]
