#!/usr/bin/env pwsh
# LexPulse Lite 기동 — KURE-v1 + Jina Reranker + Qwen2.5-7B
# Usage: .\start_lite.ps1

$env:INDEX_DIR      = "./index_official"
$env:DENSE_INDEX_DIR  = "./index_dense_e5large"     # 1차: e5-large 1024d (fastembed ONNX)
$env:DENSE_INDEX_DIR2 = "./index_dense_mpnet"       # 2차: mpnet 768d (앙상블 RRF)
# Reranker 제거: 3-way RRF가 이미 강해 reranker가 역효과 (R@5 0.215->0.230, 속도 6.5s->2.2s)
# $env:RERANKER_MODEL  = ""
$env:LLM_GGUF_PATH   = ""                           # 빈 값 → Qwen2.5-7B 자동 탐지
$env:LLM_N_CTX       = "4096"
$env:LLM_MAX_TOKENS  = "512"
$env:LLM_N_GPU_LAYERS = "-1"
$env:ANSWER_DEADLINE_SEC = "22"                     # 30초 한도, 마진 8초
$env:HOST = "0.0.0.0"
$env:PORT = "8000"

python app_lite.py
