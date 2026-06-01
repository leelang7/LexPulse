#!/usr/bin/env pwsh
# LexPulse Lite 기동 — KURE-v1 + Jina Reranker + Qwen2.5-7B
# Usage: .\start_lite.ps1

$env:INDEX_DIR      = "./index_official"
$env:DENSE_INDEX_DIR = "./index_dense_kure"         # KURE-v1 768→1024d 한국어 SOTA
$env:RERANKER_MODEL  = "jinaai/jina-reranker-v2-base-multilingual"  # CPU 재랭커
$env:LLM_GGUF_PATH   = ""                           # 빈 값 → Qwen2.5-7B 자동 탐지
$env:LLM_N_CTX       = "4096"
$env:LLM_MAX_TOKENS  = "512"
$env:LLM_N_GPU_LAYERS = "-1"
$env:ANSWER_DEADLINE_SEC = "22"                     # 30초 한도, 마진 8초
$env:HOST = "0.0.0.0"
$env:PORT = "8000"

python app_lite.py
