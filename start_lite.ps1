#!/usr/bin/env pwsh
# Lite 모드 기동 — EXAONE-3.5-7.8B + KURE-v1 한국어 SOTA 스택
# Usage: .\start_lite.ps1

$env:INDEX_DIR = "./index_official"
# fastembed mpnet (ONNX, GPU CUDA) — torch 비의존이라 안정적
$env:DENSE_INDEX_DIR = "./index_dense_mpnet"
# 한국어 cross-encoder 재랭커 (multilingual, CPU — GPU 충돌 회피)
$env:RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
# Qwen2.5-7B-Instruct Q4_K_M (사용자 정책: Qwen 우선)
$env:LLM_GGUF_PATH = ""     # 빈 값이면 local_llm.py 가 Qwen 자동 탐지
# HyDE: Qwen 으로 가상 답변 생성 → dense 검색 보강 (vague 쿼리에 효과적, MRR +10%)
$env:USE_HYDE = "1"
$env:DOC_AGGREGATE = "0"   # 자체 평가셋에서 R@5 ↓ → 비활성
$env:LLM_N_CTX = "4096"
$env:LLM_MAX_TOKENS = "300"
$env:LLM_N_GPU_LAYERS = "-1"
$env:ANSWER_DEADLINE_SEC = "18"   # 가이드 20s 한도 마진 2초
$env:HOST = "0.0.0.0"
$env:PORT = "8000"

python app_lite.py
