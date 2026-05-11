# 공정거래 의결서 AI 어시스턴트

**제2회 공정거래 AI·데이터 활용 공모전 (AI 학습모델 개발 트랙)**

공정거래위원회 의결서를 자연어로 검색하고, 근거 의결서와 함께 답변을 받을 수 있는 RAG 시스템입니다.

## 아키텍처

```
공식 의결서 ZIP (PDF + 청킹 + 메타) ─┐
                                    ├─ load_official_data.py → chunks.jsonl
자체 PDF + JSON 메타데이터 ──────────┘   (chunk_id 보존)
  ↓ build_index.py
BM25 인덱스 (kiwipiepy) + FAISS 벡터 (mpnet 다국어 768d, fastembed ONNX GPU)
  ↓ retrieval.py (Hybrid RRF + Jina Reranker v2 multilingual + doc_diversity)
Top-5 distinct docs
  ↓ local_llm.py (Qwen2.5-7B-Instruct Q4_K_M, llama-cpp CUDA, 외부 API 금지)
근거 발췌 답변
  ↓ app.py (FastAPI)
REST API (/predict 공모전 포맷, /search, /answer, /classify-violation, /predict-sanction)
```

## 공모전 트랙2 요건 매핑

| 공식 요건 | 본 시스템 |
|---|---|
| 외부 API 호출 금지 | ✅ 로컬 GGUF (urllib·SDK 둘 다 외부 호출 없음) |
| 8B 이하 LLM | ✅ Qwen2.5-7B-Instruct Q4_K_M (4.3 GB) |
| 응답 30초 이내 | ✅ 30초 deadline 마진 확보 |
| 정확히 5개 chunk_id 반환 | ✅ /predict 포맷 강제 (중복·외부 ID 검증) |
| 공식 chunk_id 보존 | ✅ load_official_data.py 가 변경 없이 통과 |
| 평가 환경 인터넷 차단 | ✅ Dockerfile 에 모델·임베딩 번들 |
| 배열 순서 = ranking | ✅ RRF score 내림차순 정렬 |

## 제출 워크플로우 (구현물 마감: 2026-05-26 14:00 KST)

```powershell
# 1. 공식 ZIP 압축 해제 → 어댑터로 변환
python src/load_official_data.py data/official chunks_official.jsonl

# 2. BM25 인덱스 빌드 (kiwipiepy 형태소)
python src/build_index.py chunks_official.jsonl index_official --model nlpai-lab/KURE-v1

# 3. 한국어 SOTA Dense 인덱스 빌드 (KURE-v1, bge-m3 fine-tune, GPU 권장)
python src/build_dense_st.py chunks_official.jsonl index_dense_kure --model nlpai-lab/KURE-v1 --batch 16

# 4. 서버 기동 (EXAONE-3.5-7.8B 자동 탐지, KURE Dense 사용)
.\start_lite.ps1
# 브라우저 http://localhost:8000/  → 시연

# 5. Docker 이미지 빌드 (평가 환경 호환)
docker build -t kftc-track2:submit .
docker save -o kftc-track2.tar kftc-track2:submit
# kftc-track2.tar 를 ftcdatacontest@korea.kr 로 전송
```

**구현물 점검 체크리스트**
- [ ] /predict 가 정확히 5 chunk_id 반환 (`curl http://localhost:8000/predict -d '{"query":"..."}'` 로 확인)
- [ ] chunk_id 모두 공식 corpus 에 존재 (재청킹·재명명 안 했는지)
- [ ] 응답 시간 < 30초 (warmup 포함)
- [ ] Docker 이미지에 인터넷 fetch 코드 없음 (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 설정됨)
- [ ] LLM 외부 API 코드 제거 (`google.generativeai`, `groq` 등 미사용)

## 1. 환경 셋업

### 권장 환경
- Python 3.10 이상
- 메모리 8GB 이상 (임베딩 단계에서 사용)
- 디스크 5GB 이상 여유 (인덱스 + 모델 캐시)

### 가상환경 생성

**Linux / macOS (bash):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 실행 정책 오류가 나면: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

## 2. 데이터 준비

대회에서 제공한 의결서 데이터(243MB)를 다운로드 받아 `data/` 폴더에 압축 해제합니다.

```
data/
├─ <사건명>.pdf
├─ <사건명>_metadata.json
├─ ...
```

### 2-1. 샘플 데이터로 빠르게 시험하기 (실데이터가 없을 때)

대회 데이터가 아직 없으면 `src/make_sample_data.py`로 합성 의결서 PDF·JSON
3건을 생성해 전체 파이프라인을 검증할 수 있습니다.

```powershell
python src/make_sample_data.py data
```

생성물: `data/sample_2024-001.pdf` 외 PDF 3개, 동명 `_metadata.json` 3개.
이후 3-1 단계부터 동일하게 실행하면 됩니다.

## 3. 단계별 실행

> 아래 명령은 **프로젝트 루트(`Kftc/`)에서 실행**하는 것을 가정합니다.

### 3-1. 전처리 (PDF + JSON → 청크)
```powershell
python src/preprocess.py data chunks.jsonl
```
출력: `chunks.jsonl` — 청크 단위 JSONL

### 3-2. 인덱스 구축 (BM25 + FAISS)

**기본 (사전학습 임베딩):**
```powershell
python src/build_index.py chunks.jsonl index
```

**fine-tuned 임베딩 사용 (3-7 단계 후):**
```powershell
python src/build_index.py chunks.jsonl index --model models/embedding_ft
```

출력: `index/bm25.pkl`, `index/faiss.bin`, `index/meta.jsonl`, `index/model_name.txt`

### 3-3. 검색 단독 테스트
```powershell
python src/retrieval.py index "다단계판매업자 미등록 위반"
```

### 3-4. 로컬 LLM 다운로드 (외부 API 금지 → 로컬 GGUF 필수)

```powershell
# 1) Qwen2.5-7B-Instruct Q4_K_M (4.47 GB)
$Url  = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
$Dest = "C:\lsc\Kftc\models\llm\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
New-Item -ItemType Directory -Path C:\lsc\Kftc\models\llm -Force | Out-Null
Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing

# 2) llama-cpp-python 설치 (CPU wheel)
pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python

# 3) 환경변수 설정
$env:LLM_GGUF_PATH = $Dest
```

> 다른 8B 이하 모델 사용 시 `LLM_GGUF_PATH` 만 교체하면 됨.
> Llama 시리즈는 사용자 정책상 미사용.

### 3-5. 학습 파이프라인 (대회 트랙: AI 학습모델 개발)

| 모델 | 역할 | 베이스 | 데이터 |
|---|---|---|---|
| Embedding (도메인 적응) | 의결서 검색 정확도 ↑ | intfloat/multilingual-e5-small | 자가합성 (질의↔본문) 360 페어 |
| 위반유형 분류기 | 임의 사실관계 → 위반유형 멀티라벨 | klue/roberta-base | 청크 본문 + 메타데이터 라벨 |
| 조치유형 예측기 | 사실관계 → 처분 카테고리 단일라벨 | klue/roberta-base | 의결서 단위 |
| Re-ranker (CrossEncoder) | RRF 후 정밀 재정렬 | BAAI/bge-reranker-base | (질의, 정답, hard negative) 트리플 |

#### 3-5-1. 학습용 페어 합성
```powershell
python src/synthetic_qa.py chunks.jsonl pairs.jsonl
```

#### 3-5-2. 임베딩 도메인 fine-tuning
```powershell
python src/finetune_embedding.py pairs.jsonl models/embedding_ft `
    --base intfloat/multilingual-e5-small --epochs 3 --batch-size 8
```

#### 3-5-3. 위반유형 멀티라벨 분류기
```powershell
python src/train_classifier.py chunks.jsonl models/violation_clf `
    --base klue/roberta-base --epochs 5
```

#### 3-5-4. 조치유형 예측기
```powershell
python src/train_sanction.py chunks.jsonl models/sanction_clf `
    --base klue/roberta-base --epochs 8
```

#### 3-5-5. Cross-encoder Re-ranker
```powershell
python src/train_reranker.py chunks.jsonl pairs.jsonl models/reranker `
    --base BAAI/bge-reranker-base --epochs 2
```

### 3-6. 공식 데이터로 전환 (대회 ZIP 사용 시)

```powershell
python src/load_official_data.py data/official chunks_official.jsonl
python src/build_index.py chunks_official.jsonl index --model models/embedding_ft
```

**중요**: 공식 chunk_id 는 절대 변경하지 않습니다 (변경 시 자동 실격).

### 3-7. API 서버 기동

```powershell
python app.py
# 또는
uvicorn app:app --host 0.0.0.0 --port 8000
```

기본 바인딩: `http://localhost:8000`

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/`        | GET  | 프론트엔드 (vanilla HTML/CSS/JS) |
| `/static/*`| GET  | 정적 자원 |
| `/health`  | GET  | 서비스 상태 + 인덱스 통계 + 학습된 모델 로드 여부 |
| `/search`  | POST | 하이브리드 검색만 (LLM 호출 없음) |
| `/answer`  | POST | 검색 + LLM 답변 + 근거 인용 |
| `/classify-violation` | POST | 위반유형 멀티라벨 분류기 추론 |
| `/predict-sanction`   | POST | 조치유형 예측기 추론 |
| `/predict` | POST | 공모전 제출 포맷 (chunk_id 5개 반환) |
| `/docs`    | GET  | Swagger UI |

#### 호출 예시 (PowerShell)

```powershell
# 헬스체크
curl http://localhost:8000/health

# 검색만
$body = @{ query = "다단계판매업자 미등록"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/search -Method POST `
    -ContentType "application/json; charset=utf-8" -Body $body

# 공모전 포맷 예측
$body = @{ query = "프랜차이즈 본사가 물품 구매를 강요하면 위법인가요?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST `
    -ContentType "application/json; charset=utf-8" -Body $body
```

## 4. 운영 배포

### 4-1. Docker

```powershell
docker build -t kftc-track2:submit .
docker run -p 8000:8000 kftc-track2:submit
```

### 4-2. systemd 서비스 (Linux)

```ini
[Unit]
Description=KFTC RAG API
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/kftc
EnvironmentFile=/opt/kftc/.env
ExecStart=/opt/kftc/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## 5. 주요 모듈 요약

| 파일 | 역할 |
|---|---|
| `src/load_official_data.py` | 공식 ZIP → chunk_id 보존 JSONL 변환 |
| `src/build_index.py` | BM25 + FAISS 인덱스 구축 |
| `src/build_dense_st.py` | SentenceTransformer 기반 Dense 인덱스 |
| `src/retrieval.py` | Hybrid 검색 + RRF + Reranker |
| `src/local_llm.py` | 로컬 GGUF LLM 추론 (외부 API 없음) |
| `src/synthetic_qa.py` | 학습용 (질의, 정답청크) 페어 합성 |
| `src/finetune_embedding.py` | 임베딩 도메인 fine-tuning |
| `src/train_classifier.py` | 위반유형 멀티라벨 분류기 학습 |
| `src/train_reranker.py` | Cross-encoder re-ranker 학습 |
| `app.py` / `app_lite.py` | FastAPI 서버 |
| `static/index.html` | 프론트엔드 진입점 |

## 6. 고도화 아이디어

- **Re-ranker 고도화** — bge-reranker-v2-m3 로 Top-30 → Top-5 정밀 재정렬
- **유사 사건 추천** — 의결서 임베딩 간 유사도로 "이 사건과 비슷한 다른 의결서" 기능
- **시계열 통계** — 업종별·기간별 심결 추이 시각화
- **답변 캐싱** — 같은 질의 반복 시 Redis로 응답 속도 개선

---

본 시스템은 BM25 + Dense + RRF 하이브리드 검색과, 근거 문단 인용을 강제하는
설명가능한 RAG(Explainable RAG) 구조를 결합하여 법률 AI의 환각 위험을 구조적으로
억제하는 것을 차별점으로 합니다.
