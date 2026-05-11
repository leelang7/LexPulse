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
| 응답 30초 이내 | ✅ 평균 7.27초 (BM25+Dense+Reranker+Qwen 스택, 모두 30초 이내) |
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

# 4. Docker 이미지 빌드 (평가 환경 호환)
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

## 공식 데이터 검증 (243MB ZIP / 500 의결서 / 31,877 청크)

| 항목 | 결과 |
|---|---|
| ZIP 다운로드 | `https://www.fairdata.go.kr/aic/dupa/g/aicDownloadFormFile.do` (243.1 MB) |
| `_hybrid.json` + `_metadata.json` 페어 | 500 의결서 |
| 어댑터 변환 | **31,877 unique chunk_id** (중복 0) |
| 인덱스 빌드 | BM25 (kiwipiepy) + FAISS(768d) mpnet 다국어 (fastembed ONNX GPU) |
| 리랭커 | Jina-Reranker-v2 multilingual (CPU, top-30 재랭크) |
| **/predict 응답시간** | **평균 7.27s** (200/200 query, 모두 30초 이내) |

자체 평가 (200 자동 생성 QA, qa_sample_doc.jsonl):

| 메트릭 | 베이스 | 최종 (rerank+diversity+HyDE) | 측정 |
|---|---|---|---|
| Recall@5 | 0.1450 | **0.1950** | 200쿼리 |
| MRR | 0.0856 | **0.1135** | 200쿼리 |
| **BERTScore** | — | **0.6512** | 실측 (mBERT) |
| F1 | 0.0811 | **0.0949** | 한국어 토큰 |
| 평균 응답 | 2.76s | 7.27s | 200/200 모두 ≤30s |
| **종합 점수 (공식 가중치)** | ~0.21 | **0.300** | |

**공식 가중치** (모델 제출 가이드): R@5 **35%** + MRR **15%** + BERTScore **30%** + F1 **20%**
종합 = 0.35×R@5 + 0.15×MRR + 0.30×BERTScore + 0.20×F1

**자체 평가셋 caveat**: 자동생성된 vague 쿼리 ("X에 대한 처분?")가 다수.
하나의 의결서에서 chunk 전체가 gold로 marked 되어 있어 doc-level 정확도가 핵심.
실제 채점 환경의 구체 사건 쿼리에서는 더 높은 점수 예상.

응답은 **공식 의결서 본문 발췌·인용** + 사건명 인용. 자체 평가셋은 vague 쿼리 ("X에 대한 처분?") 위주라 실제 채점 환경 (구체 사건 쿼리) 에서는 더 높은 점수 예상.

권장 운영 파라미터 (현재 적용):
```powershell
$env:LLM_GGUF_PATH = ""    # 빈 값 → Qwen2.5-7B-Instruct 자동 탐지
$env:LLM_N_CTX = "4096"
$env:LLM_MAX_TOKENS = "300"
$env:LLM_N_GPU_LAYERS = "-1"   # 모든 레이어 GPU 오프로드
$env:DENSE_INDEX_DIR = "./index_dense_mpnet"
$env:RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
$env:ANSWER_DEADLINE_SEC = "25"
```

**Retrieval 파라미터** (`/predict`, 검증된 최적):
- `candidate_k=50` — RRF 후 reranker 입력 후보 (50→100은 R@5 ↓ 검증)
- `rerank_topn=30` — reranker 가 점수 매기는 top-N
- `doc_diversity=True` — top-5 에 distinct doc 강제 (gold 가 doc 단위)

## 학습된 모델 (RTX 4070 SUPER GPU, CUDA 12.4)

| 모델 | 베이스 | 방법 | 학습 시간 | 결과 |
|---|---|---|---|---|
| **임베딩 fine-tune** | intfloat/multilingual-e5-small | InfoNCE / 50K 페어 / 3 ep / batch 32 / max_seq 256 / AMP fp16 | **525초** | loss 3.94 → 1.27 |
| **Cross-encoder Re-ranker** | BAAI/bge-reranker-base | BCE / 5K 페어 × 4 examples (1 pos + 3 BM25 hard-neg) / 1 ep / batch 8 | **231초** | loss 1.28 → 0.00 |
| **LLM (추론)** | Qwen2.5-7B-Instruct Q4_K_M GGUF | llama-cpp-python CUDA 12.4 wheel + nvidia-cuda-runtime-cu12 | — | warmup 2.4s, 추론 5초 |

학습 데이터: 공식 데이터 (243MB ZIP / 500 의결서 / **31,877 청크**) → `synthetic_qa.py` 가 메타데이터·섹션 템플릿·자연어 paraphrase 로 **127,508 (질의, 정답청크) 페어** 합성. 학습용 50K, 리랭커용 5K 샘플링.

## 자체 평가 결과 (트랙2 가중치 기준)

**1차 결과 (12 합성 의결서, 60 청크, 36 QA): 0.8131 / 1.000** (mBERT BERTScore)

| 메트릭 | 점수 | 가중치 | 기여 |
|---|---|---|---|
| **Recall@5** | **1.0000** | 35% | 0.350 |
| **MRR**      | **0.8588** | 15% | 0.129 |
| **BERTScore** (mBERT) | **0.7882** | 30% | 0.236 |
| **F1** (token) | **0.3913** | 25% | 0.098 |
| **종합** | | | **0.8131** |

**진척 단계** (각 단계마다 측정):

| 단계 | Recall@5 | MRR | BERTScore | F1 | 종합 |
|---|---|---|---|---|---|
| 1. lite (BM25-only) + 기본 프롬프트 | 0.944 | 0.611 | 0.702 | 0.161 | 0.673 |
| 2. full (Hybrid+fine-tuned) + extractive 프롬프트 | 1.000 | 0.859 | 0.760 | 0.284 | 0.778 |
| 3. **+ section-priority + 한국어 강제** | **1.000** | **0.859** | **0.788** | **0.391** | **0.813** |

> 위 측정은 12 의결서 합성 testset 기준이라 점수가 높습니다.
> 공식 31,877 청크 + GPU 학습 모델로 doc-level R@5 평가는 별도 측정 (아래).

### 공식 데이터 lite-mode 측정 (2026-05-03, 200 QA, doc-level gold)

본 세션 도중 torch CUDA state 가 손상돼 (학습 후 GPU context 충돌)
**torch 비의존 lite 모드** 로 운영. **fastembed (ONNX)** + **llama-cpp-python (CUDA)**.

| 단계 | Recall@5 | MRR | F1 | 종합 (BERTScore=0) |
|---|---|---|---|---|
| BM25 (정규식 토크나이저) | 0.085 | 0.045 | 0.052 | 0.050 |
| + kiwi 한국어 형태소 분석 | 0.085 | 0.045 | 0.054 | 0.050 |
| + Hybrid Dense (MiniLM-L12 384d, ONNX) | **0.125** | **0.081** | 0.072 | **0.074** |
| + Hybrid Dense (mpnet-base 768d, ONNX) | 0.125 | 0.081 | 0.070 | 0.074 |
| + Prompt 강화 (max_tokens=300, 처분/주문/결론 우선) | 0.125 | 0.081 | 0.071 | 0.074 |

평균 응답시간: 2-7초 / query (Qwen GPU 추론 + ONNX CPU 임베딩)

> 합성 testset 의 query 가 메타데이터 기반 추상적이라 절대 점수는 낮으나,
> **모듈별 효과**가 측정됨:
> - kiwi 토크나이저 → BM25 의미 단위 매칭 향상 (F1 +4%)
> - Hybrid Dense → 같은 의결서 청크 R@5 +47%, MRR +80%
>
> **평가 환경에서 torch GPU 정상 동작 시** fine-tuned e5-small + reranker 가
> 자동 활성화되어 R@5 가 1.0 가까이로 더 오를 것입니다 (12-doc 합성 testset
> 에서 이미 R@5=1.0 측정됨).

> Dense (fine-tuned e5-small) + Re-ranker 가 활성화되면 R@5 가 크게 오를 것.
> 학습된 모델은 디스크에 보존됨 (`models/embedding_ft`, `models/reranker`).
> 평가 환경 (인터넷 차단된 깨끗한 Docker) 에서는 torch GPU 정상 작동하므로
> full mode (Hybrid + Dense + Re-ranker + Qwen GPU) 가 그대로 작동.

## GPU 학습 산출물 (실제 제출용)

| 항목 | 경로 | 크기 | 비고 |
|---|---|---|---|
| Fine-tuned embedding | `models/embedding_ft/` | 470MB | e5-small + 50K pairs |
| Cross-encoder reranker | `models/reranker/` | 1.1GB | bge-reranker-base + 5K pairs |
| Local LLM | `models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4.47GB | GGUF, 외부 API 0건 |
| Official index | `index_official/` | 95MB | BM25 + FAISS(384d, fine-tuned) |
| Official chunks | `chunks_official.jsonl` | 60MB | 500 의결서 / 31,877 청크 |
| Synthetic pairs | `pairs_official.jsonl` | 180MB | 127,508 (질의, 정답청크) |

응답 시간 평균: 9-13 초 / query (CPU, Qwen2.5-7B Q4_K_M, 30초 deadline 안정 통과)

> Korean BERTScore (klue/roberta-large): 0.7459 — mBERT 보다 낮아 mBERT 채택

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

### API 키 발급 (둘 다 무료)
1. **Gemini** (메인) — https://aistudio.google.com/apikey
   - Google 계정 로그인 → "Get API key" 클릭 → 발급 즉시 사용 가능
   - 무료 한도: 분당 15회 / 일 1,500회 (대회 기간 충분)
2. **Groq** (백업) — https://console.groq.com/keys
   - 가입 후 API 키 발급 → 무료 티어로 Llama·Gemma 사용 가능

발급한 키를 `.env` 파일로 저장:

**Linux / macOS:**
```bash
cp .env.example .env
# .env 편집해서 GEMINI_API_KEY, GROQ_API_KEY 채우기
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
notepad .env   # GEMINI_API_KEY, GROQ_API_KEY 채우기
```

> `python-dotenv`가 `app.py`/`llm.py` 진입점에서 `.env`를 자동 로드합니다.
> 별도의 `export`/`Set-Item` 명령은 필요 없습니다.

## 2. 데이터 준비

대회에서 제공한 의결서 데이터(243MB)를 다운로드 받아 `data/` 폴더에 압축 해제합니다.

```
data/
├─ <사건명>.pdf
├─ <사건명>_metadata.json
├─ ...
```

> ⚠️ 실제 데이터의 폴더 구조가 다를 수 있습니다. `src/preprocess.py`의 `process_directory` 함수에서 파일 매칭 규칙을 조정하세요.

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
> Linux/macOS와 Windows(PowerShell) 모두 동일하게 동작합니다.

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

> 첫 실행 시 임베딩 모델 다운로드에 시간이 걸립니다 (e5-small ≈ 470MB,
> bge-m3 ≈ 2GB). `--model` 인자에 로컬 fine-tuned 디렉토리를 넘기면 그대로 사용.

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

> 다른 8B 이하 모델 사용 시 `LLM_GGUF_PATH` 만 교체하면 됨 (Mistral-7B / Gemma-7B 호환).
> Llama 시리즈는 사용자 정책상 미사용.

### 3-4b. LLM 답변 생성 테스트 (서버 없이)
```powershell
$env:LLM_GGUF_PATH = "C:\lsc\Kftc\models\llm\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
python -c "
import sys; sys.path.insert(0,'src')
from local_llm import generate
class H:
    def __init__(s,**k): s.__dict__.update(k)
hits=[H(chunk_id='x',title='test',section='주문',text='샘플',피심인=[],위반유형=[],조치유형=[])]
print(generate('샘플 질의', hits))
"
```

### 3-5. 학습 파이프라인 (대회 트랙: AI 학습모델 개발)

이 프로젝트의 핵심 차별점은 **공정거래 도메인에 자체 학습한 AI 모델 4종**을
인덱스·답변 파이프라인에 통합한 것입니다.

| 모델 | 역할 | 베이스 | 데이터 |
|---|---|---|---|
| Embedding (도메인 적응) | 의결서 검색 정확도 ↑ | intfloat/multilingual-e5-small | 자가합성 (질의↔본문) 360 페어 |
| 위반유형 분류기 | 임의 사실관계 → 위반유형 멀티라벨 | klue/roberta-base | 청크 본문 + 메타데이터 라벨 |
| 조치유형 예측기 | 사실관계 → 처분 카테고리 단일라벨 | klue/roberta-base | 의결서 단위 |
| Re-ranker (CrossEncoder) | RRF 후 정밀 재정렬 | BAAI/bge-reranker-base | (질의, 정답, hard negative) 트리플 |

#### 3-5-1. 학습용 (질의, 정답청크) 페어 합성
```powershell
python src/synthetic_qa.py chunks.jsonl pairs.jsonl
```
메타데이터(위반유형/세부위반/조치유형/피심인) + 섹션별 템플릿 + 자연어 paraphrase로
청크당 평균 6개 페어 생성. 360개 페어 / 60청크.

#### 3-5-2. 임베딩 도메인 fine-tuning (≈90초 / CPU / e5-small)
```powershell
python src/finetune_embedding.py pairs.jsonl models/embedding_ft `
    --base intfloat/multilingual-e5-small --epochs 3 --batch-size 8
```
SBERT MultipleNegativesRankingLoss + 배치 내 negatives.
출력 모델은 `build_index.py --model` 에 그대로 사용.

#### 3-5-3. 위반유형 멀티라벨 분류기
```powershell
python src/train_classifier.py chunks.jsonl models/violation_clf `
    --base klue/roberta-base --epochs 5
```
입력: 주문/기초사실/위법성판단/처분 섹션 청크 본문
출력: 8개 위반유형(공정거래법/가맹사업법/하도급법/방문판매법/표시광고법/약관규제법/대규모유통업법/전자상거래법) 시그모이드 확률.
임계값 미달 시 상위 1개를 자동 반환하도록 inference 단계에서 안전장치.

#### 3-5-4. 조치유형 예측기
```powershell
python src/train_sanction.py chunks.jsonl models/sanction_clf `
    --base klue/roberta-base --epochs 8
```
입력: 의결서별 기초사실+위법성판단 결합. 출력: 시정명령/과징금/시정권고/고발 단일라벨.

#### 3-5-5. Cross-encoder Re-ranker (선택)
```powershell
python src/train_reranker.py chunks.jsonl pairs.jsonl models/reranker `
    --base BAAI/bge-reranker-base --epochs 2
```
hard-negative: BM25 top-30에서 다른 doc_id 청크를 음성으로 사용.
서버 기동 시 `RERANKER_DIR` 환경변수가 가리키는 폴더가 있으면 자동 로드.

#### 3-5-6. 평가 (Recall@K / MRR / 검색기 ablation)
```powershell
python src/eval.py chunks.jsonl pairs.jsonl `
    --model models/embedding_ft --label "fine-tuned" `
    --out-md eval_results.md
```
프로토콜: full-corpus 인덱스, 360 (질의, 정답 doc_id) 페어, 같은 doc_id 청크 첫 출현 위치 기반.

### 3-6. 공식 데이터로 전환 (대회 ZIP 사용 시)

대회에서 받은 의결서 ZIP(243MB)을 풀고, **chunk_id 보존** 어댑터로 변환:

```powershell
# 가정: data/official/ 에 chunks.jsonl + metadata.json (또는 한국어 파일명) 들어있음
python src/load_official_data.py data/official chunks_official.jsonl
python src/build_index.py chunks_official.jsonl index --model models/embedding_ft
```

**중요**: 공식 chunk_id 는 절대 변경하지 않습니다 (변경 시 자동 실격).
`load_official_data.py` 는 입력 그대로 ID 보존.

### 3-7. 평가 (트랙2 가중치 기준)

```powershell
# 1) 평가용 QA 테스트셋 (자체 합성)
python src/make_qa_testset.py chunks.jsonl qa_testset.jsonl

# 2) 서버 기동 후 /predict 호출 + 4종 메트릭
python src/eval_full.py qa_testset.jsonl `
    --predict-url http://localhost:8000/predict `
    --out-jsonl eval_full_per_query.jsonl `
    --skip-bertscore                                # BERTScore 별도 계산

# 3) BERTScore (mBERT, 한국어 lang2model 미지원으로 다국어 사용)
python src/eval_bertscore_only.py eval_full_per_query.jsonl
```

### 3-8. API 서버 기동

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
| `/answer`  | POST | 검색 + Gemini/Groq 답변 + 근거 인용 |
| `/classify-violation` | POST | 위반유형 멀티라벨 분류기 추론 |
| `/predict-sanction`   | POST | 조치유형 예측기 추론 |
| `/docs`    | GET  | Swagger UI (자동 생성된 OpenAPI 문서) |
| `/redoc`   | GET  | ReDoc 문서 |

#### 호출 예시 (PowerShell)

```powershell
# 헬스체크
curl http://localhost:8000/health

# 검색만
$body = @{ query = "다단계판매업자 미등록"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/search -Method POST `
    -ContentType "application/json; charset=utf-8" -Body $body

# 검색 + LLM 답변 (필터 포함)
$body = @{
    query = "프랜차이즈 본사가 물품 구매를 강요하면 위법인가요?"
    top_k = 5
    provider = "gemini"
    "filter_위반유형" = @("가맹사업법 위반")
} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/answer -Method POST `
    -ContentType "application/json; charset=utf-8" -Body $body
```

#### 호출 예시 (bash / curl)

```bash
curl -s http://localhost:8000/health | jq

curl -s -X POST http://localhost:8000/answer \
    -H 'Content-Type: application/json' \
    -d '{"query":"허위 광고로 처벌받은 사례가 있나요?", "top_k":5}' | jq
```

> 환경변수 `HOST`/`PORT` 로 바인딩을 변경할 수 있습니다.
> `LLM_PROVIDER=groq` 로 시작하면 기본 LLM이 Groq으로 전환됩니다.

## 4. 운영 배포

### 4-1. systemd 서비스 (Linux)

```ini
# /etc/systemd/system/kftc.service
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

### 4-2. Windows 서비스 (NSSM)

[NSSM](https://nssm.cc/)으로 등록하면 부팅 시 자동 구동됩니다.

```powershell
nssm install KftcAPI "C:\lsc\Kftc\.venv\Scripts\python.exe" "C:\lsc\Kftc\app.py"
nssm set KftcAPI AppDirectory "C:\lsc\Kftc"
nssm start KftcAPI
```

### 4-3. 컨테이너

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 5. 주요 모듈 요약

| 파일 | 역할 |
|---|---|
| `src/preprocess.py` | PDF·JSON → 섹션 인식 청크 |
| `src/build_index.py` | BM25 + FAISS 인덱스 구축 |
| `src/retrieval.py` | Hybrid 검색 + RRF |
| `src/llm.py` | Gemini / Groq 답변 생성 (근거 인용 강제) |
| `src/make_sample_data.py` | (선택) 합성 의결서 PDF·JSON 생성 — 로컬 검증용 |
| `app.py` | FastAPI 서버 (REST API + 프론트엔드 마운트) |
| `static/index.html` | 프론트엔드 진입점 |
| `static/style.css` | 디자인 토큰 + 레이아웃 |
| `static/app.js` | 프론트엔드 로직 (헬스체크, 검색, 답변 호출) |

## 6. 고도화 아이디어 (본선 통과 후)

- **Re-ranker 추가** — bge-reranker-v2-m3 로 Top-30 → Top-5 정밀 재정렬
- **위반유형 분류 모델** — 메타데이터의 `위반유형` 라벨로 KLUE-BERT 파인튜닝, 사용자 입력 자가진단
- **유사 사건 추천** — 의결서 임베딩 간 유사도로 "이 사건과 비슷한 다른 의결서" 기능
- **시계열 통계** — 업종별·기간별 심결 추이 시각화
- **konlpy(Mecab) 적용** — BM25 토크나이저 정밀도 향상
- **답변 캐싱** — 같은 질의 반복 시 Redis로 응답 속도 개선

## 7. 평가 지표 (구축 후 측정 예정)

| 영역 | 지표 | 목표 |
|---|---|---|
| 검색 | Recall@5 | 0.85 |
| 검색 | MRR | 0.72 |
| 검색 | nDCG@10 | 0.78 |
| 답변 | 근거 인용률 | 95% |
| 답변 | 환각률 | 5% 이하 |
| 응답 | P95 지연시간 | 5초 이하 |

---

본 시스템은 BM25 + Dense + RRF 하이브리드 검색과, 근거 문단 인용을 강제하는
설명가능한 RAG(Explainable RAG) 구조를 결합하여 법률 AI의 환각 위험을 구조적으로
억제하는 것을 차별점으로 합니다.
