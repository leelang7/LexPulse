"""
인덱스 구축 모듈
- chunks.jsonl을 읽어 BM25 인덱스와 FAISS 벡터 인덱스를 함께 구축
- 두 인덱스는 같은 청크 순서를 공유하므로, 검색 결과는 인덱스 정수로 통합 관리

사용법:
    python build_index.py chunks.jsonl ../index/

생성물:
    index/bm25.pkl       BM25 인덱스 + tokenized corpus
    index/faiss.bin      FAISS 벡터 인덱스
    index/meta.jsonl     청크 본문·메타데이터 (검색 시 결과 복원용)
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Iterable

# sentence_transformers 를 torch 보다 먼저 임포트 (5.x 호환성 — segfault 회피)
from sentence_transformers import SentenceTransformer

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ───────────────────────── 한글 토크나이저 (BM25용) ─────────────────────────
# kiwipiepy 형태소 분석으로 합성어 분리 → BM25 매칭 정밀도 향상.
# kiwipiepy 미설치 시 정규식 fallback.

from korean_tok import korean_tokenize  # noqa: F401  (다른 모듈이 import)


# ───────────────────────── 임베딩 모델 ─────────────────────────
# 한국어 법률 텍스트 성능과 다국어 호환성을 모두 고려해 bge-m3 를 기본으로 사용.
# 메모리·속도가 부담이면 BM-K/KoSimCSE-roberta 로 교체 가능.

DEFAULT_MODEL = "BAAI/bge-m3"


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    HuggingFace 모델 ID 또는 로컬 디렉토리 경로 모두 지원.
    로컬 fine-tuned 모델을 그대로 넘기면 자동 인식. GPU 우선.
    """
    import torch as _torch
    from pathlib import Path as _P
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"[임베딩 모델 로드] {model_name}  (device={device})")
    if _P(model_name).exists():
        m = SentenceTransformer(str(_P(model_name).resolve()), device=device)
    else:
        m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = 256
    return m


# ───────────────────────── 청크 로더 ─────────────────────────

def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


# ───────────────────────── 인덱스 구축 ─────────────────────────

def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[list[str]]]:
    """BM25 인덱스 생성. 토크나이즈된 코퍼스도 함께 저장(검색 시 동일 토크나이저 필요)."""
    print("[BM25] 토크나이즈 중...")
    tokenized = [korean_tokenize(c["text"]) for c in tqdm(chunks)]
    print("[BM25] 인덱스 빌드 중...")
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def build_faiss(chunks: list[dict], model: SentenceTransformer,
                batch_size: int = 32) -> faiss.Index:
    """FAISS 벡터 인덱스 생성 (코사인 유사도 = 정규화 후 내적)."""
    texts = [c["text"] for c in chunks]
    print(f"[FAISS] {len(texts)}개 임베딩 중...")
    embs = model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    dim = embs.shape[1]
    # 데이터 규모가 수만 건 이내면 IndexFlatIP 가 가장 정확하고 단순함
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print(f"[FAISS] 차원={dim}, 벡터수={index.ntotal}")
    return index


# ───────────────────────── 저장 ─────────────────────────

def save_all(index_dir: Path, bm25: BM25Okapi, tokenized: list[list[str]],
             faiss_index: faiss.Index, chunks: list[dict]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = index_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    print(f"[저장] BM25 → {bm25_path}")

    faiss_path = index_dir / "faiss.bin"
    faiss.write_index(faiss_index, str(faiss_path))
    print(f"[저장] FAISS → {faiss_path}")

    meta_path = index_dir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[저장] 청크 메타 → {meta_path}")


# ───────────────────────── 메인 ─────────────────────────

def main(chunks_path: str, index_dir: str, model_name: str = DEFAULT_MODEL) -> None:
    chunks = load_chunks(Path(chunks_path))
    print(f"청크 수: {len(chunks)}")

    bm25, tokenized = build_bm25(chunks)
    model = load_embedding_model(model_name)
    faiss_index = build_faiss(chunks, model)

    save_all(Path(index_dir), bm25, tokenized, faiss_index, chunks)

    # 어떤 모델로 인덱스가 구축됐는지 기록 — retrieval 시 동일 모델 사용
    Path(index_dir).joinpath("model_name.txt").write_text(
        model_name, encoding="utf-8")
    print("\n인덱스 구축 완료")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", help="chunks.jsonl")
    ap.add_argument("index_dir", help="인덱스 저장 디렉토리")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF 모델 ID 또는 fine-tuned 로컬 디렉토리")
    args = ap.parse_args()
    main(args.chunks, args.index_dir, args.model)
