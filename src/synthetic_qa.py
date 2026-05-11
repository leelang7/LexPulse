"""
학습용 (질의, 정답청크) 페어 합성 모듈

전처리된 chunks.jsonl 을 입력받아, 메타데이터·본문에서 룰 기반으로
자연어 질의를 생성하고 (질의, 양성청크) 페어를 만든다.

생성 전략:
    1) 메타데이터 기반 — 위반유형/세부위반유형/조치유형/피심인 조합
    2) 섹션 기반 — 섹션마다 의도 다른 질의 (주문→처벌, 기초사실→사실관계)
    3) 사건명 기반 — 제목에서 키워드 추출

각 청크당 평균 4~6개의 (질의, 정답청크) 페어가 생성된다.
사용처:
    finetune_embedding.py     SBERT MultipleNegativesRankingLoss
    train_reranker.py          하드 네거티브 마이닝의 양성 시드
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable


# 섹션별 질의 템플릿
# {label} 자리에 위반유형/세부위반유형/조치유형/피심인 등이 들어감
SECTION_TEMPLATES = {
    "주문": [
        "{title}에서 어떤 처분이 내려졌나요?",
        "{label}로 받은 처벌은 무엇인가요?",
        "{피심인} 사건의 주문 내용은?",
        "{title} 처분 결과",
        "{피심인}이 받은 시정명령·과징금 내용",
        "{label} 사건에서 부과된 과징금 액수는 얼마인가?",
    ],
    "이유": [   # 공식 의결서의 다수 섹션 — 전체 본문 / 사실관계 / 위법성 / 처분 모두 포함
        "{label} 사건의 사실관계는 무엇인가요?",
        "{피심인}의 행위 내용은 무엇인가요?",
        "{label}이 위법한 이유는 무엇인가요?",
        "{label}에 대한 법적 판단 근거",
        "{피심인} 행위의 위법성 판단",
        "{label} 관련 적용 법조항",
        "{label}에 대한 제재 수준은?",
        "{title}의 과징금 또는 시정명령",
        "{피심인}에게 부과된 처분",
        "{title} 사건 경위",
        "{label}로 적발된 구체적인 사례",
        "{title} 결론",
    ],
    "기초사실": [
        "{label} 사건의 사실관계는 무엇인가요?",
        "{피심인}의 행위 내용",
        "{title} 사건 경위",
        "{label}로 적발된 구체적인 사례",
    ],
    "위법성판단": [
        "{label}이 위법한 이유는 무엇인가요?",
        "{label}에 대한 법적 판단 근거",
        "{피심인} 행위의 위법성 판단",
        "{label} 관련 적용 법조항",
    ],
    "처분": [
        "{label}에 대한 제재 수준은?",
        "{title}의 과징금 또는 시정명령",
        "{label}로 부과된 과징금 산정",
        "{피심인}에게 부과된 처분",
    ],
    "결론": [
        "{title}의 최종 결론",
        "{label} 사건 결론 요약",
    ],
}

# 위반유형별 자연스러운 표현 (학습 데이터 다양성 ↑)
NATURAL_PARAPHRASES = {
    "다단계판매업자 미등록": [
        "다단계판매업 등록 안 하고 영업하면",
        "다단계 미등록 영업 처벌",
        "후원수당 지급하는 다단계 등록의무",
    ],
    "부당한 구입강제": [
        "프랜차이즈 본사가 물품 구매 강요",
        "가맹본부 구입강제 행위",
        "가맹점에 특정 물품 사라고 강제",
    ],
    "허위·과장 광고": [
        "허위 광고 처벌 사례",
        "과장 광고로 받은 처분",
        "근거 없이 효능 단정한 광고",
    ],
    "가격 담합": [
        "기업 간 가격 담합",
        "사업자 간 가격 합의",
        "출고가 인상 담합",
    ],
    "시장지배적지위 남용": [
        "시장지배적사업자 끼워팔기",
        "독점 사업자가 끼워팔기 강제",
        "지배적 지위 남용 행위",
    ],
    "부당한 하도급대금 결정": [
        "하도급 단가 일률 인하",
        "협력업체 단가 부당 인하",
        "하도급법상 단가 결정 위반",
    ],
    "부당 반품": [
        "정당한 사유 없는 반품 요구",
        "대형마트 부당 반품",
        "납품업체에 반품 강요",
    ],
    "영업지역 침해": [
        "가맹점 영업지역 안에 직영점 출점",
        "가맹점 상권 침해",
        "동일 업종 직영점 신규 출점 위법",
    ],
    "고객에게 부당하게 불리한 약관": [
        "약관 불공정 조항",
        "고객 책임 일방적 확대 약관",
        "이용약관 부당성",
    ],
    "청약철회 방해": [
        "온라인 쇼핑몰 환불 거부",
        "청약철회 방해 처벌",
        "단순변심 환불 거부",
    ],
    "기만적인 비교광고": [
        "근거 없는 비교광고",
        "경쟁사 대비 우월성 단정 광고",
        "객관적 자료 없는 비교 표현",
    ],
    "부당한 거래거절": [
        "거래상대방 일방적 거래 단절",
        "정당한 사유 없는 거래거절",
        "사전 통보 없는 거래 중단",
    ],
}


def _make_metadata_queries(chunk: dict) -> list[str]:
    """메타데이터에서 자연어 질의 생성"""
    queries: list[str] = []

    위반 = chunk.get("위반유형", []) or []
    세부 = chunk.get("세부위반유형", []) or []
    조치 = chunk.get("조치유형", []) or []
    피심 = chunk.get("피심인", []) or []
    title = chunk.get("title", "")
    section = chunk.get("section", "")

    # 라벨 후보 (질의에 자연스럽게 들어갈 표현)
    label_pool: list[str] = []
    label_pool.extend(위반)
    label_pool.extend(세부)
    for s in 세부:
        label_pool.extend(NATURAL_PARAPHRASES.get(s, []))
    if not label_pool:
        return queries

    templates = SECTION_TEMPLATES.get(section, [])
    if not templates:
        return queries

    for tpl in templates:
        for label in label_pool[:3]:  # 너무 많이 만들지 않음
            try:
                q = tpl.format(label=label, title=title,
                               피심인=", ".join(피심) if 피심 else label)
                queries.append(q)
            except (KeyError, IndexError):
                continue

    # 일반 질의 (조치 + 위반)
    if 조치 and 위반:
        for c in 조치[:1]:
            for v in 위반[:1]:
                queries.append(f"{v}으로 {c}을 받은 사례")

    # 자연어 paraphrase 직접 사용
    for s in 세부:
        for p in NATURAL_PARAPHRASES.get(s, []):
            queries.append(p)

    # 중복 제거 + 셔플
    queries = list(dict.fromkeys(queries))
    random.shuffle(queries)
    # 공식 데이터의 섹션이 대부분 "이유"라 청크가 매우 많을 수 있어 청크당 4개로 제한
    return queries[:4]


def build_pairs(chunks: list[dict], seed: int = 42) -> list[dict]:
    """
    각 청크에 대해 (query, positive_chunk_id, chunk_text) 페어 생성.
    표지 청크가 있어도 제외하지 않음 — preprocess.process_document 단계에서 이미 제외됨.
    """
    random.seed(seed)
    pairs: list[dict] = []
    for c in chunks:
        for q in _make_metadata_queries(c):
            pairs.append({
                "query": q,
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "section": c["section"],
                "text": c["text"],
                "위반유형": c.get("위반유형", []),
                "세부위반유형": c.get("세부위반유형", []),
                "조치유형": c.get("조치유형", []),
            })
    return pairs


def main(chunks_path: str, out_path: str) -> None:
    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    pairs = build_pairs(chunks)

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"청크 {len(chunks)}개 → 페어 {len(pairs)}개 ({len(pairs)/max(len(chunks),1):.1f}/청크)")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python synthetic_qa.py <chunks.jsonl> <out_pairs.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
