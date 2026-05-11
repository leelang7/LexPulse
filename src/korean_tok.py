"""
한국어 형태소 분석 기반 BM25 토크나이저 (kiwipiepy).

기존 정규식 토크나이저는 "가맹사업법상" 을 한 토큰으로 둬 BM25 매칭이 약했다.
형태소 분석으로 "가맹/사업/법/상" 으로 분리하면 의미 단위 매칭 가능.

torch 비의존 — 평가 환경에서도 안전.
"""

from __future__ import annotations

import re
from functools import lru_cache

# 명사/동사/형용사/외래어/숫자 등 의미 있는 품사만 사용
_USEFUL_TAGS = {
    "NNG", "NNP", "NNB", "NR", "NP",   # 일반/고유/의존/수/대명사
    "VV", "VA", "VX",                    # 동사 / 형용사 / 보조용언
    "MAG", "MAJ",                        # 부사
    "SL", "SN", "SH",                    # 외국어 / 숫자 / 한자
    "XR",                                # 어근 (구입, 강제 등)
}

# 불용어 (도메인에 흔하지만 의미 약함)
_STOPWORDS = {
    "있", "하", "되", "이", "그", "저", "수", "것", "등", "통하", "관련",
    "대하", "위하", "따르", "있다", "한다", "하는", "되는", "있는", "한",
    "이다", "그러나", "그리고", "그런데", "또한", "또는", "이상", "이하",
    "기초", "사실",  # 너무 일반적
}

# 회피용 비-한글/숫자 토큰
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


_KIWI = None


def _get_kiwi():
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return _KIWI


@lru_cache(maxsize=10000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    kiwi = _get_kiwi()
    out: list[str] = []
    for tok in kiwi.tokenize(text):
        if tok.tag not in _USEFUL_TAGS:
            continue
        form = tok.form
        if len(form) < 2:
            continue
        if form in _STOPWORDS:
            continue
        out.append(form)
    return tuple(out)


def korean_tokenize_kiwi(text: str) -> list[str]:
    """캐시된 형태소 토크나이저."""
    if not text:
        return []
    return list(_tokenize_cached(text))


# fallback: kiwipiepy 미설치 시 정규식 사용
def korean_tokenize_regex(text: str) -> list[str]:
    _legacy_stop = {
        "그리고","그러나","또한","또는","있다","한다","하는","하여","위해","통해","대해","관한",
        "있는","이를","따라서","이에","있어","보다","관련","기재","다음","같은","이상","이하",
    }
    return [t for t in _TOKEN_RE.findall(text) if len(t) >= 2 and t not in _legacy_stop]


def korean_tokenize(text: str) -> list[str]:
    """동적 디스패치: kiwipiepy 가능하면 형태소, 아니면 정규식."""
    try:
        return korean_tokenize_kiwi(text)
    except ImportError:
        return korean_tokenize_regex(text)


if __name__ == "__main__":
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "가맹사업법상 구입강제 행위에 대한 처분은 무엇인가요?"
    print(f"입력: {sample}")
    print(f"kiwi: {korean_tokenize_kiwi(sample)}")
    print(f"regex: {korean_tokenize_regex(sample)}")
