"""
공식 데이터 (243MB ZIP) 어댑터

공정거래위원회 제2회 공모전이 제공하는 공식 의결서 데이터를 로드해
시스템 인덱스 형식(chunks.jsonl)으로 변환한다.
**chunk_id 는 공식 데이터의 것을 그대로 보존** (변경 시 자동 실격).

지원 입력 구조:

    (A) 의결서별 페어 (실제 공모전 ZIP 형식):
        <사건명>.pdf
        <사건명>_hybrid.json       # 청킹: list of {page_content, metadata{...,chunk_id}}
        <사건명>_metadata.json     # 메타: {의결서관리번호, 의결서제목, 피심인정보:[...]}

    (B) 단일 파일 (mock / 다른 포맷):
        chunks.jsonl  + metadata.json
        공개본의결서_청킹.json + 공개본의결서_메타.jsonl

사용법:
    python load_official_data.py <official_dir> <out_chunks.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


# ───────────────────────── 후보 파일명 ─────────────────────────

CHUNK_FILE_CANDIDATES = [
    "chunks.jsonl", "chunks.json",
    "chunking.jsonl", "chunking.json",
    "공개본의결서_청킹.jsonl", "공개본의결서_청킹.json",
    "decisions_chunks.jsonl",
]

META_FILE_CANDIDATES = [
    "metadata.json", "metadata.jsonl",
    "meta.json", "meta.jsonl",
    "공개본의결서_메타.json", "공개본의결서_메타.jsonl",
]


# ───────────────────────── 정규화 ─────────────────────────

def _normalize_record(raw: dict, meta_by_doc: dict[str, dict]) -> dict | None:
    """
    공식 청킹 레코드 → 시스템 표준 형식.
    누락된 키는 빈 값으로 채우되, **chunk_id 와 doc_id 는 절대 변경하지 않는다**.

    공식 포맷이 어떻든 다음 키들이 보존돼야 함:
        chunk_id, doc_id, text
    그 외 메타데이터 (title, section, 피심인 등) 는 best-effort.
    """
    # chunk_id (필수)
    chunk_id = raw.get("chunk_id") or raw.get("chunkId") or raw.get("id")
    if not chunk_id:
        return None
    chunk_id = str(chunk_id)

    # text (필수)
    text = raw.get("text") or raw.get("content") or raw.get("body") or ""
    if not text.strip():
        return None

    # doc_id (필수, 없으면 chunk_id 에서 유추)
    doc_id = (raw.get("doc_id") or raw.get("docId") or raw.get("document_id")
              or raw.get("의결서관리번호"))
    if not doc_id:
        # 흔한 패턴: "<doc_id>-<section>-<idx>" 또는 "<doc_id>_<idx>"
        m = re.match(r"^([^-_]+(?:[-_][^-_]+)?)", chunk_id)
        doc_id = m.group(1) if m else chunk_id
    doc_id = str(doc_id)

    meta = meta_by_doc.get(doc_id, {})

    return {
        "chunk_id":      chunk_id,                                # 절대 변경 금지
        "doc_id":        doc_id,
        "title":         (raw.get("title") or meta.get("의결서제목") or
                          meta.get("title") or ""),
        "section":       (raw.get("section") or raw.get("섹션") or ""),
        "chunk_idx":     int(raw.get("chunk_idx") or raw.get("idx") or 0),
        "text":          text,
        "공개일자":      meta.get("공개일자") or raw.get("공개일자") or "",
        "의결번호":      meta.get("의결번호") or raw.get("의결번호") or "",
        "피심인":        _to_list(meta.get("피심인") or raw.get("피심인") or
                                  _extract_피심인(meta)),
        "위반유형":      _to_list(meta.get("위반유형") or raw.get("위반유형")
                                  or _extract_field(meta, "위반유형")),
        "세부위반유형":  _to_list(meta.get("세부위반유형") or raw.get("세부위반유형")
                                  or _extract_field(meta, "세부위반유형")),
        "조치유형":      _to_list(meta.get("조치유형") or raw.get("조치유형")
                                  or _extract_field(meta, "조치유형")),
    }


def _to_list(v) -> list:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return [str(v)]


def _extract_피심인(meta: dict) -> list:
    info = meta.get("피심인정보") or []
    if isinstance(info, list):
        return [str(p.get("피심인기업명") or p.get("name") or "")
                for p in info if isinstance(p, dict)]
    return []


def _extract_field(meta: dict, key: str) -> list:
    info = meta.get("피심인정보") or []
    if isinstance(info, list):
        out = {str(p.get(key) or "") for p in info if isinstance(p, dict)}
        out.discard("")
        return sorted(out)
    return []


# ───────────────────────── 파일 검색·로드 ─────────────────────────

def _find_first(base: Path, names: list[str]) -> Path | None:
    for name in names:
        cand = base / name
        if cand.exists():
            return cand
    # 재귀 탐색 한 번
    for name in names:
        for f in base.rglob(name):
            return f
    return None


def _load_meta_by_doc(meta_file: Path | None) -> dict[str, dict]:
    if meta_file is None:
        return {}
    out: dict[str, dict] = {}
    if meta_file.suffix == ".jsonl":
        with open(meta_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = (obj.get("doc_id") or obj.get("의결서관리번호")
                          or obj.get("id"))
                if doc_id:
                    out[str(doc_id)] = obj
    else:
        with open(meta_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                doc_id = (obj.get("doc_id") or obj.get("의결서관리번호")
                          or obj.get("id"))
                if doc_id:
                    out[str(doc_id)] = obj
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    out[str(k)] = v
                else:
                    out[str(k)] = {"value": v}
    return out


def _iter_chunks(chunk_file: Path) -> Iterable[dict]:
    if chunk_file.suffix == ".jsonl":
        with open(chunk_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(chunk_file, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            # {chunks: [...]} 또는 dict-of-chunks
            if "chunks" in data and isinstance(data["chunks"], list):
                yield from data["chunks"]
            else:
                yield from data.values()


# ───────────────────────── 메인 ─────────────────────────

# ───────────────────────── 공식 포맷 (페어) 처리 ─────────────────────────

def _normalize_hybrid_record(raw: dict, doc_meta: dict) -> dict | None:
    """
    공식 _hybrid.json 한 항목 → 표준 chunk dict.
    형식: {"page_content": "...", "metadata": {chunk_id, section, ...}}
    """
    text = raw.get("page_content") or ""
    if not text.strip():
        return None
    md = raw.get("metadata") or {}
    chunk_id = md.get("chunk_id") or md.get("chunkId")
    if not chunk_id:
        return None
    chunk_id = str(chunk_id)

    # doc_id: chunk_id 의 "DOC-{uuid}" prefix, 또는 metadata 의 의결서관리번호
    doc_id = (doc_meta.get("의결서관리번호") or doc_meta.get("doc_id") or "")
    if not doc_id:
        # chunk_id 패턴 "DOC-{uuid}-CH-XXX" → "DOC-{uuid}"
        m = re.match(r"^(DOC-[^-]+(?:-[^-]+){0,3})-CH-\d+$", chunk_id)
        doc_id = m.group(1) if m else chunk_id
    doc_id = str(doc_id)

    피심인정보 = doc_meta.get("피심인정보") or []
    피심인 = [str(p.get("피심인기업명") or "") for p in 피심인정보 if isinstance(p, dict)]
    피심인 = [p for p in 피심인 if p]
    위반 = sorted({str(p.get("위반유형") or "") for p in 피심인정보 if isinstance(p, dict)})
    위반 = [v for v in 위반 if v]
    세부 = sorted({str(p.get("세부위반유형") or "") for p in 피심인정보 if isinstance(p, dict)})
    세부 = [v for v in 세부 if v]
    조치 = sorted({str(p.get("조치유형") or "") for p in 피심인정보 if isinstance(p, dict)})
    조치 = [v for v in 조치 if v]

    return {
        "chunk_id":      chunk_id,
        "doc_id":        doc_id,
        "title":         doc_meta.get("의결서제목") or doc_meta.get("title") or "",
        "section":       md.get("section") or md.get("Header") or "",
        "chunk_idx":     int(md.get("chunk_index") or 0),
        "text":          text,
        "공개일자":      doc_meta.get("공개일자") or "",
        "의결번호":      doc_meta.get("의결번호") or doc_meta.get("의결서관리번호") or "",
        "피심인":        피심인,
        "위반유형":      위반,
        "세부위반유형":  세부,
        "조치유형":      조치,
    }


def _process_paired_format(base: Path, out_path: Path) -> int | None:
    """
    공식 ZIP 포맷: 의결서마다 <X>_hybrid.json + <X>_metadata.json 페어.
    공통 stem (paired) 가 발견되면 이 함수가 처리하고 청크 수를 반환.
    페어가 발견되지 않으면 None.
    """
    pairs: list[tuple[Path, Path]] = []
    for h in base.rglob("*_hybrid.json"):
        m_path = h.with_name(h.name.replace("_hybrid.json", "_metadata.json"))
        if m_path.exists():
            pairs.append((h, m_path))

    if not pairs:
        return None

    print(f"[paired] {len(pairs)} 의결서 발견 (hybrid + metadata)")
    n_in = n_out = 0
    seen_chunk_ids: set[str] = set()
    duplicates = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for h_path, m_path in sorted(pairs):
            try:
                with open(m_path, encoding="utf-8") as mf:
                    doc_meta = json.load(mf)
            except Exception as e:
                print(f"[skip] meta 읽기 실패 {m_path.name}: {e}", file=sys.stderr)
                continue
            try:
                with open(h_path, encoding="utf-8") as hf:
                    chunks = json.load(hf)
            except Exception as e:
                print(f"[skip] hybrid 읽기 실패 {h_path.name}: {e}", file=sys.stderr)
                continue
            if not isinstance(chunks, list):
                continue

            for raw in chunks:
                n_in += 1
                rec = _normalize_hybrid_record(raw, doc_meta)
                if rec is None:
                    continue
                if rec["chunk_id"] in seen_chunk_ids:
                    duplicates += 1
                    continue
                seen_chunk_ids.add(rec["chunk_id"])
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"\n[ok] {len(pairs)} 의결서 / {n_in} 청크 입력 → "
          f"{n_out} 청크 저장 (중복 {duplicates} 스킵)")
    print(f"     {out_path}")
    print(f"     unique chunk_ids: {len(seen_chunk_ids)}")
    return n_out


def main(official_dir: str, out_path: str,
         chunks_filename: str | None = None,
         meta_filename: str | None = None) -> int:
    base = Path(official_dir)
    if not base.exists():
        print(f"[error] not found: {base}", file=sys.stderr)
        return 1

    # 1) 페어 형식 우선 시도 (공식 ZIP)
    n = _process_paired_format(base, Path(out_path))
    if n is not None:
        return 0 if n > 0 else 3

    # 2) 단일 파일 포맷 (mock 또는 다른 ZIP 구조)
    print("[paired] 페어 미발견, 단일 파일 포맷으로 시도")

    # 청킹 파일
    if chunks_filename:
        chunk_file = base / chunks_filename
        if not chunk_file.exists():
            chunk_file = _find_first(base, [chunks_filename])
    else:
        chunk_file = _find_first(base, CHUNK_FILE_CANDIDATES)
    if chunk_file is None:
        print(f"[error] 청킹 파일을 찾지 못했습니다 ({CHUNK_FILE_CANDIDATES})",
              file=sys.stderr)
        print(f"        검색 경로: {base}", file=sys.stderr)
        print(f"        실제 ZIP 구조에 맞춰 --chunks <파일명> 으로 지정하거나",
              file=sys.stderr)
        print(f"        load_official_data.py 의 CHUNK_FILE_CANDIDATES 추가",
              file=sys.stderr)
        return 2
    print(f"[load] 청킹 파일: {chunk_file}")

    # 메타 파일 (선택)
    if meta_filename:
        meta_file = base / meta_filename
        if not meta_file.exists():
            meta_file = _find_first(base, [meta_filename])
    else:
        meta_file = _find_first(base, META_FILE_CANDIDATES)
    if meta_file:
        print(f"[load] 메타 파일: {meta_file}")
    else:
        print(f"[load] 메타 파일 없음 (청킹 레코드의 메타데이터만 사용)")

    meta_by_doc = _load_meta_by_doc(meta_file)
    print(f"[meta] {len(meta_by_doc)} 개 문서 메타데이터")

    # 변환
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    seen_chunk_ids: set[str] = set()
    duplicates = 0
    with open(out, "w", encoding="utf-8") as fout:
        for raw in _iter_chunks(chunk_file):
            n_in += 1
            rec = _normalize_record(raw, meta_by_doc)
            if rec is None:
                continue
            if rec["chunk_id"] in seen_chunk_ids:
                duplicates += 1
                continue
            seen_chunk_ids.add(rec["chunk_id"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"\n[ok] {n_in} 입력 → {n_out} 청크 저장 (중복 {duplicates} 건 스킵)")
    print(f"     {out}")
    print(f"     unique chunk_ids: {len(seen_chunk_ids)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="공식 의결서 데이터(243MB) → 시스템 chunks.jsonl 변환")
    ap.add_argument("official_dir",
                    help="공식 ZIP 압축 해제 디렉토리 (예: ./data/official)")
    ap.add_argument("out_path", help="출력 chunks.jsonl 경로")
    ap.add_argument("--chunks", default=None,
                    help="청킹 파일명 (자동 탐색 실패 시 명시)")
    ap.add_argument("--meta", default=None,
                    help="메타 파일명 (선택)")
    args = ap.parse_args()
    sys.exit(main(args.official_dir, args.out_path, args.chunks, args.meta))
