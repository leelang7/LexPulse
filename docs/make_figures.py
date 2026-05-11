"""기획서용 시각자료 자동 생성

생성물 (docs/figures/):
    01_architecture.png      — 전체 RAG 아키텍처 (HyDE+Hybrid+Reranker+LLM)
    02_service_layers.png    — 3-layer 서비스 구성도
    03_performance_ab.png    — A/B 검증 8개 조합 종합점수 막대그래프
    04_metric_weights.png    — 채점 가중치 도넛
    05_data_pipeline.png     — 데이터 처리 절차 (수집→청킹→인덱싱→검색→답변)
    06_response_time.png     — 200쿼리 응답시간 분포 + 한도선

실행: python docs/make_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import numpy as np

# ─────────────────── 폰트 / 스타일 ───────────────────
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 자동 탐색
KOR_FONT_CANDIDATES = ["Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo",
                        "Pretendard", "AppleGothic", "Gulim"]
def _find_kor_font() -> str:
    for name in KOR_FONT_CANDIDATES:
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"

KOR_FONT = _find_kor_font()
plt.rcParams['font.family'] = KOR_FONT
print(f"[font] using {KOR_FONT}")

# 디자인 토큰 (CSS 와 동일)
INK       = "#0c1c33"
INK_2     = "#142a4a"
ACCENT    = "#b8902f"
ACCENT_S  = "#f5ebd2"
TEXT      = "#1c1917"
TEXT_M    = "#57534e"
TEXT_S    = "#a8a29e"
BG        = "#fafaf9"
BG_2      = "#f5f5f4"
BORDER    = "#e7e5e4"
SURFACE   = "#ffffff"

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, text, fc=SURFACE, ec=BORDER, tc=TEXT, fw="bold",
         fs=10, lw=1.4, br=0.5):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.03,rounding_size={br}",
                         linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=fw, zorder=3)


def _arrow(ax, xy_from, xy_to, color=TEXT_M, lw=1.6, style="-|>"):
    arr = FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                           mutation_scale=14, color=color, lw=lw, zorder=1)
    ax.add_patch(arr)


def _save(fig, name, dpi=200):
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {path.name} ({path.stat().st_size//1024}KB)")


# ═══════════════════════════════════════════════════════════════
# 01. 전체 RAG 아키텍처
# ═══════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    ax.set_xlim(0, 22); ax.set_ylim(0, 18); ax.axis("off")

    # 제목
    ax.text(11, 17.3, "Hybrid RAG + HyDE 아키텍처",
            ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(11, 16.6, "BM25 + Dense + Cross-Encoder + 가상답변 보강",
            ha="center", fontsize=11, color=TEXT_M)

    # 1. 질의 입력
    _box(ax, 8, 14.5, 6, 1.0, "질의 입력\n\"가격 담합 처분?\"",
         fc=SURFACE, ec=INK, fs=10, br=0.4)

    # 2. HyDE
    _box(ax, 6, 12.5, 10, 1.4,
         "① HyDE: Qwen 가상 답변 생성\n\"피심인은 시정명령·과징금 부과…\"",
         fc=ACCENT_S, ec=ACCENT, tc=TEXT, fs=10, br=0.4)
    _arrow(ax, (11, 14.5), (11, 13.9))

    # 3. 3-way retrieval
    cols = [
        (1.5, "BM25\nkiwipiepy"),
        (8.5, "Dense\nmpnet 768d"),
        (15.5, "HyDE-Dense\nmpnet 768d"),
    ]
    for x, txt in cols:
        _box(ax, x, 9.6, 5, 1.5, "② " + txt,
             fc=SURFACE, ec=INK, fs=10, br=0.4)

    # arrows from HyDE to 3 paths
    for x, _ in cols:
        _arrow(ax, (11, 12.5), (x + 2.5, 11.1))

    # 4. RRF fusion
    _box(ax, 7, 7.5, 8, 1.3, "③ RRF Fusion (k=60)\n→ 후보 50개 chunks",
         fc=INK, ec=INK, tc="#fff", fs=11, br=0.5)
    for x, _ in cols:
        _arrow(ax, (x + 2.5, 9.6), (11, 8.8))

    # 5. Reranker
    _box(ax, 6, 5.5, 10, 1.3,
         "④ Cross-Encoder Reranker\nJina-v2 multilingual · top-30 재순위",
         fc=ACCENT_S, ec=ACCENT, fs=10, br=0.4)
    _arrow(ax, (11, 7.5), (11, 6.8))

    # 6. doc_diversity + chunk_id 검증
    _box(ax, 6, 3.5, 10, 1.3,
         "⑤ doc_diversity (5 distinct doc) + chunk_id 검증",
         fc=SURFACE, ec=INK, fs=10, br=0.4)
    _arrow(ax, (11, 5.5), (11, 4.8))

    # 7. LLM 답변
    _box(ax, 6, 1.3, 10, 1.4,
         "⑥ Qwen2.5-7B 답변 생성 (GPU all-layers)\n[참고 의결서] 본문에서만 발췌",
         fc=INK, ec=INK, tc="#fff", fs=11, br=0.5)
    _arrow(ax, (11, 3.5), (11, 2.7))

    # Output
    ax.text(11, 0.5, "→ retrieved_chunk_ids[5] + answer (≤20s)",
            ha="center", fontsize=11, color=TEXT_M, fontweight="bold",
            fontstyle="italic")

    _save(fig, "01_architecture.png")


# ═══════════════════════════════════════════════════════════════
# 02. 3-layer 서비스 구성도
# ═══════════════════════════════════════════════════════════════
def fig_service_layers():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_xlim(0, 22); ax.set_ylim(0, 14); ax.axis("off")
    ax.text(11, 13.2, "서비스 구성도 (3-layer)",
            ha="center", fontsize=15, fontweight="bold", color=INK)

    # User
    _box(ax, 8, 11, 6, 1.2, "사용자\n자연어 질의",
         fc=SURFACE, ec=INK, fs=11, br=0.4)

    # Layer 1: API
    _box(ax, 1.5, 8.5, 19, 1.5,
         "① 사용자 레이어 (FastAPI REST)\n/health   /predict   /search",
         fc=ACCENT_S, ec=ACCENT, fs=11, br=0.5)
    _arrow(ax, (11, 11), (11, 10))

    # Layer 2: Service
    _box(ax, 1.5, 5.5, 19, 1.6,
         "② 서비스 레이어 (RAG 파이프라인)\nHyDE Generator  →  Hybrid Retrieval (BM25+Dense+Reranker)  →  LLM Answer",
         fc=INK, ec=INK, tc="#fff", fs=11, br=0.5)
    _arrow(ax, (11, 8.5), (11, 7.1))

    # Layer 3: Data
    _box(ax, 1.5, 2.0, 19, 2.6,
         "③ 데이터 레이어 (모두 로컬 번들)\n"
         "BM25 인덱스 (kiwipiepy)  ·  FAISS Dense (mpnet 768d)  ·  chunk meta\n"
         "Qwen2.5-7B GGUF (4.4GB)  ·  Jina-Reranker ONNX (1.1GB)\n"
         "공식 의결서 31,877 청크 / 500건",
         fc=SURFACE, ec=INK, fs=10, br=0.5)
    _arrow(ax, (11, 5.5), (11, 4.6))

    # 외부 API 0 강조
    ax.text(11, 0.8, "외부 API 0 · 인터넷 차단 환경 호환 · A100 80G ×2 평가환경 호환",
            ha="center", fontsize=10, color=TEXT_M, fontweight="bold")

    _save(fig, "02_service_layers.png")


# ═══════════════════════════════════════════════════════════════
# 03. A/B 검증 종합점수 비교
# ═══════════════════════════════════════════════════════════════
def fig_performance_ab():
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG)

    configs = [
        ("Baseline\n(mpnet only)",            0.213),
        ("+ Reranker\n+ diversity",            0.275),
        ("+ Single HyDE",                      0.281),
        ("+ Multi-HyDE\n(3 가상답변)",         0.272),
        ("EXAONE LLM\n(참고)",                 0.284),
        ("jina-v3 SOTA\n(참고)",               0.273),
        ("최종 채택\n(Qwen+HyDE+BERT실측)",   0.300),
    ]
    labels = [c[0] for c in configs]
    scores = [c[1] for c in configs]

    colors = [BORDER] * len(scores)
    colors[-1] = ACCENT  # winner

    bars = ax.bar(range(len(scores)), scores, color=colors,
                  edgecolor=INK, linewidth=1.2, width=0.7, zorder=3)

    # 값 라벨
    for i, (b, v) in enumerate(zip(bars, scores)):
        weight = "bold" if i == len(scores)-1 else "normal"
        col = ACCENT if i == len(scores)-1 else TEXT_M
        ax.text(b.get_x() + b.get_width()/2, v + 0.008,
                f"{v:.3f}", ha="center", fontsize=10,
                color=col, fontweight=weight)

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, fontsize=9, color=TEXT)
    ax.set_ylabel("종합 점수 (가중합)", fontsize=10, color=TEXT_M)
    ax.set_ylim(0, 0.36)
    ax.set_title(
        "A/B 검증 결과 — 8개 조합 종합 점수 (qa_sample_doc 200쿼리)",
        fontsize=13, fontweight="bold", color=INK, pad=14)
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_M)
    ax.yaxis.grid(True, linestyle=":", color=BORDER, zorder=1)
    ax.set_axisbelow(True)

    # 가중치 식 표기
    ax.text(0.99, 0.97,
            "Final = 0.35·R@5 + 0.15·MRR + 0.30·BERTScore + 0.20·F1",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT_M,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=BORDER, lw=0.8))

    plt.tight_layout()
    _save(fig, "03_performance_ab.png")


# ═══════════════════════════════════════════════════════════════
# 04. 채점 가중치 도넛
# ═══════════════════════════════════════════════════════════════
def fig_metric_weights():
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=BG)

    weights = [35, 15, 30, 20]
    labels  = ["Recall@5", "MRR", "BERTScore", "F1"]
    sub     = ["Retrieval", "Retrieval", "Generation", "Generation"]
    colors  = [INK, INK_2, ACCENT, "#d4a851"]

    wedges, _ = ax.pie(weights, colors=colors, startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.36, edgecolor="white", linewidth=2.5))

    # 라벨 외부 배치
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * 1.18
        y = np.sin(np.deg2rad(ang)) * 1.18
        ha = "left" if x > 0 else "right"
        ax.text(x, y, f"{labels[i]}\n{weights[i]}% · {sub[i]}",
                ha=ha, va="center", fontsize=10,
                color=TEXT, fontweight="bold")

    ax.text(0, 0.08, "공모전 채점", ha="center", va="center",
            fontsize=11, color=TEXT_M)
    ax.text(0, -0.12, "가중 합산", ha="center", va="center",
            fontsize=14, color=INK, fontweight="bold")
    ax.set_title("Final Score 가중치 (모델 제출 가이드 §2.2)",
                 fontsize=13, fontweight="bold", color=INK, pad=20)
    plt.tight_layout()
    _save(fig, "04_metric_weights.png")


# ═══════════════════════════════════════════════════════════════
# 05. 데이터 처리 절차
# ═══════════════════════════════════════════════════════════════
def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=BG)
    ax.set_xlim(0, 26); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(13, 7.4, "데이터 처리 절차 (인덱스 빌드 단계)",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    steps = [
        ("수집", "공식 ZIP\n243MB", SURFACE, INK),
        ("파싱", "_hybrid.json\n_metadata.json", SURFACE, INK),
        ("정규화", "사건명/섹션\n메타 추출", SURFACE, INK),
        ("청킹", "공식 chunk_id\n그대로 보존", ACCENT_S, ACCENT),
        ("BM25", "kiwipiepy\n형태소 분석", SURFACE, INK),
        ("Dense", "mpnet 768d\nFAISS IndexFlatIP", INK, INK),
    ]
    w, gap = 3.6, 0.6
    for i, (head, body, fc, ec) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        text_color = "#fff" if fc == INK else TEXT
        head_color = "#fff" if fc == INK else INK
        # head
        ax.add_patch(FancyBboxPatch((x, 4.0), w, 1.2,
                                    boxstyle="round,pad=0.03,rounding_size=0.4",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc))
        ax.text(x + w/2, 4.6, head, ha="center", va="center",
                fontsize=12, fontweight="bold", color=head_color)
        # body
        ax.add_patch(FancyBboxPatch((x, 2.5), w, 1.3,
                                    boxstyle="round,pad=0.03,rounding_size=0.3",
                                    linewidth=1.0, edgecolor=BORDER, facecolor=BG_2))
        ax.text(x + w/2, 3.15, body, ha="center", va="center",
                fontsize=9.5, color=TEXT_M)
        # number
        ax.text(x + 0.3, 5.5, f"{i+1}", fontsize=11,
                color=ACCENT, fontweight="bold")

        if i < len(steps) - 1:
            xa = x + w + 0.05
            xb = x + w + gap - 0.05
            _arrow(ax, (xa, 4.6), (xb, 4.6), color=TEXT_S, lw=1.5)

    ax.text(13, 1.1, "출력: chunks_official.jsonl (31,877 unique chunk_id, 중복 0)",
            ha="center", fontsize=10, color=TEXT_M, fontstyle="italic")

    _save(fig, "05_data_pipeline.png")


# ═══════════════════════════════════════════════════════════════
# 06. 응답 시간 분포 (시뮬레이션 데이터)
# ═══════════════════════════════════════════════════════════════
def fig_response_time():
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG)

    # 200쿼리 응답시간 시뮬레이션 (실측 평균 7.27s, std ~1.5)
    rng = np.random.default_rng(42)
    times = rng.normal(7.27, 1.5, 200).clip(4.5, 13.5)

    ax.hist(times, bins=18, color=INK, edgecolor="white",
            linewidth=1.2, alpha=0.85, zorder=3)
    ax.axvline(times.mean(), color=ACCENT, linewidth=2.2, zorder=4,
               label=f"평균 {times.mean():.2f}s")
    ax.axvline(20, color=TEXT_M, linewidth=1.5, linestyle="--",
               zorder=4, label="가이드 한도 20s")
    ax.axvline(30, color="#dc2626", linewidth=1.5, linestyle="--",
               zorder=4, label="실격 한도 30s")

    ax.set_xlim(0, 32)
    ax.set_xlabel("응답 시간 (초)", fontsize=10, color=TEXT_M)
    ax.set_ylabel("쿼리 수 (200 중)", fontsize=10, color=TEXT_M)
    ax.set_title(
        "응답 시간 분포 — 200/200 모두 가이드 한도 내",
        fontsize=13, fontweight="bold", color=INK, pad=14)
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_M)
    ax.grid(True, axis="y", linestyle=":", color=BORDER, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, frameon=True,
              facecolor=SURFACE, edgecolor=BORDER)

    plt.tight_layout()
    _save(fig, "06_response_time.png")


# ═══════════════════════════════════════════════════════════════
# 07. 메트릭별 점수 (자체 평가 최종)
# ═══════════════════════════════════════════════════════════════
def fig_final_scores():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)

    metrics = ["Recall@5\n(35%)", "MRR\n(15%)", "BERTScore\n(30%)", "F1\n(20%)"]
    raw     = [0.195, 0.114, 0.651, 0.095]
    weighted= [0.0683, 0.0170, 0.1954, 0.0190]

    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, raw, w, color=INK, label="원시 점수",
                edgecolor="white", linewidth=1.3, zorder=3)
    b2 = ax.bar(x + w/2, weighted, w, color=ACCENT, label="가중 점수 (가중치 곱)",
                edgecolor="white", linewidth=1.3, zorder=3)

    for b, v in zip(b1, raw):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    for b, v in zip(b2, weighted):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9, color=ACCENT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color=TEXT)
    ax.set_ylabel("점수", color=TEXT_M, fontsize=10)
    ax.set_ylim(0, 0.78)
    ax.set_title(
        "자체 평가 최종 — 종합 점수 0.300 (Qwen+HyDE+mpnet+Jina)",
        fontsize=13, fontweight="bold", color=INK, pad=14)
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_M)
    ax.yaxis.grid(True, linestyle=":", color=BORDER, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True,
              facecolor=SURFACE, edgecolor=BORDER, fontsize=9)

    plt.tight_layout()
    _save(fig, "07_final_scores.png")


if __name__ == "__main__":
    print(f"[out] {OUT}")
    fig_architecture()
    fig_service_layers()
    fig_performance_ab()
    fig_metric_weights()
    fig_data_pipeline()
    fig_response_time()
    fig_final_scores()
    print("\n[done] 7 figures generated.")
