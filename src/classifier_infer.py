"""
학습된 분류기 추론 헬퍼

train_classifier.py 와 train_sanction.py 가 저장한 디렉토리를 그대로 받아
입력 텍스트에 대한 예측을 반환한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class ViolationPrediction:
    labels: list[str]                  # 임계값 이상으로 활성화된 위반유형들
    scores: dict[str, float]           # 라벨별 시그모이드 확률


@dataclass
class SanctionPrediction:
    label: str                         # 예측된 조치유형
    score: float                       # 정답 클래스 softmax 확률
    distribution: dict[str, float]     # 모든 클래스 확률


class ViolationClassifier:
    """위반유형 멀티라벨 분류기 추론"""

    def __init__(self, model_dir: str | Path, device: str | None = None,
                 threshold: float = 0.30, min_top_k: int = 1):
        """
        threshold: 시그모이드 확률 임계값. 학습 데이터가 적을 때 모델이 underconfident
                   일 수 있어 0.30 으로 기본값 낮춤.
        min_top_k: 임계값 미달이라도 상위 K개는 항상 반환 (UI에서 빈 결과 방지).
        """
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "label_map.json", encoding="utf-8") as f:
            self.labels: list[str] = json.load(f)["labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.min_top_k = min_top_k

    @torch.no_grad()
    def predict(self, text: str) -> ViolationPrediction:
        enc = self.tokenizer(text, truncation=True, padding=True,
                             max_length=512, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[0].cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

        # 1) 임계값 통과한 라벨
        passed = [(l, float(p)) for l, p in zip(self.labels, probs) if p >= self.threshold]
        # 2) 임계값 미달 시 상위 K개 보장
        if not passed:
            top_idx = np.argsort(-probs)[:self.min_top_k]
            passed = [(self.labels[i], float(probs[i])) for i in top_idx]

        passed.sort(key=lambda x: x[1], reverse=True)
        labels = [l for l, _ in passed]
        scores = {l: float(p) for l, p in zip(self.labels, probs)}
        return ViolationPrediction(labels=labels, scores=scores)


class SanctionPredictor:
    """조치유형 단일라벨 예측"""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "label_map.json", encoding="utf-8") as f:
            m = json.load(f)
            self.labels: list[str] = m["labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> SanctionPrediction:
        enc = self.tokenizer(text, truncation=True, padding=True,
                             max_length=512, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[0].cpu().numpy()
        # softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        return SanctionPrediction(
            label=self.labels[idx],
            score=float(probs[idx]),
            distribution={l: float(p) for l, p in zip(self.labels, probs)},
        )
