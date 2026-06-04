"""
e5 도메인 파인튜닝 (transformers AutoModel 직접 구현) - GPU 안전 가드 내장

sentence-transformers 가 이 환경에서 segfault 나므로 transformers 만으로
InfoNCE (in-batch negatives) 파인튜닝을 직접 구현한다.

GPU 안전장치 (이전 e5-large 7시간 hang 재발 방지):
    - nvidia DLL 선등록 (torch CUDA segfault 회피)
    - VRAM 사용 캡 (avatar 등 타 프로세스와 공존)
    - 속도 가드: 초반 step 평균이 임계 초과 시 자동 abort
    - unbuffered flush 로 실시간 로그
    - 작은 모델(e5-small) + fp16 + 적정 batch

사용:
    python -u finetune_e5_transformers.py pairs_official.jsonl models/e5small_ft \
        --base intfloat/multilingual-e5-small --epochs 1 --batch 32 \
        --max-samples 30000 --vram-frac 0.35 --abort-step-sec 0.8
"""
from __future__ import annotations

# ── nvidia DLL 선등록 (이 환경 torch CUDA segfault 회피) ──
import os
import importlib.util
from pathlib import Path as _P

# 메모리 단편화 완화 (학습 시작 전에 설정해야 함)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

for _sub in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cuda_nvrtc",
             "nvidia.cudnn", "nvidia.cufft", "nvidia.nvjitlink"):
    _s = importlib.util.find_spec(_sub)
    if _s and _s.submodule_search_locations:
        _binp = _P(list(_s.submodule_search_locations)[0]) / "bin"
        if _binp.exists():
            try:
                os.add_dll_directory(str(_binp))
            except Exception:
                pass
            os.environ["PATH"] = str(_binp) + os.pathsep + os.environ.get("PATH", "")

import argparse
import json
import random
import sys
import time

# Windows cp949 콘솔에서 한글/특수문자 출력 깨짐·크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def log(msg):
    """unbuffered 즉시 출력."""
    print(msg, flush=True)
    sys.stdout.flush()


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class PairDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = d.get("query", "").strip()
                t = d.get("text", "").strip()
                if q and t:
                    self.pairs.append((q, t))
        if max_samples and len(self.pairs) > max_samples:
            random.seed(42)
            self.pairs = random.sample(self.pairs, max_samples)
        log(f"[data] {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        q, t = self.pairs[i]
        return f"query: {q}", f"passage: {t[:512]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-samples", type=int, default=30000)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--vram-frac", type=float, default=0.35,
                    help="이 프로세스가 쓸 VRAM 비율 상한 (avatar 공존)")
    ap.add_argument("--abort-step-sec", type=float, default=0.8,
                    help="초반 step 평균이 이 값 초과면 자동 중단 (hang 방지)")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        log("[FATAL] CUDA 불가 - 중단")
        sys.exit(1)
    device = "cuda"

    # ── VRAM 캡: 타 프로세스(avatar)와 공존 ──
    try:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac, 0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"[gpu] VRAM 캡 {a.vram_frac*100:.0f}% of {total_gb:.1f}GB "
            f"= {total_gb*a.vram_frac:.1f}GB")
    except Exception as e:
        log(f"[gpu] VRAM 캡 실패(무시): {e}")

    log(f"[init] base={a.base} batch={a.batch} accum={a.grad_accum} "
        f"max_samples={a.max_samples}")
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModel.from_pretrained(a.base).to(device)
    model.train()

    ds = PairDataset(a.pairs, max_samples=a.max_samples)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=0)  # num_workers=0: Windows 안정성

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    total_steps = (len(dl) // a.grad_accum) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(total_steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    log(f"[train] batches/epoch={len(dl)} optim_steps={total_steps}")
    t0 = time.time()
    step = 0
    step_times = []
    opt.zero_grad()

    for ep in range(a.epochs):
        running = 0.0
        for bi, (queries, passages) in enumerate(dl):
            st = time.time()
            q_enc = tok(list(queries), padding=True, truncation=True,
                        max_length=a.max_len, return_tensors="pt").to(device)
            p_enc = tok(list(passages), padding=True, truncation=True,
                        max_length=a.max_len, return_tensors="pt").to(device)

            with torch.amp.autocast("cuda"):
                q_emb = F.normalize(mean_pool(model(**q_enc).last_hidden_state,
                                              q_enc["attention_mask"]), dim=-1)
                p_emb = F.normalize(mean_pool(model(**p_enc).last_hidden_state,
                                              p_enc["attention_mask"]), dim=-1)
                sim = q_emb @ p_emb.t() / a.temp
                labels = torch.arange(sim.size(0), device=device)
                loss = F.cross_entropy(sim, labels) / a.grad_accum

            scaler.scale(loss).backward()
            running += loss.item() * a.grad_accum

            if (bi + 1) % a.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
                step += 1
                dt = time.time() - st
                step_times.append(dt)

                # ── 속도 가드: 초반 30 step 평균이 임계 초과 → 자동 중단 ──
                if step == 30:
                    avg30 = sum(step_times) / len(step_times)
                    if avg30 > a.abort_step_sec:
                        log(f"[ABORT] step 평균 {avg30:.2f}s > {a.abort_step_sec}s "
                            f"- GPU 과부하/경쟁 추정. 학습 중단 (모델 미저장).")
                        log("[hint] batch, vram-frac, 또는 타 GPU작업 종료 후 재시도")
                        sys.exit(2)
                    log(f"[guard] 초반 30step 평균 {avg30:.3f}s/step - 정상 진행")

                if step % 50 == 0:
                    avg = running / (50 * a.grad_accum)
                    running = 0.0
                    el = time.time() - t0
                    eta = el / step * (total_steps - step)
                    vram = torch.cuda.memory_allocated(0) / 1e9
                    log(f"  step {step}/{total_steps} loss={avg:.4f} "
                        f"{el:.0f}s eta {eta:.0f}s vram={vram:.1f}GB")

    out = _P(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    log(f"[save] {out} (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
