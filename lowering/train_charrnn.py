#!/usr/bin/env python3
"""CharRNN 을 로컬 말뭉치로 학습해 checkpoints/charrnn.pt 를 만든다.

지금까지 CharRNN 은 랜덤 초기화라 "다음 문자" 예측이 균등분포(2%)였다.
학습을 시키면 같은 파이프라인으로 의미 있는 예측을 볼 수 있다.
말뭉치는 컨테이너에 이미 있는 /usr/share/common-licenses/* 라 네트워크가 필요없다.

    python3 train_charrnn.py [--minutes 5]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models import _data                      # noqa: E402
from models import model_charrnn as M         # noqa: E402


def build_dataset(device):
    """말뭉치를 문자 인덱스 하나의 긴 배열로 만든다.

    윈도우는 학습 중에 **임의 위치**에서 잘라낸다. 16자 배수 경계로만 자르면
    그 정렬의 윈도우만 외워서, 학습 정확도는 100% 인데 실제 입력(임의 오프셋)
    에서는 자신 있게 틀린다 — 칩 실측으로 확인한 실패다.
    """
    text = _data._corpus()
    vocab = _data.char_vocab(M.VOCAB)
    ids = torch.tensor([vocab[c] for c in text if c in vocab], dtype=torch.long)
    print(f"  말뭉치 {len(text):,}자 / 어휘 {len(vocab)}자 / "
          f"유효 문자 {len(ids):,}개 (임의 오프셋 샘플링)")
    return ids.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  장치 {device}")
    ids = build_dataset(device)
    offs = torch.arange(M.SEQ_LEN, device=device)
    hi = len(ids) - M.SEQ_LEN - 1

    model = M.CharRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    step = 0
    best = float("inf")
    while time.time() - t0 < args.minutes * 60:
        starts = torch.randint(0, hi, (args.batch,), device=device)
        xb = ids[starts[:, None] + offs[None, :]]
        yb = ids[starts + M.SEQ_LEN]
        h = torch.zeros(xb.shape[0], M.HIDDEN, device=device)
        c = torch.zeros(xb.shape[0], M.HIDDEN, device=device)
        logits, _, _ = model(xb, h, c)
        loss = lossf(logits, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            l = float(loss)
            best = min(best, l)
            acc = float((logits.argmax(-1) == yb).float().mean())
            print(f"  step {step:6d}  loss {l:.4f}  정확도 {acc*100:5.1f}%  "
                  f"({time.time()-t0:.0f}s)")

    out = ROOT / "checkpoints" / "charrnn.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)
    # 균등분포 대비 얼마나 나아졌는지
    import math
    print(f"\n  저장 {out}")
    print(f"  최종 loss {float(loss):.4f}  (균등분포 기준선 {math.log(M.VOCAB):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
