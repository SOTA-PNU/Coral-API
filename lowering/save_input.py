#!/usr/bin/env python3
"""모델이 실제로 보는 입력 텐서를 PNG 로 저장한다.

    python3 save_input.py [<model> ...]      # 지정 안 하면 이미지 모델 전부

아스키 아트는 대략의 형태만 보여주므로, 32x32 로 줄어든 뒤 정규화까지 거친
"모델의 눈에 보이는 그림" 을 그대로 파일로 남긴다.

    build/<model>/input_preview.png      원본 해상도 (예: 32x32)
    build/<model>/input_preview_x8.png   보기 좋게 확대 (nearest, 최소 256px)

정규화는 되돌려서 저장한다. ImageNet 정규화면 mean/std 를 역산하고,
[0,1] 이면 그대로 쓴다. 판별은 값의 범위로 한다.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace/lowering-project")
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def denorm(x):
    """(3,H,W) 또는 (1,H,W) 실수 텐서를 [0,1] 화면값으로."""
    if x.shape[0] == 3 and (x.min() < -0.05 or x.max() > 1.05):
        # ImageNet 정규화로 보인다 -> 역산
        return np.clip(x * STD + MEAN, 0, 1)
    return np.clip(x, 0, 1)


def save(model):
    bdir = ROOT / "build" / model
    man = json.loads((bdir / "manifest.json").read_text())
    inp = man["abi"]["inputs"][0]
    if inp["dtype"] != "f32" or len(inp["shape"]) != 4:
        print(f"  - {model}: 이미지 입력이 아님 {inp['shape']} {inp['dtype']}")
        return
    _, c, h, w = inp["shape"]
    x = np.fromfile(bdir / inp["file"], dtype=np.float32).reshape(c, h, w)
    img = denorm(x)
    arr = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    if c == 1:
        arr = arr[:, :, 0]
    im = Image.fromarray(arr)
    p1 = bdir / "input_preview.png"
    im.save(p1)
    k = max(1, -(-256 // max(h, w)))          # 256px 이상이 되도록 정수배 확대
    p2 = bdir / f"input_preview_x{k}.png"
    im.resize((w * k, h * k), Image.NEAREST).save(p2)
    print(f"  ✓ {model}: {w}x{h}x{c}  ->  {p1.name}, {p2.name} ({w*k}x{h*k})")


def main():
    models = sys.argv[1:]
    if not models:
        models = [d.name for d in sorted((ROOT / "build").iterdir())
                  if (d / "manifest.json").is_file()]
    for m in models:
        try:
            save(m)
        except Exception as e:
            print(f"  ! {m}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
