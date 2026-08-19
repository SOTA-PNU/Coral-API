#!/usr/bin/env python3
"""다크넷 가중치를 얹은 YOLOv3-tiny 가 bus.jpg 를 제대로 탐지하는지 호스트에서 확인.

파이프라인에 태우기 전에 여기서 먼저 맞아야 한다. 버스 1대와 사람 여럿이
나오면 가중치 매핑이 정확한 것이다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/lowering-project")
sys.path.insert(0, str(ROOT))
from models.darknet import DarknetModel  # noqa: E402


def letterbox(img_path, size):
    """다크넷 전처리: 비율 유지 축소 + 회색(0.5) 패딩."""
    from PIL import Image
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((size, size, 3), 0.5, dtype=np.float32)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(im, np.float32) / 255.0
    t = torch.from_numpy(canvas).permute(2, 0, 1)[None]
    return t.contiguous(), (w, h, s, ox, oy)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def decode(outs, model, size, meta, conf_thr=0.25):
    W0, H0, s, ox, oy = meta
    heads = [m for m in model.module_list if hasattr(m, "mask")]
    dets = []
    for o, head in zip(outs, heads):
        t = o[0].numpy()
        na, nc = len(head.mask), head.num_classes
        _, gh, gw = t.shape
        t = t.reshape(na, 5 + nc, gh, gw)
        anchors = [head.anchors[i] for i in head.mask]
        for a in range(na):
            obj = sigmoid(t[a, 4])
            cls = sigmoid(t[a, 5:])
            ci = cls.argmax(axis=0)
            cs = cls.max(axis=0)
            score = obj * cs
            ys, xs = np.where(score > conf_thr)
            for y, x in zip(ys, xs):
                bx = (sigmoid(t[a, 0, y, x]) + x) / gw * size
                by = (sigmoid(t[a, 1, y, x]) + y) / gh * size
                bw = anchors[a][0] * np.exp(t[a, 2, y, x])
                bh = anchors[a][1] * np.exp(t[a, 3, y, x])
                # letterbox 되돌리기
                x0 = (bx - bw / 2 - ox) / s
                y0 = (by - bh / 2 - oy) / s
                dets.append((float(score[y, x]), int(ci[y, x]),
                             x0, y0, bw / s, bh / s))
    return dets


def nms(dets, iou_thr=0.45):
    dets = sorted(dets, reverse=True)
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            if d[1] != k[1]:
                continue
            ax0, ay0, aw, ah = d[2], d[3], d[4], d[5]
            bx0, by0, bw, bh = k[2], k[3], k[4], k[5]
            ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
            iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
            inter = ix * iy
            iou = inter / (aw * ah + bw * bh - inter + 1e-9)
            if iou > iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 416
    names = (ROOT / "data/darknet/coco.names").read_text().split()
    m = DarknetModel(ROOT / "data/darknet/yolov3-tiny.cfg")
    m.load_darknet_weights(ROOT / "data/darknet/yolov3-tiny.weights")
    m.eval()
    x, meta = letterbox(ROOT / "data/photos/bus.jpg", size)
    with torch.no_grad():
        outs = m(x)
    dets = nms(decode(outs, m, size, meta))
    print(f"  입력 {size}x{size} (letterbox), 원본 {meta[0]}x{meta[1]}")
    print(f"  탐지 {len(dets)}건")
    for sc, ci, x0, y0, w, h in dets:
        print(f"    {names[ci]:12s} {sc*100:5.1f}%   "
              f"박스 ({x0:6.0f},{y0:6.0f}) {w:5.0f}x{h:5.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
