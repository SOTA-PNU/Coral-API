#!/usr/bin/env python3
"""칩 출력을 기준 출력과 대조해 합격/불합격을 판정한다.

    usage: check_result.py <model> <run.log> [--expect-argmax N]

두 가지를 본다
--------------
1) 의미 검사 (있으면 이쪽이 판정 기준)
   분류 모델 -> top-1 클래스가 같은가
   탐지 모델 -> 탐지 목록(개수/클래스)이 같고 박스 IoU 가 충분히 큰가

2) 수치 검사 (참고)
   상대 L2 오차 ||칩 - 참조|| / ||참조||

왜 나누는가
-----------
모델이 하는 일이 정확한데도 수치 오차만으로 불합격이 나오는 일이 있었다.
YOLOv3-tiny 는 출력 172,380 개 중 대부분이 objectness 가 낮아 버려지는 격자이고,
거기서 나는 오차는 탐지 결과에 영향이 없다. 실제로 상대 L2 1.5e-2 인데
탐지는 호스트와 완전히 같았다(사람 4 + 버스 1).

그렇다고 허용치를 그냥 올리면 진짜 회귀를 놓친다. 그래서 판정은 의미 기준으로
하되, 수치 오차는 항상 같이 찍고 허용치를 넘으면 경고를 남긴다.

합계의 상대오차는 쓰지 않는다. 출력 원소가 서로 상쇄되면(CharRNN 의 은닉
상태 h 는 -1~1 값 128개의 합이 -2.9) 분모가 0 에 가까워져 지표가 무너진다.
"""

import os
import json
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_TOL = 1e-2
ROOT = Path(os.environ.get("CORAL_ROOT", "/workspace/lowering-project"))

# 모델별 수치 허용치. 기본값으로 판정 못 하는 것만 근거와 함께 적는다.
#
# charrnn: LSTM 16스텝 전개라 스텝마다 재양자화가 일어나고, 은닉/셀 상태가
#   다음 스텝 입력으로 되먹임되어 오차가 증폭된다. 실측 상대 L2 4~6%.
MODEL_TOL = {"charrnn": 1e-1}

CLASSIFIERS = {"lenet5", "lenet5_s1", "lenet5_s2", "resnet18", "vgg_cifar100"}
DETECTORS = {"yolov3_tiny"}


def grab(log, name):
    m = re.search(rf"^{re.escape(name)} = \[([^\]]*)\]", log, re.M)
    return None if not m else m.group(1).split()


def load_outputs(model, manifest):
    bdir = ROOT / "build" / model
    chip = np.fromfile(bdir / "chip_output.bin", dtype=np.float32)
    chips, refs, off = [], [], 0
    for o in manifest["abi"]["outputs"]:
        n = o["bytes"] // 4
        chips.append(chip[off:off + n].reshape(o["shape"]))
        refs.append(np.fromfile(bdir / o["file"],
                                dtype=np.float32).reshape(o["shape"]))
        off += n
    return chips, refs


def check_classifier(model, chips, refs):
    a, b = int(np.argmax(chips[0])), int(np.argmax(refs[0]))
    top5a = list(np.argsort(-chips[0].ravel())[:5])
    top5b = list(np.argsort(-refs[0].ravel())[:5])
    ok = a == b
    lines = [f"      top-1: 칩 {a}, 참조 {b}  {'일치' if ok else '불일치'}",
             f"      top-5 순서 {'일치' if top5a == top5b else '다름'}"]
    return ok, lines


def check_detector(model, chips, refs):
    import importlib.util
    import torch
    sys.path.insert(0, str(ROOT))
    from models import _data
    from models.darknet import DarknetModel
    spec = importlib.util.spec_from_file_location("vy", ROOT / "verify_yolo.py")
    vy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vy)

    size = 416
    dn = DarknetModel(ROOT / "data/darknet/yolov3-tiny.cfg")
    meta = _data.letterbox_meta("bus", size)
    names = (ROOT / "data/darknet/coco.names").read_text().split()

    def det(outs):
        t = [torch.from_numpy(np.ascontiguousarray(o)) for o in outs]
        return vy.nms(vy.decode(t, dn, size, meta))

    W0, H0 = meta[0], meta[1]

    def clip(x, y, w, h):
        """박스를 이미지 경계로 자른다.

        화면 밖으로 잘린 물체(이 사진의 버스)는 모델이 프레임 바깥까지
        박스를 외삽하는데, 그 영역은 실재하지 않으면서 IoU 차이만 부풀린다.
        탐지 평가에서 이미지 경계로 clip 하는 것은 표준 관행이다.
        """
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W0), x + w), min(float(H0), y + h)
        return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)

    da, db = det(chips), det(refs)
    lines = [f"      탐지 수: 칩 {len(da)}, 참조 {len(db)}"]
    ok = len(da) == len(db)
    if ok:
        for (sa, ca, *ba), (sb, cb, *bb) in zip(da, db):
            xa, ya, wa, ha = clip(*ba)
            xb, yb, wb, hb = clip(*bb)
            ix = max(0, min(xa + wa, xb + wb) - max(xa, xb))
            iy = max(0, min(ya + ha, yb + hb) - max(ya, yb))
            inter = ix * iy
            iou = inter / (wa * ha + wb * hb - inter + 1e-9)
            same = ca == cb and iou >= 0.9
            ok = ok and same
            lines.append(f"      {names[ca]:10s} IoU {iou:.4f}  "
                         f"점수 칩 {sa*100:5.1f}% 참조 {sb*100:5.1f}%  "
                         f"{'일치' if same else '다름'}")
    return ok, lines


def main():
    model = sys.argv[1]
    log = Path(sys.argv[2]).read_text(errors="replace")
    expect_argmax = None
    if "--expect-argmax" in sys.argv:
        expect_argmax = int(sys.argv[sys.argv.index("--expect-argmax") + 1])

    bdir = ROOT / "build" / model
    manifest = json.loads((bdir / "manifest.json").read_text())

    stage = grab(log, "inference_stage")
    code = grab(log, "inference_status_code")
    cycles = re.search(r"^cycles = (\d+)", log, re.M)

    if stage is None:
        print(f"  ✗ {model}: 시뮬레이터 출력이 없습니다 (실행 실패 또는 타임아웃)")
        return 1
    if int(stage[0]) != 9 or (code and int(code[0]) != 0):
        msg = re.search(r"^status_msg = '(.*)'", log, re.M | re.S)
        print(f"  ✗ {model}: stage={int(stage[0])} (9 여야 함), "
              f"status={int(code[0]) if code else '?'}")
        if msg:
            print(f"      {msg.group(1)[:220]}")
        return 1
    if not (bdir / "chip_output.bin").is_file():
        print(f"  ✗ {model}: chip_output.bin 이 없습니다")
        return 1

    chips, refs = load_outputs(model, manifest)
    tol = MODEL_TOL.get(model, DEFAULT_TOL)

    # --- 수치 (참고) ---
    num_lines, num_ok = [], True
    for i, (c, r) in enumerate(zip(chips, refs)):
        d = (c - r).ravel()
        rel = float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-12))
        over = rel > tol
        num_ok = num_ok and not over
        num_lines.append(f"      출력{i}: 원소 {c.size:>6d}개  상대L2 {rel:.2e} "
                         f"(허용 {tol:.0e})  최대차 {np.abs(d).max():.3e}"
                         f"{'  초과' if over else ''}")

    # --- 의미 (판정) ---
    if model in CLASSIFIERS:
        sem_ok, sem_lines = check_classifier(model, chips, refs)
        task = "분류: top-1 일치"
    elif model in DETECTORS:
        sem_ok, sem_lines = check_detector(model, chips, refs)
        task = "탐지: 목록/클래스/IoU 일치"
    else:
        sem_ok, sem_lines, task = None, [], None

    ok = num_ok if sem_ok is None else sem_ok
    if expect_argmax is not None:
        am = grab(log, "inference_argmax")
        got = int(am[0]) if am else -1
        if got != expect_argmax:
            ok = False
            sem_lines.append(f"      argmax: 칩 {got}, 기대 {expect_argmax}  다름")
        else:
            sem_lines.append(f"      argmax: {got}  기대값과 일치")

    cyc = int(cycles.group(1)) if cycles else 0
    print(f"  {'✓' if ok else '✗'} {model}: {cyc:,} 사이클")
    if task:
        print(f"    [판정] {task}")
        for l in sem_lines:
            print(l)
        print(f"    [참고] 수치 대조")
    for l in num_lines:
        print(l)
    if task and not num_ok:
        print("      (수치는 허용치를 넘지만 결과는 같다. 대부분 버려지는 "
              "출력에서 나는 재양자화 반올림 차이다.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
