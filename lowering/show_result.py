#!/usr/bin/env python3
"""칩이 낸 출력을 사람이 읽을 수 있게 해석한다.

    usage: show_result.py <model>

입력은 build/<model>/input*.bin (export_tosa.py 가 저장한 실제 데이터),
출력은 build/<model>/chip_output.bin (run_probe 가 칩 메모리에서 덤프한 것).
"기준 출력과 합계가 맞는가" 만으로는 모델이 무엇을 인식했는지 알 수 없어서,
모델별로 의미 단위까지 풀어준다.
"""

import os
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CORAL_ROOT", "/workspace/lowering-project"))
SHADES = " .:-=+*#%@"
BOXES = "─│┌┐└┘"


def _color_on():
    """아스키 아트에 ANSI 색을 쓸지.

    coral.sh 가 sed 로 들여쓰기를 붙이며 파이프를 거치므로 isatty 만으로는
    안 되고, CI(GITHUB_ACTIONS 로그 뷰어는 256색을 렌더링한다)와 강제
    플래그를 함께 본다. NO_COLOR 는 관례대로 최우선.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if any(os.environ.get(k) for k in ("FORCE_COLOR", "CLICOLOR_FORCE",
                                       "GITHUB_ACTIONS")):
        return True
    return sys.stdout.isatty()


COLOR = _color_on()


def paint(row, rgb=None):
    """명암 문자 한 줄에 색을 입힌다.

    rgb([w,3] 0~255)가 있으면 픽셀 원색을 24bit 트루컬러 전경색으로,
    없으면(흑백 이미지) 256색 그레이스케일(232~255)로. 문자 자체는
    그대로 두어 색을 지운 로그(real.out 를 grep 하는 경우 등)에서도
    그림이 읽히게 한다. 탐지 박스 문자는 이미지와 구분되게 노랑.
    """
    if not COLOR:
        return row
    out, cur = [], None
    for x, ch in enumerate(row):
        if ch in BOXES:
            esc = "\033[38;5;226m"
        elif rgb is not None:
            r, g, b = rgb[x]
            esc = f"\033[38;2;{r};{g};{b}m"
        else:
            i = max(0, SHADES.find(ch))
            esc = f"\033[38;5;{232 + round(i / (len(SHADES) - 1) * 23)}m"
        if esc != cur:
            out.append(esc)
            cur = esc
        out.append(ch)
    out.append("\033[0m")
    return "".join(out)

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light "
        "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy_bear hair_drier toothbrush").split()



def tr(v, nd=2):
    """반올림하지 않고 0 방향으로 버려서 문자열로.

    softmax 결과는 수학적으로 1 이 될 수 없는데(분모에 다른 클래스의 지수항이
    항상 양수로 남는다) 반올림하면 100.00% 로 보여 오해를 준다.
    예: 99.997787% -> 99.99%
    """
    f = 10 ** nd
    return f"{int(v * f) / f:.{nd}f}"



def score_rows(logits, label_of, k=5, width=34, window=18.0):
    """분류 결과를 '점수 + 1위와의 격차'로 보여준다.

    softmax 는 쓰지 않는다. 지수를 취하는 순간 잘 학습된 모델에서는 1위가
    99.99%, 나머지가 0.00% 로 뭉개져 "2위가 얼마나 근소했는가" 가 사라진다.
    실제로 CIFAR-100 튤립은 1·2위 점수 차가 12.35 인데 softmax 로는 둘 다
    100.00 / 0.00 으로만 보였다.

    막대는 1위 기준 window(로짓 12) 안에서의 상대 위치다. 모델이 확신하면
    막대가 크게 벌어지고, 헷갈리면 다 같이 길어진다 — 분포 모양이 그대로 보인다.
    """
    order = np.argsort(-logits)[:k]
    top = float(logits[order[0]])
    lines = []
    for r in order:
        v = float(logits[r])
        frac = max(0.0, min(1.0, (v - (top - window)) / window))
        bar = "█" * max(1, int(frac * width))   # 빈 막대는 두지 않는다
        gap = "" if r == order[0] else f"Δ {tr(top - v, 3):>7}"
        lines.append(f"    {label_of(r):16s} score {tr(v, 3):>9}  {gap:>10}  {bar}")
    margin = top - float(logits[order[1]]) if len(order) > 1 else float("inf")
    return order, lines, margin


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def ascii_art(img, w=28, invert=False):
    """[H,W] 흑백 또는 [3,H,W] 컬러 배열을 아스키 명암으로.

    (rows, rgb) 를 돌려준다. rgb 는 문자 격자와 같은 [h,w,3] 0~255 배열로
    paint() 가 트루컬러에 쓰고, 흑백 입력이면 None 이다. 정규화(imagenet
    등)로 틀어진 값 범위는 전체 min/max 로 되돌려 근사한다.
    """
    a = np.asarray(img, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo + 1e-9)
    if invert:
        a = 1.0 - a
    rgb = None
    if a.ndim == 3 and a.shape[0] == 1:    # [1,H,W] 단일 채널은 흑백으로
        a = a[0]
    if a.ndim == 3:                        # [3,H,W]
        h_step = max(1, a.shape[1] // 28)
        w_step = max(1, a.shape[2] // w)
        a = a[:, ::h_step, ::w_step]
        rgb = np.clip(a.transpose(1, 2, 0) * 255, 0, 255).astype(int)
        a = a.mean(axis=0)                 # 명암 문자용 휘도
    else:
        h_step = max(1, a.shape[0] // 28)
        w_step = max(1, a.shape[1] // w)
        a = a[::h_step, ::w_step]
    rows = []
    for r in a:
        rows.append("".join(SHADES[min(len(SHADES) - 1, int(v * len(SHADES)))]
                            for v in r))
    return rows, rgb


def print_art(img, w=28):
    rows, rgb = ascii_art(img, w=w)
    for i, row in enumerate(rows):
        print("    " + paint(row, rgb[i] if rgb is not None else None))



def identify_photo(img):
    """입력 텐서가 data/photos 의 어느 사진인지 되찾는다.

    manifest 에 출처가 안 남아 있어서, 실제 파일을 같은 크기로 읽어
    가장 가까운 것을 고른다. 정규화 방식(imagenet / [0,1])도 함께 판별한다.
    """
    sys.path.insert(0, str(ROOT))
    from models import _data
    size = img.shape[-1]
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    best = None
    for q in _data.photo_paths():
        raw = _data._load_png(q, size).numpy()[0]
        for tag, cand in (("[0,1]", raw), ("imagenet", (raw - mean) / std)):
            d = float(np.abs(cand - img).mean())
            if best is None or d < best[0]:
                best = (d, q.name, tag)
    if best and best[0] < 1e-4:
        return f"{best[1]} ({best[2]} 정규화)"
    return "출처 불명 (합성/아이콘 등)"


def load(model):
    man = json.loads((ROOT / "build" / model / "manifest.json").read_text())
    chip = np.fromfile(ROOT / "build" / model / "chip_output.bin", dtype=np.float32)
    outs, off = [], 0
    for o in man["abi"]["outputs"]:
        n = o["bytes"] // 4
        outs.append(chip[off:off + n].reshape(o["shape"]))
        off += n
    ins = []
    for i in man["abi"]["inputs"]:
        dt = {"f32": np.float32, "i64": np.int64, "i32": np.int32,
              "i8": np.int8}[i["dtype"]]
        ins.append(np.fromfile(ROOT / "build" / model / i["file"],
                               dtype=dt).reshape(i["shape"]))
    return man, ins, outs


# ------------------------------------------------------------------ 모델별 ---

def show_lenet5(model, man, ins, outs):
    img = ins[0][0, 0]
    print("  입력: MNIST 손글씨 28x28 (실제 테스트 이미지)")
    print_art(img)
    logits = outs[0][0]
    print("\n  추론 결과")
    order, lines, margin = score_rows(logits, lambda r: f"숫자 {r}", k=4)
    for l in lines:
        print(l)
    print(f"\n  → 예측: {order[0]}   (2위와 {tr(margin,3)} 차이)")


def show_charrnn(model, man, ins, outs):
    sys.path.insert(0, str(ROOT))
    from models import _data
    vocab = _data.char_vocab(64)
    inv = {v: k for k, v in vocab.items()}
    ids = ins[0][0]
    text = "".join(inv.get(int(t), "?") for t in ids)
    print(f"  입력: 문자 인덱스 {len(ids)}개 (실제 말뭉치에서)")
    print(f"    시퀀스 = {ids.tolist()}")
    print(f"    텍스트 = {text!r}")
    logits = outs[0][0]
    print("\n  다음 문자 예측 (상위 5)")
    order, lines, margin = score_rows(logits, lambda r: repr(inv.get(int(r), "?")))
    for l in lines:
        print(l)
    print(f"\n  → 이어질 문자: {inv.get(int(order[0]), '?')!r}   "
          f"(2위와 {tr(margin,3)} 차이)")
    print(f"  → 갱신된 은닉 상태 h[0:4] = "
          f"{np.array2string(outs[1][0][:4], precision=4)}")
    print(f"     셀 상태      c[0:4] = "
          f"{np.array2string(outs[2][0][:4], precision=4)}")


def show_vgg(model, man, ins, outs):
    img = ins[0][0]
    print(f"  입력: {identify_photo(img)}  -> {img.shape[1]}x{img.shape[2]}")
    print_art(img, w=32)
    feat = outs[0][0].reshape(outs[0].shape[1], -1).mean(axis=1)
    order = np.argsort(-feat)
    nz = int((feat > 0).sum())
    print(f"\n  추론 결과: 512차원 특징 벡터 "
          f"(VGG16 backbone 은 분류기가 없어 특징까지만 낸다)")
    print(f"    활성 채널 {nz}/512   최대 {feat.max():.4f}   평균 {feat.mean():.4f}")
    print("    가장 강하게 반응한 채널 상위 8개")
    for r in order[:8]:
        bar = "█" * int(feat[r] / (feat.max() + 1e-9) * 34)
        print(f"      ch{r:3d}  {feat[r]:8.4f}  {bar}")


def show_yolo(model, man, ins, outs):
    img = ins[0][0]
    print(f"  입력: {identify_photo(img)}  -> {img.shape[1]}x{img.shape[2]}")
    print_art(img, w=36)
    print(f"\n  추론 결과: 2개 스케일의 탐지 맵 "
          f"(NMS 는 모델 밖이라 원시 예측을 그대로 해석)")
    dets = []
    for si, o in enumerate(outs):
        _, ch, gh, gw = o.shape
        na = 3
        step = ch // na                      # 85 = 5(box+obj) + 80(class)
        nc = step - 5
        t = o[0].reshape(na, step, gh, gw)
        obj = sigmoid(t[:, 4])               # [na,gh,gw]
        cls = t[:, 5:]                       # [na,nc,gh,gw]
        print(f"    스케일 {si}: {gh}x{gw} 격자 x {na} anchor "
              f"= {na*gh*gw}개 후보, 클래스 {nc}개")
        print(f"      objectness  최대 {obj.max():.4f}  평균 {obj.mean():.4f}")
        for a in range(na):
            for y in range(gh):
                for x in range(gw):
                    ci = int(np.argmax(cls[a, :, y, x]))
                    score = float(obj[a, y, x]) * float(
                        sigmoid(cls[a, ci, y, x]))
                    dets.append((score, si, a, y, x, ci, float(obj[a, y, x])))
    dets.sort(reverse=True)
    print("\n    점수 상위 5개 후보 (objectness x class 확률)")
    for sc, si, a, y, x, ci, ob in dets[:5]:
        name = COCO[ci] if ci < len(COCO) else f"class{ci}"
        print(f"      스케일{si} 격자({y},{x}) anchor{a}  "
              f"{name:14s} 점수 {sc:.4f}  obj {ob:.4f}")
    print("\n    (랜덤 초기화 가중치 모델이라 클래스 자체는 의미가 없다."
          "\n     확인하려는 것은 두 헤드가 격자 전체에 걸쳐 값을 만들어냈는가다.)")



def show_resnet(model, man, ins, outs):
    """ImageNet 1000 클래스 분류. 클래스 이름은 torchvision 메타에서 가져온다."""
    import torchvision
    cats = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    sys.path.insert(0, str(ROOT))
    from models import _data
    paths = _data.photo_paths()
    src = paths[0].name if paths else "?"
    img = ins[0][0]
    print(f"  입력: 실제 사진 {src}  ({img.shape[1]}x{img.shape[2]}, ImageNet 정규화)")
    print_art(img, w=44)
    logits = outs[0][0]
    print("\n  추론 결과 (ImageNet 1000 클래스)")
    order, lines, margin = score_rows(logits, lambda r: cats[r][:22])
    for l in lines:
        print(l)
    print(f"\n  → 예측: {cats[order[0]]}   (2위와 {tr(margin,3)} 차이)")



def show_yolo_real(model, man, ins, outs):
    """다크넷 사전학습 YOLOv3-tiny. 실제 탐지 결과를 박스까지 풀어서 보여준다."""
    import importlib.util
    sys.path.insert(0, str(ROOT))
    from models import _data
    from models.darknet import DarknetModel

    spec = importlib.util.spec_from_file_location("vy", ROOT / "verify_yolo.py")
    vy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vy)

    size = ins[0].shape[-1]
    names = (ROOT / "data/darknet/coco.names").read_text().split()
    # anchors 만 필요하므로 가중치는 읽지 않는다 (35 MB 절약).
    dn = DarknetModel(ROOT / "data/darknet/yolov3-tiny.cfg")
    meta = _data.letterbox_meta("bus", size)

    import torch
    tos = [torch.from_numpy(np.ascontiguousarray(o)) for o in outs]
    dets = vy.nms(vy.decode(tos, dn, size, meta))

    W0, H0, sc, ox, oy = meta
    print(f"  입력: bus.jpg {W0}x{H0} -> letterbox {size}x{size} "
          f"(다크넷 전처리)")

    # 아스키 위에 박스를 겹쳐 그린다. 좌표는 원본 기준이라 격자로 환산한다.
    AW, AH = 56, 30
    rows, rgb = ascii_art(ins[0][0], w=AW)
    rows = [list(r) for r in rows]
    ah = len(rows)
    aw = len(rows[0]) if ah else 0
    for _, ci, x0, y0, bw, bh in dets:
        # 원본 좌표 -> letterbox 좌표 -> 아스키 격자
        gx0 = int((x0 * sc + ox) / size * aw)
        gx1 = int(((x0 + bw) * sc + ox) / size * aw)
        gy0 = int((y0 * sc + oy) / size * ah)
        gy1 = int(((y0 + bh) * sc + oy) / size * ah)
        gx0, gx1 = max(0, min(aw - 1, gx0)), max(0, min(aw - 1, gx1))
        gy0, gy1 = max(0, min(ah - 1, gy0)), max(0, min(ah - 1, gy1))
        for x in range(gx0, gx1 + 1):
            rows[gy0][x] = "─"
            rows[gy1][x] = "─"
        for y in range(gy0, gy1 + 1):
            rows[y][gx0] = "│"
            rows[y][gx1] = "│"
        rows[gy0][gx0] = "┌"; rows[gy0][gx1] = "┐"
        rows[gy1][gx0] = "└"; rows[gy1][gx1] = "┘"
    for i, r in enumerate(rows):
        print("    " + paint("".join(r), rgb[i]))

    print(f"\n  탐지 {len(dets)}건 (objectness x class, NMS 적용)")
    for scr, ci, x0, y0, bw, bh in dets:
        bar = "█" * int(scr * 30)
        print(f"    {names[ci]:10s} detection score {tr(scr*100,2):>6}%  "
              f"박스 ({x0:6.0f},{y0:6.0f}) {bw:5.0f}x{bh:5.0f}  {bar}")



def show_cifar100(model, man, ins, outs):
    """CIFAR-100 100 클래스 분류."""
    sys.path.insert(0, str(ROOT))
    from models import _data
    cls = _data.cifar100_classes()
    idx = (man.get("calibration") or {}).get("sample_index", 0)
    truth = (man.get("calibration") or {}).get("label")
    img = ins[0][0]
    print(f"  입력: CIFAR-100 test[{idx}]"
          + (f"  정답 = {truth}" if truth else "")
          + f"   ({img.shape[1]}x{img.shape[2]})")
    print_art(img, w=32)
    logits = outs[0][0]
    print("\n  추론 결과 (CIFAR-100 100 클래스)")
    order, lines, margin = score_rows(logits, lambda r: cls[r])
    for l in lines:
        print(l)
    got = cls[order[0]]
    mark = "정답" if truth and got == truth else ("오답" if truth else "")
    print(f"\n  → 예측: {got}   (2위와 {tr(margin,3)} 차이)  {mark}")


RENDER = {"resnet18": show_resnet, "vgg_cifar100": show_cifar100, "yolov3_tiny": show_yolo_real, "lenet5": show_lenet5, "lenet5_s1": show_lenet5,
          "lenet5_s2": show_lenet5, "charrnn": show_charrnn,
          "vgg_small": show_vgg, "vgg_backbone": show_vgg,
          "yolov3_small": show_yolo, "yolov3": show_yolo}


def main():
    model = sys.argv[1]
    man, ins, outs = load(model)
    par = man.get("params")
    if isinstance(par, dict):
        par = par.get("total") or par.get("count") or next(iter(par.values()), None)
    head = f" {model}" + (f"   파라미터 {par:,}" if isinstance(par, int) else "")
    print("=" * 62)
    print(head)
    print("=" * 62)
    RENDER.get(model, lambda *a: print("  (해석기 없음)"))(model, man, ins, outs)
    print()


if __name__ == "__main__":
    raise SystemExit(main())
