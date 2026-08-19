"""모델 공용 데이터 소스.

오프라인 환경이라 ImageNet/COCO/대규모 코퍼스가 없다. 대신 시스템에 실재하는
데이터를 최대한 활용하고, 부족한 부분만 자연 이미지 통계를 흉내 낸 합성으로
채운다. 순수 torch.randn 보다 값 분포가 현실에 가까워 calibration scale 이
덜 왜곡된다.

    이미지  : PNG 실파일 + MNIST 업스케일 + 1/f 합성  (3종 혼합)
    텍스트  : /usr/share/common-licenses/GPL-3       (실제 영문)
    MNIST   : data/MNIST                              (실데이터)

calibration 은 "값의 범위를 재는" 작업이므로 데이터의 **분포**가 중요하고
의미(라벨)는 중요하지 않다. 그래서 도식 PNG 라도 백색잡음보다 낫다.
"""

from pathlib import Path
import glob
import struct

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_MNIST_RAW = _ROOT / "data" / "MNIST" / "raw"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- MNIST ---

def _read_idx_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"unexpected magic {magic} in {path}"
        buf = f.read(count * rows * cols)
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    return t.view(count, 1, rows, cols).float().div_(255.0)   # [N,1,28,28] [0,1]


_mnist_cache = {}


def mnist_images(train=True):
    """[N,1,28,28] float [0,1]. torchvision 없이 idx 파일을 직접 읽는다."""
    key = "train" if train else "test"
    if key not in _mnist_cache:
        name = ("train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte")
        _mnist_cache[key] = _read_idx_images(_MNIST_RAW / name)
    return _mnist_cache[key]


# ---------------------------------------------------------------- 이미지 ---

_png_cache = None


def _png_paths(limit=256):
    global _png_cache
    if _png_cache is None:
        found = []
        for pat in ("/workspace/coralnpu/**/*.png", "/workspace/iree/**/*.png",
                    "/workspace/torch-mlir/**/*.png"):
            found += glob.glob(pat, recursive=True)
        found = [p for p in found if "/.git/" not in p]
        _png_cache = sorted(set(found))[:limit]
    return _png_cache


def _load_png(path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
    return t.view(1, size, size, 3).permute(0, 3, 1, 2).float().div_(255.0)


def _synth_natural(size, gen):
    """1/f 스펙트럼을 흉내 낸 합성 이미지.

    저해상도 노이즈를 bilinear 업샘플하면 공간 상관이 생겨 실제 사진의
    저주파 우세 특성에 가까워진다. 백색잡음은 이 성질이 없다.
    """
    low = torch.randn(1, 3, max(size // 32, 2), max(size // 32, 2), generator=gen)
    img = F.interpolate(low, size=(size, size), mode="bilinear",
                        align_corners=False)
    img = img + 0.25 * torch.randn(1, 3, size, size, generator=gen)
    lo, hi = img.amin(), img.amax()
    return (img - lo) / (hi - lo + 1e-8)                       # [0,1]


def _mnist_as_rgb(size, gen):
    """MNIST 숫자를 컬러 이미지로. 실제 필기 텍스처와 희소성을 갖는다."""
    imgs = mnist_images(train=True)
    idx = int(torch.randint(0, imgs.shape[0], (1,), generator=gen).item())
    one = imgs[idx:idx + 1]                                    # [1,1,28,28]
    up = F.interpolate(one, size=(size, size), mode="bilinear",
                       align_corners=False)
    tint = 0.4 + 0.6 * torch.rand(1, 3, 1, 1, generator=gen)
    return (up * tint).clamp(0, 1)


def image_batches(size=224, n=32, normalize="imagenet", seed=0):
    """3종 소스를 번갈아 내보낸다. yield 는 (tensor,) 튜플.

    normalize : "imagenet" -> torchvision 계열 전처리
                None       -> [0,1] 그대로 (darknet/YOLO 계열 관례)
    """
    gen = torch.Generator().manual_seed(seed)
    pngs = _png_paths()
    for i in range(n):
        which = i % 3
        if which == 0 and pngs:
            img = _load_png(pngs[i % len(pngs)], size)
        elif which == 1:
            img = _mnist_as_rgb(size, gen)
        else:
            img = _synth_natural(size, gen)
        if normalize == "imagenet":
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        yield (img.contiguous(),)


def image_example(size=224, normalize="imagenet"):
    """대표 입력 1장. calibration 과 같은 파이프라인을 쓴다."""
    return next(iter(image_batches(size=size, n=1, normalize=normalize, seed=7)))


def image_source_info(size, normalize):
    return {"source": "PNG 실파일 + MNIST 업스케일 + 1/f 합성 (3종 혼합)",
            "png_pool": len(_png_paths()), "size": size,
            "normalize": normalize or "[0,1]",
            "real_data": True,
            "note": "라벨 없음. 값 분포만 현실적이며 정확도 평가에는 쓸 수 없음"}


# ---------------------------------------------------------------- 텍스트 ---

_TEXT_CANDIDATES = [
    "/usr/share/common-licenses/GPL-3",
    "/usr/share/common-licenses/Apache-2.0",
    "/usr/share/common-licenses/BSD",
]

_text_cache = None


def _corpus():
    global _text_cache
    if _text_cache is None:
        parts = []
        for p in _TEXT_CANDIDATES:
            try:
                parts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
        _text_cache = "\n".join(parts) or ("hello world " * 4096)
    return _text_cache


def char_vocab(vocab_size):
    """말뭉치에서 빈도 상위 vocab_size 문자로 사전을 만든다."""
    from collections import Counter
    counts = Counter(_corpus())
    chars = [ch for ch, _ in counts.most_common(vocab_size)]
    chars.sort()
    return {ch: i for i, ch in enumerate(chars)}


def text_sequences(vocab_size=64, seq_len=16, n=32, seed=0):
    """실제 영문에서 잘라낸 문자 인덱스 시퀀스. yield 는 [1, seq_len] int64."""
    stoi = char_vocab(vocab_size)
    text = _corpus()
    ids = [stoi[ch] for ch in text if ch in stoi]
    gen = torch.Generator().manual_seed(seed)
    hi = max(len(ids) - seq_len - 1, 1)
    for _ in range(n):
        start = int(torch.randint(0, hi, (1,), generator=gen).item())
        chunk = ids[start:start + seq_len]
        chunk += [0] * (seq_len - len(chunk))
        yield torch.tensor(chunk, dtype=torch.int64).unsqueeze(0)


def text_source_info(vocab_size, seq_len):
    return {"source": "/usr/share/common-licenses (GPL-3, Apache-2.0, BSD)",
            "corpus_chars": len(_corpus()), "vocab": vocab_size,
            "seq_len": seq_len, "real_data": True,
            "note": "실제 영문 문자 빈도 분포. 학습된 모델이 아니므로 정확도는 무의미"}

# --------------------------------------------------------------- 실사진 ---
#
# data/photos/ 의 실제 사진. 파일명에 ImageNet 정답 클래스가 들어 있어
# (예: n02123045_tabby.jpg) 추론 결과를 사람이 검증할 수 있다.
# _png_paths() 가 긁어오는 아이콘 이미지와 달리 의미 있는 입력이다.

_PHOTO_DIR = Path(__file__).resolve().parent.parent / "data" / "photos"


def photo_paths():
    if not _PHOTO_DIR.is_dir():
        return []
    return sorted(p for p in _PHOTO_DIR.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def photo_label(path):
    """n02123045_tabby.jpg -> 'tabby'"""
    stem = Path(path).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def photo_by_name(name):
    """파일명(확장자 제외)으로 사진을 고른다. 없으면 예외."""
    for p in photo_paths():
        if p.stem == name or p.name == name:
            return p
    raise FileNotFoundError(
        f"data/photos 에 {name} 이 없습니다. 있는 것: "
        f"{[p.name for p in photo_paths()]}")


def real_photo(size=224, normalize="imagenet", index=0, name=None):
    """실제 사진 1장을 모델 입력 텐서로. 반환은 (tensor,) 튜플."""
    if name is not None:
        img = _load_png(photo_by_name(name), size)
    else:
        paths = photo_paths()
        if not paths:
            raise FileNotFoundError(f"{_PHOTO_DIR} 에 사진이 없습니다")
        img = _load_png(paths[index % len(paths)], size)
    if normalize == "imagenet":
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return (img.contiguous(),)


def real_photo_batches(size=224, n=16, normalize="imagenet"):
    """캘리브레이션용. 사진을 순환하며 내보낸다."""
    paths = photo_paths()
    for i in range(n):
        img = _load_png(paths[i % len(paths)], size)
        if normalize == "imagenet":
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        yield (img.contiguous(),)


def letterbox_photo(name, size):
    """다크넷 전처리: 비율을 유지해 줄이고 남는 곳을 회색(0.5)으로 채운다.

    YOLO 계열은 정사각형으로 눌러 담으면 종횡비가 망가져 탐지가 나빠진다.
    반환은 ((tensor,), meta) 가 아니라 (tensor,) — 모델 계약에 맞춘다.
    meta 가 필요하면 letterbox_meta 를 쓴다.
    """
    return (_letterbox_impl(name, size)[0],)


def letterbox_meta(name, size):
    """(원본폭, 원본높이, 축소비, x오프셋, y오프셋). 박스를 원본 좌표로 되돌릴 때 쓴다."""
    return _letterbox_impl(name, size)[1]


def _letterbox_impl(name, size):
    from PIL import Image
    im = Image.open(photo_by_name(name)).convert("RGB")
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((size, size, 3), 0.5, dtype=np.float32)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(im, np.float32) / 255.0
    t = torch.from_numpy(canvas).permute(2, 0, 1)[None].contiguous()
    return t, (w, h, s, ox, oy)


def letterbox_batches(size, n=16):
    """캘리브레이션용. data/photos 의 사진을 순환한다."""
    for i, q in enumerate(photo_paths() * ((n // max(1, len(photo_paths()))) + 1)):
        if i >= n:
            break
        yield (_letterbox_impl(q.stem, size)[0],)


# ------------------------------------------------------------- CIFAR-100 ---
#
# CIFAR-100 은 원래 32x32 라 32x32 입력 모델에 정확히 맞는다. 큰 사진을
# 32x32 로 줄이면 생기는 축소 왜곡이 없고, 라벨이 있어 추론 결과를 검증할 수 있다.
#
# 원본 배포처(cs.toronto.edu)는 연결이 자주 끊겨 169 MB 를 끝까지 받기 어려웠다.
# HuggingFace 미러의 test 스플릿 parquet(23.7 MB)만 받아서 쓴다.

CIFAR100_MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)

# 원본 meta 의 fine_label_names 순서 (알파벳순, 고정값)
CIFAR100_CLASSES = (
    "apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy "
    "bridge bus butterfly camel can castle caterpillar cattle chair chimpanzee "
    "clock cloud cockroach couch crab crocodile cup dinosaur dolphin elephant "
    "flatfish forest fox girl hamster house kangaroo keyboard lamp lawn_mower "
    "leopard lion lizard lobster man maple_tree motorcycle mountain mouse mushroom "
    "oak_tree orange orchid otter palm_tree pear pickup_truck pine_tree plain plate "
    "poppy porcupine possum rabbit raccoon ray road rocket rose sea seal shark "
    "shrew skunk skyscraper snail snake spider squirrel streetcar sunflower "
    "sweet_pepper table tank telephone television tiger tractor train trout tulip "
    "turtle wardrobe whale willow_tree wolf woman worm").split()

_cifar_df = None
_CIFAR_PARQUET = (Path(__file__).resolve().parent.parent
                  / "data" / "cifar100" / "test.parquet")


def _cifar100_df():
    global _cifar_df
    if _cifar_df is None:
        import pandas as pd
        _cifar_df = pd.read_parquet(_CIFAR_PARQUET)
    return _cifar_df


def cifar100_classes():
    return CIFAR100_CLASSES


def cifar100_label(index=0):
    return CIFAR100_CLASSES[int(_cifar100_df().iloc[index]["fine_label"])]


def cifar100_image(index=0):
    """PIL 이미지 하나 (32x32 RGB)."""
    import io
    from PIL import Image
    rec = _cifar100_df().iloc[index]["img"]
    raw = rec["bytes"] if isinstance(rec, dict) else rec
    return Image.open(io.BytesIO(raw)).convert("RGB")


def cifar100_example(index=0, normalize="cifar100"):
    """CIFAR-100 test 샘플 1장. 반환은 (tensor,) 튜플."""
    img = cifar100_image(index)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(1, 32, 32, 3).permute(0, 3, 1, 2).float().div_(255.0)
    if normalize == "cifar100":
        t = (t - CIFAR100_MEAN) / CIFAR100_STD
    elif normalize == "imagenet":
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return (t.contiguous(),)


def cifar100_batches(n=16, normalize="cifar100"):
    for i in range(n):
        yield cifar100_example(index=i, normalize=normalize)
