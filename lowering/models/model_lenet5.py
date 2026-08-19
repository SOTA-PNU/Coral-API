"""LeNet-5 (MNIST 28x28) — Coral NPU lowering 파이프라인용.

모든 모델 파일은 아래 컨트랙트를 모듈 레벨 함수로 제공한다.
    get_model()               -> nn.Module (eval 상태)
    get_example_inputs()      -> tuple[Tensor, ...]
    get_calibration_batches() -> Iterable[tuple[Tensor, ...]]
    get_calibration_info()    -> dict   (manifest 기록용)
"""

from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"
_CKPT = _ROOT / "checkpoints" / "lenet5_mnist.pt"


class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        # 학습된 체크포인트가 이 relu5 를 포함한다. 기준값
        # [0,...,14.269,...,0] 의 음수 0 이 여기서 나오므로 제거하지 말 것.
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.pool1(self.relu1(self.conv1(x)))
        y = self.pool2(self.relu2(self.conv2(y)))
        y = torch.flatten(y, 1)
        y = self.relu3(self.fc1(y))
        y = self.relu4(self.fc2(y))
        return self.relu5(self.fc3(y))


# --------------------------------------------------------------------------
# 파이프라인 컨트랙트
# --------------------------------------------------------------------------

def get_model():
    model = Lenet5()
    ckpt = torch.load(_CKPT, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    return model.eval()


def get_example_inputs(sample_index=0):
    """랜덤이 아니라 실제 MNIST test 샘플. 기준 출력(argmax=7)의 근거."""
    from . import _data
    img = _data.mnist_images(train=False)[sample_index:sample_index + 1]
    return (img.contiguous(),)                          # [1,1,28,28] f32


def get_calibration_batches(n=128):
    from . import _data
    imgs = _data.mnist_images(train=True)
    for i in range(min(n, imgs.shape[0])):
        yield (imgs[i:i + 1].contiguous(),)             # 배치 1 (ABI 고정)


def get_calibration_info():
    return {"source": "MNIST train split (idx 파일 직접 read)",
            "transform": "uint8 / 255 -> [0,1]  (ToTensor 와 동일)",
            "samples": 128, "real_data": True}
