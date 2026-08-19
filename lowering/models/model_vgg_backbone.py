"""VGG16 conv 백본 — classifier 없음. Coral 배포용 변형.

전체 VGG16 은 int8 131.9 MiB 로 Coral DDR(128 MiB) 를 넘어 링크 자체가 안 된다.
classifier 3층(파라미터의 89.4%)을 떼면 14.0 MiB 가 되어 여유 있게 들어간다.

출력이 클래스 로짓이 아니라 특징맵이다:
    [1, 3, 224, 224]  ->  [1, 512, 7, 7]
"""

import torch
import torch.nn as nn

from .model_vgg import make_features


class VGGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = make_features()

    def forward(self, x):
        return self.features(x)


# --------------------------------------------------------------------------
# 파이프라인 컨트랙트
# --------------------------------------------------------------------------

def get_model():
    return VGGBackbone().eval()


def get_example_inputs():
    from . import _data
    return _data.image_example(size=224, normalize="imagenet")


def get_calibration_batches(n=32):
    from . import _data
    yield from _data.image_batches(size=224, n=n, normalize="imagenet", seed=0)


def get_calibration_info():
    from . import _data
    return _data.image_source_info(224, "imagenet")
