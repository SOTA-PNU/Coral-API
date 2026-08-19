"""VGG16 — Coral NPU lowering 파이프라인용.

크기 주의
---------
  전체 VGG16   : 138.4 M param → int8 131.9 MiB  → Coral DDR(128 MiB) 초과
  conv 백본만  :  14.7 M param → int8  14.0 MiB  → 여유

conv 가 연산의 99.2%, fc 가 파라미터의 89.4% 다. classifier 를 떼면
메모리는 89% 줄지만 연산은 0.8% 밖에 안 준다.
Coral 실행은 model_vgg_backbone 을, 호스트 검증은 이 파일을 쓴다.
"""

import torch
import torch.nn as nn

CONFIG_VGG16 = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def make_features(config=None):
    """conv/relu/maxpool 스택. model_vgg_backbone 에서도 재사용한다."""
    layers, in_channels = [], 3
    for value in (config or CONFIG_VGG16):
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv2d(in_channels, value, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = value
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super().__init__()
        self.features = features
        # Dropout 은 eval 에서 항등이지만 TOSA 에 배율 1.0 짜리 rescale 을
        # 남긴다(낭비). 추론 전용이므로 넣지 않는다.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# --------------------------------------------------------------------------
# 파이프라인 컨트랙트
# --------------------------------------------------------------------------

def get_model():
    return VGG(features=make_features(), num_classes=1000).eval()


def get_example_inputs():
    from . import _data
    return _data.image_example(size=224, normalize="imagenet")


def get_calibration_batches(n=32):
    from . import _data
    yield from _data.image_batches(size=224, n=n, normalize="imagenet", seed=0)


def get_calibration_info():
    from . import _data
    return _data.image_source_info(224, "imagenet")
