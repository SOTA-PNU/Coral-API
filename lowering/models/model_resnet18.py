"""ResNet18, ImageNet 사전학습. 진짜 이미지 분류.

왜 이 모델인가
--------------
지금까지의 VGG/YOLO/CharRNN 은 가중치가 랜덤이라 "수치가 맞는가" 만 검증할 수
있었다. 의미 있는 추론을 보려면 학습된 가중치가 필요한데, VGG16 은 분류기까지
포함하면 138 M 파라미터(int8 로도 138 MB)라 시뮬레이터 DDR 128 MiB 에 안 들어간다.
ResNet18 은 11.7 M 라 여유롭게 들어가면서 ImageNet 1000 클래스 분류를 그대로 한다.

    [1, 3, 224, 224]  ->  [1, 1000]   ImageNet 클래스 로짓

연산량 약 1.8 GMAC. 실측 3.4 M 사이클/초 기준 약 1시간.
"""

import torch
import torchvision

from . import _data


def get_model():
    m = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    # BatchNorm 은 eval 이라 상수로 접히고, PT2E 양자화가 conv 에 흡수한다.
    return m


def get_example_inputs():
    # data/photos/ 의 첫 사진. 파일명에 정답 클래스가 들어 있다.
    return _data.real_photo(size=224, normalize="imagenet", index=0)


def get_calibration_batches(n=16):
    yield from _data.real_photo_batches(size=224, n=n, normalize="imagenet")


def get_calibration_info():
    paths = _data.photo_paths()
    return {"source": "data/photos 실제 사진",
            "count": len(paths),
            "labels": [_data.photo_label(p) for p in paths],
            "real_data": True}
