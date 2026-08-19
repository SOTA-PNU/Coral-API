"""VGG16-BN, CIFAR-100 사전학습. 32x32 입력으로 100 클래스 분류.

왜 이 모델인가
--------------
model_vgg_backbone / model_vgg_small 은 랜덤 초기화 + 분류기 없음이라
512차원 특징만 나왔다. 의미 있는 분류를 보려면 학습된 가중치가 필요한데,
ImageNet VGG16 은 분류기까지 138 M 파라미터(int8 로도 138 MB)라 시뮬레이터
DDR 128 MiB 에 안 들어간다.

CIFAR-100 판 VGG16-BN 은 15.3 M 이라 여유롭게 들어가고, 입력이 원래 32x32 라
큰 사진을 줄이며 생기는 왜곡도 없다.

    [1, 3, 32, 32]  ->  [1, 100]   CIFAR-100 클래스 로짓

가중치: torch.hub chenyaofo/pytorch-cifar-models (공개 정확도 약 72%,
호스트 실측 300장에서 74.7%).
"""

import torch

from . import _data

SAMPLE_INDEX = 16          # CIFAR-100 test 몇 번째 이미지를 쓸지


def get_model():
    m = torch.hub.load("chenyaofo/pytorch-cifar-models",
                       "cifar100_vgg16_bn", pretrained=True, verbose=False)
    return m.eval()


def get_example_inputs():
    return _data.cifar100_example(index=SAMPLE_INDEX, normalize="cifar100")


def get_calibration_batches(n=16):
    yield from _data.cifar100_batches(n=n, normalize="cifar100")


def get_calibration_info():
    return {"source": "CIFAR-100 test 스플릿",
            "sample_index": SAMPLE_INDEX,
            "label": _data.cifar100_label(SAMPLE_INDEX),
            "weights": "chenyaofo/pytorch-cifar-models cifar100_vgg16_bn",
            "real_data": True}
