"""VGG16 backbone, 32x32 축소판.

구조는 model_vgg_backbone 과 완전히 동일하고 입력 해상도만 낮춘다.
backbone 은 전부 conv/pool 이라 해상도에 무관하게 그대로 동작한다.

    [1, 3, 32, 32]  ->  [1, 512, 1, 1]      (maxpool 5회, 32 -> 1)

왜 필요한가
-----------
원 해상도(224)는 약 15 GMAC 이라 Coral 시뮬레이터에서 약 900억 사이클,
수십 시간이 걸린다. 32x32 는 연산량이 1/49 이라 완주 검증이 현실적이다.
파이프라인(양자화 -> TOSA -> IREE -> ELF -> 에뮬레이터)이 동일하므로
정확도 검증 목적에는 이쪽이 맞고, 원 해상도는 따로 장시간 돌린다.
"""

from . import _data
from . import model_vgg_backbone as _full

INPUT_SIZE = 32


def get_model():
    return _full.get_model()


def get_example_inputs():
    # CIFAR-100 은 원래 32x32 라 이 모델 입력에 정확히 맞는다.
    # (큰 사진을 32x32 로 줄이면 형태만 겨우 남는다.)
    return _data.cifar100_example(index=16, normalize="cifar100")


def get_calibration_batches(n=16):
    yield from _data.cifar100_batches(n=n, normalize="cifar100")
