"""YOLOv3-tiny, 96x96 축소판.

구조는 model_yolov3 과 완전히 동일하고 입력 해상도만 낮춘다.
96 은 32 의 배수라 두 출력 헤드가 그대로 성립한다.

    [1, 3, 96, 96]  ->  [1, 255, 3, 3]  (coarse, /32)
                        [1, 255, 6, 6]  (fine,   /16)

원 해상도 416 은 약 2.8 GMAC 이라 시뮬레이터에서 10시간이 넘는다.
96 은 연산량이 약 1/19 이라 완주 검증이 가능하다.
"""

from . import _data
from . import model_yolov3 as _full

INPUT_SIZE = 96


def get_model(num_classes=80):
    return _full.get_model(num_classes)


def get_example_inputs():
    # 아이콘 대신 실제 사진(ultralytics 표준 테스트 이미지 bus.jpg).
    return _data.real_photo(size=INPUT_SIZE, normalize=None, name="bus")


def get_calibration_batches(n=16):
    yield from _data.real_photo_batches(size=INPUT_SIZE, n=n, normalize=None)
