"""LeNet5, MNIST test 샘플 1 번 입력.

"칩이 정말 연산하는가"를 확인하기 위한 변형이다. 모델·가중치는 동일하고
입력 이미지만 다르다. 연산이 실제로 일어난다면 argmax 가 그 이미지의
정답 레이블을 따라 바뀌어야 한다.
"""

from . import model_lenet5 as _base

SAMPLE_INDEX = 1


def get_model():
    return _base.get_model()


def get_example_inputs():
    return _base.get_example_inputs(sample_index=SAMPLE_INDEX)


def get_calibration_batches(n=128):
    return _base.get_calibration_batches(n)
