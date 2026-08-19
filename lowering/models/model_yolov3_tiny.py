"""YOLOv3-tiny, 다크넷 COCO 사전학습 가중치. 진짜 물체 탐지.

기존 model_yolov3.py 는 TOSA 로 내려가게 하려고 손으로 쓴 근사 구조라
다크넷 가중치를 얹을 수 없었다(층 구성 불일치). 이 모델은 cfg 를 직접 읽어
층을 만들므로 .weights 를 순서대로 그대로 부을 수 있다.

    [1, 3, 416, 416]  ->  [1, 255, 13, 13]   coarse (/32)
                          [1, 255, 26, 26]   fine   (/16)

    255 = 3 anchor x (4 box + 1 objectness + 80 COCO class)

sigmoid/anchor 디코딩과 NMS 는 모델 밖(호스트)에서 한다. 다크넷도 [yolo] 층은
학습 손실용이라 추론 시에는 원시 맵을 내보내는 것이 자연스럽고, 모델을
단순하게 유지해야 TOSA 로 내려간다.

호스트 검증 (verify_yolo.py, float): bus.jpg 에서 person 4 + bus 1 탐지.
"""

from pathlib import Path

from . import _data
from .darknet import DarknetModel

INPUT_SIZE = 416
_DN = Path(__file__).resolve().parent.parent / "data" / "darknet"
CFG = _DN / "yolov3-tiny.cfg"
WEIGHTS = _DN / "yolov3-tiny.weights"


def get_model():
    m = DarknetModel(CFG)
    m.load_darknet_weights(WEIGHTS)
    return m.eval()


def get_example_inputs():
    return _data.letterbox_photo("bus", INPUT_SIZE)


def get_calibration_batches(n=16):
    yield from _data.letterbox_batches(INPUT_SIZE, n)


def get_calibration_info():
    return {"source": "data/photos letterbox 전처리",
            "input_size": INPUT_SIZE,
            "weights": "darknet yolov3-tiny COCO 사전학습",
            "real_data": True}
