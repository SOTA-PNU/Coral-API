"""YOLOv3-tiny — Coral NPU lowering 파이프라인용 정의.

설계 근거
---------
* 활성화가 LeakyReLU(0.1) 다. YOLOv5 v4.0 이후의 SiLU 와 달리 조각 선형이라
  정수 도메인 구현이 현실적이다.
* NMS / 박스 디코딩은 **의도적으로 제외**했다. 데이터 의존 제어흐름이라
  torch.export 가 처리하지 못한다. raw feature map 만 내보내고 후처리는
  호스트에서 한다.
* 출력이 2개(13x13, 26x26)다. 단일 출력 가정을 깨는 것이 이 모델의 역할이다.
* TorchToTosa 에 필요한 op 이 모두 등록되어 있음을 확인했다:
  AtenLeakyReluOp / AtenUpsampleNearest2dOp / AtenCatOp / AtenSliceTensorOp

크기 (int8 기준)
----------------
  num_classes=80 (COCO) : 약 8.7 M 파라미터 → 8.7 MB
  num_classes=20 (VOC)  : 약 8.4 M 파라미터 → 8.4 MB
Coral DDR 128 MiB 에 여유롭게 들어간다.
"""

import torch
import torch.nn as nn

# 입력 해상도. 416 이 기본이지만 연산량을 줄이려면 320 으로 낮춘다.
# (320 은 416 대비 약 0.59 배 연산량)
DEFAULT_INPUT_SIZE = 416
EXAMPLE_INPUT_SHAPE = (1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)


class ConvBlock(nn.Module):
    """Conv + BatchNorm + LeakyReLU.

    PT2E 의 XNNPACKQuantizer 에 conv_bn 애노테이터가 있어 BN 은 conv 로 접힌다.
    LeakyReLU 는 애노테이터가 없으므로 별도 op 으로 남는다(현재로선 float 왕복).
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class NearestUpsample2x(nn.Module):
    """2배 nearest upsample 을 reshape/expand 로 직접 구현한다.

    왜 nn.Upsample 을 안 쓰는가:
      nn.Upsample 과 F.interpolate 는 모두
      `torch.aten.__interpolate.size_list_scale_list` 로 내려가는데,
      이 op 은 TorchToTosa 에서 legalize 되지 않는다(실측 확인).
      TorchToTosa 에 AtenUpsampleNearest2dOp 패턴은 등록돼 있지만
      PyTorch 가 그 op 을 만들어주지 않는다.

    이 구현은 tosa.reshape + tosa.tile 로 내려간다.
    """

    def forward(self, x):
        n, c, h, w = x.shape
        return (x.reshape(n, c, h, 1, w, 1)
                 .expand(n, c, h, 2, w, 2)
                 .reshape(n, c, h * 2, w * 2))


class YOLOv3Tiny(nn.Module):
    """YOLOv3-tiny backbone + head. 후처리 없음.

    forward(x) -> (out_large, out_small)
        out_large : [N, 3*(5+C), H/32, W/32]   coarse  (416 -> 13x13)
        out_small : [N, 3*(5+C), H/16, W/16]   fine    (416 -> 26x26)

    채널 3*(5+C) 의 의미:
        3   = 스케일당 anchor 수
        5   = tx, ty, tw, th, objectness
        C   = 클래스 수
    """

    def __init__(self, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_ch = num_anchors * (5 + num_classes)

        # ---- backbone ------------------------------------------------------
        self.b1 = ConvBlock(3, 16, 3)
        self.b2 = ConvBlock(16, 32, 3)
        self.b3 = ConvBlock(32, 64, 3)
        self.b4 = ConvBlock(64, 128, 3)
        self.b5 = ConvBlock(128, 256, 3)      # ← 여기 출력이 fine 경로로 분기
        self.b6 = ConvBlock(256, 512, 3)
        self.b7 = ConvBlock(512, 1024, 3)

        self.pool = nn.MaxPool2d(2, 2)

        # 6번째 pool 은 stride 1 이라 크기가 유지된다. darknet 은 오른쪽/아래에만
        # zero padding 을 넣으므로 ZeroPad2d 로 그 비대칭을 재현한다.
        self.pad_right_bottom = nn.ZeroPad2d((0, 1, 0, 1))
        self.pool_stride1 = nn.MaxPool2d(2, 1)

        # ---- head: coarse (13x13) -----------------------------------------
        self.neck = ConvBlock(1024, 256, 1)   # ← 여기서도 분기
        self.head_large = ConvBlock(256, 512, 3)
        self.pred_large = nn.Conv2d(512, out_ch, 1)

        # ---- head: fine (26x26) -------------------------------------------
        self.lateral = ConvBlock(256, 128, 1)
        self.upsample = NearestUpsample2x()
        self.head_small = ConvBlock(128 + 256, 256, 3)
        self.pred_small = nn.Conv2d(256, out_ch, 1)

    def forward(self, x):
        x = self.pool(self.b1(x))     # 416 -> 208
        x = self.pool(self.b2(x))     # 208 -> 104
        x = self.pool(self.b3(x))     # 104 ->  52
        x = self.pool(self.b4(x))     #  52 ->  26
        route = self.b5(x)            #  26x26, 256ch  (fine 경로로 나감)
        x = self.pool(route)          #  26 ->  13
        x = self.b6(x)
        x = self.pool_stride1(self.pad_right_bottom(x))   # 13 유지
        x = self.b7(x)                #  13x13, 1024ch

        neck = self.neck(x)                               # 13x13, 256ch

        out_large = self.pred_large(self.head_large(neck))

        lat = self.upsample(self.lateral(neck))           # 13 -> 26, 128ch
        fused = torch.cat([lat, route], dim=1)            # 26x26, 384ch
        out_small = self.pred_small(self.head_small(fused))

        return out_large, out_small


def get_model(num_classes=80):
    return YOLOv3Tiny(num_classes=num_classes).eval()


def get_example_inputs(input_size=DEFAULT_INPUT_SIZE):
    """darknet 계열은 ImageNet 정규화 없이 [0,1] 을 그대로 쓴다."""
    from . import _data
    return _data.image_example(size=input_size, normalize=None)


def get_calibration_batches(n=16, input_size=DEFAULT_INPUT_SIZE):
    from . import _data
    yield from _data.image_batches(size=input_size, n=n, normalize=None, seed=2)


def get_calibration_info():
    from . import _data
    info = _data.image_source_info(DEFAULT_INPUT_SIZE, None)
    info["activation"] = "LeakyReLU(0.1) — quantizer 애노테이터 없음, float 왕복 발생"
    return info
