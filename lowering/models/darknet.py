"""다크넷 .cfg / .weights 를 PyTorch 로 읽어들인다.

왜 cfg 를 파싱하나
------------------
다크넷 .weights 는 층 순서대로 실수를 이어붙인 평면 바이너리다. 헤더 20바이트
뒤로 각 conv 층의 값이 순서대로 나올 뿐 어느 층 것인지 표시가 없다. 그래서
모델의 층 구성이 cfg 와 정확히 일치해야만 순서대로 부어넣을 수 있다.
손으로 옮겨 적으면 어긋나기 쉬우므로 cfg 를 직접 읽어 모듈을 만든다.

TOSA 로 내려갈 수 있게 바꾼 곳
------------------------------
* nn.Upsample -> reshape/expand (aten.__interpolate 는 TOSA legalize 실패)
* stride=1 maxpool 의 비대칭 -inf 패딩 -> shift 4장의 원소별 max
  (tosa.pad 는 상수 패딩만 되고, 큰 음수 상수를 넣으면 양자화 범위가 망가진다.
   edge 복제와 -inf 패딩은 max 연산에 대해 결과가 같다.)
* [yolo] 층은 디코딩(sigmoid/anchor/NMS)을 하지 않고 원시 맵을 그대로 낸다.
  후처리는 호스트에서 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------- cfg 파싱 ---

def parse_cfg(path):
    blocks, cur = [], None
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if cur is not None:
                blocks.append(cur)
            cur = {"type": line[1:-1]}
        else:
            k, _, v = line.partition("=")
            cur[k.strip()] = v.strip()
    if cur is not None:
        blocks.append(cur)
    return blocks


# ------------------------------------------------------------ 보조 모듈 ---

class Upsample2x(nn.Module):
    """nearest 2배 확대. nn.Upsample 은 TOSA legalize 가 안 된다."""

    def forward(self, x):
        n, c, h, w = x.shape
        return (x.reshape(n, c, h, 1, w, 1)
                 .expand(n, c, h, 2, w, 2)
                 .reshape(n, c, h * 2, w * 2))


class MaxPoolStride1(nn.Module):
    """2x2 / stride 1 maxpool. 다크넷은 오른쪽·아래를 -inf 로 패딩한다.

    같은 결과를 shift 4장의 원소별 max 로 만든다. slice/concat/maximum 만
    쓰므로 TOSA 로 내려간다. 가장자리는 복제인데, max 에 대해서는 -inf
    패딩과 결과가 동일하다 (복제값은 이미 창 안에 있는 값이라 최대를 못 바꾼다).
    """

    def forward(self, x):
        right = torch.cat([x[..., 1:], x[..., -1:]], dim=-1)
        down = torch.cat([x[..., 1:, :], x[..., -1:, :]], dim=-2)
        diag = torch.cat([right[..., 1:, :], right[..., -1:, :]], dim=-2)
        return torch.maximum(torch.maximum(x, right),
                             torch.maximum(down, diag))


class Route(nn.Module):
    """[route] 는 앞선 층의 출력을 가져오거나(1개) 채널 방향으로 잇는다(2개)."""

    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class YoloHead(nn.Module):
    """[yolo] 는 출력 지점 표시일 뿐. 디코딩은 호스트에서 한다."""

    def __init__(self, anchors, mask, num_classes):
        super().__init__()
        self.anchors = anchors
        self.mask = mask
        self.num_classes = num_classes


# ----------------------------------------------------------------- 본체 ---

class DarknetModel(nn.Module):
    def __init__(self, cfg_path):
        super().__init__()
        self.blocks = parse_cfg(cfg_path)
        self.net = self.blocks[0]
        self.module_list = nn.ModuleList()
        self.meta = []                       # 층별 정보 (가중치 로드에 필요)

        prev_ch = int(self.net.get("channels", 3))
        out_ch = []                          # 각 층의 출력 채널 수

        for b in self.blocks[1:]:
            t = b["type"]
            if t == "convolutional":
                bn = int(b.get("batch_normalize", 0))
                filt = int(b["filters"])
                size = int(b["size"])
                stride = int(b["stride"])
                pad = (size - 1) // 2 if int(b.get("pad", 0)) else 0
                seq = nn.Sequential()
                seq.add_module("conv", nn.Conv2d(prev_ch, filt, size, stride,
                                                 pad, bias=not bn))
                if bn:
                    seq.add_module("bn", nn.BatchNorm2d(filt))
                if b.get("activation") == "leaky":
                    seq.add_module("act", nn.LeakyReLU(0.1, inplace=False))
                self.module_list.append(seq)
                self.meta.append(("conv", bn, filt))
                prev_ch = filt

            elif t == "maxpool":
                size, stride = int(b["size"]), int(b["stride"])
                if stride == 1:
                    self.module_list.append(MaxPoolStride1())
                else:
                    self.module_list.append(nn.MaxPool2d(size, stride))
                self.meta.append(("maxpool", 0, prev_ch))

            elif t == "upsample":
                self.module_list.append(Upsample2x())
                self.meta.append(("upsample", 0, prev_ch))

            elif t == "route":
                idx = [int(x) for x in b["layers"].split(",")]
                cur = len(self.module_list)
                idx = [i if i >= 0 else cur + i for i in idx]
                prev_ch = sum(out_ch[i] for i in idx)
                self.module_list.append(Route(idx))
                self.meta.append(("route", 0, prev_ch))

            elif t == "yolo":
                a = [int(x) for x in b["anchors"].replace(" ", "").split(",")]
                anchors = list(zip(a[0::2], a[1::2]))
                mask = [int(x) for x in b["mask"].split(",")]
                self.module_list.append(
                    YoloHead(anchors, mask, int(b["classes"])))
                self.meta.append(("yolo", 0, prev_ch))

            else:
                raise ValueError(f"모르는 블록 [{t}]")

            out_ch.append(prev_ch)

        self.out_ch = out_ch

    def forward(self, x):
        cache, outs = [], []
        for mod, (kind, _, _) in zip(self.module_list, self.meta):
            if kind == "route":
                if len(mod.layers) == 1:
                    x = cache[mod.layers[0]]
                else:
                    x = torch.cat([cache[i] for i in mod.layers], dim=1)
            elif kind == "yolo":
                outs.append(x)
            else:
                x = mod(x)
            cache.append(x)
        return tuple(outs)

    # ------------------------------------------------------- 가중치 로드 ---
    def load_darknet_weights(self, path):
        """다크넷 바이너리를 순서대로 붓는다.

        층당 순서 (darknet 의 load_convolutional_weights):
            bias -> (BN 이면) scale, running_mean, running_var -> weight
        BN 이 있는 층은 conv 에 bias 가 없고, 읽은 bias 가 BN 의 beta 다.
        """
        with open(path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=3)
            major, minor = int(header[0]), int(header[1])
            if major * 10 + minor >= 2:
                np.fromfile(f, dtype=np.int64, count=1)      # seen
            else:
                np.fromfile(f, dtype=np.int32, count=1)
            w = np.fromfile(f, dtype=np.float32)

        p = 0
        n_conv = 0
        for mod, (kind, bn, _) in zip(self.module_list, self.meta):
            if kind != "conv":
                continue
            conv = mod.conv
            n_out = conv.out_channels
            if bn:
                b = mod.bn
                for tgt in (b.bias, b.weight, b.running_mean, b.running_var):
                    tgt.data.copy_(torch.from_numpy(w[p:p + n_out]))
                    p += n_out
            else:
                conv.bias.data.copy_(torch.from_numpy(w[p:p + n_out]))
                p += n_out
            n = conv.weight.numel()
            conv.weight.data.copy_(
                torch.from_numpy(w[p:p + n]).view_as(conv.weight))
            p += n
            n_conv += 1

        if p != w.size:
            raise ValueError(
                f"가중치 크기 불일치: {p} 개를 읽었는데 파일에는 {w.size} 개. "
                f"모델 구성이 cfg 와 다릅니다.")
        return {"conv_layers": n_conv, "floats": int(w.size),
                "version": f"{major}.{minor}"}
