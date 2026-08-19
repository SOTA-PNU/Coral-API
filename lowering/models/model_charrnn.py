"""CharRNN — Coral NPU lowering 파이프라인용.

왜 nn.LSTM 을 쓰지 않는가
-------------------------
nn.LSTM 은 가중치를 _flat_weights 리스트로 묶어 aten.lstm 에 통째로 넘기는데,
torch-mlir 이 그 리스트를 처리하지 못한다(실측):

    TOSA FAIL: Heterogeneous lists are not supported:
               expected vtensor, got <class 'torch.nn.parameter.Parameter'>

그래서 LSTM 게이트를 nn.Linear 로 직접 편다. 얻는 것:
  * aten.lstm 이 사라지고 linear/sigmoid/tanh/mul/add 만 남는다 (전부 TOSA 지원)
  * Linear 는 PT2E 애노테이터가 있어 int8 로 양자화된다
  * 출력 중첩 튜플이 사라져 ABI 가 3 in / 3 out 으로 평탄해진다

sigmoid/tanh 는 양자화 애노테이터가 없어 float 으로 남는다.
YOLOv3 의 LeakyReLU 와 같은 상황이며 현 단계에서는 허용한다.

시퀀스 길이 주의
----------------
torch.export 는 파이썬 루프를 **펼친다**. SEQ_LEN 이 길면 그래프가 폭발하므로
짧게(16~32) 시작할 것.
"""

from pathlib import Path

import torch
import torch.nn as nn

VOCAB = 64
HIDDEN = 128
SEQ_LEN = 16


class CharRNN(nn.Module):
    """
    LSTM 셀을 Linear 로 편 char-level RNN.

    forward(seq, h, c) -> (logits, h, c)
        seq    : [1, SEQ_LEN] int64  문자 인덱스
        h, c   : [1, HIDDEN]  f32    은닉/셀 상태
        logits : [1, VOCAB]   f32    다음 문자 분포
    """

    def __init__(self, vocab=VOCAB, hidden=HIDDEN):
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.i2h = nn.Linear(hidden, hidden * 4)   # i, f, g, o 게이트 한꺼번에
        self.h2h = nn.Linear(hidden, hidden * 4)
        self.decoder = nn.Linear(hidden, vocab)

    def forward(self, seq, h, c):
        x = self.embed(seq)                        # [1, SEQ, H]
        hid = self.hidden
        for t in range(x.shape[1]):                # export 가 펼친다
            xt = x[:, t]                           # [1, H]
            gates = self.i2h(xt) + self.h2h(h)     # [1, 4H]
            # chunk/split 대신 명시적 슬라이싱 (aten.slice.Tensor 로 내려감)
            i = torch.sigmoid(gates[:, 0 * hid:1 * hid])
            f = torch.sigmoid(gates[:, 1 * hid:2 * hid])
            g = torch.tanh(gates[:, 2 * hid:3 * hid])
            o = torch.sigmoid(gates[:, 3 * hid:4 * hid])
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.decoder(h), h, c


# --------------------------------------------------------------------------
# 파이프라인 컨트랙트
# --------------------------------------------------------------------------

_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "charrnn.pt"


def get_model():
    """학습된 체크포인트가 있으면 불러온다.

    train_charrnn.py 가 컨테이너 안의 말뭉치(/usr/share/common-licenses/*)로
    만든다. 없으면 랜덤 초기화로 떨어지는데, 그 경우 다음 문자 예측은
    균등분포(1/64)에 가까워 의미가 없다.
    """
    m = CharRNN()
    if _CKPT.is_file():
        m.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    return m.eval()


def get_example_inputs():
    from . import _data
    seq = next(iter(_data.text_sequences(VOCAB, SEQ_LEN, n=1, seed=7)))
    return (seq, torch.zeros(1, HIDDEN), torch.zeros(1, HIDDEN))


def get_calibration_batches(n=32):
    from . import _data
    for seq in _data.text_sequences(VOCAB, SEQ_LEN, n=n, seed=1):
        yield (seq, torch.zeros(1, HIDDEN), torch.zeros(1, HIDDEN))


def get_calibration_info():
    from . import _data
    return _data.text_source_info(VOCAB, SEQ_LEN)
