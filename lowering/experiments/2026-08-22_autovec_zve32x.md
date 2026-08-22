# 실험: RVV 자동 벡터화 (Phase 1)

목적: "컴파일러 자동 벡터화만으로 충분한가?"에 대한 실측 베이스라인.
논문 표의 가운데 줄(스칼라 / auto-vec / 커스텀 lowering).

## 결과 (lenet5, MPACT coralnpu_v2 시뮬레이터)

| 빌드 플래그 | 사이클 | argmax | 정확도 |
|---|---|---|---|
| `+m,+f,+zicsr,+zmmul` (스칼라) | 2,650,300 | 7 | ✓ |
| `…,+zve32x,+zvl256b` | 1,990,079 | 0 | ✗ 무효 (아래 참고) |
| `…,+zve32x,+zvl128b` | **2,032,503 (-23.3%, 1.30x)** | 7 | ✓ 상대L2 0.00, 최대차 0 |

재현: `CORAL_CPU_FEATURES="+m,+f,+zicsr,+zmmul,+zve32x,+zvl128b" ./build_coral.sh <model>`

## 배운 것

1. **시뮬레이터 RVV 는 VLEN=128 (vlenb=16).** 문서의 256-bit·64 레지스터는 커스텀
   SIMD 쪽 스펙이고 표준 RVV 백엔드와 다르다. `+zvl256b` 를 주면 LLVM 이 VLMAX=8
   가정 코드를 내고, 시뮬레이터는 vl 을 4 로 잘라 절반만 처리한다 — 트랩도 에러도
   없이 결과만 조용히 틀린다. 정확도 게이트(check_result) 없었으면 "25% 개선"으로
   오인했을 것.
2. 시뮬레이터는 커널이 쓰는 RVV 47종 명령을 전부 구현하고 있다 (Zve32x 서브셋
   보강 불필요). 단, pybind `ReadRegister("vl")` 은 CSR raw 값을 읽어 0 으로
   보인다 — 디버깅 시 함정 (업스트림 coralnpu_top.cc ReadRegister 가 GetUint32
   대신 AsUint64 를 써야 함).
3. **왜 1.3배뿐인가** — 스텝 추적(scratch `trace_dispatch.py`, pybind Step +
   ReadRegister 로 구현)으로 `main_dispatch_1_conv_24x24x6x5x5_i8xi8xi32` 를 한
   명령씩 본 결과: 내부 루프(lb/mul/lw/add/sw ×7,200)는 **스칼라 그대로**, LLVM 은
   재양자화 에필로그(vnclipu 등)만 벡터화했다. 벡터 명령 구성도 vslideup/down·
   vmv 같은 재배치 위주, vmacc 류 산술은 거의 없음. 자동 벡터화는 int8 내적 본체를
   만들지 못한다 → 명시적 MAC 엔진 lowering 이 필요하다는 실측 근거.

## 추가: charrnn (2026-08-22)

| 빌드 | 사이클 | 정확도 |
|---|---|---|
| 스칼라 | 20,361,185 | 기준 대비 상대L2 4.0/5.7/6.1% (LSTM 재양자화, 정상 대역) |
| `+zve32x,+zvl128b` | **13,639,474 (-33.0%, 1.49x)** | **스칼라 출력과 비트 단위 동일** (상대L2 0, 최대차 0) |

LSTM(행렬-벡터 곱 위주)이 conv 보다 자동 벡터화 이득이 크다. 두 모델 모두
결과는 스칼라와 동일 — 자동 벡터화는 수치적으로 안전하지만 이득은 1.3~1.5x 에
그친다.

함정: `check_result.py` 의 모델별 허용치(`MODEL_TOL["charrnn"]=1e-1`)와
분류기/탐지기 집합은 모델 *이름* 으로 찾는다. `charrnn_zvl128` 처럼 변종 이름을
쓰면 기본 허용치 1e-2 로 떨어져 정상 출력이 ✗ 로 찍힌다. 변종 판정은 스칼라
칩 출력과 직접 비교(위 표)로 확인했다.

## 다음

- charrnn 등 다른 모델의 auto-vec 수치 추가 (같은 플래그).
- Phase 2: TOSA conv/matmul → CoralNPU 외적 MAC(VDOT) ukernel lowering.
  비교축: 스칼라 / auto-vec(본 실험) / 커스텀 / (가능하면) TFLM 수제 커널.
