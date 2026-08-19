#!/usr/bin/env bash
#
# PyTorch 모델 하나를 Coral NPU 에뮬레이터에서 돌리기까지 전 과정을 실행한다.
#
#   ./coral.sh lenet5                  한 모델 (양자화 → TOSA → IREE → ELF → 실행 → 검증)
#   ./coral.sh lenet5 charrnn          여러 모델
#   ./coral.sh --real                  학습된 가중치 4개 (의미 있는 추론). 약 30분
#   ./coral.sh --quick                 빠른 것만 (lenet5, charrnn). 약 80초
#   ./coral.sh --all                   위 + 수치 검증용까지 전부
#   ./coral.sh --gate                  회귀 게이트: 같은 가중치에 입력 3종 → argmax 7/2/1
#   ./coral.sh --run-only lenet5       이미 빌드된 것 실행/검증만
#
# 옵션
#   --deadline N    시뮬레이터 데드라인 초 (기본 10800)
#   --keep-going    한 모델이 실패해도 다음으로 진행
#   --no-show       추론 결과 해석 출력을 생략
#
# 단계
#   ①② export_tosa.py  : PT2E 양자화 + TOSA MLIR + 기준 출력 저장
#   ③④ build_coral.sh  : iree-compile(VMFB+RV32 커널) → Coral ELF 링크
#   ⑤   run_probe       : MPACT 시뮬레이터 실행, 전역 심볼 덤프
#   ⑥   check_result.py : 기준 출력과 대조해 합격/불합격 판정
set -uo pipefail

ROOT="${CORAL_ROOT:-/workspace/lowering-project}"
SIM="${CORALNPU_DIR:-/workspace/coralnpu}"
DEADLINE=10800
RUN_ONLY=0
KEEP_GOING=0
SHOW=1
MODELS=()

# 학습된 가중치로 의미 있는 추론을 하는 4개. 합계 약 30분.
#   lenet5        MNIST 학습        손글씨 숫자 인식
#   charrnn       GPL-3 로컬 학습   다음 문자 예측
#   vgg_cifar100  CIFAR-100 사전학습 VGG16-BN, 100 클래스 분류
#   yolov3_tiny   COCO 다크넷        물체 탐지
REAL_MODELS=(lenet5 charrnn vgg_cifar100 yolov3_tiny)
# 빠른 것만. 약 80초.
QUICK_MODELS=(lenet5 charrnn)
# 수치 검증용(랜덤 가중치) 까지 포함. 원 해상도 VGG 는 수 시간이라 뺐다.
ALL_MODELS=(lenet5 charrnn vgg_cifar100 yolov3_tiny resnet18 vgg_small yolov3_small)
# 게이트: 모델·가중치는 같고 입력만 다르다. 실제 연산이 일어나는지 확인한다.
GATE_MODELS=(lenet5 lenet5_s1 lenet5_s2)
GATE_EXPECT=(7 2 1)

while [ $# -gt 0 ]; do
  case "$1" in
    --all)        MODELS=("${ALL_MODELS[@]}") ;;
    --real)       MODELS=("${REAL_MODELS[@]}") ;;
    --quick)      MODELS=("${QUICK_MODELS[@]}") ;;
    --gate)       MODELS=("${GATE_MODELS[@]}"); GATE=1 ;;
    --run-only)   RUN_ONLY=1 ;;
    --keep-going) KEEP_GOING=1 ;;
    --no-show)    SHOW=0 ;;
    --deadline)   shift; DEADLINE="$1" ;;
    -h|--help)    sed -n "2,21p" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)           echo "알 수 없는 옵션: $1"; exit 2 ;;
    *)            MODELS+=("$1") ;;
  esac
  shift
done

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "모델을 지정하세요.  ./coral.sh --help"
  exit 2
fi

# 에뮬레이터 실행 예상 시간(초). 실측값이며 진행 표시용이다.
expect_sec() {
  case "$1" in
    lenet5|lenet5_s1|lenet5_s2) echo 5 ;;
    charrnn)                    echo 10 ;;
    vgg_small|yolov3_small)     echo 200 ;;
    vgg_cifar100)               echo 190 ;;
    resnet18)                   echo 1350 ;;
    yolov3_tiny)                echo 1600 ;;
    *)                          echo 0 ;;
  esac
}
EXPECT_SEC=1

hr() { printf '%s\n' "────────────────────────────────────────────────────────────"; }

FAILED=()
PASSED=()
START_ALL=$SECONDS

for idx in "${!MODELS[@]}"; do
  m="${MODELS[$idx]}"
  hr
  echo "▶ $m"
  hr
  t0=$SECONDS

  if [ "$RUN_ONLY" -eq 0 ]; then
    if [ ! -f "$ROOT/models/model_$m.py" ]; then
      echo "  ✗ models/model_$m.py 가 없습니다"
      FAILED+=("$m"); [ "$KEEP_GOING" -eq 1 ] && continue || break
    fi

    echo "  [1/4] 양자화 + TOSA"
    if ! ( cd "$ROOT" && python3 export_tosa.py "$m" ) > "$ROOT/build_$m.export.log" 2>&1; then
      tail -15 "$ROOT/build_$m.export.log" | sed 's/^/      /'
      FAILED+=("$m"); [ "$KEEP_GOING" -eq 1 ] && continue || break
    fi
    grep -E "ABI|양자화오차" "$ROOT/build_$m.export.log" | sed 's/^/    /'

    echo "  [2/4] IREE 컴파일 + ELF 링크"
    if ! "$ROOT/build_coral.sh" "$m" > "$ROOT/build_$m.build.log" 2>&1; then
      tail -15 "$ROOT/build_$m.build.log" | sed 's/^/      /'
      FAILED+=("$m"); [ "$KEEP_GOING" -eq 1 ] && continue || break
    fi
    grep -E "ELF [0-9]+ bytes" "$ROOT/build_$m.build.log" | sed 's/^/    /'
  fi

  ELF="$ROOT/build/$m/elf/model_coral_elf"
  if [ ! -f "$ELF" ]; then
    echo "  ✗ $ELF 가 없습니다 (먼저 --run-only 없이 실행하세요)"
    FAILED+=("$m"); [ "$KEEP_GOING" -eq 1 ] && continue || break
  fi

  echo "  [3/4] 에뮬레이터 실행 (데드라인 ${DEADLINE}s)"
  LOG="$ROOT/run_$m.log"
  # 출력 전체를 파일로 받아 둔다. 합계만으로는 모델이 무엇을 인식했는지
  # 알 수 없어서, show_result.py 가 이 원본을 해석한다.
  NOUT=$(python3 -c "
import json,sys
d=json.load(open('$ROOT/build/$m/manifest.json'))
print(sum(o['bytes']//4 for o in d['abi']['outputs']))")
  EXP=$(expect_sec "$m")
  [ "$EXP" -gt 0 ] && echo "        예상 약 $((EXP / 60))분 $((EXP % 60))초"
  # 시뮬레이터는 실행 중 아무것도 출력하지 않는다(sim.run() 이 GIL 을 잡고 있어
  # 파이썬 쪽 진행 폴링이 돌지 않는다). 멈춘 것처럼 보이지 않도록 셸에서 경과를 찍는다.
  ( cd "$SIM" && timeout $((DEADLINE + 600)) bazel run --config=npusim \
      //sw/coralnpu_sim:run_probe -- "$DEADLINE" 900 "$ELF" \
      inference_stage:uint32:1 inference_status_code:uint32:1 \
      inference_argmax:int32:1 inference_out_count:uint32:1 \
      inference_out_sum:float32:4 \
      "inference_output:float32:${NOUT}:$ROOT/build/$m/chip_output.bin" \
      status_msg:int8:512 bm_peak_kb:uint32:1 ) > "$LOG" 2>&1 &
  SIMPID=$!
  SIMT0=$SECONDS
  while kill -0 "$SIMPID" 2>/dev/null; do
    sleep 15
    kill -0 "$SIMPID" 2>/dev/null || break
    EL=$((SECONDS - SIMT0))
    if [ "$EXP" -gt 0 ]; then
      printf "\r        실행 중 %3d분 %02d초 / 예상 %d분   " \
             $((EL / 60)) $((EL % 60)) $((EXP / 60))
    else
      printf "\r        실행 중 %3d분 %02d초   " $((EL / 60)) $((EL % 60))
    fi
  done
  wait "$SIMPID" 2>/dev/null
  printf "\r        실행 완료 %d분 %02d초%-20s\n" \
         $(((SECONDS - SIMT0) / 60)) $(((SECONDS - SIMT0) % 60)) ""

  echo "  [4/4] 기준 출력과 대조"
  EXPECT=()
  if [ "${GATE:-0}" = "1" ]; then EXPECT=(--expect-argmax "${GATE_EXPECT[$idx]}"); fi
  if python3 "$ROOT/check_result.py" "$m" "$LOG" "${EXPECT[@]}"; then
    PASSED+=("$m")
    if [ "$SHOW" -eq 1 ] && [ -f "$ROOT/build/$m/chip_output.bin" ]; then
      python3 "$ROOT/show_result.py" "$m" 2>&1 | sed 's/^/  /'
    fi
  else
    FAILED+=("$m")
    [ "$KEEP_GOING" -eq 1 ] || { echo "  (--keep-going 을 주면 계속 진행합니다)"; break; }
  fi
  echo "  소요 $((SECONDS - t0))초"
done

hr
echo "합격 ${#PASSED[@]}개: ${PASSED[*]:-없음}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "불합격 ${#FAILED[@]}개: ${FAILED[*]}"
fi
echo "전체 소요 $((SECONDS - START_ALL))초"
[ ${#FAILED[@]} -eq 0 ]
