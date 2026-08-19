#!/usr/bin/env bash
# 모델 하나를 TOSA(mlirbc) 에서 Coral NPU ELF 까지 굽는다.
#
#   usage: build_coral.sh <model>        # build/<model>/model.mlirbc 필요
#
# 단계
#   ③ iree-compile : TOSA -> 바이트코드 VMFB(스케줄+가중치) + RV32 정적 라이브러리 커널
#   ④ cmake        : VMFB 를 objcopy 로 .ddr_data 에 넣고 ELF 링크
#
# VM 은 바이트코드를 쓴다. EmitC(vm-c) 는 생성 shim 이 VM ABI 의
# iree_vm_ref_t 슬롯을 8정렬로 가정하는데 RV32 에서는 정렬이 4라
# "ref type mismatch" 가 난다 (상류 static_library 샘플도 바이트코드 판만
# 테스트한다). 커널은 어느 쪽이든 --iree-llvmcpu-link-static 으로 컴파일되므로
# 속도 차이는 없다.
#
# 컴파일러는 반드시 런타임과 같은 소스에서 빌드된 것을 쓴다. pip 배포판은
# 리비전이 달라 VM ABI 가 어긋난다.
set -euo pipefail

MODEL="${1:?usage: build_coral.sh <model>}"
ROOT="${CORAL_ROOT:-/workspace/lowering-project}"
BUILD="$ROOT/build/$MODEL"
ELF="$BUILD/elf"

IREE_BUILD_DIR="${IREE_BUILD_DIR:-/workspace/iree-build}"
IREE_SRC_DIR="${IREE_SRC_DIR:-/workspace/iree}"
CORALNPU_DIR="${CORALNPU_DIR:-/workspace/coralnpu}"
IREE_COMPILE="$IREE_BUILD_DIR/tools/iree-compile"
# output_base 해시는 사용자명·워크스페이스 경로에서 나오므로 고정하지 않고 bazel 에 묻는다.
BAZEL_EXT="${CORAL_BAZEL_OUTPUT_BASE:-$(cd "$CORALNPU_DIR" && bazel info output_base)}"
CORAL_CRT="$BAZEL_EXT/execroot/coralnpu_hw/bazel-out/k8-fastbuild/bin/toolchain/crt/libcrt.lo"
RISCV_ROOT="$BAZEL_EXT/external/toolchain_coralnpu_v2"

[ -f "$BUILD/model.mlirbc" ] || { echo "!! $BUILD/model.mlirbc 없음"; exit 1; }
mkdir -p "$ELF"

echo "=== [3] iree-compile ($MODEL) -> VMFB + 커널 ==="
( cd "$BUILD" && "$IREE_COMPILE" model.mlirbc \
    --iree-input-type=auto \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv32-unknown-elf \
    --iree-llvmcpu-target-cpu=generic-rv32 \
    --iree-llvmcpu-target-abi=ilp32 \
    --iree-llvmcpu-target-cpu-features=+m,+f,+zicsr,+zmmul \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-link-static \
    --iree-llvmcpu-static-library-output-path=model.o \
    --iree-vm-target-index-bits=32 \
    --iree-vm-emit-polyglot-zip=false \
    --iree-opt-level=O2 \
    -o model.vmfb )
ls -la "$BUILD/model.vmfb" "$BUILD/model.o" | awk '{printf "    %10d  %s\n", $5, $9}'

echo "=== [4] 링커 스크립트 ==="
BASE_LD="$ROOT/coral_elf/coral_itcm.ld"
[ -f "$BASE_LD" ] || { echo "!! $BASE_LD 없음"; exit 1; }
# 항상 DDR 판을 쓴다.
#
# VMFB 블롭(.ddr_data)은 코드와 같은 PT_LOAD 세그먼트에 있어야 시뮬레이터가
# 확실히 로드한다. ITCM 판에서는 .ddr_data 가 별도 DDR 세그먼트로 떨어져
# 로드 여부가 불확실했다. DDR 판은 코드/상수/블롭이 모두 한 세그먼트에 모인다.
# ITCM(1 MiB) 용량 제한도 함께 사라진다.
python3 "$ROOT/coral_elf/make_ld.py" "$BASE_LD" "$ELF/coral.ld" > /dev/null

echo "=== [4] cmake configure ==="
# VM 종류는 configure 시점에 정해지므로, EmitC 로 잡혀 있던 캐시는 버린다.
if [ -f "$ELF/CMakeCache.txt" ] && ! grep -q "MODEL_VM_KIND.*bytecode" "$ELF/configure.log" 2>/dev/null; then
  rm -f "$ELF/CMakeCache.txt"
fi
if [ ! -f "$ELF/CMakeCache.txt" ]; then
  cmake -S "$ROOT/coral_elf" -B "$ELF" -G Ninja \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_TOOLCHAIN_FILE="$IREE_SRC_DIR/build_tools/cmake/generic_riscv32.cmake" \
    -DIREE_REPO_ROOT="$IREE_SRC_DIR" \
    -DRISCV_TOOLCHAIN_ROOT="$RISCV_ROOT" \
    -DIREE_HOST_BIN_DIR="$IREE_BUILD_DIR/tools" \
    -DMODEL_DIR="$BUILD" \
    -DCORAL_CRT="$CORAL_CRT" \
    -DCORAL_LINKER_SCRIPT="$ELF/coral.ld" > "$ELF/configure.log" 2>&1 \
    || { tail -25 "$ELF/configure.log"; exit 1; }
  grep -o "MODEL_VM_KIND = .*" "$ELF/configure.log" | sed 's/^/    /'
fi

echo "=== [4] gen_shapes ==="
python3 "$ROOT/coral_elf/gen_shapes.py" --build-dir "$BUILD" --output "$ELF/model_shapes.h"

echo "=== [4] 링크 ==="
if ! cmake --build "$ELF" --target model_coral_elf -j 16 > "$ELF/link.log" 2>&1; then
  grep -E "error:|Error|overflow|not fit|undefined" "$ELF/link.log" | head -10
  exit 1
fi
ls -la "$ELF/model_coral_elf" | awk '{printf "    ELF %d bytes\n", $5}'
