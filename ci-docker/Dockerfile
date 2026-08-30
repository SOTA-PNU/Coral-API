# syntax=docker/dockerfile:1.7
#
# Coral NPU 로워링 파이프라인 — 완전 from-scratch 빌드
#
# torch-mlir(+LLVM/MLIR), IREE, CoralNPU 시뮬레이터를 소스에서 전부 빌드하고
# 파이프라인 코드를 심어 자립 실행 가능한 이미지를 만든다.
#
# 빌드 컨텍스트는 **저장소 루트**다 (lowering/ 을 담아야 하므로).
# 따라서 이 파일의 COPY 경로는 전부 저장소 루트 기준으로 쓴다 — ci-docker/... 접두사 필수.
#
# 빌드 시간: 첫 빌드 3~6시간 (LLVM 2회 + bazel). 이후는 BuildKit 캐시 마운트
# (ccache / bazel repo·disk 캐시)로 크게 단축된다. 캐시는 `docker rmi` 로
# 이미지를 지워도 남으므로 "매번 이미지 삭제" 정책과 양립한다.
# 진짜 무캐시 검증이 필요하면 build.sh --no-cache.

FROM ubuntu:22.04

# 재현을 위한 고정 리비전 (2026-08-28 검증 조합)
ARG TORCH_MLIR_COMMIT=3da108c1ac2ccbfec69d2e07fbab9618f37c821e
ARG IREE_COMMIT=af08a7c817dbfafbd6872670e2c792d4732319f7
ARG CORALNPU_COMMIT=cc7e9fb6f99a83008753773cf7b66421a9588757
ARG BAZEL_VERSION=8.6.0

ENV DEBIAN_FRONTEND=noninteractive \
    WS=/workspace \
    VENV=/opt/venv \
    CCACHE_DIR=/ccache \
    BAZEL_OUTPUT_BASE=/opt/bazel-out

# ---------------------------------------------------------------- 1. 시스템
# clang: MPACT 소스가 -ffp-model=strict 를 쓴다. lld: LLVM 링크 시간 단축.
# default-jre: MPACT 디코더의 antlr genrule 이 java 를 요구한다.
# ccache: /usr/lib/ccache 심볼릭 링크 경로는 bazel CC 툴체인 설정과도 맞춘다.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang-14 lld-14 llvm-14-dev \
      git ca-certificates curl wget unzip zip patch pkg-config \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      default-jre-headless ccache libssl-dev zlib1g-dev \
 && for t in clang clang++ lld ld.lld llvm-ar llvm-ranlib; do \
      ln -sf /usr/bin/${t}-14 /usr/bin/${t} 2>/dev/null || true; done \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- 2. bazel
RUN curl -fsSL -o /usr/local/bin/bazel \
      "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64" \
 && chmod +x /usr/local/bin/bazel \
 && ln -sf /usr/local/bin/bazel /usr/bin/bazel

# ---------------------------------------------------------------- 3. 파이썬
# cmake/ninja 는 pip 판을 쓴다 (검증된 환경과 동일).
# torch 는 CPU 휠 — 파이프라인은 CPU 경로만 쓰고 이미지가 2.5GB 작아진다.
COPY ci-docker/requirements.txt /tmp/requirements.txt
RUN python3.10 -m venv $VENV \
 && $VENV/bin/pip install --no-cache-dir --upgrade pip wheel \
 && $VENV/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 torchvision==0.23.0 \
 && $VENV/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PATH=$VENV/bin:$PATH VIRTUAL_ENV=$VENV

# ---------------------------------------------------------------- 4. torch-mlir
# 로컬 패치가 핵심이다: 양자화 relu/pooling 의 TOSA lowering 과 정수 융합
# (MatchQuantizedOps +366, TorchToTosa +544, FuseQuantizedOps +31).
# 이게 없으면 lenet5 는 'failed to legalize torch.aten.relu' 로 죽고
# 풀링이 f32 로 남아 완전 정수화가 깨진다.
COPY ci-docker/patches/torch-mlir.patch /tmp/torch-mlir.patch
RUN git clone --filter=blob:none https://github.com/llvm/torch-mlir $WS/torch-mlir \
 && cd $WS/torch-mlir \
 && git checkout ${TORCH_MLIR_COMMIT} \
 && (git submodule update --init --depth 1 --jobs 4 \
        externals/llvm-project externals/stablehlo \
      || git submodule update --init --jobs 4 \
        externals/llvm-project externals/stablehlo) \
 && git apply --verbose /tmp/torch-mlir.patch

RUN --mount=type=cache,target=/ccache \
    cd $WS/torch-mlir && cmake -GNinja -Bbuild externals/llvm-project/llvm \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE=$VENV/bin/python \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
      -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=$WS/torch-mlir \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_USE_LINKER=lld \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DTORCH_MLIR_ENABLE_STABLEHLO=ON \
      -DTORCH_MLIR_ENABLE_TOSA=ON \
      -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
      -DTORCH_MLIR_ENABLE_ONNX_C_IMPORTER=OFF \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
 && cmake --build build --target TorchMLIRPythonModules torch-mlir-opt -j"$(nproc)" \
 && ccache -s | tail -3

# 파이프라인은 pip 휠이 아니라 이 소스 빌드를 PYTHONPATH 로 쓴다.
# 빌드 트리의 .py 는 소스로 가는 심볼릭 링크라 소스 트리를 지우면 안 된다.
ENV PYTHONPATH=$WS/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:$WS/torch-mlir/projects/pt1/python

# ---------------------------------------------------------------- 5. IREE
COPY ci-docker/patches/iree.patch /tmp/iree.patch
RUN git clone --filter=blob:none https://github.com/iree-org/iree.git $WS/iree \
 && cd $WS/iree \
 && git checkout ${IREE_COMMIT} \
 && (git submodule update --init --recursive --depth 1 --jobs 4 \
      || git submodule update --init --recursive --jobs 4) \
 && git apply --verbose /tmp/iree.patch

RUN --mount=type=cache,target=/ccache \
    cd $WS/iree && cmake -GNinja -B$WS/iree-build . \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DIREE_BUILD_COMPILER=ON \
      -DIREE_BUILD_SAMPLES=ON \
      -DIREE_BUILD_TESTS=OFF \
      -DIREE_BUILD_PYTHON_BINDINGS=OFF \
      -DIREE_ENABLE_ASSERTIONS=OFF \
      -DIREE_DEFAULT_CPU_LLVM_TARGETS="X86;ARM;AArch64;RISCV" \
 && cmake --build $WS/iree-build --target iree-compile iree-run-module -j"$(nproc)" \
 && ccache -s | tail -3

# ---------------------------------------------------------------- 6. CoralNPU
# 로컬 패치: .bazelrc 의 npusim 설정(clang 플랫폼 + execroot -I.), sw BUILD 의
# py_binary 규칙. 그리고 저장소에 없는 실행 스크립트(run_probe.py 등)와
# //sim:coralnpu_v2_sim 별칭을 파일로 심는다.
COPY ci-docker/patches/coralnpu.patch /tmp/coralnpu.patch
COPY ci-docker/files/coralnpu/ /tmp/coralnpu-files/
RUN git clone --filter=blob:none https://github.com/google-coral/coralnpu.git $WS/coralnpu \
 && cd $WS/coralnpu \
 && git checkout ${CORALNPU_COMMIT} \
 && git apply --verbose /tmp/coralnpu.patch \
 && cp -r /tmp/coralnpu-files/. $WS/coralnpu/ \
 && echo "startup --output_base=${BAZEL_OUTPUT_BASE}" > $WS/coralnpu/.bazelrc.user

# 시뮬레이터와 RISC-V CRT 를 미리 굽는다. 이 output_base 가 이미지에 남아야
# 런타임의 bazel run 이 재빌드 없이 즉시 실행된다 (build_coral.sh 는
# `bazel info output_base` 로 경로를 알아내므로 하드코딩이 없다).
RUN --mount=type=cache,target=/bazel-repo-cache \
    --mount=type=cache,target=/bazel-disk-cache \
    cd $WS/coralnpu && bazel build --config=npusim \
      --repository_cache=/bazel-repo-cache --disk_cache=/bazel-disk-cache \
      //sw/coralnpu_sim:run_probe //toolchain/crt:crt \
 && bazel shutdown

# ---------------------------------------------------------------- 7. 파이프라인
# 코드는 마지막에 넣는다 — 커밋마다 바뀌므로 앞 레이어 캐시를 살린다.
COPY lowering/ $WS/lowering/
ENV CORAL_ROOT=$WS/lowering \
    CORALNPU_DIR=$WS/coralnpu \
    IREE_BUILD_DIR=$WS/iree-build \
    IREE_SRC_DIR=$WS/iree \
    LANG=C.UTF-8

# 환경이 성립하는지 굽는 시점에 확인한다. 깨지면 이미지를 내보내지 않는다.
RUN python3 -c "import torch, numpy; print('torch', torch.__version__)" \
 && python3 -c "from torch_mlir import fx; assert 'build/tools' in fx.__file__, fx.__file__; print('torch_mlir', fx.__file__)" \
 && $WS/iree-build/tools/iree-compile --version \
 && (cd $WS/coralnpu && bazel info output_base) \
 && test -f "${BAZEL_OUTPUT_BASE}/execroot/coralnpu_hw/bazel-out/k8-fastbuild/bin/toolchain/crt/libcrt.lo" \
 && java -version 2>&1 | head -1

WORKDIR $WS/lowering
CMD ["./coral.sh", "--quick", "--keep-going"]
