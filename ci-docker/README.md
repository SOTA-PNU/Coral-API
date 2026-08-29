# coral-ci — 매번 처음부터 굽는 CI

파이프라인이 도는 환경(torch-mlir·IREE·CoralNPU 시뮬레이터)을 **소스에서 전부
빌드하는** 도커 이미지를 만들고, 그 안에서 로워링 파이프라인을 실행한 뒤
**이미지를 삭제한다.** 매 CI 실행마다 반복한다.

기존 CI(`coral-pr.yml`)와의 차이: 그쪽은 러너 호스트에 미리 구축된
`/workspace/*` 툴체인에 의존한다. 이쪽은 러너에 **docker 와 데이터셋만**
있으면 되고, 툴체인 구성이 Dockerfile 로 완전히 문서화·검증된다.

## 배치

    /mnt/data/jiho/
    ├── Coral-API/                  ← 이 저장소 (git clone)
    │   ├── ci-docker/              ← 이 디렉터리
    │   └── lowering/               ← 파이프라인 (이미지에 COPY 됨)
    └── coral-data/data/            ← 데이터셋 122MB (저장소 밖, 런타임 마운트)

## 최초 1회 설정 (호스트에서)

    sudo mkdir -p /mnt/data/jiho && sudo chown $USER /mnt/data/jiho
    cd /mnt/data/jiho
    git clone https://github.com/SOTA-PNU/Coral-API.git
    cd Coral-API

    # 데이터셋 준비 — 기존 환경에서 복사하는 편이 빠르다
    SRC=<기존 lowering-project/data 경로> ./ci-docker/fetch-data.sh

    # GitHub 러너를 이 호스트에 등록 (레이블 coral-docker 필수)
    #   Settings → Actions → Runners → New self-hosted runner
    #   ./config.sh --url https://github.com/SOTA-PNU/Coral-API \
    #     --labels coral-docker --name coral-host

## 실행

CI: Actions → "coral docker CI (from scratch)" → Run workflow
(주 1회 일요일 자동 실행 — 무캐시 재현성 검증)

손으로:

    ./ci-docker/run-ci.sh              # --quick (lenet5, charrnn)
    ./ci-docker/run-ci.sh --real       # 학습 가중치 4종
    ./ci-docker/run-ci.sh --no-cache --quick   # 완전 무캐시 검증
    KEEP_IMAGE=1 ./ci-docker/run-ci.sh # 디버깅: 이미지 보존

## 빌드 시간 — 알고 쓸 것

| 구간 | 첫 빌드 | 캐시 이후 |
|---|---|---|
| LLVM/MLIR + torch-mlir | 1.5~3시간 | 수 분 (ccache) |
| IREE (번들 LLVM) | 1~2시간 | 수 분 (ccache) |
| CoralNPU bazel (RISC-V 툴체인 + MPACT) | ~1시간 | 수 분 (disk cache) |
| **합계** | **3~6시간** | **10~30분** |

`docker rmi` 는 **이미지**를 지울 뿐 **BuildKit 빌드 캐시**는 남긴다. 그래서
"매번 이미지 삭제" 정책을 지키면서도 재빌드가 빠르다. 진짜 무캐시 검증은
`--no-cache` 로 명시할 때만 한다(주 1회 스케줄이 이 역할).

캐시가 디스크를 먹으면: `docker builder prune --filter until=168h`

## 재현 재료 (이 디렉터리의 핵심)

고정 리비전 — Dockerfile 의 ARG:

| 저장소 | 커밋 |
|---|---|
| torch-mlir | `3da108c1` (llvm-project `ea2f5081`, stablehlo `a93085ce`) |
| IREE | `af08a7c8` |
| CoralNPU | `cc7e9fb6` |

`patches/` — 업스트림에 없는 로컬 수정. **이게 없으면 파이프라인이 동작하지 않는다.**

- `torch-mlir.patch` (1149줄): 양자화 relu 의 TOSA lowering(없으면 lenet5 가
  `failed to legalize torch.aten.relu` 로 죽는다), max/avg 풀링 정수 융합
  (없으면 풀링이 f32 로 남아 완전 정수화가 깨진다), MatchQuantizedOps 확장.
- `iree.patch` (246줄): static_library 샘플 — RV32 정적 링크 경로.
- `coralnpu.patch` (57줄): `.bazelrc` 의 npusim 설정(clang 플랫폼 + execroot
  `-I.`), `sw/coralnpu_sim/BUILD` 의 py_binary 규칙.

`files/coralnpu/` — 저장소에 아예 없는 파일. 시뮬레이터 실행 스크립트
(`run_probe.py` 등)와 `//sim:coralnpu_v2_sim` 별칭.

## 설계 메모

- **pip 판 torch-mlir 을 넣지 않는다.** 소스 빌드와 같은 네임스페이스 패키지라
  섞이면 pip 판이 먼저 잡혀 양자화 relu lowering 이 조용히 사라진다.
  `requirements.txt` 주석 참고.
- **torch 는 CPU 휠.** 파이프라인은 CPU 경로만 쓴다. 이미지 2.5GB 절약.
- **bazel output_base 를 `/opt/bazel-out` 에 고정** (`.bazelrc.user`).
  `build_coral.sh` 가 `bazel info output_base` 로 조회하므로 경로 하드코딩이
  없고, 이미지 안에서 재빌드 없이 즉시 실행된다.
- **빌드 시점 자가검증**: `torch_mlir` 이 소스 빌드에서 로드되는지, iree-compile·
  java·libcrt.lo 가 있는지 Dockerfile 안에서 확인한다. 깨지면 이미지가 안 나온다.
- **파이프라인 COPY 는 맨 마지막 레이어** — 커밋마다 바뀌므로 툴체인 캐시를 살린다.

## 알려진 제약

- resnet18 은 벡터 빌드에서 IREE riscv32 바인딩 버그로 정확도가 깨진다
  (`../repro/iree-rv32-binding/` 재현 참고). 스칼라 피처로 빌드해야 한다.
- 데이터셋의 `darknet/`(YOLO 가중치)·`photos/`(캘리브레이션 사진)는 자동
  다운로드가 없다. `fetch-data.sh` 는 MNIST/CIFAR-100 만 받는다.
