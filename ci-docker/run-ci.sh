#!/usr/bin/env bash
#
# 빌드 → 실행 → 이미지 삭제. CI 가 부르는 진입점이며 손으로도 쓸 수 있다.
#
#   ./run-ci.sh                     # --quick + --gate (기본)
#   ./run-ci.sh --real              # 학습 가중치 4종
#   ./run-ci.sh --no-cache --quick  # 무캐시 완전 재빌드 검증
#
# 환경변수
#   CORAL_DATA   데이터셋 디렉터리 (기본 /mnt/data/jiho/coral-data/data)
#   OUT_DIR      아티팩트 수집 위치 (기본 ./artifacts)
#   KEEP_IMAGE=1 이미지를 지우지 않는다 (디버깅용)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CORAL_DATA="${CORAL_DATA:-/mnt/data/jiho/coral-data/data}"
OUT_DIR="${OUT_DIR:-$HERE/artifacts}"
TAG="coral-ci:$(date +%s)-$$"
NO_CACHE=""
MODELS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --no-cache) NO_CACHE="--no-cache" ;;
    *)          MODELS+=("$1") ;;
  esac
  shift
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=(--quick)

CID=""
cleanup() {
  local rc=$?
  [ -n "$CID" ] && docker rm -f "$CID" >/dev/null 2>&1
  if [ "${KEEP_IMAGE:-0}" != "1" ]; then
    echo "── 이미지 삭제: $TAG"
    docker rmi -f "$TAG" >/dev/null 2>&1
    # 태그만 지우면 dangling 레이어가 남는다. 빌드 캐시는 보존한다.
    docker image prune -f >/dev/null 2>&1
  else
    echo "── KEEP_IMAGE=1 — $TAG 유지"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

echo "════ 1/3 이미지 빌드 ($TAG) ════"
echo "  첫 빌드는 3~6시간(LLVM 2회+bazel). 이후는 BuildKit 캐시로 단축."
# lowering/ 을 컨텍스트에 넣기 위해 저장소 루트에서 빌드한다.
DOCKER_BUILDKIT=1 docker build $NO_CACHE \
  -f "$HERE/Dockerfile" -t "$TAG" \
  --progress=plain \
  "$REPO_ROOT" || { echo "!! 빌드 실패"; exit 1; }

echo "════ 2/3 파이프라인 실행: ${MODELS[*]} ════"
[ -d "$CORAL_DATA" ] || { echo "!! 데이터셋 없음: $CORAL_DATA (fetch-data.sh 참고)"; exit 1; }
CID=$(docker create -v "$CORAL_DATA":/workspace/lowering/data "$TAG" \
        ./coral.sh "${MODELS[@]}" --keep-going)
docker start -a "$CID"; RC=$?

echo "════ 3/3 아티팩트 수집 → $OUT_DIR ════"
mkdir -p "$OUT_DIR"
for p in run_lenet5.log run_charrnn.log run_lenet5_s1.log run_lenet5_s2.log \
         build_lenet5.export.log build_lenet5.build.log \
         build_charrnn.export.log build_charrnn.build.log; do
  docker cp "$CID:/workspace/lowering/$p" "$OUT_DIR/" 2>/dev/null
done
docker cp "$CID:/workspace/lowering/build" "$OUT_DIR/build" 2>/dev/null && \
  find "$OUT_DIR/build" -type f ! -name 'manifest.json' ! -name 'chip_output.bin' -delete 2>/dev/null
echo "수집 완료: $(find "$OUT_DIR" -type f 2>/dev/null | wc -l)개 파일"

echo "════ 결과: $([ $RC -eq 0 ] && echo 합격 || echo 불합격) (exit $RC) ════"
exit $RC
