#!/usr/bin/env bash
# CI 가 마운트할 데이터셋을 호스트에 준비한다. 한 번만 실행하면 된다.
#
#   ./fetch-data.sh                    # 기본 위치에 받는다
#   SRC=/path/to/existing/data ./fetch-data.sh   # 기존 데이터를 복사
#
# 데이터셋(122MB)은 저장소에 넣지 않는다. 체크포인트(768KB)는 lowering/ 에 있다.
set -euo pipefail
DEST="${DEST:-/mnt/data/jiho/coral-data/data}"
mkdir -p "$DEST"

if [ -n "${SRC:-}" ]; then
  echo "기존 데이터 복사: $SRC → $DEST"
  cp -a "$SRC/." "$DEST/"
else
  echo "torchvision 으로 MNIST/CIFAR-100 을 내려받습니다 (약 87MB)"
  python3 - "$DEST" <<'PY'
import sys, torchvision
d = sys.argv[1]
torchvision.datasets.MNIST(d, download=True)
torchvision.datasets.CIFAR100(d + "/cifar100", download=True)
print("완료")
PY
  echo
  echo "!! 아직 수동으로 채워야 하는 것:"
  echo "   $DEST/darknet/   yolov3-tiny.cfg, yolov3-tiny.weights, coco.names"
  echo "   $DEST/photos/    캘리브레이션용 사진 (파일명에 ImageNet 클래스)"
  echo "   기존 환경에서 가져오는 편이 빠릅니다: SRC=<기존 data 경로> $0"
fi
echo "준비됨: $DEST ($(du -sh "$DEST" | cut -f1))"
