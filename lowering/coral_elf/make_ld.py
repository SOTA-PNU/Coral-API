#!/usr/bin/env python3
"""ITCM 판 coral.ld 에서 DDR 판을 만든다.

왜 필요한가
-----------
ITCM 은 1 MiB 뿐이라 조금만 큰 모델도 .text 가 안 들어간다
(CharRNN 은 LSTM 16스텝 언롤 때문에 .text 가 1.55 MB).

시뮬레이터의 DDR 은 0x80000000 부터 128 MiB 이고 권한이
kReadWriteExecute 다 (sw/coralnpu_sim/coralnpu_v2_sim_utils.py). 그래서
코드와 상수를 통째로 DDR 에 두고 실행할 수 있다. ITCM 만큼 빠르진 않지만
용량 제한이 사라진다.

바꾸는 것
---------
  * DDR 의 길이를 실제 값(128M)으로, 권한에 x 추가
  * ITCM 에 있던 출력 섹션(.text/.rodata/.data.rel.ro 등)을 DDR 로
  * .heap 은 DDR 에서 남은 공간 전부를 가져가도록

DTCM(.data/.bss/.stack) 은 그대로 둔다 — 작고 자주 쓰여서 빠른 편이 낫다.
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()

# 1) DDR 실제 크기/권한
t = re.sub(r"DDR\(rw\):\s*ORIGIN\s*=\s*0x80000000,\s*LENGTH\s*=\s*\S+",
           "DDR(rwx): ORIGIN = 0x80000000, LENGTH = 128M", t)

# 2) ITCM 배치를 DDR 로
n_itcm = t.count("} > ITCM")
t = t.replace("} > ITCM", "} > DDR")
t = t.replace(". = ORIGIN(ITCM);", ". = ORIGIN(DDR);")

# 3) heap 은 DDR 잔여 전부
old_heap = re.search(r"\.heap : ALIGN\(16\) \{.*?\} > DDR", t, re.S)
if not old_heap:
    print("!! .heap 블록을 찾지 못했습니다", file=sys.stderr)
    sys.exit(1)
new_heap = """.heap : ALIGN(16) {
      __heap_start__ = .;
      __heap_start = .;
      /* 코드/상수/가중치를 DDR 에 올린 뒤 남는 공간을 전부 힙으로. */
      . = ORIGIN(DDR) + LENGTH(DDR) - 64;
      __heap_end__ = .;
      __heap_end = .;
    } > DDR"""
t = t[:old_heap.start()] + new_heap + t[old_heap.end():]

open(dst, "w").write(t)
print(f"[make_ld] {n_itcm}개 섹션을 ITCM -> DDR 로, heap 은 DDR 잔여 전부 -> {dst}")
