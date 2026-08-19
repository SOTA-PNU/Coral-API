#!/usr/bin/env python3
"""EmitC 생성 코드의 VM ABI ref 슬롯 정렬을 32비트 타깃에 맞게 고친다.

배경
----
IREE 의 EmitC 코드젠은 인자 버퍼의 iree_vm_ref_t 슬롯을 항상 8바이트 경계에
놓는다 (생성 코드의 iree_host_align(x, 8)). 런타임이 쓰는 iree_vm_abi_*_t 는
평범한 C 구조체라 자연 정렬을 따르는데, 32비트에서 iree_vm_ref_t 는
크기 8 / 정렬 4다. Coral(RV32) 실측:

    offsetof(iree_vm_abi_ICrD_t, a1) == 12    <- 런타임이 읽는 위치
    EmitC 가 쓰는 위치                == 16    <- iree_host_align(12, 8)

대부분의 시그니처는 ref 앞의 필드들이 이미 8의 배수라 차이가 없다. 문제가
되는 것은 ref 가 8의 배수가 아닌 오프셋에 오는 경우뿐이고, 이 모델이 쓰는
24개 shim 중 아래 3개가 그렇다:

  ICrD     (hal.fence.join)   i64(0..8) + count(8..12) -> ref @12
  iICrD    (hal.fence.await)  i32,i64,count            -> ref @20
  rIirIIi  (hal.buffer.*)     r,I,i                    -> ref @20

주의: 가변 원소가 i64 를 포함하는 시그니처(rIrrrICrIID 의 {ref,i64,i64} 등)는
원소 자체의 정렬이 8이므로 원래대로 두어야 한다. 그래서 전체 일괄 치환이
아니라 위 3개 함수 안에서만 바꾼다.

IREE 체크아웃은 건드리지 않는다. iree-compile 산출물만 손본다.
"""
import re
import sys

TARGETS = [
    "module_call_0ICrD_r_2_import_shim",
    "module_call_0iICrD_i_1_import_shim",
    "module_call_0rIirIIi_r_import_shim",
]

path = sys.argv[1]
src = open(path).read()

pat_size = re.compile(
    r"(\w+ = iree_host_align\(\w+, )8(\);\s*\n\s*\w+ = sizeof\(iree_vm_ref_t\);)")
pat_write = re.compile(
    r"(\w+) = iree_host_align\((\w+), 8\);(\s*\n\s*)(\w+) = \(uint8_t\*\) \1;"
    r"(\s*\n\s*)(\w+) = \(iree_vm_ref_t\*\) \4;")

total = 0
for name in TARGETS:
    # 정의(본문 있는 것)를 찾는다: "static ... name(...) {"
    m = re.search(r"static iree_status_t " + re.escape(name) + r"\([^;]*?\)\s*\{", src)
    if not m:
        print(f"  ! {name} 없음 (건너뜀)")
        continue
    start = m.start()
    end = src.index("\n}\n", start) + 3
    body = src[start:end]

    body, n1 = pat_size.subn(r"\g<1>4\g<2>", body)
    def rep(mm):
        return (f"{mm.group(1)} = iree_host_align({mm.group(2)}, 4);{mm.group(3)}"
                f"{mm.group(4)} = (uint8_t*) {mm.group(1)};{mm.group(5)}"
                f"{mm.group(6)} = (iree_vm_ref_t*) {mm.group(4)};")
    body, n2 = pat_write.subn(rep, body)

    src = src[:start] + body + src[end:]
    total += n1 + n2
    print(f"  {name}: 크기계산 {n1}곳, 쓰기 {n2}곳")

open(path, "w").write(src)
print(f"총 {total}곳 수정")
