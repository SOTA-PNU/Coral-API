# IREE riscv32 디스패치 바인딩 주소 불일치 재현

## 증상
`tp_i8.mlir` (20줄: i8 pad+transpose 64x56x56→58x58x64)를 riscv32
`+zve32x,+zvl128b` 벡터로 컴파일해 MPACT coralnpu_v2 시뮬레이터에서 실행하면
결과 버퍼가 미기록(0) 상태로 반환된다. 같은 파일의 x86 호스트 실행은 정확.
스칼라(rv32, 벡터 피처 없이) 빌드도 정확.

## 핵심 증거 (메모리 스캔)
실행 후 시뮬 메모리를 스캔하면 **올바르게 전치된 데이터가 정확히 한 주소에
존재**한다 — 커널 계산은 완벽하고, 기록 주소만 VM/소비자가 읽는 버퍼와
다르다 (예시 실행: 기록 0x80207800, 소비 버퍼는 0 유지).
ResNet18 전체 그래프에서는 소비 버퍼 자리의 이전 텐서 잔재가 읽혀
"입력에 없는 값"으로 나타난다 (상대L2 0.997).

무죄 판정 완료: MPACT 시뮬 RVV 명령 22종 KAT(레지스터 그룹 교차·vl 초과
소스 슬라이드 포함), RV32 스칼라 경로, 커널 어셈 의미론.

## 재현 절차
1. iree-compile tp_i8.mlir (riscv32-unknown-elf, generic-rv32, ilp32,
   +m,+f,+zicsr,+zmmul,+zve32x,+zvl128b, link-static, vm-target-index-bits=32)
2. Coral-API lowering 하네스로 ELF 링크 (build/tpi8 방식, manifest dtype i8)
3. run_probe 로 시뮬 실행 → chip_output 이 ref 와 불일치(내부 전부 0)
4. 검증: scripts 의 find_writes.py 로 메모리 스캔 → 전치 데이터의 실제 위치 확인

관련 우회: 이 커널 형태가 생기는 그래프(resnet18)는 당분간 스칼라 피처로
빌드한다. lenet5/charrnn/vgg 는 이 커널 형태가 없어 벡터 정상.
