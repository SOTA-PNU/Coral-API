import sys, os
R = "/workspace/coralnpu/bazel-bin/sw/coralnpu_sim/run_probe.runfiles"
sys.path.insert(0, f"{R}/coralnpu_hw/sw/coralnpu_sim")
for dep in ("coralnpu_pip_deps_numpy", "coralnpu_pip_deps_pyelftools"):
    root = f"{R}/{dep}"; sp = os.path.join(root, "site-packages")
    sys.path.insert(0, sp if os.path.isdir(sp) else root)
import numpy as np
from coralnpu_v2_sim_utils import CoralNPUV2Simulator
from run_probe import preload_high_segments
L = "/workspace/lowering-project"
ELF = f"{L}/build/tpi8/elf/model_coral_elf"
sim = CoralNPUV2Simulator(semihost_htif=False)
entry, syms = sim.get_elf_entry_and_symbol(ELF, ["inference_output", "inference_stage"])
sim.load_program(ELF, entry); preload_high_segments(sim, ELF)
sim.run(); sim.wait()
stage = int(sim.read_memory(syms["inference_stage"], 4).view(np.uint32)[0])
print("stage =", stage)
# 바늘: 기대 출력의 내부 첫 64바이트 (out[1,1,0:64] = 입력의 [:,0,0])
i = np.load("/tmp/claude-1002/-workspace/878a10e6-4285-4204-baa3-89244afd0ea9/scratchpad/tpi8_in.npy")
needle = i[:,0,0].astype(np.int8).tobytes()
print("바늘 16B:", needle[:16].hex())
hits = []
CH = 1 << 20
for base in range(0x80000000, 0x80000000 + 24*CH, CH):
    buf = bytes(sim.read_memory(base, CH))
    off = -1
    while True:
        off = buf.find(needle, off+1)
        if off < 0: break
        hits.append(base + off)
print("발견 주소:", [hex(h) for h in hits[:8]], f"(총 {len(hits)})")
# 기대 출력 버퍼 위치: 0 으로 남은 곳 — inference_output 심볼 근처가 아니라 heap 의 결과 버퍼.
# 참고로 입력 텐서(연속 CHW)의 위치도 찾자: 입력 첫 64B
nin = i.reshape(-1)[:64].astype(np.int8).tobytes()
ih = []
for base in range(0x80000000, 0x80000000 + 24*CH, CH):
    buf = bytes(sim.read_memory(base, CH))
    off = buf.find(nin)
    if off >= 0: ih.append(base + off)
print("입력 텐서 위치:", [hex(h) for h in ih[:4]])
