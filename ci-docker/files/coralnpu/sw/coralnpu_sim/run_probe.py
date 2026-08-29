# run_any 와 같지만, 실행 중에 주기적으로 진행 상황을 찍고 데드라인에서 halt 한다.
# 무한 루프(사이클은 늘지만 stage/PC 가 고정)와 단순히 느린 것을 구분하기 위한 것.
#
# usage: run_probe <deadline_sec> <poll_sec> <prog.elf> [name[:dtype[:count]] ...]
import sys
import threading
import time

import numpy as np
from coralnpu_v2_sim_utils import CoralNPUV2Simulator

_DT = {
    "int8": np.int8, "uint8": np.uint8,
    "int16": np.int16, "uint16": np.uint16,
    "int32": np.int32, "uint32": np.uint32,
    "float32": np.float32,
}



def preload_high_segments(sim, elf_path):
    """ITCM/DTCM 밖(EXTMEM/DDR)에 있는 PT_LOAD 세그먼트를 직접 써 넣는다.

    Coral 의 부트스트랩은 ELF 의 로드 가능 섹션을 ITCM 과 DTCM 으로만
    memcpy 한다 (developers.google.com/coral/guides/software/bare-metal-prog).
    DDR 은 부트로더가 따로 채워야 하는 영역이라, 시뮬레이터의 load_program
    만으로는 .ddr_data 가 0 으로 남는다. 여기서 그 역할을 대신한다.

    ELF32 리틀엔디안 프로그램 헤더를 직접 읽는다 (pyelftools 의존 없이).
    """
    blob = open(elf_path, "rb").read()
    e_phoff = int.from_bytes(blob[28:32], "little")
    e_phentsize = int.from_bytes(blob[42:44], "little")
    e_phnum = int.from_bytes(blob[44:46], "little")
    loaded = []
    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        if int.from_bytes(blob[o:o + 4], "little") != 1:      # PT_LOAD 만
            continue
        p_offset = int.from_bytes(blob[o + 4:o + 8], "little")
        p_vaddr = int.from_bytes(blob[o + 8:o + 12], "little")
        p_filesz = int.from_bytes(blob[o + 16:o + 20], "little")
        if p_filesz == 0 or p_vaddr < 0x20000000:             # TCM 은 로더가 함
            continue
        payload = np.frombuffer(blob[p_offset:p_offset + p_filesz],
                                dtype=np.uint8).copy()
        sim.write_memory(p_vaddr, payload)
        loaded.append((p_vaddr, p_filesz))
    for addr, size in loaded:
        back = sim.read_memory(addr, 8)
        print(f"[preload] 0x{addr:08x} <- {size} bytes, 즉시 읽기: "
              f"{list(int(b) for b in back)}", flush=True)
    return loaded


def main():
    deadline = float(sys.argv[1])
    poll = float(sys.argv[2])
    elf = sys.argv[3]
    specs = [s.split(":") for s in sys.argv[4:]]
    names = [s[0] for s in specs]
    # 진행 관측용 심볼도 같이 찾아둔다
    watch = ["inference_stage", "device_step", "bm_allocs", "bm_frees"]
    for w in watch:
        if w not in names:
            names.append(w)

    sim = CoralNPUV2Simulator(semihost_htif=False)
    entry, syms = sim.get_elf_entry_and_symbol(elf, names)
    sim.load_program(elf, entry)
    preload_high_segments(sim, elf)

    def u32(name):
        a = syms.get(name, 0)
        if not a:
            return None
        try:
            return int(sim.read_memory(a, 4).view(np.uint32)[0])
        except Exception:
            return None

    done = threading.Event()

    # sim.run() 은 이 빌드에서 블로킹이다. 메인 스레드에서 부르면 폴링 루프가
    # 아예 돌지 않아 진행 상황도, 데드라인 halt 도 동작하지 않는다.
    def waiter():
        try:
            sim.run()
            sim.wait()
        finally:
            done.set()

    t0 = time.time()
    th = threading.Thread(target=waiter, daemon=True)
    th.start()

    prev_cycles = -1
    halted = False
    while not done.wait(poll):
        el = time.time() - t0
        try:
            cyc = sim.get_cycle_count()
        except Exception:
            cyc = -1
        try:
            pc = sim.read_register("pc")
        except Exception:
            pc = "?"
        print(f"[{el:7.0f}s] cycles={cyc} pc={pc} "
              f"stage={u32('inference_stage')} dev={u32('device_step')} "
              f"allocs={u32('bm_allocs')} frees={u32('bm_frees')} "
              f"(+{cyc - prev_cycles if prev_cycles >= 0 else 0} cycles)",
              flush=True)
        prev_cycles = cyc
        if el > deadline:
            print("!! 데드라인 도달 -> halt", flush=True)
            sim.halt()
            halted = True
            done.wait(30)
            break

    print("halted =", halted)
    print("cycles =", sim.get_cycle_count())
    for s in specs:
        name = s[0]
        dt = _DT[s[1]] if len(s) > 1 else np.float32
        cnt = int(s[2]) if len(s) > 2 else 8
        addr = syms.get(name, 0)
        if addr == 0:
            print(f"{name}: symbol not found")
            continue
        data = sim.read_memory(addr, np.dtype(dt).itemsize * cnt).view(dt)
        if len(s) > 3:          # name:dtype:count:저장경로
            data.tofile(s[3])
            print(f"[dump] {name} -> {s[3]} ({data.nbytes} bytes)", flush=True)
        if dt is np.int8 and cnt > 32:
            raw = bytes([int(b) & 0xFF for b in data])
            raw = raw.split(bytes(1))[0]
            print(name, '=', repr(raw.decode('utf-8', 'replace')))
        else:
            print(f"{name} =", data[:min(cnt, 32)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
