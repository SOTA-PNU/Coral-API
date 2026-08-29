# Generic CoralNPU sim runner: load an ELF, run it, print cycle count,
# and optionally dump symbols given as name[:dtype[:count]] (defaults float32, 8).
# Usage: bazel run --config=npusim //sw/coralnpu_sim:run_any -- prog.elf [output:int16:1024 ...]
import sys
import numpy as np
from coralnpu_v2_sim_utils import CoralNPUV2Simulator

_DT = {
    "int8": np.int8, "uint8": np.uint8,
    "int16": np.int16, "uint16": np.uint16,
    "int32": np.int32, "uint32": np.uint32,
    "float32": np.float32,
}


def main():
    if len(sys.argv) < 2:
        print("usage: run_any <prog.elf> [name[:dtype[:count]] ...]")
        return 1
    elf = sys.argv[1]
    specs = [s.split(":") for s in sys.argv[2:]]
    names = [s[0] for s in specs]

    sim = CoralNPUV2Simulator(semihost_htif=False)
    entry, syms = sim.get_elf_entry_and_symbol(elf, names)
    sim.load_program(elf, entry)
    sim.run()
    sim.wait()
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
        print(f"{name} =", data[:min(cnt, 32)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
