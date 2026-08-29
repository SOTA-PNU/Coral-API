# Run hello_world_add_floats.elf on the CoralNPU MPACT simulator via the Python API.
# Usage: bazel run --config=npusim //sw/coralnpu_sim:run_hello -- /abs/path/program.elf
import sys
import numpy as np
from coralnpu_v2_sim_utils import CoralNPUV2Simulator


def main():
    if len(sys.argv) < 2:
        print("usage: run_hello <program.elf>")
        return 1
    elf = sys.argv[1]

    # Defaults: highmem layout, exit_on_ebreak=True, semihost_htif=True.
    sim = CoralNPUV2Simulator()

    # Look up symbol addresses from the ELF, then load the program.
    entry, syms = sim.get_elf_entry_and_symbol(elf, ["input1", "input2", "output"])
    sim.load_program(elf, entry)

    # Write inputs into memory (same values the cocotb tutorial uses).
    in1 = np.arange(1, 9, dtype=np.float32)
    in2 = 0.5 * np.ones(8, dtype=np.float32)
    sim.write_memory(syms["input1"], in1)
    sim.write_memory(syms["input2"], in2)

    # Run to completion.
    sim.run()
    sim.wait()

    # Read back and print the result.
    out = sim.read_memory(syms["output"], 4 * 8).view(np.float32)
    print("input1 =", in1)
    print("input2 =", in2)
    print("output =", out)
    print("cycles =", sim.get_cycle_count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
