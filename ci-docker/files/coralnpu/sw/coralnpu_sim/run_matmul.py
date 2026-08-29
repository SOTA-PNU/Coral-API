# Run the matmul ELF, OVERRIDE its built-in A,B,num1,num2, then read back C,result.
# Usage: bazel run --config=npusim //sw/coralnpu_sim:run_matmul -- /abs/matmul.elf
import sys
import numpy as np
from coralnpu_v2_sim_utils import CoralNPUV2Simulator


def main():
    elf = sys.argv[1]
    sim = CoralNPUV2Simulator()

    entry, s = sim.get_elf_entry_and_symbol(elf, ["A", "B", "C", "num1", "num2", "result", "data"])
    sim.load_program(elf, entry)

    # --- write ALL inputs BEFORE the single run ---
    A = np.eye(4, dtype=np.int8)
    B = np.arange(16, dtype=np.int8).reshape(4, 4)
    sim.write_memory(s["A"], A.flatten())
    sim.write_memory(s["B"], B.flatten())
    # num1/num2 are int8_t (1 byte). pass 1-element int8 numpy arrays (NOT a bare int, NO dtype= kwarg).
    sim.write_memory(s["num1"], np.array([3], dtype=np.int8))
    sim.write_memory(s["num2"], np.array([10], dtype=np.int8))
    sim.write_memory(s['data'], np.array([1,2,3,4], dtype=np.int8))

    sim.run()     # runs main() once: matmul AND result = num2 / num1
    sim.wait()

    C = sim.read_memory(s["C"], 4 * 16).view(np.int32).reshape(4, 4)
    result = sim.read_memory(s["result"], 1).view(np.int8)[0]   # result is int8 -> read 1 byte
    data = sim.read_memory(s["data"], 4).view(np.int8)  # data is int8[4] -> read 4 bytes

    print("C = A @ B =\n", C)
    print("result = num2 % num1 = 10 % 3 =", result)
    print("data =", data)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
