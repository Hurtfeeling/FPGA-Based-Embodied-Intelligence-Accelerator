"""Independent UQ16.48 -> binary32 RNE checks for the row-sum pipeline.

Run from any directory with Python and Icarus Verilog installed.
"""
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "Softmax" / "tb" / "build" / "sum_to_fp32"
N = 8192


def reference(x):
    lead = x.bit_length() - 1
    shift = lead - 23
    mantissa, remainder = divmod(x, 1 << shift)
    half = 1 << (shift - 1)
    mantissa += remainder > half or (remainder == half and mantissa & 1)
    if mantissa >= 1 << 24:
        mantissa >>= 1
        lead += 1
    return ((lead - 48 + 127) << 23) | (mantissa & 0x7FFFFF)


def main():
    rng = random.Random(0x51)
    low, high = 1 << 48, 4096 << 48
    values = [low, high, low + 1, high - 1]
    for exponent in range(48, 61):
        # Adjacent fixed-point codes around powers of two and RNE ties.
        for delta in range(-3, 4):
            x = (1 << exponent) + delta
            if low <= x <= high:
                values.append(x)
        for mantissa in [1, 2, 3, 0x7FFFFE, 0x7FFFFF]:
            for delta in [-1, 0, 1]:
                x = ((1 << exponent) + (mantissa << (exponent - 23))
                     + (1 << (exponent - 24)) + delta)
                if low <= x <= high:
                    values.append(x)
    while len(values) < N:
        values.append(rng.randrange(low, high + 1))
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "vectors.mem").write_text(
        "".join(f"{x:016x}{reference(x):08x}\n" for x in values), encoding="ascii")
    output = BUILD / "sum_to_fp32.vvp"
    subprocess.run(["iverilog", "-g2005", "-Wall", "-s", "tb_softmax_sum_to_fp32",
                    "-o", str(output), "Softmax/rtl/softmax_sum_to_fp32.v",
                    "Softmax/tb/tb_softmax_sum_to_fp32.v"], cwd=ROOT, check=True)
    subprocess.run(["vvp", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
