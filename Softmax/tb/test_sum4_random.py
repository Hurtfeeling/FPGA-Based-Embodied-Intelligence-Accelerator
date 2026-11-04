"""Randomized four-lane CSA sum checks with an independent integer RNE reference."""
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "Softmax" / "tb" / "build" / "sum4_random"


def quantize(x):
    exponent = (x >> 23) & 255
    if exponent == 0:
        return 0
    mantissa = (1 << 23) | (x & 0x7FFFFF)
    shift = exponent - 102
    if shift >= 0:
        return mantissa << shift
    truncated, remainder = divmod(mantissa, 1 << -shift)
    half = 1 << (-shift - 1)
    return truncated + (remainder > half or (remainder == half and truncated & 1))


def main():
    rng = random.Random(0x4816)
    lengths = [1024, 1, 2, 3, 4, 7, 8, 13]
    lengths += [rng.randrange(1, 129) for _ in range(56)]
    boundaries = [0, 0x80000000, 0x26FFFFFF, 0x27000000, 0x27000001,
                  0x27C00000, 0x28200000, 0x28600000, 0x3F800000]
    beats, sums = [], []
    for row, length in enumerate(lengths):
        total = 0
        for _ in range(length):
            values = []
            for lane in range(4):
                if row == 0:
                    x = 0x3F800000
                elif rng.randrange(5) == 0:
                    x = rng.choice(boundaries)
                else:
                    x = (rng.randrange(127) << 23) | rng.randrange(1 << 23)
                values.append(x)
                total += quantize(x)
            beats.append(sum(x << (32 * lane) for lane, x in enumerate(values)))
        sums.append(total)
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, words, digits in [("data", beats, 32), ("sums", sums, 16), ("lengths", lengths, 3)]:
        (BUILD / f"{name}.mem").write_text(
            "".join(f"{x:0{digits}x}\n" for x in words), encoding="ascii")
    (BUILD / "count.vh").write_text(f"localparam NBEATS={len(beats)};\n", encoding="ascii")
    output = BUILD / "sum4_random.vvp"
    subprocess.run(["iverilog", "-g2005", "-Wall", "-I", str(BUILD),
                    "-s", "tb_softmax_sum4_random", "-o", str(output),
                    "Softmax/rtl/softmax_sum4.v", "Softmax/tb/tb_softmax_sum4_random.v"],
                   cwd=ROOT, check=True)
    subprocess.run(["vvp", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
