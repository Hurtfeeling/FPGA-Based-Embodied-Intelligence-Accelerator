"""Bit-exact checks for the four-lane pipelined binary32 multiplier.

Run from any directory: python Softmax/tb/test_mul4.py
Requires only Python's standard library and Icarus Verilog. The reference
uses binary64 multiplication followed by binary32 rounding: a product of
two binary32 significands has at most 48 bits, so binary64 represents the
finite product exactly before the independent binary32 conversion.
"""
from pathlib import Path
import math
import random
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "Softmax" / "tb" / "build" / "mul4"
N_BEATS = 4096


def value(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def bits(number):
    try:
        return struct.unpack("<I", struct.pack("<f", number))[0]
    except OverflowError:
        return 0xFF800000 if number < 0 else 0x7F800000


def flush(word):
    return word & 0x80000000 if (word >> 23) & 255 == 0 else word


def reference(a, b):
    product = value(flush(a)) * value(flush(b))
    return 0x7FC00000 if math.isnan(product) else flush(bits(product))


def generate():
    rng = random.Random(0x250F0032)
    # Pair every corner with every common scale. This includes signed zero,
    # DAZ, minimum-normal rounding, overflow, infinities, and both NaN classes.
    corners = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
               0x00800000, 0x80800000, 0x00800001, 0x80800001,
               0x00800002, 0x80800002, 0x3F000000, 0xBF000000,
               0x3F7FFFFE, 0xBF7FFFFE, 0x3F7FFFFF, 0xBF7FFFFF,
               0x3F800000, 0xBF800000, 0x3F800001, 0xBF800001,
               0x40000000, 0xC0000000, 0x7F7FFFFF, 0xFF7FFFFF,
               0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
               0x7F800001, 0xFF800001]
    cases = [(corners[index:index + 4], scale, 15)
             for scale in corners for index in range(0, len(corners), 4)]
    while len(cases) < N_BEATS:
        if len(cases) % 4 == 0:
            # Deliberately concentrate exponent sums near the FTZ boundary;
            # pure random words rarely exercise rounding into min-normal.
            scale_exp = rng.randrange(1, 127)
            lane_exp = 127 - scale_exp
            scale = (rng.randrange(2) << 31) | (scale_exp << 23) | rng.getrandbits(23)
            lanes = [(rng.randrange(2) << 31) |
                     (max(1, lane_exp + rng.randrange(-2, 3)) << 23) |
                     rng.getrandbits(23) for _ in range(4)]
        else:
            scale = rng.getrandbits(32)
            lanes = [rng.getrandbits(32) for _ in range(4)]
        cases.append((lanes, scale, rng.randrange(16) if len(cases) % 3 else 15))

    # Lock down the subtle RNE-before-FTZ case independently of random input.
    assert reference(0x00800000, 0x3F7FFFFF) == 0x00800000
    assert reference(0x00800000, 0x3F7FFFFE) == 0
    assert reference(0x80800000, 0x3F7FFFFF) == 0x80800000
    words = []
    for lanes, scale, keep in cases:
        data = sum(word << (32 * lane) for lane, word in enumerate(lanes))
        expected = sum((reference(word, scale) if (keep >> lane) & 1 else 0) << (32 * lane)
                       for lane, word in enumerate(lanes))
        words.append((keep << 288) | (scale << 256) | (data << 128) | expected)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "vectors.mem").write_text("".join(f"{word:073x}\n" for word in words), encoding="ascii")


def main():
    generate()
    executable = BUILD / "mul4.vvp"
    subprocess.run(["iverilog", "-g2005", "-Wall", "-s", "tb_softmax_mul4",
                    "-P", f"tb_softmax_mul4.N_BEATS={N_BEATS}", "-o", str(executable),
                    "Softmax/rtl/softmax_fp32_mul4.v", "Softmax/tb/tb_softmax_mul4.v"],
                   cwd=ROOT, check=True)
    result = subprocess.run(["vvp", str(executable)], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    passed = [line for line in result.stdout.splitlines() if line.startswith("PASS mul4:")]
    if result.returncode or len(passed) != 1:
        raise RuntimeError(f"mul4 simulation failed ({result.returncode}):\n{result.stdout}")
    print(passed[0])


if __name__ == "__main__":
    main()
