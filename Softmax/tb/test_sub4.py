"""Bit-exact randomized test for the actual pipelined Softmax subtractor."""
from pathlib import Path
import math
import random
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "Softmax" / "tb" / "build" / "sub4"
N = 8192


def value(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


def bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def daz(x):
    return x & 0x80000000 if (x >> 23) & 255 == 0 else x


def finite_random(rng):
    while True:
        x = rng.getrandbits(32)
        if (x >> 23) & 255 != 255:
            return x


def main():
    rng = random.Random(0x250F032)
    BUILD.mkdir(parents=True, exist_ok=True)
    cases = []
    corners = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
               0x00800000, 0x80800000, 0x00800001, 0x80800001,
               0x00800002, 0x80800002, 0x00FFFFFF, 0x80FFFFFF,
               0x01000000, 0x81000000, 0x01000001, 0x81000001,
               0x3F000000, 0xBF000000, 0x3F7FFFFF, 0xBF7FFFFF,
               0x3F800000, 0xBF800000, 0x3F800001, 0xBF800001,
               0x7F7FFFFF, 0xFF7FFFFF]
    # Exhaustive corner pairing includes cancellation into subnormal range,
    # signed zeros, DAZ, opposite-sign overflow, and huge exponent gaps.
    for x in corners:
        for y in corners:
            maximum = y if value(daz(x)) <= value(daz(y)) else x
            minimum = x if value(daz(x)) <= value(daz(y)) else y
            cases.append(([minimum, maximum, minimum, maximum], maximum, 15))
    while len(cases) < N:
        if len(cases) % 4 == 0:
            exponent = rng.choice([1, 2, 3, rng.randrange(1, 255)])
            sign = rng.randrange(2) << 31
            base = sign | (exponent << 23) | rng.randrange(1 << 23)
            values = [base]
            for _ in range(3):
                delta = rng.randrange(-16, 17)
                values.append(max(sign | (exponent << 23), min(base + delta, sign | (exponent << 23) | 0x7FFFFF)))
        else:
            values = [finite_random(rng) for _ in range(4)]
        maximum = max(values, key=lambda x: value(daz(x)))
        keep = rng.randrange(16) if len(cases) % 13 == 0 else 15
        if len(cases) % 17 == 0:
            values[2] = 0xFF800000
        cases.append((values, maximum, keep))
    words = []
    for values, maximum, keep in cases:
        expected, out_keep = 0, keep
        for lane, x in enumerate(values):
            if not (keep >> lane) & 1 or (x >> 23) & 255 == 255:
                result = 0
                out_keep &= ~(1 << lane)
            else:
                result = daz(bits(value(daz(x)) - value(daz(maximum))))
            expected |= result << (32 * lane)
        data = sum(x << (32 * lane) for lane, x in enumerate(values))
        words.append((out_keep << 292) | (expected << 164) | (keep << 160) | (maximum << 128) | data)
    (BUILD / "vectors.mem").write_text("".join(f"{x:074x}\n" for x in words), encoding="ascii")
    subprocess.run(["iverilog", "-g2005", "-Wall", "-I", "Softmax/rtl", "-s", "tb_softmax_sub4",
                    "-o", str(BUILD / "sub4.vvp"), "Softmax/rtl/softmax_fp32_sub4.v",
                    "Softmax/tb/tb_softmax_sub4.v"], cwd=ROOT, check=True)
    subprocess.run(["vvp", str(BUILD / "sub4.vvp")], cwd=ROOT, check=True)
    print(f"PASS sub4: {N * 4} lanes checked against independent binary32 subtraction, DAZ/RNE/FTZ")


if __name__ == "__main__":
    main()
