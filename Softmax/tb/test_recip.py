"""Bit-exact reciprocal checks across all binary32 exponent ranges."""
from pathlib import Path
import math
import random
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "Softmax" / "tb" / "build" / "recip"
N = 4096


def value(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


def bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def flush(x):
    return x & 0x80000000 if (x >> 23) & 255 == 0 else x


def reference(x):
    z = value(flush(x))
    if math.isnan(z):
        return 0x7FC00000
    if math.isinf(z):
        return x & 0x80000000
    if z == 0:
        return (x & 0x80000000) | 0x7F800000
    return flush(bits(1 / z))


def main():
    rng = random.Random(0x251)
    values = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000,
              0x00800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
              0xFF800000, 0x7FC00000, 0xFFC00000]
    # Exact powers, their next representable neighbor, and the largest
    # significand in every finite exponent bucket, with both signs.
    values += [(exponent << 23) | mantissa | sign
               for exponent in range(1, 255)
               for mantissa in [0, 1, 0x7FFFFF]
               for sign in [0, 0x80000000]]
    values += [bits(rng.uniform(1, 4096)) for _ in range(1024)]
    while len(values) < N:
        values.append(rng.getrandbits(32))
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "vectors.mem").write_text(
        "".join(f"{x:08x}{reference(x):08x}\n" for x in values), encoding="ascii")
    output = BUILD / "recip.vvp"
    subprocess.run(["iverilog", "-g2005", "-Wall", "-s", "tb_softmax_recip",
                    "-o", str(output), "Softmax/rtl/softmax_recip.v",
                    "Softmax/tb/tb_softmax_recip.v"], cwd=ROOT, check=True)
    subprocess.run(["vvp", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
