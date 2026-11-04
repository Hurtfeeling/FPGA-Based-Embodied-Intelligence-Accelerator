"""Independent binary32/reference checks for the synthesizable math datapath.

Run from any directory: python Softmax/tb/test_math.py
Only Python's standard library and Icarus Verilog are required.
"""
from pathlib import Path
import math
import random
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "Softmax" / "tb" / "build" / "math"
RNG = random.Random(0x5F7A)
N_ARITH = 12000
N_EXP = 1024
N_NORM = 512
N_RECIP = 256
N_UQ = 1024


def bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def value(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


def flush(x):
    return x & 0x80000000 if (x >> 23) & 255 == 0 else x


def ref_binary(a, b, op):
    a, b = value(flush(a)), value(flush(b))
    result = a - b if op == "sub" else a * b
    if math.isnan(result):
        return 0x7FC00000
    return flush(bits(result))


def ref_uq(x):
    if not x:
        return 0
    lead = x.bit_length() - 1
    shift = lead - 23
    if shift > 0:
        mant, rem = divmod(x, 1 << shift)
        half = 1 << (shift - 1)
        mant += rem > half or (rem == half and mant & 1)
    else:
        mant = x << -shift
    if mant >= 1 << 24:
        mant >>= 1
        lead += 1
    return ((lead - 48 + 127) << 23) | (mant & 0x7FFFFF)


def write_mem(name, words, digits):
    (BUILD / name).write_text("".join(f"{word:0{digits}x}\n" for word in words), encoding="ascii")


def generate():
    BUILD.mkdir(parents=True, exist_ok=True)
    corners = [0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x00800001,
               0x3F000000, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x40000000,
               0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000]
    pairs = [(a, b) for a in corners for b in corners]
    while len(pairs) < N_ARITH:
        if len(pairs) % 3:
            a, b = RNG.getrandbits(32), RNG.getrandbits(32)
        else:
            a, b = bits(RNG.uniform(-1e7, 1e7)), bits(RNG.uniform(-1e7, 1e7))
        pairs.append((a, b))
    write_mem("arith.mem", [(a << 96) | (b << 64) | (ref_binary(a, b, "sub") << 32) |
                           ref_binary(a, b, "mul") for a, b in pairs], 32)
    uq = [0, 1, (1 << 48), 4096 << 48, (1 << 60) - 1]
    uq += [RNG.randrange(1 << 60) for _ in range(N_UQ - len(uq))]
    write_mem("uq.mem", [(a << 32) | ref_uq(a) for a in uq], 24)

    exp_cases = []
    exp_words = []
    for i in range(N_EXP):
        maximum = bits(RNG.uniform(-30.0, 30.0))
        keep = 15 if i % 13 else RNG.randrange(1, 16)
        if i == 0:
            maximum = bits(2.0)
            lanes = [bits(2.0 + math.log(x)) for x in [1.0, .75, .5, .25]]
        elif i == 1:
            maximum = 0x7F7FFFFF
            lanes = [maximum, 0xFF7FFFFF, 0xFF800000, 0]
        elif i == 2:
            maximum = 0
            lanes = [0, 0x80000000, 1, 0x80000001]
        elif i == 3:
            maximum = 0
            lanes = [bits(-126 * math.log(2) + d) for d in [-1e-4, -1e-6, 1e-6, 1e-4]]
        else:
            lanes = [bits(value(maximum) - RNG.uniform(0, 100)) for _ in range(4)]
            if i % 9 == 0:
                lanes[0] = maximum
            if i % 17 == 0:
                lanes[2] = 0xFF800000
        data = sum(x << (32 * lane) for lane, x in enumerate(lanes))
        exp_words.append((keep << 160) | (maximum << 128) | data)
        expected = []
        for lane, x in enumerate(lanes):
            if not (keep >> lane) & 1 or x == 0xFF800000:
                expected.append(0.0)
            else:
                z = value(ref_binary(x, maximum, "sub"))
                expected.append(math.exp(z))
        exp_cases.append(expected)
    write_mem("exp.mem", exp_words, 41)

    norm_words, norm_expected = [], []
    for i in range(N_NORM):
        keep = 15 if i % 7 else RNG.randrange(1, 16)
        scale = bits(1 / RNG.uniform(1, 4096))
        lanes = [bits(2.0 ** RNG.uniform(-126, 0)) for _ in range(4)]
        data = sum(x << (32 * lane) for lane, x in enumerate(lanes))
        norm_words.append((keep << 160) | (scale << 128) | data)
        norm_expected.append(sum((ref_binary(x, scale, "mul") if (keep >> lane) & 1 else 0)
                                 << (32 * lane) for lane, x in enumerate(lanes)))
    write_mem("norm.mem", norm_words, 41)
    recip = [bits(2.0 ** i) for i in range(13)]
    recip += [bits(x) for x in [480, 640, 1023, 4095, 1.0000001192092896]]
    recip += [bits(RNG.uniform(1, 4096)) for _ in range(N_RECIP - len(recip))]
    write_mem("recip.mem", [(x << 32) | bits(1.0 / value(x)) for x in recip], 16)
    return exp_cases, norm_expected


def main():
    exp_expected, norm_expected = generate()
    output = BUILD / "math.vvp"
    sources = ["Softmax/rtl/softmax_fp32_sub4.v", "Softmax/rtl/softmax_fp32_mul4.v", "Softmax/rtl/softmax_exp4.v",
               "Softmax/rtl/softmax_norm4.v", "Softmax/rtl/softmax_recip.v",
               "Softmax/tb/tb_softmax_math.v"]
    subprocess.run(["iverilog", "-g2005", "-I", "Softmax/rtl", "-s", "tb_softmax_math", "-o", str(output), *sources], cwd=ROOT, check=True)
    subprocess.run(["vvp", str(output)], cwd=ROOT, check=True)
    worst = 0.0
    observed = (BUILD / "exp_results.txt").read_text().splitlines()
    assert len(observed) == N_EXP
    for index, line in enumerate(observed):
        tag, packed = (int(x, 16) for x in line.split())
        assert tag == index, (tag, index)
        for lane, ref in enumerate(exp_expected[index]):
            got_bits = (packed >> (32 * lane)) & 0xFFFFFFFF
            got = value(got_bits)
            assert math.isfinite(got) and 0 <= got <= 1, (index, lane, got)
            if ref == 0:
                assert got_bits == 0, (index, lane, got)
            elif ref < 2.0 ** -126 * (1 + 1e-5):
                # FTZ boundary classification can move by the range-reduction
                # rounding error; compare absolute rather than relative error.
                assert abs(got - ref) <= 2.0 ** -126 * (1 + 1e-5)
            else:
                relative = abs(got - ref) / ref
                worst = max(worst, relative)
                assert relative < 1e-5, (index, lane, got, ref, relative)
                if ref == 1:
                    assert got_bits == 0x3F800000
    observed = (BUILD / "norm_results.txt").read_text().splitlines()
    assert len(observed) == N_NORM
    for index, line in enumerate(observed):
        tag, data = (int(x, 16) for x in line.split())
        assert tag == index and data == norm_expected[index], (index, tag, hex(data), hex(norm_expected[index]))
    print(f"PASS math: {N_ARITH} exact sub/mul pairs, {N_UQ} fixed conversions, {N_RECIP} reciprocals")
    print(f"PASS streams: {N_EXP * 4} exp lanes, {N_NORM * 4} normalize lanes; worst exp relative error {worst:.9g}")


if __name__ == "__main__":
    main()
