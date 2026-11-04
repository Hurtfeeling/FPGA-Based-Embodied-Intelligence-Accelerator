"""Compile Verilog-2005 and check public AXI Softmax results against Python.

Requires Python 3 and Icarus Verilog; run from any working directory.
Default: arithmetic, CSA, AXI-Lite, and small end-to-end/regression cases.
--large adds a complete 1024 x 1024 matrix, including all DDR transfers.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "Softmax" / "rtl"
TB = ROOT / "Softmax" / "tb"
BUILD = TB / "build" / "axi"
FP32_MIN_NORMAL = 2.0**-126
ABS_TOL = 1.0e-7
REL_TOL = 5.0e-5
ROW_SUM_TOL = 5.0e-5


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    cols: int
    pattern: str = "random"
    expect_code: int = 0
    fault: int = 0
    config: int = 0
    inject_at: int = 0
    reset_mid: int = 0
    dense: int = 0


CASES = [
    Case("single", 1, 1, "uniform"),
    Case("signed_zero_subnormal", 1, 3, "zeros"),
    Case("tail_two_rows", 2, 5),
    Case("mask_tail", 3, 3, "mask"),
    Case("four_lane_example", 3, 1024, "example"),
    Case("random_1024", 3, 1024),
    Case("maximum_4096", 3, 4096),
    Case("sparse_mask_4096", 1, 4096, "sparse"),
    Case("exp_underflow", 2, 5, "underflow"),
    Case("extreme_finite", 2, 5, "extreme"),
    Case("uniform_1024", 2, 1024, "uniform"),
    Case("read_error_recovery", 3, 1024, expect_code=4, fault=1, inject_at=300),
    Case("write_error_recovery", 3, 1024, expect_code=5, fault=2),
    Case("nan_recovery", 2, 5, "nan", expect_code=2),
    Case("positive_inf_recovery", 2, 5, "positive_inf", expect_code=2),
    Case("all_mask_recovery", 2, 5, "all_mask", expect_code=3),
    Case("invalid_k", 1, 5, expect_code=1, config=1),
    Case("unaligned_address", 1, 5, expect_code=1, config=2),
    Case("overlapping_addresses", 2, 5, expect_code=1, config=3),
    Case("address_overflow", 1, 5, expect_code=1, config=4),
    Case("reset_active_job", 3, 1024, reset_mid=1),
]
LARGE_CASE = Case("matrix_1024x1024", 1024, 1024, "example", dense=1)


def fbits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fvalue(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def daz_value(bits: int) -> float:
    return fvalue(bits & 0x80000000 if (bits >> 23) & 255 == 0 else bits)


def row_bits(case: Case, row: int, rng: random.Random) -> list[int]:
    k = case.cols
    p = case.pattern
    if p == "uniform":
        return [fbits(float(row % 7 - 3))] * k
    if p == "zeros":
        raw = [0, 0x80000000, 0x80000001, 0x00000001]
        return [raw[col % 4] for col in range(k)]
    if p == "example":
        offset = (row % 16) / 8.0
        group = [fbits(2.0 + math.log(e) + offset) for e in (1.0, 0.75, 0.5, 0.25)]
        return [group[col % 4] for col in range(k)]
    if p == "sparse":
        return [0xFF800000] * (k - 1) + [fbits(-7.5)]
    if p == "underflow":
        values = [0.0, -200.0, -87.0, -90.0, -1000.0]
        return [fbits(values[col % 5] + row) for col in range(k)]
    if p == "extreme":
        values = ([0x7F7FFFFF, 0xFF7FFFFF, 0, fbits(1.0), 0xFF800000]
                  if row % 2 == 0 else [0xFF7FFFFF] * 5)
        return [values[col % 5] for col in range(k)]
    values = [fbits(rng.uniform(-16.0, 16.0) + row * 0.125) for _ in range(k)]
    if p == "mask":
        for col in range(1, k, 2):
            values[col] = 0xFF800000
    elif p == "nan" and row == 0:
        values[min(2, k-1)] = 0x7FC00001
    elif p == "positive_inf" and row == 0:
        values[min(2, k-1)] = 0x7F800000
    elif p == "all_mask" and row == 0:
        values = [0xFF800000] * k
    return values


def write_vectors(case: Case, path: Path) -> array:
    """Use double-precision math as an independent stable-softmax oracle."""
    expected = array("d")
    rng = random.Random(0x10244096)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for row in range(case.rows):
            raw = row_bits(case, row, rng)
            if case.expect_code:
                # The bench explicitly recovers with a new uniform job.
                expected.extend([1.0 / case.cols] * case.cols)
            else:
                x = [daz_value(b) for b in raw]
                maximum = max(x)
                e = [math.exp(v - maximum) for v in x]
                total = math.fsum(e)
                expected.extend(v / total if v / total >= FP32_MIN_NORMAL else 0.0 for v in e)
            raw.extend([0x7FC00001] * ((-len(raw)) % 4))
            for col in range(0, len(raw), 4):
                word = sum(raw[col + lane] << (32 * lane) for lane in range(4))
                stream.write(f"{word:032x}\n")
    return expected


def command(args: list[str], timeout: float) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}")
    return result.stdout


def compile_tb(name: str, sources: list[Path], timeout: float, params: dict | None = None) -> Path:
    executable = BUILD / (name + ("_" + "_".join(f"{k}{v}" for k, v in params.items()) if params else "") + ".vvp")
    cmd = ["iverilog", "-g2005", "-I", str(RTL), "-s", name, "-o", str(executable)]
    for key, value in (params or {}).items():
        cmd.extend(["-P", f"{name}.{key}={value}"])
    cmd.extend(str(p) for p in sources)
    output = command(cmd, timeout)
    if output.strip():
        print(output.rstrip(), flush=True)
    return executable


def run_units(timeout: float) -> None:
    for name, sources in [
        ("tb_softmax_axil_regs", [RTL / "softmax_axil_regs.v", TB / "tb_softmax_axil_regs.v"]),
        ("tb_softmax_sum4", [RTL / "softmax_sum4.v", TB / "tb_softmax_sum4.v"]),
    ]:
        print(f"RUN {name}", flush=True)
        sim = compile_tb(name, sources, timeout)
        print(command(["vvp", str(sim)], timeout).rstrip(), flush=True)
    for script in ("test_sub4.py", "test_mul4.py", "test_sum_to_fp32.py",
                   "test_recip.py", "test_sum4_random.py", "test_math.py"):
        print(f"RUN {script} reference tests", flush=True)
        print(command([sys.executable, str(TB / script)], timeout).rstrip(), flush=True)


def compare(case: Case, path: Path, expected: array) -> dict:
    words_per_row = (case.cols + 3) // 4
    max_abs = max_rel = max_row_error = 0.0
    row_values = []
    value_index = 0
    line_count = 0
    errors = []
    with path.open(encoding="ascii") as stream:
        for word_index, line in enumerate(stream):
            text = line.strip()
            if not text or text.startswith("//") or text.startswith("@"):
                continue
            try:
                word = int(text, 16)
            except ValueError as exc:
                raise AssertionError(f"{case.name}: unknown output bits {text}") from exc
            line_count += 1
            for lane in range(4):
                column = (word_index % words_per_row) * 4 + lane
                raw = (word >> (32 * lane)) & 0xFFFFFFFF
                if column >= case.cols:
                    if raw:
                        errors.append(f"padding word={word_index} lane={lane}: {raw:08x}")
                    continue
                value = fvalue(raw)
                reference = expected[value_index]
                row_values.append(value)
                absolute = abs(value - reference)
                relative = absolute / reference if reference else 0.0
                max_abs = max(max_abs, absolute)
                max_rel = max(max_rel, relative)
                if (not math.isfinite(value) or value < 0 or
                        absolute > ABS_TOL + REL_TOL * reference or
                        (reference == 0.0 and raw != 0)):
                    if len(errors) < 12:
                        errors.append(f"row={value_index // case.cols} col={column} got={value:.9g} ref={reference:.9g}")
                value_index += 1
                if column == case.cols - 1:
                    row_error = abs(math.fsum(row_values) - 1.0)
                    max_row_error = max(max_row_error, row_error)
                    if row_error > ROW_SUM_TOL and len(errors) < 12:
                        errors.append(f"row={value_index // case.cols - 1} sum error={row_error:.9g}")
                    row_values.clear()
    if line_count != case.rows * words_per_row or value_index != len(expected):
        errors.append(f"output size words={line_count} values={value_index}")
    if errors:
        raise AssertionError(f"{case.name} numerical comparison failed:\n" + "\n".join(errors))
    return {"max_abs_error": max_abs, "max_relative_error": max_rel, "max_row_sum_error": max_row_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", help="include a complete 1024x1024 matrix")
    parser.add_argument("--case", action="append", help="run only the named case(s)")
    parser.add_argument("--skip-unit", action="store_true", help="skip already-verified arithmetic/CSA/AXIL units")
    parser.add_argument("--overlap", type=int, choices=(0, 1), default=1)
    parser.add_argument("--timeout", type=float, default=900, help="timeout per process, in seconds")
    args = parser.parse_args()
    for tool in ("iverilog", "vvp"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Required tool is not on PATH: {tool}")
    BUILD.mkdir(parents=True, exist_ok=True)
    cases = CASES + ([LARGE_CASE] if args.large else [])
    if args.case:
        unknown = set(args.case) - {case.name for case in cases}
        if unknown:
            parser.error(f"unknown case(s): {sorted(unknown)}; use --large for matrix_1024x1024")
        cases = [case for case in cases if case.name in args.case]
    if not args.skip_unit:
        run_units(args.timeout)
    sources = sorted(RTL.glob("*.v")) + [
        ROOT / "GEMM" / "rtl" / "gemm_axi_dma128.v",
        ROOT / "GEMM" / "rtl" / "gemm_axi_duplex128.v",
        TB / "tb_softmax_accelerator.v",
    ]
    sims = {}
    results = []
    for case in cases:
        large = int(case.rows * ((case.cols + 3) // 4) > 4096)
        if large not in sims:
            sims[large] = compile_tb("tb_softmax_accelerator", sources, args.timeout,
                                    {"LARGE": large, "DMA_BURST": 256 if large else 17,
                                     "OVERLAP": args.overlap})
        vector_path = BUILD / f"{case.name}.input.mem"
        output_path = BUILD / f"{case.name}.output.mem"
        expected = write_vectors(case, vector_path)
        plusargs = [
            f"+INPUT={vector_path.as_posix()}", f"+OUTPUT={output_path.as_posix()}",
            f"+ROWS={case.rows}", f"+COLS={case.cols}", f"+FAULT={case.fault}",
            f"+EXPECT_CODE={case.expect_code}", f"+CONFIG={case.config}",
            f"+INJECT_AT={case.inject_at}", f"+RESET_MID={case.reset_mid}",
            f"+DENSE={case.dense}",
            f"+MAX_CYCLES={max(100000, case.rows * ((case.cols + 3) // 4) * 40 + 50000)}",
        ]
        print(f"RUN {case.name}: {case.rows}x{case.cols}, overlap={args.overlap}", flush=True)
        started = time.monotonic()
        log = command(["vvp", str(sims[large]), *plusargs], args.timeout)
        (BUILD / f"{case.name}.log").write_text(log, encoding="utf-8")
        if "PASS AXI matrix" not in log:
            raise AssertionError(f"Missing completion marker:\n{log}")
        metrics = compare(case, output_path, expected)
        cycle_match = re.search(r"cycles=(\d+)", log)
        result = {"case": case.name, "rows": case.rows, "cols": case.cols,
                  "overlap": args.overlap, "cycles": int(cycle_match.group(1)) if cycle_match else None,
                  "seconds": time.monotonic() - started, **metrics}
        results.append(result)
        print(log.rstrip(), flush=True)
        print(f"  numerical PASS: max_abs={metrics['max_abs_error']:.3g}, "
              f"row_sum={metrics['max_row_sum_error']:.3g}, {result['seconds']:.2f}s", flush=True)
    summary = BUILD / f"results_overlap{args.overlap}.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(results)} end-to-end cases; report: {summary}", flush=True)


if __name__ == "__main__":
    main()
