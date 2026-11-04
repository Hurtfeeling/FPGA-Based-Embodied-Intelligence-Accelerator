"""Generate the paired 1024-segment Q1.30 exp2 ROM (standard library only)."""
from pathlib import Path
import math


def main():
    target = Path(__file__).resolve().parents[1] / "rtl" / "exp_lut.mem"
    endpoints = [round(2.0 ** (-i / 1024.0) * (1 << 30)) for i in range(1025)]
    words = [(endpoints[i] << 32) | (endpoints[i] - endpoints[i + 1]) for i in range(1024)]
    target.write_text("".join(f"{word:016x}\n" for word in words), encoding="ascii")
    print(f"Wrote {target}: 1024 x 64-bit paired base/drop values")


if __name__ == "__main__":
    main()
