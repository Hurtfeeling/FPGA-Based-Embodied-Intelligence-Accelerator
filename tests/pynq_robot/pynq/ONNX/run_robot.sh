#!/bin/bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$BASE/onnxruntime-linux-aarch64-1.16.3/lib:${LD_LIBRARY_PATH:-}"
export OMP_WAIT_POLICY=PASSIVE
python3 "$BASE/../GEMM/set_cpu_highest.py"
exec taskset -c 0,1,2,3 python3 -u "$BASE/run_robot.py" "$@"
