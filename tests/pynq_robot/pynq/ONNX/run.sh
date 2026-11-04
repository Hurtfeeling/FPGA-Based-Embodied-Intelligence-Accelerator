#!/bin/bash
# Usage: sudo bash run.sh cpu MODEL [infer_board options]
#        sudo bash run.sh hybrid MODEL CPU_BASELINE.npy [infer_board options]
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 2 ]]; then
  echo 'Usage: run.sh cpu MODEL [options] | run.sh hybrid MODEL BASELINE.npy [options]' >&2
  exit 2
fi
MODE="$1"; MODEL="$2"; shift 2
EXTRA=()
case "$MODE" in
  cpu) ;;
  hybrid)
    [[ $# -ge 1 ]] || { echo 'hybrid requires a CPU baseline NPY' >&2; exit 2; }
    BASELINE="$1"; shift
    MANIFEST="${MODEL%.onnx}.manifest.json"
    [[ -f "$MANIFEST" ]] || { echo "Missing matching model manifest: $MANIFEST" >&2; exit 2; }
    EXTRA=(--runtime-root "$BASE/../GEMM"
           --runtime-config "$BASE/runtime_cma80_native_softmax.json"
           --cpu-baseline "$BASELINE")
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
export LD_LIBRARY_PATH="$BASE/onnxruntime-linux-aarch64-1.16.3/lib:${LD_LIBRARY_PATH:-}"
export SMOL_ORT_EXECUTION_MODE=sequential SMOL_ORT_INTRA_OP_THREADS=4 SMOL_ORT_INTER_OP_THREADS=1
export SMOL_ORT_GRAPH_OPT=disable SMOL_ORT_CPU_ARENA=0 SMOL_ORT_MEMORY_PATTERN=0
export SMOL_ORT_ALLOW_SPINNING=1 OMP_WAIT_POLICY=PASSIVE
python3 "$BASE/../GEMM/set_cpu_highest.py"
PREFIX="$BASE/results/${MODE}_$(date +%Y%m%d_%H%M%S)_$$"
echo "Result prefix: $PREFIX"
exec taskset -c 0,1,2,3 python3 -u "$BASE/infer_board.py" \
  --mode "$MODE" --model "$MODEL" --inputs "$BASE/validation_0.npz" \
  --library "$BASE/libsmolvla_ort_sweep.so" --output-prefix "$PREFIX" \
  --warmups 1 --repeats 3 "${EXTRA[@]}" "$@"
