#!/bin/bash
# Usage: bash build_runtime.sh /path/to/onnxruntime-linux-aarch64-1.16.3 [output.so]
set -eu
cd "$(dirname "$0")"
ORT_DIR="${1:?Pass the ORT directory containing include/ and lib/}"
ORT_DIR="$(cd "$ORT_DIR" && pwd)"
OUTPUT_LIB="${2:-libsmolvla_ort_cpu.so}"
test -f "$ORT_DIR/include/onnxruntime_cxx_api.h"
test -f "$ORT_DIR/lib/libonnxruntime.so"
g++ -std=c++14 -O2 -fPIC -shared ort_cpu_bridge.cpp \
    -I"$ORT_DIR/include" -L"$ORT_DIR/lib" -lonnxruntime \
    -Wl,-rpath,"$ORT_DIR/lib" -o "$OUTPUT_LIB"
/usr/bin/python3 -m py_compile infer_board.py
printf 'Built %s with ORT from %s\n' "$OUTPUT_LIB" "$ORT_DIR"
