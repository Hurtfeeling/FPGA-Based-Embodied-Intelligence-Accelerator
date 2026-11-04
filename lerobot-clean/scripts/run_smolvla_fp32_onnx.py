#!/usr/bin/env python
"""Run standard SmolVLA FP32 ONNX directly with ONNX Runtime."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(args.inputs, allow_pickle=False) as arrays:
        feed = {v.name: arrays[v.name] for v in session.get_inputs()}
        started = time.monotonic()
        actions = session.run(["actions"], feed)[0]
        report = {"shape": list(actions.shape), "dtype": str(actions.dtype),
                  "all_finite": bool(np.isfinite(actions).all()), "seconds": time.monotonic() - started}
        key = "reference_actions" if "reference_actions" in arrays else "pytorch_fp32_actions"
        if key in arrays:
            error = np.abs(actions - arrays[key])
            report.update({"mae_vs_pytorch_fp32": float(error.mean()), "max_error_vs_pytorch_fp32": float(error.max())})
    if actions.dtype != np.float32 or not np.isfinite(actions).all():
        raise ValueError("Expected finite FP32 actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, actions=actions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
