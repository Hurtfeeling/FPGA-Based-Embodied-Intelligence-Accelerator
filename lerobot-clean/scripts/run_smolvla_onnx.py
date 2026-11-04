#!/usr/bin/env python
"""Run a recorded SmolVLA ONNX example without PyTorch or robot hardware."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new file")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(args.inputs, allow_pickle=False) as arrays:
        feed = {value.name: arrays[value.name] for value in session.get_inputs()}
        actions = session.run(["actions"], feed)[0]
        report = {"shape": list(actions.shape), "all_finite": bool(np.isfinite(actions).all())}
        if "reference_actions" in arrays:
            error = np.abs(actions - arrays["reference_actions"])
            report["mae_vs_w8a8_reference"] = float(error.mean())
            report["max_abs_error_vs_w8a8_reference"] = float(error.max())
    if args.output is not None:
        with args.output.open("wb") as stream:
            np.save(stream, actions, allow_pickle=False)
        report["output"] = str(args.output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
