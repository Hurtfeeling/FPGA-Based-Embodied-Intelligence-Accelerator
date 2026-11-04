#!/usr/bin/env python
"""Run W8A8+FP8 ONNX with the FP32-internal FP8-boundary reference executor."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx

from w8a8_fp8_runtime import FP8_DTYPE, FP8Session, Float8, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Original six-input NPZ or FP8 bytes plus exponents")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true", help="Include decoded FP32 normalized actions in the output NPZ")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    session = FP8Session(args.model)
    feed = {}
    with np.load(args.inputs, allow_pickle=False) as arrays:
        for value in session.model.graph.input:
            name = value.name
            if name.endswith("__fp8_exponent"):
                continue
            if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT8E4M3FN:
                if name + "__fp8_exponent" in arrays:
                    if arrays[name].dtype != np.uint8:
                        raise TypeError(f"Encoded FP8 input {name} must contain UINT8 bit patterns")
                    feed[name] = arrays[name].view(FP8_DTYPE)
                    feed[name + "__fp8_exponent"] = arrays[name + "__fp8_exponent"]
                else:
                    value = encode(arrays[name])
                    feed[name], feed[name + "__fp8_exponent"] = value.data, value.exponent
            else:
                feed[name] = arrays[name]
    started = time.monotonic()
    result = session.run(feed, progress=lambda done, total: print(f"{done}/{total}", flush=True))
    name = session.model.graph.output[0].name
    output = {"actions_fp8_bits": result[name].view(np.uint8), "actions_exponent": result[name + "__fp8_exponent"]}
    if args.decode:
        output["decoded_actions"] = Float8(result[name], result[name + "__fp8_exponent"]).decode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **output)
    print(json.dumps({"output": str(args.output), "shape": list(result[name].shape),
                      "action_dtype": str(result[name].dtype), "seconds": time.monotonic() - started,
                      "native_fp8_execution": False}, indent=2))


if __name__ == "__main__":
    main()
