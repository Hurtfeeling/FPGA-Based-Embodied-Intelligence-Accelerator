#!/usr/bin/env python
"""Run the strict INT8 ONNX with the integer CPU reference interpreter."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from strict_int8_runtime import IntegerSession


def encode_external(array):
    """Host-side input conversion; this is outside integer model inference."""
    values = np.asarray(array, np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Inputs must be finite")
    maximum = float(np.max(np.abs(values), initial=0))
    exponent = int(np.ceil(np.log2(maximum / 127))) if maximum else 0
    codes = np.clip(np.rint(values / np.exp2(exponent)), -127, 127).astype(np.int8)
    return codes, np.asarray(exponent, np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Recorded six-input NPZ, or integer codes plus exponents")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true", help="Also decode normalized floating actions outside the integer runtime")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    session = IntegerSession(args.model)
    feed = {}
    with np.load(args.inputs, allow_pickle=False) as arrays:
        for value in session.model.graph.input:
            name = value.name
            if name in feed:
                continue
            if name.endswith("__exponent"):
                feed[name] = np.asarray(arrays[name], np.int32)
            elif name + "__exponent" in {v.name for v in session.model.graph.input}:
                if name + "__exponent" in arrays:
                    feed[name] = arrays[name]
                    feed[name + "__exponent"] = arrays[name + "__exponent"]
                else:
                    feed[name], feed[name + "__exponent"] = encode_external(arrays[name])
            else:
                feed[name] = arrays[name]
    started = time.monotonic()
    result = session.run(feed, progress=lambda done, total: print(f"{done}/{total}", flush=True))
    name = session.model.graph.output[0].name
    if args.decode:
        result["normalized_actions"] = result[name].astype(np.float64) * np.exp2(result[name + "__exponent"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    print(json.dumps({"file": str(args.output), "seconds": time.monotonic() - started,
                      "action_shape": list(result[name].shape), "action_dtype": str(result[name].dtype)}, indent=2))


if __name__ == "__main__":
    main()
