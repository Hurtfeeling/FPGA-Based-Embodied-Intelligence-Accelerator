#!/usr/bin/env python
"""Export a compact, custom-domain ONNX with INT8 weights and activations.

This is a custom integer model, not an ONNX Runtime QDQ model. Execute it with
strict_int8_runtime.IntegerSession, or implement its custom-domain kernels.
"""

import argparse
import collections
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from scripts.strict_int8_runtime import DOMAIN, FRACTION, UNIT, IntegerSession, Q8

FLOAT_TYPES = {T.FLOAT, T.DOUBLE, T.FLOAT16, T.BFLOAT16}
SUPPORTED = {
    "FromInteger", "ToInteger", "MatMul", "MatMulWeights", "Add", "Sub", "Mul", "Div",
    "Abs", "Neg", "Equal", "Greater", "Less", "LessOrEqual", "Round", "Clip", "Reciprocal",
    "Sqrt", "Pow", "Sin", "Cos", "Sigmoid", "Tanh", "Softmax", "LayerNormalization", "Conv",
    "Resize", "Where", "Concat", "ScatterND", "Pad", "Range", "ConstantOfShape", "ReduceMean",
    "ReduceSum", "ReduceMax", "ReduceMin", "Gather", "Reshape", "Transpose", "Unsqueeze",
    "Squeeze", "Expand", "Slice", "Split", "Identity",
    "RMSNormalization", "RotarySin", "RotaryCos",
}


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def is_float(array):
    return array.dtype.kind == "f" or str(array.dtype) == "bfloat16"


def encode(array, per_channel=False):
    """Offline quantization; floating point is used only during conversion."""
    values = np.asarray(array, np.float64)
    if np.isnan(values).any() or np.any(values > 1e30):
        raise ValueError("NaN and positive infinity are not supported")
    mask = values < -1e30
    values = np.where(mask, 0, values)
    axes = tuple(range(1, values.ndim)) if per_channel and values.ndim > 1 else None
    maximum = np.max(np.abs(values), axis=axes, keepdims=axes is not None, initial=0)
    safe = np.maximum(maximum, np.finfo(np.float64).tiny)
    exponent = np.where(maximum > 0, np.ceil(np.log2(safe / 127)), 0).astype(np.int32)
    codes = np.clip(np.rint(values / np.exp2(exponent)), -127, 127).astype(np.int8)
    codes = np.where(mask, np.int8(-128), codes)
    return Q8(codes, exponent)


def dead_code(nodes, outputs):
    needed = set(outputs)
    live = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            live.append(node)
            needed.update(n for n in node.input if n)
    return list(reversed(live)), needed


def canonicalize(model):
    """Recover logical products and discard old floating quantizer machinery."""
    model = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type
             for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    types.update({t.name: t.data_type for t in model.graph.initializer})
    producers = {o: n for n in model.graph.node for o in n.output}
    users = collections.defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            users[name].append(node)

    def next_node(name, op):
        matches = [n for n in users[name] if n.op_type == op]
        if len(matches) != 1:
            raise ValueError(f"Expected one {op} after {name}, got {len(matches)}")
        return matches[0]

    def operand(name):
        node = producers.get(name)
        if node is None or node.op_type != "Cast":
            return None
        node = producers[node.input[0]]
        if node.op_type == "Add":
            node = producers[node.input[0]]
        for op in ("Clip", "Round", "Div"):
            if node.op_type != op:
                raise ValueError(f"Unrecognized dynamic quantizer: {name}, {node.op_type}")
            result = node.input[0]
            if op != "Div":
                node = producers[result]
        return result

    replacements = {}
    products = 0
    for node in model.graph.node:
        if node.op_type != "MatMulInteger":
            continue
        cast = next_node(node.output[0], "Cast")
        first = next_node(cast.output[0], "Mul")
        second = next_node(first.output[0], "Mul")
        left = operand(node.input[0])
        right = operand(node.input[1])
        if left is None:
            raise ValueError("Could not recover the left matrix operand")
        if right is None:
            scale = next(name for name in second.input if name != first.output[0])
            new = helper.make_node("MatMulWeights", [left, node.input[1], scale], list(second.output))
        else:
            new = helper.make_node("MatMul", [left, right], list(second.output))
        replacements[second.output[0]] = new
        products += 1
    aliases = {}
    nodes = []
    for original in model.graph.node:
        node = copy.deepcopy(replacements.get(original.output[0], original))
        node.input[:] = [aliases.get(n, n) for n in node.input]
        if (node.op_type == "Cast" and attrs(node)["to"] in FLOAT_TYPES
                and types.get(original.input[0]) in FLOAT_TYPES):
            aliases[node.output[0]] = node.input[0]
            continue
        nodes.append(node)
    # Resolve aliases again: replacement products can refer to operands whose
    # final floating Cast was defined earlier in the source graph.
    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name
    for node in nodes:
        node.input[:] = [resolve(n) for n in node.input]
    outputs = [resolve(v.name) for v in model.graph.output]
    nodes, _ = dead_code(nodes, outputs)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in [*model.graph.input, *model.graph.value_info]
              if v.type.tensor_type.HasField("shape")
              and all(d.HasField("dim_value") for d in v.type.tensor_type.shape.dim)}
    live = []
    for node in nodes:
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(attrs(node)["value"])
            continue
        if node.op_type == "Shape" and node.input[0] in shapes:
            a = attrs(node)
            constants[node.output[0]] = np.asarray(shapes[node.input[0]][a.get("start", 0):a.get("end")], np.int64)
            continue
        if (all(n in constants for n in node.input) and node.op_type not in ("MatMul", "MatMulWeights")
                and sum(constants[n].size for n in node.input) < 2_000_000
                and not (node.op_type in ("Transpose", "Identity")
                         and constants[node.input[0]].size > 4096)):
            try:
                with np.errstate(all="ignore"):
                    evaluator = ReferenceEvaluator(node, opsets={"": 18})
                    result = evaluator.run(None, {n: constants[n] for n in node.input})
            except (NotImplementedError, ValueError, TypeError, RuntimeError):
                pass
            else:
                for name, value in zip(node.output, result, strict=True):
                    constants[name] = value
                continue
        live.append(node)
    # Keep normalization and rotary phases inside single integer kernels.
    # Quantizing their temporary inverse standard deviations or 1..10000
    # timescales as ordinary INT8 activations would introduce zero divisors.
    producers = {o: n for n in live for o in n.output}
    fused = {}
    rms_count, rope_count = 0, 0
    for node in live:
        if node.op_type != "Mul":
            continue
        for weight, hidden in (node.input, list(reversed(node.input))):
            if weight not in constants:
                continue
            normalized = producers.get(hidden)
            if normalized is None or normalized.op_type != "Mul":
                continue
            for x, reciprocal in (normalized.input, list(reversed(normalized.input))):
                division = producers.get(reciprocal)
                if division is None or division.op_type != "Div":
                    continue
                root = producers.get(division.input[1])
                if root is None or root.op_type != "Sqrt":
                    continue
                addition = producers.get(root.input[0])
                if addition is None or addition.op_type != "Add":
                    continue
                for mean_name, epsilon_name in (addition.input, list(reversed(addition.input))):
                    mean = producers.get(mean_name)
                    epsilon = constants.get(epsilon_name)
                    if mean is None or mean.op_type != "ReduceMean" or epsilon is None or epsilon.size != 1:
                        continue
                    square = producers.get(mean.input[0])
                    if square is None or square.op_type != "Pow" or square.input[0] != x:
                        continue
                    power = constants.get(square.input[1])
                    numerator = constants.get(division.input[0])
                    axes = constants.get(mean.input[1]) if len(mean.input) > 1 else np.array(attrs(mean).get("axes"))
                    if (power is None or power.size != 1 or float(power.item()) != 2 or numerator is None
                            or numerator.size != 1 or float(numerator.item()) != 1 or axes.tolist() != [-1]):
                        continue
                    fused[node.output[0]] = helper.make_node("RMSNormalization", [x, weight], list(node.output),
                                                            epsilon=float(epsilon.item()))
                    rms_count += 1
    def raw_position(name):
        for _ in range(8):
            parent = producers.get(name)
            if parent is None:
                return name if types.get(name) not in FLOAT_TYPES else None
            if parent.op_type == "Cast" and types.get(parent.input[0]) not in FLOAT_TYPES:
                return parent.input[0]
            if parent.op_type not in ("Cast", "Unsqueeze", "Identity"):
                return name if types.get(name) not in FLOAT_TYPES else None
            name = parent.input[0]
        return None
    def strip_dimensions(name):
        for _ in range(8):
            parent = producers.get(name)
            if parent is None or parent.op_type not in ("Cast", "Unsqueeze", "Identity"):
                break
            name = parent.input[0]
        return name
    for node in live:
        if node.op_type not in ("Sin", "Cos"):
            continue
        division = producers.get(strip_dimensions(node.input[0]))
        if division is None or division.op_type != "Div":
            continue
        positions = raw_position(division.input[0])
        if positions is None:
            continue
        denominator = strip_dimensions(division.input[1])
        constant = constants.get(denominator)
        half = None
        if constant is not None and constant.size > 1:
            half = "__rope_half_" + str(constant.size)
            constants[half] = np.asarray(constant.size, np.int64)
            types[half] = T.INT64
        else:
            power = producers.get(denominator)
            if power is None or power.op_type != "Pow":
                continue
            base = constants.get(power.input[0])
            product = producers.get(power.input[1])
            if base is None or base.size != 1 or float(base.item()) != 10000 or product is None or product.op_type != "Mul":
                continue
            for name in product.input:
                interval = producers.get(name)
                if interval is not None and interval.op_type == "Range":
                    half = raw_position(interval.input[1]) or interval.input[1]
        if half is not None:
            fused[node.output[0]] = helper.make_node("Rotary" + node.op_type, [positions, half], list(node.output),
                                                    period_q20=int(round(2 * math.pi * UNIT)))
            rope_count += 1
    live = [fused.get(n.output[0], n) for n in live]
    live, needed = dead_code(live, outputs)
    constants = {k: v for k, v in constants.items() if k in needed}
    print("Recovered", products, "INT8 products;", len(live), "logical runtime nodes", flush=True)
    print("Fused integer kernels:", rms_count, "RMS normalizations;", rope_count, "rotary trigonometric nodes", flush=True)
    print("Operations:", dict(collections.Counter(n.op_type for n in live)), flush=True)
    return live, constants, types, outputs


def audit(model):
    onnx.checker.check_model(model)
    allowed = {T.INT8, T.UINT8, T.INT16, T.UINT16, T.INT32, T.UINT32, T.INT64, T.UINT64, T.BOOL}
    for tensor in model.graph.initializer:
        if tensor.data_type not in allowed:
            raise ValueError(f"Non-integer initializer: {tensor.name}")
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.type.tensor_type.elem_type not in allowed:
            raise ValueError(f"Non-integer value: {value.name}")
    for node in model.graph.node:
        if node.domain == DOMAIN:
            if node.op_type.removeprefix("I8") not in SUPPORTED:
                raise ValueError(f"Missing runtime kernel: {node.op_type}")
        elif node.op_type == "Cast" and attrs(node)["to"] not in allowed:
            raise ValueError("Floating Cast remains")
        for a in node.attribute:
            if a.type in (onnx.AttributeProto.FLOAT, onnx.AttributeProto.FLOATS):
                raise ValueError(f"Floating attribute: {node.name}/{a.name}")
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type not in allowed:
                raise ValueError(f"Floating tensor attribute: {node.name}/{a.name}")


def export(model):
    nodes, constants, types, source_outputs = canonicalize(model)
    initializers, graph_nodes, value_info = [], [], []
    quantized = set()
    scale32 = {n.input[2] for n in nodes if n.op_type == "MatMulWeights"}
    parameter_records = []
    original_parameters = {t.name for t in model.graph.initializer if t.data_type in FLOAT_TYPES or t.data_type == T.INT8}

    def info(name, kind, shape=None):
        value_info.append(helper.make_tensor_value_info(name, kind, shape))

    for name, array in constants.items():
        if name in scale32:
            values = np.asarray(array, np.float64)
            maximum = float(np.max(np.abs(values)))
            exponent = int(math.ceil(math.log2(maximum / ((1 << 30) - 1)))) if maximum else 0
            mantissa = np.rint(values / (2.0 ** exponent)).astype(np.int32)
            initializers.extend([numpy_helper.from_array(mantissa, name),
                                 numpy_helper.from_array(np.array(exponent, np.int32), name + "__exponent")])
            info(name, T.INT32, list(array.shape))
            info(name + "__exponent", T.INT32, [])
        elif is_float(array):
            value = encode(array, per_channel=array.ndim > 1)
            initializers.extend([numpy_helper.from_array(value.codes, name),
                                 numpy_helper.from_array(value.exponent, name + "__exponent")])
            quantized.add(name)
            info(name, T.INT8, list(array.shape))
            info(name + "__exponent", T.INT32, list(value.exponent.shape))
        else:
            initializers.append(numpy_helper.from_array(array, name))
        if name in original_parameters and name not in scale32:
            parameter_records.append({"name": name, "elements": int(array.size), "storage": "INT8",
                                      "bytes": int(array.size)})
    # Shared nonlinear tables are constants, not widened copies of parameters.
    angles = np.arange(16384) * (2 * np.pi / 16384)
    grid = np.arange(8193) / 256 - 16
    tables = {
        "__lut_sin": np.rint(np.sin(angles) * 64).astype(np.int8),
        "__lut_cos": np.rint(np.cos(angles) * 64).astype(np.int8),
        "__lut_tanh": np.rint(np.tanh(grid) * 64).astype(np.int8),
        "__lut_sigmoid": np.clip(np.rint(128 / (1 + np.exp(-grid))), 0, 127).astype(np.int8),
        "__lut_exp": np.rint(np.exp(np.arange(4097) / 256 - 16) * (1 << 24)).astype(np.int32),
    }
    rope_frequencies = np.zeros((257, 256), np.int32)
    for half in range(1, 257):
        rope_frequencies[half, :half] = np.rint(np.power(10000, -np.arange(half) / half) * (1 << 30)).astype(np.int32)
    tables["__lut_rope_frequencies"] = rope_frequencies
    graph_inputs = []
    for source in model.graph.input:
        value = copy.deepcopy(source)
        if value.type.tensor_type.elem_type in FLOAT_TYPES:
            quantized.add(value.name)
            value.type.tensor_type.elem_type = T.INT8
            graph_inputs.extend([value, helper.make_tensor_value_info(value.name + "__exponent", T.INT32, [])])
        else:
            graph_inputs.append(value)
    for index, original in enumerate(nodes):
        node = copy.deepcopy(original)
        a = attrs(node)
        if node.op_type == "Shape":
            graph_nodes.append(node)
            continue
        input_kinds = [2 if n in scale32 else int(n in quantized) for n in node.input]
        output_kinds = [int(types.get(n) in FLOAT_TYPES) for n in node.output]
        if node.op_type in ("MatMul", "MatMulWeights"):
            output_kinds = [1]
        if not any(input_kinds) and not any(output_kinds):
            graph_nodes.append(node)
            continue
        op = node.op_type
        if op == "Cast":
            op = "FromInteger" if a["to"] in FLOAT_TYPES else "ToInteger"
            if op == "FromInteger":
                a.pop("to")
        if op not in SUPPORTED:
            raise ValueError(f"No strictly integer kernel for {op}: {node.name}")
        if op == "Pow":
            base = constants.get(node.input[0])
            exponent = constants.get(node.input[1])
            if base is not None and base.size == 1 and (exponent is None or exponent.size != 1):
                key = "__lut_power_" + hashlib.sha256(np.asarray(base, np.float64).tobytes()).hexdigest()[:12]
                values = np.power(float(base.item()), np.arange(16385) / 4096 - 2)
                tables[key] = np.rint(np.clip(values, 0, (1 << 31) - 1) * UNIT).astype(np.int64)
                a["power_table"] = key
        if op in ("Sin", "Cos"):
            a["period_q20"] = int(round(2 * math.pi * UNIT))
        if op == "ConstantOfShape":
            fill = numpy_helper.to_array(a.pop("value")) if "value" in a else np.array([0], np.float32)
            value = encode(fill)
            a["fill_code"], a["fill_exponent"] = int(value.codes.item()), int(value.exponent.item())
        clean = {}
        for key, value in a.items():
            if isinstance(value, float):
                clean[key + "_q20"] = int(round(value * UNIT))
            else:
                clean[key] = value
        inputs, outputs = [], []
        for name, kind in zip(node.input, input_kinds, strict=True):
            inputs.append(name)
            if kind in (1, 2):
                inputs.append(name + "__exponent")
        for name, kind in zip(node.output, output_kinds, strict=True):
            outputs.append(name)
            if kind:
                quantized.add(name)
                outputs.append(name + "__exponent")
                info(name, T.INT8)
                info(name + "__exponent", T.INT32, [])
            else:
                info(name, types[name])
        table_names = []
        if op in ("Sin", "Cos", "Sigmoid", "Tanh"):
            table_names = ["__lut_" + op.lower()]
        elif op in ("RotarySin", "RotaryCos"):
            table_names = ["__lut_rope_frequencies", "__lut_" + op.removeprefix("Rotary").lower()]
        elif op == "Softmax":
            table_names = ["__lut_exp"]
        elif "power_table" in clean:
            table_names = [clean["power_table"]]
        if table_names:
            inputs.extend(table_names)
            clean["table_names"] = table_names
        graph_nodes.append(helper.make_node("I8" + op, inputs, outputs, domain=DOMAIN,
                                            name=f"int8_{index}_{op}", input_kinds=input_kinds,
                                            output_kinds=output_kinds, **clean))
    for name, array in tables.items():
        initializers.append(numpy_helper.from_array(array, name))
    # Unrolled Euler steps reuse the same learned tensors. Never serialize a
    # fresh copy of a constant-folded weight for every use of that weight.
    signatures, aliases, distinct = {}, {}, []
    for tensor in initializers:
        if tensor.name.startswith("__lut_"):
            distinct.append(tensor)
            continue
        key = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(tensor.raw_data).digest())
        if key in signatures:
            aliases[tensor.name] = signatures[key]
        else:
            signatures[key] = tensor.name
            distinct.append(tensor)
    for node in graph_nodes:
        node.input[:] = [aliases.get(name, name) for name in node.input]
    value_info = [v for v in value_info if v.name not in aliases]
    initializers = distinct
    outputs = []
    for source, name in zip(model.graph.output, source_outputs, strict=True):
        if name not in quantized:
            raise ValueError("Output did not become INT8")
        shape = [d.dim_value for d in source.type.tensor_type.shape.dim]
        outputs.extend([helper.make_tensor_value_info(name, T.INT8, shape),
                        helper.make_tensor_value_info(name + "__exponent", T.INT32, [])])
    graph = helper.make_graph(graph_nodes, "SmolVLA_strict_INT8", graph_inputs, outputs,
                              initializer=initializers, value_info=value_info)
    result = helper.make_model(graph, producer_name="lerobot-strict-int8-v1", ir_version=9,
                               opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(DOMAIN, 1)])
    helper.set_model_props(result, {
        "format": "strict-int8-custom-v1", "weights": "INT8", "activations": "INT8",
        "scaling": "INT32 exponents; per-channel GEMM scales use INT32 mantissa+exponent; -128=negative-infinity",
        "runtime": "scripts/strict_int8_runtime.py; default ONNX Runtime is unsupported",
        "accumulation": "INT32 dot products; INT64 integer scratch for reductions/rescaling",
        "nonlinear": "integer lookup tables and integer square root; no float inference",
    })
    audit(result)
    # Compute learned parameter storage from actual source tensors, including
    # constant-folded aliases; every surviving floating parameter is INT8.
    weight_elements = sum(math.prod(t.dims) for t in initializers if t.data_type == T.INT8 and not t.name.startswith("__lut_"))
    return result, {"int8_parameter_and_constant_elements": weight_elements,
                    "int8_parameter_and_constant_bytes": weight_elements,
                    "source_parameter_records": parameter_records,
                    "output_name": source_outputs[0]}


def encode_feed(example, model):
    feed = {}
    for source in model.graph.input:
        if source.name.endswith("__exponent"):
            continue
        array = example[source.name]
        if source.type.tensor_type.elem_type == T.INT8:
            value = encode(array)
            feed[source.name], feed[source.name + "__exponent"] = value.codes, value.exponent
        else:
            feed[source.name] = array
    return feed


def validate(path, examples, output, report):
    session = IntegerSession(path)
    validation = []
    for index, example in enumerate(examples):
        feed = encode_feed(example, session.model)
        started = time.monotonic()
        def progress(done, total):
            print(f"Sample {index}: {done}/{total} nodes", flush=True)
        result = session.run(feed, progress=progress, checkpoint=output / f".validation_checkpoint_{index}.npz")
        name = report["output_name"]
        codes, exponent = result[name], result[name + "__exponent"]
        # Decode only outside the integer runtime, for an offline comparison.
        actions = codes.astype(np.float64) * np.exp2(exponent)
        error = np.abs(actions - example["onnx_actions"])
        record = {"sample": index, "seconds": time.monotonic() - started,
                  "mae_vs_w8a8": float(error.mean()), "max_error_vs_w8a8": float(error.max()),
                  "output_dtype": str(codes.dtype), "output_exponent": int(exponent),
                  "unique_action_codes": int(np.unique(codes).size), "float_runtime_outputs": 0}
        validation.append(record)
        np.savez(output / f"validation_int8_{index}.npz", **feed, actions_int8=codes,
                 actions_exponent=exponent, decoded_actions=actions, reference_actions=example["onnx_actions"])
        report["validation"] = validation
        (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Validation:", record, flush=True)
    return validation


def finish_package(output, report):
    path = output / "smolvla_strict_int8.onnx"
    model = onnx.load(path)
    audit(model)
    by_type = collections.defaultdict(lambda: {"tensors": 0, "elements": 0, "bytes": 0})
    for tensor in model.graph.initializer:
        record = by_type[T.DataType.Name(tensor.data_type)]
        record["tensors"] += 1
        record["elements"] += math.prod(tensor.dims)
        record["bytes"] += len(tensor.raw_data)
    report["initializer_storage"] = dict(by_type)
    report["file"] = str(path.resolve())
    report["bytes"] = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report["sha256"] = digest.hexdigest()
    report["accuracy_accepted_for_deployment"] = False
    report["accuracy_note"] = "Recorded action errors are large. Runtime completion is not task success; integer-specific QAT/calibration is still needed."
    protocol = {
        "domain": DOMAIN, "version": 1,
        "activation_encoding": "Signed INT8 code times 2**INT32_exponent. Code -128 is reserved for attention negative infinity.",
        "learned_weight_encoding": "INT8 codes. Original W8A8 matrix scales are INT32 mantissa plus INT32 exponent.",
        "input_kinds": {"0": "one ordinary integer/index/bool tensor", "1": "INT8 codes followed by INT32 exponent", "2": "INT32 scale mantissa followed by INT32 exponent"},
        "output_kinds": {"0": "ordinary integer/index/bool tensor", "1": "INT8 activation followed by INT32 exponent"},
        "table_names": "Lookup-table tensors are explicit trailing inputs, in the order of the table_names string-list attribute.",
        "kernels": sorted({n.op_type for n in model.graph.node if n.domain == DOMAIN}),
        "model_inputs": [{"name": v.name, "dtype": T.DataType.Name(v.type.tensor_type.elem_type),
                          "shape": [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]}
                         for v in model.graph.input],
        "model_outputs": [{"name": v.name, "dtype": T.DataType.Name(v.type.tensor_type.elem_type)} for v in model.graph.output],
        "floating_computation_in_model": False,
        "host_boundary": "Encoding float input examples and optionally decoding output codes are external to integer model inference.",
        "runtime": "strict_int8_runtime.py. Optional PyTorch CPU _int_mm accelerates INT8 dot products with exact INT32 results; NumPy integer fallback is slower.",
        "onnxruntime": "This package does not include an ONNX Runtime custom-op .so. Default ONNX Runtime cannot load this domain.",
    }
    (output / "integer_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    errors = [v["mae_vs_w8a8"] for v in report["validation"]]
    (output / "README.md").write_text(
        "# SmolVLA 严格 INT8 自定义 ONNX\n\n"
        f"模型文件：`smolvla_strict_int8.onnx`，{report['bytes']:,} 字节。"
        "权重、层间激活和动作输出均为 INT8；累加为 INT32，定点临时计算可用 INT64；"
        "量化参数及索引使用必要的整数类型。图内没有浮点张量或浮点属性。\n\n"
        "这是 `org.lerobot.strictint8` 自定义算子模型，使用附带的 `strict_int8_runtime.py`。"
        "不含 ONNX Runtime 的 C++ 自定义算子动态库；不能直接用默认 ONNX Runtime 加载。"
        "未进行 GPU 验证。协议、算子清单和输入输出类型见 `integer_protocol.json`。\n\n"
        f"三组完整输入已通过整数运行器验证，归一化动作 MAE 为 {min(errors):.4f}–{max(errors):.4f}。"
        "误差较大；目前不应把它当作已通过任务精度验证的机器人策略。"
        "运行通过不代表任务成功，需要进一步进行适配此整数结构的校准/QAT。\n\n"
        "图内保留原模型的两路图像处理、上下文编码和十步 Euler 更新。"
        "状态输入仍需使用原策略的外部归一化，语言输入为外部 tokenizer 的整数结果。"
        "输入/输出编码为 `code * 2**exponent`；浮点样例的输入编码和可选动作解码在模型之外完成。\n\n"
        "现有本地环境运行示例（输出路径必须不存在）：\n\n```bash\n"
        "/home/ial-mohd/workspace/so101-lab/venvs/tools/bin/uv run --no-project "
        "--python /home/ial-mohd/workspace/so101-lab/venvs/isaaclab/bin/python python "
        f"{output / 'run_smolvla_strict_int8.py'} "
        f"--model {path} --inputs {output / 'validation_int8_0.npz'} "
        f"--output {output / 'actions_check.npz'} --decode\n```\n"
    )
    (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-frames", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validation_frames < 1:
        parser.error("--validation-frames must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.validate_only:
        parser.error("Output is nonempty; choose a new directory")
    source, output = args.source.resolve(), args.output.resolve()
    samples = sorted(source.parent.glob("validation_*.npz"))[:args.validation_frames]
    if not samples:
        parser.error("Source needs recorded validation_*.npz inputs")
    examples = []
    for path in samples:
        with np.load(path, allow_pickle=False) as arrays:
            examples.append({k: arrays[k] for k in arrays.files})
    output.mkdir(parents=True, exist_ok=True)
    path = output / "smolvla_strict_int8.onnx"
    if not args.validate_only:
        model = onnx.load(source)
        converted, report = export(model)
        del model
        onnx.save(converted, path)
        report.update({"source": str(source), "file": str(path), "bytes": path.stat().st_size,
                       "strict_int8_audit_passed": True, "all_learned_storage_int8": True,
                       "all_logical_activations_int8": True, "float_tensors": 0, "float_attributes": 0,
                       "accumulator_dtype": "INT32", "scratch_dtype": "INT64", "scale_dtype": "INT32",
                       "operator_counts": dict(collections.Counter((n.domain + ":" + n.op_type) for n in converted.graph.node)),
                       "runtime_validation_passed": False, "validation_status": "pending",
                       "onnxruntime_directly_supported": False, "gpu_validated": False})
        (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.copy2(Path(__file__).with_name("strict_int8_runtime.py"), output / "strict_int8_runtime.py")
        print("Saved", path, path.stat().st_size, "bytes", flush=True)
        del converted
    else:
        report = json.loads((output / "export_report.json").read_text())
    shutil.copy2(Path(__file__).with_name("strict_int8_runtime.py"), output / "strict_int8_runtime.py")
    report["runtime_validation_passed"] = False
    report["validation_status"] = "running"
    report.pop("validation_error", None)
    (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        validate(path, examples, output, report)
    except Exception as error:
        report["validation_status"] = "failed"
        report["validation_error"] = str(error)
        (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["runtime_validation_passed"] = True
    report["validation_status"] = "passed"
    report["runtime_sha256"] = hashlib.sha256((output / "strict_int8_runtime.py").read_bytes()).hexdigest()
    shutil.copy2(Path(__file__).with_name("run_smolvla_strict_int8.py"), output / "run_smolvla_strict_int8.py")
    finish_package(output, report)


if __name__ == "__main__":
    main()
