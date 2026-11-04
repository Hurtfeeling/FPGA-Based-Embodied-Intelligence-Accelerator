#!/usr/bin/env python
"""Export W8A8 with E4M3FN floating boundaries and unchanged INT8 products.

Custom FP8 operators use FP32 reference arithmetic internally. This is not a
native all-FP8 hardware graph and cannot be loaded by default ONNX Runtime.
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

from scripts.w8a8_fp8_runtime import DOMAIN, FP8_DTYPE, FP8Session, Float8, attributes, encode

OLD_FLOATS = {T.FLOAT, T.DOUBLE, T.FLOAT16, T.BFLOAT16}
FP8_TYPE = T.FLOAT8E4M3FN


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def floating(array):
    return array.dtype.kind == "f" or str(array.dtype) == "bfloat16"


def live_nodes(nodes, outputs):
    needed, kept = set(outputs), []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(n for n in node.input if n)
    return list(reversed(kept)), needed


def canonicalize(source):
    model = onnx.shape_inference.infer_shapes(source)
    types = {v.name: v.type.tensor_type.elem_type
             for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    types.update({v.name: v.data_type for v in model.graph.initializer})
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(attributes(node)["value"])
    producers = {o: n for n in model.graph.node for o in n.output}
    users = collections.defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            users[name].append(node)
    fused, replaced_scales = {}, set()
    def consumer(name, op):
        candidates = [n for n in users[name] if n.op_type == op]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {op} after {name}")
        return candidates[0]
    # Preserve the integer accumulator as INT32 until its two scales have
    # been applied in a fused reference kernel, then round the result to FP8.
    for node in model.graph.node:
        if node.op_type != "MatMulInteger":
            continue
        cast = consumer(node.output[0], "Cast")
        first = consumer(cast.output[0], "Mul")
        second = consumer(first.output[0], "Mul")
        left_scale = next(n for n in first.input if n != cast.output[0])
        right_scale = next(n for n in second.input if n != first.output[0])
        fused[second.output[0]] = helper.make_node("RescaleMatMulInteger", [node.output[0], left_scale, right_scale], list(second.output))
    quantizers = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or attributes(node)["to"] not in (T.INT8, T.UINT8):
            continue
        current = producers.get(node.input[0])
        unsigned = attributes(node)["to"] == T.UINT8
        if unsigned:
            if current is None or current.op_type != "Add":
                continue
            offset = constants.get(current.input[1])
            if offset is None or offset.size != 1 or float(offset.item()) != 128:
                continue
            current = producers.get(current.input[0])
        if current is None or current.op_type != "Clip":
            continue
        lower, upper = (constants.get(n) for n in current.input[1:])
        if lower is None or upper is None or float(lower.item()) != -127 or float(upper.item()) != 127:
            continue
        rounded = producers.get(current.input[0])
        division = producers.get(rounded.input[0]) if rounded is not None and rounded.op_type == "Round" else None
        if division is None or division.op_type != "Div":
            continue
        scale_name = division.input[1]
        scale = producers.get(scale_name)
        if scale is None or scale.op_type != "Where":
            continue
        scale_div = producers.get(scale.input[1])
        maximum = producers.get(scale_div.input[0]) if scale_div is not None and scale_div.op_type == "Div" else None
        if maximum is None or maximum.op_type != "ReduceMax":
            continue
        axis = constants[maximum.input[1]]
        if axis.size != 1:
            raise ValueError("Expected one per-row/column quantizer axis")
        fused[node.output[0]] = helper.make_node("QuantizeINT8", [division.input[0]],
                                                [node.output[0], scale_name],
                                                axis=int(axis.item()), unsigned=int(unsigned))
        replaced_scales.add(scale_name)
        quantizers += 1
    aliases, nodes = {}, []
    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name
    for original in model.graph.node:
        if original.op_type == "Constant" or (original.output[0] in replaced_scales):
            continue
        node = copy.deepcopy(fused.get(original.output[0], original))
        node.input[:] = [resolve(n) for n in node.input]
        if (node.op_type == "Cast" and attributes(node)["to"] in OLD_FLOATS
                and types.get(original.input[0]) in OLD_FLOATS):
            aliases[node.output[0]] = node.input[0]
            continue
        nodes.append(node)
    for node in nodes:
        node.input[:] = [resolve(n) for n in node.input]
    output_names = [resolve(v.name) for v in model.graph.output]
    nodes, _ = live_nodes(nodes, output_names)
    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in [*model.graph.input, *model.graph.value_info]
              if v.type.tensor_type.HasField("shape")
              and all(d.HasField("dim_value") for d in v.type.tensor_type.shape.dim)}
    kept = []
    for node in nodes:
        if node.op_type == "Shape" and node.input[0] in shapes:
            a = attributes(node)
            constants[node.output[0]] = np.asarray(shapes[node.input[0]][a.get("start", 0):a.get("end")], np.int64)
            continue
        if (all(n in constants for n in node.input) and node.op_type not in ("QuantizeINT8", "MatMulInteger")
                and sum(constants[n].size for n in node.input) < 1_000_000
                and not any(constants[n].dtype == np.int8 for n in node.input)):
            try:
                with np.errstate(all="ignore"):
                    result = ReferenceEvaluator(node, opsets={"": 18}).run(None, {n: constants[n] for n in node.input})
            except (NotImplementedError, ValueError, TypeError, RuntimeError):
                pass
            else:
                constants.update(zip(node.output, result, strict=True))
                continue
        kept.append(node)
    # Fuse the original RMS module, keeping its internal reciprocal in the
    # allowed FP32 reference kernel instead of exposing an FP8 zero divisor.
    producers = {o: n for n in kept for o in n.output}
    rms = {}
    for node in kept:
        if node.op_type != "Mul":
            continue
        for weight, hidden in (node.input, list(reversed(node.input))):
            if weight not in constants:
                continue
            product = producers.get(hidden)
            if product is None or product.op_type != "Mul":
                continue
            for x, reciprocal in (product.input, list(reversed(product.input))):
                inverse = producers.get(reciprocal)
                if inverse is None or inverse.op_type != "Div":
                    continue
                root = producers.get(inverse.input[1])
                addition = producers.get(root.input[0]) if root is not None and root.op_type == "Sqrt" else None
                if addition is None or addition.op_type != "Add":
                    continue
                for mean_name, epsilon_name in (addition.input, list(reversed(addition.input))):
                    mean, epsilon = producers.get(mean_name), constants.get(epsilon_name)
                    if mean is None or mean.op_type != "ReduceMean" or epsilon is None or epsilon.size != 1:
                        continue
                    square = producers.get(mean.input[0])
                    if square is None or square.op_type != "Pow" or square.input[0] != x:
                        continue
                    power, numerator = constants.get(square.input[1]), constants.get(inverse.input[0])
                    axes = constants.get(mean.input[1]) if len(mean.input) > 1 else None
                    if (power is None or float(power.item()) != 2 or numerator is None or float(numerator.item()) != 1
                            or axes is None or axes.tolist() != [-1]):
                        continue
                    rms[node.output[0]] = helper.make_node("RMSNormalization", [x, weight], list(node.output),
                                                          epsilon=float(epsilon.item()))
    kept = [rms.get(n.output[0], n) for n in kept]
    kept, needed = live_nodes(kept, output_names)
    constants = {k: np.asarray(v) for k, v in constants.items() if k in needed}
    for name, node in fused.items():
        if node.op_type == "QuantizeINT8":
            types[name] = T.UINT8 if attributes(node)["unsigned"] else T.INT8
    print(f"Prepared {len(kept)} nodes; fused {quantizers} original INT8 quantizers and {len(rms)} RMS modules", flush=True)
    print("Operations:", dict(collections.Counter(n.op_type for n in kept)), flush=True)
    return kept, constants, types, output_names


def audit(model, source):
    onnx.checker.check_model(model)
    for tensor in model.graph.initializer:
        if tensor.data_type in OLD_FLOATS:
            raise ValueError(f"Non-FP8 floating initializer: {tensor.name}")
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.type.tensor_type.elem_type in OLD_FLOATS:
            raise ValueError(f"Non-FP8 floating boundary: {value.name}")
    for node in model.graph.node:
        if not node.domain and node.op_type == "Cast" and attributes(node)["to"] in OLD_FLOATS:
            raise ValueError("Non-FP8 floating Cast remains")
        for a in node.attribute:
            if a.type in (onnx.AttributeProto.FLOAT, onnx.AttributeProto.FLOATS):
                raise ValueError(f"FP32 attribute remains: {node.name}/{a.name}")
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type in OLD_FLOATS:
                raise ValueError(f"Non-FP8 constant remains: {node.name}")
    actual = {v.name: v for v in model.graph.initializer}
    weights = 0
    for tensor in source.graph.initializer:
        if tensor.data_type == T.INT8:
            if tensor.name not in actual or actual[tensor.name].SerializeToString() != tensor.SerializeToString():
                raise ValueError(f"Original INT8 weight changed: {tensor.name}")
            weights += 1
    old_products = {n.name: n.SerializeToString() for n in source.graph.node if n.op_type == "MatMulInteger"}
    new_products = {n.name: n.SerializeToString() for n in model.graph.node if n.op_type == "MatMulInteger"}
    if old_products != new_products:
        raise ValueError("Original MatMulInteger nodes were modified")
    return {"int8_weights_byte_identical": weights, "matmul_integer_nodes_identical": len(new_products),
            "fp16_bf16_fp32_fp64_tensors": 0, "floating_attributes": 0}


def compact_constants(model, source):
    """Share repeated index/FP8 constants without touching integer products."""
    protected = {v.name for v in source.graph.initializer if v.data_type == T.INT8}
    protected.update(name for n in source.graph.node if n.op_type == "MatMulInteger" for name in n.input)
    signatures, aliases, kept = {}, {}, []
    for tensor in model.graph.initializer:
        if tensor.name in protected:
            kept.append(tensor)
            continue
        key = (tensor.data_type, tuple(tensor.dims), hashlib.sha256(tensor.raw_data).digest())
        if key in signatures:
            aliases[tensor.name] = signatures[key]
        else:
            signatures[key] = tensor.name
            kept.append(tensor)
    for node in model.graph.node:
        node.input[:] = [aliases.get(name, name) for name in node.input]
    info = [v for v in model.graph.value_info if v.name not in aliases]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    model.graph.value_info.extend(info)
    return len(aliases)


def convert(source):
    nodes, constants, types, output_names = canonicalize(source)
    initializers, result_nodes, information = [], [], []
    fp8 = set()
    for name, array in constants.items():
        if floating(array):
            value = encode(array, per_channel=array.ndim > 1, per_element=name.endswith("weight_scale"))
            initializers.extend([numpy_helper.from_array(value.data, name),
                                 numpy_helper.from_array(value.exponent, name + "__fp8_exponent")])
            fp8.add(name)
            information.append(helper.make_tensor_value_info(name, FP8_TYPE, list(array.shape)))
            information.append(helper.make_tensor_value_info(name + "__fp8_exponent", T.INT32, list(value.exponent.shape)))
        else:
            initializers.append(numpy_helper.from_array(array, name))
    # Restore original protobufs verbatim for the INT8 weight-integrity check.
    original_i8 = {v.name: v for v in source.graph.initializer if v.data_type == T.INT8}
    initializers = [copy.deepcopy(original_i8.get(v.name, v)) for v in initializers]
    graph_inputs = []
    for original in source.graph.input:
        value = copy.deepcopy(original)
        if value.type.tensor_type.elem_type in OLD_FLOATS:
            value.type.tensor_type.elem_type = FP8_TYPE
            fp8.add(value.name)
            graph_inputs.extend([value, helper.make_tensor_value_info(value.name + "__fp8_exponent", T.INT32, [])])
        else:
            graph_inputs.append(value)
    for index, original in enumerate(nodes):
        node = copy.deepcopy(original)
        if node.op_type == "Shape":
            result_nodes.append(node)
            continue
        kinds = [int(name in fp8) for name in node.input]
        output_kinds = [int(types.get(name) in OLD_FLOATS) for name in node.output]
        if node.op_type == "QuantizeINT8":
            output_kinds = [0, 1]
        if not any(kinds) and not any(output_kinds):
            result_nodes.append(node)
            continue
        op, a = node.op_type, attributes(node)
        if op == "Cast":
            op = "FromInteger" if a["to"] in OLD_FLOATS else "ToInteger"
            if op == "FromInteger":
                a.pop("to")
        if op == "ConstantOfShape":
            fill = numpy_helper.to_array(a.pop("value")) if "value" in a else np.array([0], np.float32)
            value = encode(fill)
            a["fill_bits"] = int(value.data.view(np.uint8).item())
            a["fill_exponent"] = int(value.exponent.item())
        clean = {}
        for key, value in a.items():
            if isinstance(value, float):
                encoded = encode(np.array(value, np.float32))
                clean[key + "__fp8_bits"] = int(encoded.data.view(np.uint8).item())
                clean[key + "__fp8_exponent"] = int(encoded.exponent.item())
            else:
                clean[key] = value
        inputs, outputs = [], []
        for name, kind in zip(node.input, kinds, strict=True):
            inputs.append(name)
            if kind:
                inputs.append(name + "__fp8_exponent")
        for name, kind in zip(node.output, output_kinds, strict=True):
            outputs.append(name)
            if kind:
                fp8.add(name)
                outputs.append(name + "__fp8_exponent")
                information.append(helper.make_tensor_value_info(name, FP8_TYPE, None))
                information.append(helper.make_tensor_value_info(name + "__fp8_exponent", T.INT32, None))
            else:
                information.append(helper.make_tensor_value_info(name, types[name], None))
        result_nodes.append(helper.make_node("F8" + op, inputs, outputs, name=f"fp8_{index}_{op}",
                                              domain=DOMAIN, input_kinds=kinds, output_kinds=output_kinds, **clean))
    graph_outputs = []
    for original, name in zip(source.graph.output, output_names, strict=True):
        if name not in fp8:
            raise ValueError("Output did not become FP8")
        graph_outputs.extend([helper.make_tensor_value_info(name, FP8_TYPE,
                              [d.dim_value for d in original.type.tensor_type.shape.dim]),
                              helper.make_tensor_value_info(name + "__fp8_exponent", T.INT32, [])])
    graph = helper.make_graph(result_nodes, "SmolVLA_W8A8_FP8", graph_inputs, graph_outputs,
                              initializer=initializers, value_info=information)
    result = helper.make_model(graph, producer_name="lerobot-w8a8-fp8-boundaries", ir_version=9,
                               opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid(DOMAIN, 1)])
    helper.set_model_props(result, {
        "quantization": "Original W8A8 INT8 weights and MatMulInteger nodes preserved",
        "floating_tensors": "FLOAT8E4M3FN codes times 2**INT32 exponent",
        "execution": "FP32 reference kernels rounded to FP8 at operator boundaries; not native FP8 execution",
        "attention_mask": "E4M3FN negative NaN bit pattern 0xff is a custom finite negative-mask token",
        "runtime": "w8a8_fp8_runtime.py; not supported by default ONNX Runtime",
    })
    shared = compact_constants(result, source)
    print("Shared repeated constants:", shared, flush=True)
    return result, audit(result, source)


def validate(path, examples, output, report):
    session = FP8Session(path)
    report["validation"] = []
    for index, example in enumerate(examples):
        feed = {}
        for value in session.model.graph.input:
            if value.name.endswith("__fp8_exponent"):
                continue
            array = example[value.name]
            if value.type.tensor_type.elem_type == FP8_TYPE:
                encoded = encode(array)
                feed[value.name], feed[value.name + "__fp8_exponent"] = encoded.data, encoded.exponent
            else:
                feed[value.name] = array
        started = time.monotonic()
        result = session.run(feed, progress=lambda done, total: print(f"Sample {index}: {done}/{total}", flush=True))
        name = session.model.graph.output[0].name
        decoded = Float8(result[name], result[name + "__fp8_exponent"]).decode()
        if not np.isfinite(decoded).all():
            raise ValueError("Model produced non-finite actions")
        error = np.abs(decoded - example["onnx_actions"])
        record = {"sample": index, "seconds": time.monotonic() - started,
                  "mae_vs_w8a8_onnx": float(error.mean()), "max_error_vs_w8a8_onnx": float(error.max()),
                  "action_dtype": str(result[name].dtype), "boundary_fp32_tensors": 0,
                  "internal_floating_compute": "FP32 reference simulation"}
        report["validation"].append(record)
        # NumPy does not preserve ml_dtypes FP8 metadata in NPZ. Save bytes
        # explicitly and document their E4M3FN interpretation.
        packed = {k: v.view(np.uint8) if v.dtype == FP8_DTYPE else v for k, v in feed.items()}
        np.savez(output / f"validation_fp8_{index}.npz", **packed,
                 action_fp8_bits=result[name].view(np.uint8), action_exponent=result[name + "__fp8_exponent"],
                 decoded_actions=decoded, reference_actions=example["onnx_actions"])
        print("Validation:", record, flush=True)
        (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")


def finish_package(output, report):
    path = output / "smolvla_w8a8_fp8.onnx"
    model = onnx.load(path)
    source = onnx.load(report["source"])
    report.update(audit(model, source))
    report["bytes"] = path.stat().st_size
    report["sha256"] = sha256(path)
    report["source_unchanged"] = sha256(report["source"]) == report["source_sha256"]
    if not report["source_unchanged"]:
        raise ValueError("Original W8A8 file changed")
    storage = collections.defaultdict(lambda: {"tensors": 0, "bytes": 0})
    for tensor in model.graph.initializer:
        record = storage[T.DataType.Name(tensor.data_type)]
        record["tensors"] += 1
        record["bytes"] += len(tensor.raw_data)
    report["initializer_storage"] = dict(storage)
    report["initializer_types"] = dict(collections.Counter(T.DataType.Name(t.data_type) for t in model.graph.initializer))
    for filename in ("w8a8_fp8_runtime.py", "run_smolvla_w8a8_fp8.py"):
        shutil.copy2(Path(__file__).with_name(filename), output / filename)
    report["runtime_sha256"] = sha256(output / "w8a8_fp8_runtime.py")
    errors = [r["mae_vs_w8a8_onnx"] for r in report["validation"]]
    protocol = {
        "domain": DOMAIN, "version": 1, "fp8_type": "FLOAT8E4M3FN",
        "float_encoding": "physical value = E4M3FN data * 2**INT32_exponent",
        "input_kinds": {"0": "one integer/index/bool tensor", "1": "FP8 tensor followed by INT32 exponent"},
        "output_kinds": {"0": "one integer/index/bool tensor", "1": "FP8 tensor followed by INT32 exponent"},
        "quantization_parameters": "INT32 power-of-two exponents; INT8 zero-points and range limits remain exact integers",
        "attention_mask": "FP8 byte 0xff is a negative finite-mask token; decode it as -FP32_MAX, excluding it from scale selection",
        "quantize_int8": "Exact signed range [-127,127], optional UINT8 offset 128; FP8 scale uses a per-row/column INT32 exponent",
        "execution": "Decode to FP32 internally, compute the reference result, then round to FP8 at floating operator boundaries",
        "integer_execution": "Original MatMulInteger with INT8/UINT8 operands and exact INT32 accumulation",
        "native_fp8_hardware_execution": False, "default_onnxruntime_supported": False,
        "npz_encoding": "FP8 arrays are stored as UINT8 bit patterns with accompanying exponents; interpret bytes as E4M3FN, not numeric UINT8 pixels",
        "inputs": [{"name": v.name, "dtype": T.DataType.Name(v.type.tensor_type.elem_type)} for v in model.graph.input],
        "outputs": [{"name": v.name, "dtype": T.DataType.Name(v.type.tensor_type.elem_type)} for v in model.graph.output],
    }
    (output / "fp8_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "README.md").write_text(
        "# SmolVLA W8A8 + FP8\n\n"
        f"`smolvla_w8a8_fp8.onnx`：{report['bytes']:,} 字节。"
        f"原 W8A8 的 {report['int8_weights_byte_identical']} 份 INT8 权重逐字节保留，"
        f"{report['matmul_integer_nodes_identical']} 个 `MatMulInteger` 节点完整保留。"
        "其余浮点张量使用 ONNX `FLOAT8E4M3FN`，配套 INT32 二进制缩放指数；"
        "没有 FP16、BF16、FP32 或 FP64 张量/浮点属性残留。\n\n"
        "这是自定义 FP8 算子边界模型。附带运行器内部使用 FP32 参考计算，再舍入为 FP8；"
        "不是原生全 FP8 硬件执行，不含 ONNX Runtime 自定义算子动态库，默认 ONNX Runtime 不能直接加载。"
        "未进行 GPU 内核或性能验证。FP8 字节和算子协议见 `fp8_protocol.json`。\n\n"
        f"三组记录输入已完成完整推理，相对原 W8A8 的归一化动作 MAE 为 {min(errors):.6f}–{max(errors):.6f}。"
        "这是离线数值误差，不是机器人任务成功率；不能据此认定控制精度达标。\n\n"
        "保留原两路图像、上下文编码、缓存注意力和十步 Euler 更新。"
        "状态仍需使用原策略的外部归一化，语言 tokens/mask 仍为 INT64/BOOL。"
        "原始浮点样例在运行器入口编码为 FP8，动作可在出口解码；这些输入/输出转换在模型之外。\n\n"
        "本机运行示例（输出文件必须不存在）：\n\n```bash\n"
        "/home/ial-mohd/workspace/so101-lab/venvs/tools/bin/uv run --no-project "
        "--python /home/ial-mohd/workspace/so101-lab/venvs/isaaclab/bin/python python "
        f"{output / 'run_smolvla_w8a8_fp8.py'} --model {path} "
        f"--inputs {output / 'validation_fp8_0.npz'} --output {output / 'actions_check.npz'} --decode\n```\n"
        "依赖：onnx、numpy、ml_dtypes。可选 PyTorch 用于加速整数 GEMM 和 FP32 参考卷积；"
        "没有 PyTorch 时使用较慢的 NumPy 参考路径。\n"
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
        parser.error("Source directory needs validation_*.npz recorded examples")
    examples = []
    for sample in samples:
        with np.load(sample, allow_pickle=False) as arrays:
            examples.append({k: arrays[k] for k in arrays.files})
    output.mkdir(parents=True, exist_ok=True)
    path = output / "smolvla_w8a8_fp8.onnx"
    if not args.validate_only:
        model = onnx.load(source)
        converted, integrity = convert(model)
        onnx.save(converted, path)
        report = {"source": str(source), "file": str(path), "source_sha256": sha256(source),
                  "bytes": path.stat().st_size, "fp8_format": "FLOAT8E4M3FN", **integrity,
                  "operator_counts": dict(collections.Counter(n.domain + ":" + n.op_type for n in converted.graph.node)),
                  "initializer_types": dict(collections.Counter(T.DataType.Name(t.data_type) for t in converted.graph.initializer)),
                  "runtime_validation_passed": False, "validation_status": "pending",
                  "native_fp8_execution": False, "fp32_internal_reference_arithmetic": True,
                  "default_onnxruntime_supported": False, "gpu_validated": False}
        (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        del model, converted
        print("Saved", path, report["bytes"], "bytes", flush=True)
    else:
        report = json.loads((output / "export_report.json").read_text())
    shutil.copy2(Path(__file__).with_name("w8a8_fp8_runtime.py"), output / "w8a8_fp8_runtime.py")
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
    report["sha256"] = sha256(path)
    report["runtime_sha256"] = sha256(output / "w8a8_fp8_runtime.py")
    finish_package(output, report)


if __name__ == "__main__":
    main()
