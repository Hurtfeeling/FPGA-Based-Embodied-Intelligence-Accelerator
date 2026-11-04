#!/usr/bin/env python
"""Export original SmolVLA as W8A8 matrix products plus exclusively FP32 floats."""

import argparse
import collections
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as functional
from onnx import TensorProto as T, numpy_helper
from onnx.reference import ReferenceEvaluator

from scripts.export_smolvla_fp32_onnx import FP32ExecutionMode, load_original, sha256, specialize_vision
from scripts.export_smolvla_w8a8_onnx import IntegerAttentionMatmul, IntegerLinear, QuantizedLinear
from scripts.quantize_smolvla_w8a8 import quantize_activation, quantize_weight


def exact_int8_matmul(left, right):
    """INT8 x INT8 -> INT32, including broadcasting and CUDA kernel padding."""
    shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
    left = left.expand(*shape, m, k).reshape(-1, m, k)
    right = right.expand(*shape, k, n).reshape(-1, k, n)
    results = []
    for a, b in zip(left.unbind(), right.unbind(), strict=True):
        if a.is_cuda:
            # CUDA _int_mm needs aligned dimensions and more than 16 rows.
            pm, pk, pn = ((max(32, size) + 31) // 32 * 32 for size in (m, k, n))
            a = functional.pad(a, (0, pk - k, 0, pm - m)).contiguous()
            b = functional.pad(b, (0, pn - n, 0, pk - k)).contiguous()
            result = torch._int_mm(a, b)[:m, :n]
        else:
            result = torch._int_mm(a.contiguous(), b.contiguous())
        results.append(result)
    return torch.stack(results).reshape(*shape, m, n)


class ExactIntegerLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, codes, weight_scale, bias):
        if torch.onnx.is_in_onnx_export():
            # The symbolic below emits MatMulInteger. Avoid tracing CUDA
            # alignment/padding machinery that is only needed by the reference.
            return IntegerLinear.forward(ctx, value, codes, weight_scale, bias)
        maximum = value.abs().amax(-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
        qvalue = quantize_activation(value, scale)
        product = exact_int8_matmul(qvalue.reshape(-1, qvalue.shape[-1]), codes.T)
        result = product.reshape(*value.shape[:-1], codes.shape[0]).float() * scale * weight_scale
        return result if bias is None else result + bias

    symbolic = staticmethod(IntegerLinear.symbolic)


class ExactIntegerAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        if torch.onnx.is_in_onnx_export():
            return IntegerAttentionMatmul.forward(ctx, left, right)
        left_max = left.abs().amax(-1, keepdim=True)
        right_max = right.abs().amax(-2, keepdim=True)
        left_scale = torch.where(left_max > 0, left_max / 127, torch.ones_like(left_max))
        right_scale = torch.where(right_max > 0, right_max / 127, torch.ones_like(right_max))
        product = exact_int8_matmul(
            quantize_activation(left, left_scale), quantize_activation(right, right_scale)
        )
        return product.float() * left_scale * right_scale

    symbolic = staticmethod(IntegerAttentionMatmul.symbolic)


class W8A8FP32Mode(FP32ExecutionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.matmul:
            return ExactIntegerAttention.apply(*args)
        return super().__torch_function__(func, types, args, kwargs)


class ExactQuantizedLinear(QuantizedLinear):
    def forward(self, value):
        return ExactIntegerLinear.apply(value, self.codes, self.weight_scale, self.bias)


class QuantizedLanguageEmbedding(torch.nn.Module):
    """Gather actual INT8 rows before restoring just the selected vectors to FP32."""

    def __init__(self, original):
        super().__init__()
        codes, scale = quantize_weight(original.weight.detach().cpu())
        self.register_buffer("codes", codes.to(original.weight.device))
        self.register_buffer("weight_scale", scale.to(original.weight.device))

    def forward(self, tokens):
        selected = functional.embedding(tokens, self.codes)
        scale = functional.embedding(tokens, self.weight_scale[:, None])
        return selected.float() * scale


def quantize_language_embedding(policy):
    model = policy.model.vlm_with_expert.get_vlm_model().text_model
    model.embed_tokens = QuantizedLanguageEmbedding(model.embed_tokens)


def quantize_linears(policy):
    records = {}
    for name, module in list(policy.named_modules()):
        if not isinstance(module, torch.nn.Linear) or name.endswith("lm_head"):
            continue
        # Quantize on CPU so serialized codes can be independently reproduced.
        codes, scale = quantize_weight(module.weight.cpu())
        records[name] = {"weight_shape": list(codes.shape), "weight_elements": codes.numel()}
        replacement = ExactQuantizedLinear(module, codes, scale)
        parent, _, child = name.rpartition(".")
        setattr(policy.get_submodule(parent) if parent else policy, child, replacement)
    return records


class ActionChunk(torch.nn.Module):
    def __init__(self, policy, image_keys, quantized=True):
        super().__init__()
        self.policy = policy
        self.image_keys = image_keys
        self.quantized = quantized

    def forward(self, *inputs):
        images = inputs[: len(self.image_keys)]
        state, tokens, mask, noise = inputs[len(self.image_keys) :]
        batch = dict(zip(self.image_keys, images, strict=True))
        batch.update(
            {
                "observation.state": state,
                "observation.language.tokens": tokens,
                "observation.language.attention_mask": mask,
            }
        )
        with W8A8FP32Mode() if self.quantized else FP32ExecutionMode():
            return self.policy.predict_action_chunk(batch, noise=noise)


def simplify_static_shapes(model):
    """Fold small structural constants; retain floating arithmetic and INT8 products."""
    graph = model.graph
    types = {v.name: v.type.tensor_type.elem_type for v in [*graph.input, *graph.value_info, *graph.output]}
    types.update({v.name: v.data_type for v in graph.initializer})
    shapes = {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in [*graph.input, *graph.value_info, *graph.output]
        if v.type.tensor_type.HasField("shape")
        and all(d.HasField("dim_value") for d in v.type.tensor_type.shape.dim)
    }
    constants = {
        t.name: numpy_helper.to_array(t)
        for t in graph.initializer
        if t.data_type in (T.INT64, T.INT32, T.BOOL) and math.prod(t.dims) <= 8192
    }
    protected = {v.name for v in [*graph.input, *graph.output, *graph.initializer]}
    protected.update(name for n in graph.node if n.op_type == "MatMulInteger" for name in n.input)
    original_products = [n.SerializeToString() for n in graph.node if n.op_type == "MatMulInteger"]
    structural = {
        "Identity",
        "Cast",
        "Gather",
        "GatherElements",
        "Concat",
        "Reshape",
        "Transpose",
        "Unsqueeze",
        "Squeeze",
        "Slice",
        "Expand",
        "Tile",
        "ConstantOfShape",
    }
    integer_math = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Pow",
        "Equal",
        "Less",
        "LessOrEqual",
        "Greater",
        "GreaterOrEqual",
        "And",
        "Or",
        "Not",
        "Range",
        "CumSum",
        "ReduceSum",
        "ReduceProd",
        "ReduceMax",
        "ReduceMin",
        "Max",
        "Min",
        "Neg",
    }
    aliases, nodes = {}, []
    folded = 0

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    for node in graph.node:
        if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for a in node.attribute):
            raise ValueError("Static simplification expects a flat graph")
        node.input[:] = [resolve(name) for name in node.input]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Constant":
            if "value" in attrs:
                value = numpy_helper.to_array(attrs["value"])
                if value.size <= 8192:
                    constants[node.output[0]] = value
            nodes.append(node)
            continue
        if (
            node.op_type == "Identity" or (node.op_type == "Cast" and types.get(node.input[0]) == attrs["to"])
        ) and node.output[0] not in protected:
            aliases[node.output[0]] = node.input[0]
            continue
        results = None
        if node.op_type == "Shape" and node.input[0] in shapes:
            shape = shapes[node.input[0]]
            results = [np.asarray(shape[attrs.get("start", 0) : attrs.get("end")], dtype=np.int64)]
        elif node.op_type == "Size" and node.input[0] in shapes:
            results = [np.asarray(math.prod(shapes[node.input[0]]), dtype=np.int64)]
        elif (
            node.op_type in structural | integer_math
            and all(n in constants for n in node.input)
            and sum(constants[n].size for n in node.input) <= 16384
        ):
            values = [constants[n] for n in node.input]
            integer_only = all(v.dtype.kind in "biu" for v in values)
            if node.op_type in structural or integer_only:
                # Structural operations only rearrange/cast constants. No
                # floating Add/Mul/Div/norm/Softmax/matrix product is evaluated.
                try:
                    with np.errstate(all="raise"):
                        candidate = ReferenceEvaluator(node, opsets={"": 18}).run(
                            None, dict(zip(node.input, values, strict=True))
                        )
                    if all(v.size <= 8192 and v.dtype.kind in "biuf" for v in candidate):
                        results = candidate
                except (ValueError, TypeError, RuntimeError, NotImplementedError, FloatingPointError):
                    pass
        if results is not None:
            for name, value in zip(node.output, results, strict=True):
                constants[name] = value
                nodes.append(
                    onnx.helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value))
                )
            folded += 1
        else:
            nodes.append(node)
    signatures, shared = {}, 0
    unique = []
    for node in nodes:
        node.input[:] = [resolve(name) for name in node.input]
        if node.op_type == "Constant":
            attr = next((a for a in node.attribute if a.name == "value"), None)
            if attr is not None:
                value = numpy_helper.to_array(attr.t)
                key = (value.dtype.str, value.shape, value.tobytes())
                if key in signatures and node.output[0] not in protected:
                    aliases[node.output[0]] = signatures[key]
                    shared += 1
                    continue
                signatures.setdefault(key, node.output[0])
        unique.append(node)
    needed = {v.name for v in graph.output}
    kept = []
    for node in reversed(unique):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(n for n in node.input if n)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    del graph.value_info[:]
    assert [n.SerializeToString() for n in graph.node if n.op_type == "MatMulInteger"] == original_products
    return onnx.shape_inference.infer_shapes(model, strict_mode=True), {
        "structural_operations_folded": folded,
        "identical_constants_shared": shared,
        "nodes_after_static_simplification": len(graph.node),
        "matmul_integer_nodes_preserved_exactly": True,
        "floating_arithmetic_constant_folding": False,
    }


def audit(path, checkpoint=None):
    """Simplify static shapes, verify weights and persist audited FP32/INT8 types."""
    onnx.checker.check_model(str(path), full_check=True)
    model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
    model, simplification = simplify_static_shapes(model)
    allowed = {T.FLOAT, T.INT8, T.UINT8, T.INT32, T.INT64, T.BOOL}
    counts = collections.Counter()
    inferred = collections.Counter()

    def tensor_check(tensor):
        if tensor.data_type not in allowed:
            raise ValueError(f"Forbidden tensor dtype {T.DataType.Name(tensor.data_type)}: {tensor.name}")
        counts[T.DataType.Name(tensor.data_type)] += 1

    def graph_check(graph):
        for tensor in graph.initializer:
            tensor_check(tensor)
        for value in [*graph.input, *graph.value_info, *graph.output]:
            if value.type.HasField("tensor_type"):
                dtype = value.type.tensor_type.elem_type
                if dtype not in allowed:
                    raise ValueError(f"Forbidden value dtype {dtype}: {value.name}")
                inferred[T.DataType.Name(dtype)] += 1
        for node in graph.node:
            if node.domain not in ("", "ai.onnx"):
                raise ValueError(f"Custom operator: {node.domain}/{node.op_type}")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor_check(attr.t)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for tensor in attr.tensors:
                        tensor_check(tensor)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    graph_check(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for nested in attr.graphs:
                        graph_check(nested)
                if attr.name in ("to", "dtype", "stash_type") and attr.i not in allowed:
                    raise ValueError(f"Forbidden dtype attribute: {node.name}/{attr.name}={attr.i}")

    graph_check(model.graph)
    if model.functions:
        raise ValueError("Unexpected local ONNX functions")
    operators = collections.Counter(node.op_type for node in model.graph.node)
    if not operators["MatMulInteger"] or operators["MatMul"] or operators["Gemm"]:
        raise ValueError("Expected integer matrix products with no floating MatMul/Gemm")
    checked_codes = checked_floats = 0
    if checkpoint is not None:
        from safetensors import safe_open

        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
            for tensor in model.graph.initializer:
                name = tensor.name.removeprefix("policy.")
                actual = numpy_helper.to_array(tensor)
                if tensor.data_type == T.INT8:
                    if not name.endswith(".codes"):
                        raise ValueError(f"Unexpected INT8 initializer: {name}")
                    original = saved.get_tensor(name.removesuffix(".codes") + ".weight")
                    expected, _ = quantize_weight(original)
                    np.testing.assert_array_equal(actual, expected.numpy())
                    checked_codes += 1
                elif tensor.data_type == T.FLOAT:
                    if name.endswith(".weight_scale"):
                        original = saved.get_tensor(name.removesuffix(".weight_scale") + ".weight")
                        _, expected = quantize_weight(original)
                    else:
                        expected = saved.get_tensor(name).float()
                    np.testing.assert_array_equal(actual, expected.numpy())
                    checked_floats += 1
    result = {
        **simplification,
        "onnx_full_check_passed": True,
        "all_floating_tensors_fp32": True,
        "forbidden_float_types_found": [],
        "custom_operators": 0,
        "floating_matmul_operators": operators["MatMul"] + operators["Gemm"],
        "operator_counts": dict(operators),
        "all_tensor_type_counts": dict(counts),
        "inferred_value_type_counts": dict(inferred),
        "initializer_type_counts": dict(
            collections.Counter(T.DataType.Name(t.data_type) for t in model.graph.initializer)
        ),
        "initializer_bytes_by_type": {
            T.DataType.Name(dtype): sum(
                len(t.raw_data) for t in model.graph.initializer if t.data_type == dtype
            )
            for dtype in {t.data_type for t in model.graph.initializer}
        },
        "int8_weights_verified_against_checkpoint": checked_codes,
        "fp32_weights_and_scales_verified": checked_floats,
        "intermediate_shape_records": len(model.graph.value_info),
        "shape_information_persisted": True,
    }
    # Persist the audited intermediate types/shapes for the runtime loader.
    # Shape inference only annotates the existing graph and parameters.
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    del model
    gc.collect()
    return result


def prepare_examples(policy, checkpoint, count, device):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    training = json.loads((checkpoint / "train_config.json").read_text())
    info = training["dataset"]
    dataset = LeRobotDataset(info["repo_id"], root=info["root"], video_backend="pyav")
    pre, _ = make_pre_post_processors(
        policy.config, str(checkpoint), preprocessor_overrides={"device_processor": {"device": device}}
    )
    image_keys = list(policy.config.image_features)
    names = [key.removeprefix("observation.images.") for key in image_keys]
    names += ["state", "language_tokens", "language_mask", "noise"]
    episode_tasks = dataset.meta.episodes["tasks"]
    task_episodes = collections.defaultdict(list)
    excluded = set(info.get("exclude_episodes") or [])
    for ep in info.get("episodes") or range(dataset.num_episodes):
        if ep not in excluded:
            task_episodes[episode_tasks[ep][0] if episode_tasks[ep] else ""].append(ep)
    held_out = []
    for episodes in task_episodes.values():
        n_eval = math.ceil(len(episodes) * info.get("eval_split", 0))
        if n_eval:
            held_out.extend(episodes[-n_eval:])
    candidates = sorted(held_out or [ep for episodes in task_episodes.values() for ep in episodes])
    selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"])
    examples, records = [], []
    for index, ep in enumerate(selected):
        ep_frames = np.flatnonzero(episode_indices == ep)
        frame = int(ep_frames[len(ep_frames) // 2])
        batch = pre(dataset[frame])
        seed = 2000 + index
        noise = torch.randn(
            1,
            policy.config.chunk_size,
            policy.config.max_action_dim,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        values = tuple(batch[key] for key in image_keys) + (
            batch["observation.state"],
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            noise,
        )
        examples.append(values)
        records.append({"dataset_frame_index": frame, "episode_index": ep, "noise_seed": seed})
    details = {
        "dataset": info,
        "held_out_episodes": held_out,
        "samples": records,
        "uses_training_eval_split": bool(held_out),
        "images_from_dataset_without_extra_resize": True,
    }
    return image_keys, names, examples, details


def runtime_validate(path, output, report, write_report):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    print("Creating standard ONNX Runtime CPU session", flush=True)
    started = time.monotonic()
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    report["ort_session_initialization_seconds"] = time.monotonic() - started
    print("ORT CPU ready", flush=True)
    report["validation"] = []
    for index, sample in enumerate(report["examples"]["samples"]):
        with np.load(output / f"reference_{index}.npz", allow_pickle=False) as arrays:
            feed = {name: arrays[name] for name in report["input_names"]}
            reference = arrays["reference_actions"]
            baseline = arrays["fp32_actions"]
        started = time.monotonic()
        actual = session.run(["actions"], feed)[0]
        inference_seconds = time.monotonic() - started
        if actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise ValueError("Expected finite FP32 ONNX actions")
        error, quantization_error = np.abs(actual - reference), np.abs(actual - baseline)
        np.savez(
            output / f"validation_{index}.npz",
            **feed,
            reference_actions=reference,
            fp32_actions=baseline,
            onnx_actions=actual,
        )
        record = {
            **sample,
            "seconds": inference_seconds,
            "mae_vs_exact_w8a8_reference": float(error.mean()),
            "max_abs_error_vs_exact_w8a8_reference": float(error.max()),
            "mae_vs_unquantized_fp32": float(quantization_error.mean()),
            "max_abs_error_vs_unquantized_fp32": float(quantization_error.max()),
            "all_finite": True,
            "within_reference_limits": (
                float(error.mean()) <= report["validation_limits"]["mae"]
                and float(error.max()) <= report["validation_limits"]["max_abs"]
            ),
        }
        cpu_path = output / f"cpu_reference_{index}.npz"
        if cpu_path.exists():
            with np.load(cpu_path, allow_pickle=False) as arrays:
                cpu_error = np.abs(actual - arrays["cpu_w8a8_actions"])
            record.update(
                {
                    "mae_vs_pytorch_cpu_w8a8": float(cpu_error.mean()),
                    "max_abs_error_vs_pytorch_cpu_w8a8": float(cpu_error.max()),
                }
            )
        report["validation"].append(record)
        write_report()
        print("ORT validation", record, flush=True)
    report["reference_alignment_passed"] = all(v["within_reference_limits"] for v in report["validation"])
    report["runtime_execution_passed"] = True


def package(output, checkpoint, report):
    for name in (
        "tokenizer",
        "config.json",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        source = checkpoint / name
        if source.is_dir():
            shutil.copytree(source, output / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, output / name)
    shutil.copy2(Path(__file__).with_name("run_smolvla_onnx.py"), output / "run_smolvla_onnx.py")
    io = json.dumps(report["inputs"], indent=2, ensure_ascii=False)
    (output / "README.md").write_text(
        "# SmolVLA W8A8 + FP32 ONNX\n\n"
        f"训练检查点：`{checkpoint}`。\n\n"
        "`smolvla_w8a8_fp32.onnx` 是完整的固定形状动作块模型（ONNX opset 18），包含全部相机、"
        "上下文编码、KV 复用、10 次 Euler 更新和动作维度裁剪。\n\n"
        "Linear 权重为实际 INT8，逐输出通道对称量化；Linear 激活逐 token 动态量化；"
        "注意力 QK/PV 的两个输入均动态量化为 INT8。MatMulInteger 使用 INT32 累加。"
        "为兼容标准 ONNX Runtime CPU，左操作数以 UINT8(q_int8 + 128)、零点 128 等价编码。"
        "LayerNorm/RMSNorm、Softmax、卷积、嵌入、偏置、缩放、残差、时间嵌入及其余浮点计算均为 FP32。"
        "图中没有 FP16、BF16、FP8 或 FP64 张量/类型转换，也没有自定义算子。"
        "token、索引和掩码仍使用所需的整数/BOOL。\n\n"
        f"语言 Embedding 存储：{'INT8，先 Gather 再仅反量化选中的行至 FP32' if report.get('language_embedding_int8') else 'FP32'}。\n\n"
        f"文件体积 {report['bytes']:,} 字节；其中 INT8 权重 "
        f"{report['initializer_bytes_by_type']['INT8']:,} 字节，FP32 参数及尺度 "
        f"{report['initializer_bytes_by_type']['FLOAT']:,} 字节，其余为索引、计算图和常量。"
        "静态形状/索引计算与相同常量已精简，浮点算术未做常量折叠。\n\n"
        "相机顺序及原始名称映射以 policy_preprocessor.json 为准。图像输入是 RGB [0,1]，"
        "直接使用训练数据预处理后的实际图像尺寸；512×512 等比例缩放/补边和 [-1,1] 转换在图内。"
        "state 输入需先归一化；语言分词及 padding 在图外。noise 为外部标准正态噪声。"
        "输出 actions 为归一化 FP32 动作 [1,50,6]，使用随附 postprocessor 还原机器人动作单位。\n\n"
        f"输入规格：\n\n```json\n{io}\n```\n\n"
        "标准 ONNX Runtime CPU 运行示例：\n\n"
        "```bash\npython run_smolvla_onnx.py --model smolvla_w8a8_fp32.onnx "
        "--inputs validation_0.npz --output actions.npy\n```\n\n"
        "仅需 numpy、onnxruntime，无需自定义运行库。config.json/train_config.json 是来源配置记录，"
        "不能把本目录作为普通 PyTorch checkpoint 加载。\n\n"
        "export_report.json 包含图类型审计、权重与源 checkpoint 的逐值核对、校验和、"
        "留出样本上的 ONNX/精确 INT32 W8A8 参考误差及 W8A8/未量化 FP32 误差。"
        "参考执行采用真正的 INT8×INT8→INT32 运算，AMP/TF32 均关闭。"
        "runtime_execution_passed 记录标准 ORT 的有限 FP32 输出检查；reference_alignment_passed "
        "单独记录是否满足与 PyTorch 参考之间的误差阈值。动态 INT8 量化对 FP32 内核的舍入差异敏感，"
        "CPU/GPU 与不同运行库的输出可能存在差异，具体误差如实记录。"
        "离线样本误差不代表机器人任务成功率；GPU 执行性能未验证。\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vlm-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-frames", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--quantize-language-embedding",
        action="store_true",
        help="Store language rows in INT8; dequantize after Gather",
    )
    args = parser.parse_args()
    if args.validation_frames < 1:
        parser.error("Need at least one validation sample")
    if args.output.exists() and any(args.output.iterdir()) and not args.validate_only:
        parser.error("Output is not empty; choose a new directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "smolvla_w8a8_fp32.onnx"
    report_path = output / "export_report.json"
    started = time.monotonic()

    def write_report():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if not args.validate_only:
        source_hash = sha256(checkpoint / "model.safetensors")
        policy, missing = load_original(checkpoint, args.vlm_assets.resolve(), args.device)
        print("Loaded original checkpoint as FP32", flush=True)
        image_keys, names, examples, example_details = prepare_examples(
            policy, checkpoint, args.validation_frames, args.device
        )
        wrapper = ActionChunk(policy, image_keys, quantized=False).eval()
        with torch.inference_mode():
            baselines = [wrapper(*values).cpu().numpy() for values in examples]
        records = quantize_linears(policy)
        if args.quantize_language_embedding:
            quantize_language_embedding(policy)
        wrapper.quantized = True
        references = []
        with torch.inference_mode():
            for index, values in enumerate(examples):
                reference = wrapper(*values).cpu().numpy()
                if not np.isfinite(reference).all():
                    raise ValueError("Non-finite W8A8 reference")
                references.append(reference)
                np.savez(
                    output / f"reference_{index}.npz",
                    **{name: value.cpu().numpy() for name, value in zip(names, values, strict=True)},
                    reference_actions=reference,
                    fp32_actions=baselines[index],
                )
                print(
                    "W8A8 reference",
                    index,
                    "MAE vs FP32",
                    float(np.abs(reference - baselines[index]).mean()),
                    flush=True,
                )
        specialize_vision(policy)
        with torch.inference_mode():
            for values, reference in zip(examples, references, strict=True):
                np.testing.assert_array_equal(wrapper(*values).cpu().numpy(), reference)
            torch.onnx.export(
                wrapper,
                examples[0],
                str(path),
                input_names=names,
                output_names=["actions"],
                opset_version=18,
                dynamo=False,
                do_constant_folding=False,
                external_data=False,
            )
        report = {
            "source_checkpoint": str(checkpoint),
            "source_weights_sha256": source_hash,
            "precision": "W8A8 matrix products; all remaining floats FP32",
            "opset": 18,
            "input_names": names,
            "image_feature_order": image_keys,
            "inputs": [
                {"name": name, "shape": list(value.shape), "dtype": str(value.cpu().numpy().dtype)}
                for name, value in zip(names, examples[0], strict=True)
            ],
            "outputs": [{"name": "actions", "shape": list(references[0].shape), "dtype": "float32"}],
            "linear_layers": records,
            "examples": example_details,
            "weight_quantization": "per-output-channel symmetric INT8, nearest-even, [-127,127]",
            "language_embedding_int8": args.quantize_language_embedding,
            "activation_quantization": "dynamic per-token; attention per-row/per-column",
            "integer_accumulation": "INT32; reference uses exact INT8 integer products",
            "left_operand_encoding": "UINT8(q_signed+128), zero_point=128; equivalent signed INT8",
            "unquantized_operators": "FP32 Conv, embedding, LayerNorm/RMSNorm, Softmax, residuals, time embedding",
            "time_embedding": "FP32 export-local implementation",
            "forward_fp32_guard": True,
            "amp_enabled": False,
            "tf32_enabled": False,
            "vision_specialization_bit_exact": True,
            "unused_missing_lm_head_keys": missing,
            "single_file_onnx": True,
            "validation_limits": {"mae": 0.01, "max_abs": 0.1},
            "runtime_validation_passed": False,
            "validation_status": "pending",
        }
        write_report()
        del policy, wrapper, examples, values
        gc.collect()
        torch.cuda.empty_cache()
        print("ONNX exported", path.stat().st_size, "bytes", flush=True)
    else:
        report = json.loads(report_path.read_text())
    report["validation_status"] = "running"
    report["runtime_validation_passed"] = False
    report.pop("validation_error", None)
    write_report()
    try:
        report.update(audit(path, checkpoint))
        write_report()
        runtime_validate(path, output, report, write_report)
        report["source_checkpoint_unchanged"] = (
            sha256(checkpoint / "model.safetensors") == report["source_weights_sha256"]
        )
        if not report["source_checkpoint_unchanged"]:
            raise ValueError("Source checkpoint changed during export")
        report.update(
            {
                "runtime_validation_passed": report["reference_alignment_passed"],
                "validation_status": (
                    "passed"
                    if report["reference_alignment_passed"]
                    else "completed_with_reference_difference"
                ),
                "runtime_provider_validated": "CPUExecutionProvider",
                "gpu_runtime_validated": False,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "elapsed_seconds": time.monotonic() - started,
                "output": str(path),
            }
        )
        package(output, checkpoint, report)
        write_report()
    except Exception as error:
        report["validation_status"] = "failed"
        report["validation_error"] = str(error)
        write_report()
        raise
    print("W8A8/FP32 export and validation completed", flush=True)


if __name__ == "__main__":
    main()
