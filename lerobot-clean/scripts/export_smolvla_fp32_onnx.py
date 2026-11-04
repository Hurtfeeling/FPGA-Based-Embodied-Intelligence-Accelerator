#!/usr/bin/env python
"""Export original SmolVLA checkpoint weights as standard all-FP32 ONNX."""

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import MethodType

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto as T, numpy_helper
from torch.overrides import TorchFunctionMode

NAMES = ("wrist", "front", "state", "language_tokens", "language_mask", "noise")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FP32ExecutionMode(TorchFunctionMode):
    """Enforce FP32 floating outputs and ONNX-safe integer mask reductions."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (getattr(func, "__name__", "") in ("sum", "cumsum", "prod") and args
                and isinstance(args[0], torch.Tensor)
                and args[0].dtype in (torch.bool, torch.int8, torch.uint8, torch.int16)):
            # PyTorch already promotes these reductions to INT64. Make the
            # input conversion explicit so ONNX does not infer a BOOL output.
            args = (args[0].to(torch.int64), *args[1:])
        result = func(*args, **kwargs)
        def check(value):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and value.dtype != torch.float32:
                    raise TypeError(f"Non-FP32 forward result from {func}: {value.dtype}")
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
        check(result)
        return result


def fp32_time_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 or time.ndim not in (1, 2):
        raise ValueError("Expected an even embedding width and rank-1/rank-2 times")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * (2 * math.pi)
    radians = time.to(torch.float32)[..., None] * scaling_factor
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1)


class ActionChunk(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, wrist, front, state, language_tokens, language_mask, noise):
        batch = {"observation.images.wrist": wrist, "observation.images.front": front,
                 "observation.state": state, "observation.language.tokens": language_tokens,
                 "observation.language.attention_mask": language_mask}
        with FP32ExecutionMode():
            return self.policy.predict_action_chunk(batch, noise=noise)


def load_original(checkpoint, assets, device):
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla import modeling_smolvla

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    config.load_vlm_weights = False
    config.vlm_model_name = str(assets)
    policy = modeling_smolvla.SmolVLAPolicy(config).eval()
    tensors = load_file(checkpoint / "model.safetensors", device="cpu")
    missing, unexpected = policy.load_state_dict(tensors, strict=False, assign=True)
    if unexpected or any(not name.endswith("lm_head.weight") for name in missing):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    del tensors
    policy = policy.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    eager_count = 0
    for module in policy.modules():
        if type(module).__name__ == "SmolVLMVisionAttention":
            module.config._attn_implementation = "eager"
            eager_count += 1
    if not eager_count:
        raise ValueError("Unsupported vision attention implementation")
    # Export-local replacement: the shared implementation requests FP64.
    # No repository policy source or checkpoint is modified.
    modeling_smolvla.create_sinusoidal_pos_embedding = fp32_time_embedding
    for name, tensor in [*policy.named_parameters(), *policy.named_buffers()]:
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise TypeError(f"Non-FP32 model tensor: {name}")
    return policy, missing


def specialize_vision(policy):
    from transformers.modeling_outputs import BaseModelOutput

    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    side = 512 // vision.patch_size
    if side != vision.embeddings.num_patches_per_side:
        raise ValueError("Fixed 512px image grid does not match model positions")
    vision.register_buffer("onnx_position_ids", torch.arange(side * side, device=next(vision.parameters()).device)[None])
    def forward(self, pixel_values, patch_attention_mask=None, **kwargs):
        if patch_attention_mask is not None:
            raise ValueError("Only the all-valid fixed patch grid is supported")
        patches = self.embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        hidden = patches + self.embeddings.position_embedding(self.onnx_position_ids)
        hidden = self.encoder(inputs_embeds=hidden, attention_mask=None).last_hidden_state
        return BaseModelOutput(last_hidden_state=self.post_layernorm(hidden))
    vision.forward = MethodType(forward, vision)


def audit(path, checkpoint=None):
    # Pass the filename to the C++ checker, avoiding serialization of another
    # complete 1.6GB protobuf through Python.
    onnx.checker.check_model(str(path), full_check=True)
    model = onnx.load(path)
    forbidden = {T.FLOAT16, T.BFLOAT16, T.DOUBLE, T.FLOAT8E4M3FN, T.FLOAT8E4M3FNUZ,
                 T.FLOAT8E5M2, T.FLOAT8E5M2FNUZ, T.INT8, T.UINT8}
    for tensor in model.graph.initializer:
        if tensor.data_type in forbidden:
            raise ValueError(f"Non-FP32 or quantized initializer: {tensor.name}")
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.tensor_type.elem_type in forbidden:
            raise ValueError(f"Non-FP32/quantized graph value: {value.name}")
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise ValueError(f"Custom operator remains: {node.domain}/{node.op_type}")
        if node.op_type in ("MatMulInteger", "ConvInteger", "QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"):
            raise ValueError(f"Quantized operator remains: {node.name}")
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type in forbidden:
                raise ValueError(f"Non-FP32 constant: {node.name}/{attr.name}")
            if attr.name in ("to", "dtype", "stash_type") and attr.type == onnx.AttributeProto.INT and attr.i in forbidden:
                raise ValueError(f"Non-FP32 dtype attribute: {node.name}/{attr.name}")
    matched = 0
    if checkpoint is not None:
        from safetensors import safe_open
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
            keys = set(saved.keys())
            for tensor in model.graph.initializer:
                if tensor.data_type != T.FLOAT:
                    continue
                name = tensor.name.removeprefix("policy.")
                if name not in keys:
                    raise ValueError(f"Floating initializer is not an original checkpoint parameter: {tensor.name}")
                original = saved.get_tensor(name).float().numpy()
                np.testing.assert_array_equal(numpy_helper.to_array(tensor), original)
                matched += 1
    result = {"onnx_full_check_passed": True, "all_floating_tensors_fp32": True,
              "quantized_operators": 0, "custom_operators": 0,
              "checkpoint_initializers_exact_after_fp32_promotion": matched,
              "operator_counts": dict(collections.Counter(n.op_type for n in model.graph.node)),
              "initializer_types": dict(collections.Counter(T.DataType.Name(t.data_type) for t in model.graph.initializer)),
              "fp32_parameter_elements": sum(math.prod(t.dims) for t in model.graph.initializer if t.data_type == T.FLOAT)}
    del model
    gc.collect()
    return result


def validate(path, sample_paths, output, atol, rtol, report=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    print("Creating standard ONNX Runtime CPU session", flush=True)
    started = time.monotonic()
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    print("ORT session ready after", time.monotonic() - started, "seconds", flush=True)
    results = []
    for index, sample_path in enumerate(sample_paths):
        with np.load(sample_path, allow_pickle=False) as arrays:
            feed = {name: arrays[name] for name in NAMES}
            reference = arrays["pytorch_fp32_actions"]
        started = time.monotonic()
        actual = session.run(["actions"], feed)[0]
        if actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise ValueError("ONNX output must be finite FP32")
        error = np.abs(actual - reference)
        record = {"sample": index, "seconds": time.monotonic() - started,
                  "mae_vs_pytorch_fp32": float(error.mean()), "max_error_vs_pytorch_fp32": float(error.max()),
                  "all_finite": True, "output_dtype": str(actual.dtype)}
        np.savez(output / f"validation_fp32_{index}.npz", **feed, reference_actions=reference, onnx_actions=actual)
        print("ONNX Runtime FP32 validation:", record, flush=True)
        results.append(record)
        if report is not None:
            report["validation"] = results
            (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    return results


def finish_package(output, report):
    path = output / "smolvla_fp32.onnx"
    report["output"] = str(path.resolve())
    report["bytes"] = path.stat().st_size
    report["sha256"] = sha256(path)
    with np.load(output / "pytorch_reference_0.npz", allow_pickle=False) as arrays:
        report["inputs"] = [{"name": n, "shape": list(arrays[n].shape), "dtype": str(arrays[n].dtype)} for n in NAMES]
    report["floating_parameter_bytes"] = report["fp32_parameter_elements"] * 4
    errors = [v["mae_vs_pytorch_fp32"] for v in report["validation"]]
    maximum = max(v["max_error_vs_pytorch_fp32"] for v in report["validation"])
    (output / "README.md").write_text(
        "# SmolVLA FP32 ONNX\n\n"
        f"`smolvla_fp32.onnx`：{report['bytes']:,} 字节，ONNX opset 18，固定形状。"
        "直接从原始训练 checkpoint 加载，BF16 参数逐值提升至 FP32；不使用 INT8/FP8 模型反量化。"
        "所有浮点参数与计算均为 FP32；token/位置索引及布尔掩码仍使用必要的整数/BOOL 类型。\n\n"
        "没有量化算子、自定义算子、FP16/BF16/FP8/FP64 张量。可直接由标准 ONNX Runtime 加载，"
        "无需自定义运行库。导出期间关闭 AMP/TF32，并对前向输出 dtype 做 FP32 检查。"
        "原时间嵌入函数内部使用 FP64，导出进程中单独替换为 FP32 实现；仓库策略代码未修改。\n\n"
        f"{len(report['validation'])} 组输入已在 ONNX Runtime CPU 验证，相对 PyTorch FP32 参考的动作 MAE "
        f"为 {min(errors):.8g}–{max(errors):.8g}，最大绝对误差 {maximum:.8g}。"
        "这是导出数值一致性检查，不是机器人任务成功率验证。\n\n"
        "输入：wrist/front 为 RGB FP32 [1,3,480,640]、范围 [0,1]；state 为已归一化 FP32 [1,6]；"
        "language_tokens 为 INT64 [1,48]，language_mask 为 BOOL [1,48]；noise 为 FP32 [1,50,32]。"
        "输出 actions 为归一化 FP32 [1,50,6]。包含两路图像处理、上下文编码、KV 缓存复用和十步 Euler 更新。"
        "外部 tokenizer、状态归一化和动作反归一化使用附带 sidecar 文件。\n\n"
        "标准 ONNX Runtime 调用：\n\n```python\n"
        "import numpy as np\nimport onnxruntime as ort\n"
        "session = ort.InferenceSession('smolvla_fp32.onnx', providers=['CPUExecutionProvider'])\n"
        "with np.load('validation_fp32_0.npz') as inputs:\n"
        "    actions = session.run(['actions'], {v.name: inputs[v.name] for v in session.get_inputs()})[0]\n```\n\n"
        "本机已验证环境运行（输出文件必须不存在）：\n\n```bash\n"
        "PYTHONPATH=/home/ial-mohd/lerobot/outputs/tools/onnx_fp32_deps "
        "/home/ial-mohd/workspace/so101-lab/venvs/tools/bin/uv run --no-project "
        "--python /home/ial-mohd/miniforge3/envs/lerobot/bin/python python "
        f"{output / 'run_smolvla_fp32_onnx.py'} --model {path} "
        f"--inputs {output / 'validation_fp32_0.npz'} --output {output / 'actions_check.npz'}\n```\n"
    )
    (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-assets", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True, help="Directory of original six-input validation_*.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-frames", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.00001)
    parser.add_argument("--rtol", type=float, default=0.00001)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validation_frames < 1:
        parser.error("Need at least one validation frame")
    if args.output.exists() and any(args.output.iterdir()) and not args.validate_only:
        parser.error("Output is nonempty; choose a new directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "smolvla_fp32.onnx"
    report_path = output / "export_report.json"
    if not args.validate_only:
        source_hash = sha256(checkpoint / "model.safetensors")
        policy, missing = load_original(checkpoint, args.vlm_assets.resolve(), args.device)
        wrapper = ActionChunk(policy).eval()
        sample_paths = sorted(args.examples.glob("validation_*.npz"))[:args.validation_frames]
        if not sample_paths:
            parser.error("No recorded inputs found")
        examples = []
        for sample_path in sample_paths:
            with np.load(sample_path, allow_pickle=False) as arrays:
                examples.append({name: arrays[name] for name in NAMES})
        original_refs = []
        with torch.inference_mode():
            for example in examples:
                tensors = tuple(torch.from_numpy(example[name]).to(args.device) for name in NAMES)
                original_refs.append(wrapper(*tensors).cpu().numpy())
        specialize_vision(policy)
        references = []
        with torch.inference_mode():
            for index, example in enumerate(examples):
                tensors = tuple(torch.from_numpy(example[name]).to(args.device) for name in NAMES)
                reference = wrapper(*tensors).cpu().numpy()
                np.testing.assert_array_equal(reference, original_refs[index])
                references.append(output / f"pytorch_reference_{index}.npz")
                np.savez(references[-1], **example, pytorch_fp32_actions=reference)
        print("Original checkpoint FP32 references prepared; fixed vision is bit-exact", flush=True)
        with torch.inference_mode():
            torch.onnx.export(wrapper, tuple(torch.from_numpy(examples[0][n]).to(args.device) for n in NAMES),
                              str(path), input_names=list(NAMES), output_names=["actions"],
                              opset_version=18, dynamo=False, do_constant_folding=False, external_data=False)
        del policy, wrapper, tensors
        gc.collect()
        torch.cuda.empty_cache()
        print("Standard FP32 ONNX export completed", path.stat().st_size, "bytes", flush=True)
        report = {"source_checkpoint": str(checkpoint), "source_weights_sha256": source_hash,
                  "source_is_quantized": False, "precision": "FP32", "opset": 18,
                  "time_embedding_precision": "FP32 export-local replacement of original FP64 helper",
                  "forward_dtype_guard": True, "tf32_enabled": False, "amp_enabled": False,
                  "standard_onnxruntime_supported": True, "fixed_shapes": True,
                  "input_names": list(NAMES), "output_shape": [1, 50, 6],
                  "output": str(path), "bytes": path.stat().st_size,
                  "unused_missing_lm_head_keys": missing, "vision_specialization_bit_exact": True,
                  "validation_status": "pending", "runtime_validation_passed": False,
                  "tolerances": {"atol": args.atol, "rtol": args.rtol}}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    else:
        report = json.loads(report_path.read_text())
        references = sorted(output.glob("pytorch_reference_*.npz"))[:args.validation_frames]
    report["validation_status"] = "running"
    report["runtime_validation_passed"] = False
    report.pop("validation_error", None)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report.update(audit(path, checkpoint))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        report["validation"] = validate(path, references, output, args.atol, args.rtol, report)
    except Exception as error:
        report["validation_status"] = "failed"
        report["validation_error"] = str(error)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["runtime_validation_passed"] = True
    report["validation_status"] = "passed"
    report["sha256"] = sha256(path)
    report["source_checkpoint_unchanged"] = sha256(checkpoint / "model.safetensors") == report["source_weights_sha256"]
    for name in ("tokenizer", "policy_preprocessor.json", "policy_postprocessor.json",
                 "policy_preprocessor_step_5_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        source = checkpoint / name
        if source.is_dir():
            shutil.copytree(source, output / name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, output / name)
    shutil.copy2(Path(__file__).with_name("run_smolvla_fp32_onnx.py"), output / "run_smolvla_fp32_onnx.py")
    finish_package(output, report)
    print("FP32 export and validation completed", flush=True)


if __name__ == "__main__":
    main()
