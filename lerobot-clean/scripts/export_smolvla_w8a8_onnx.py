#!/usr/bin/env python
"""Export the custom SmolVLA W8A8 bundle as standard ONNX integer operators."""

import argparse
import collections
import json
import os
import shutil
import time
from pathlib import Path
from types import MethodType

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto as T, helper, numpy_helper
from torch.onnx.symbolic_helper import _is_none
from torch.overrides import TorchFunctionMode

from scripts.quantize_smolvla_w8a8 import (
    load_reference,
    quantize_activation,
    sha256,
    simulated_int8_matmul,
)


def constant(g, value, dtype=torch.float32):
    return g.op("Constant", value_t=torch.tensor(value, dtype=dtype))


def dynamic_codes(g, value, axis, unsigned=False):
    value = g.op("Cast", value, to_i=T.FLOAT)
    maximum = g.op("ReduceMax", g.op("Abs", value), constant(g, [axis], torch.int64), keepdims_i=1)
    scale = g.op("Where", g.op("Greater", maximum, constant(g, 0.0)),
                 g.op("Div", maximum, constant(g, 127.0)), constant(g, 1.0))
    rounded = g.op("Round", g.op("Div", value, scale))
    clipped = g.op("Clip", rounded, constant(g, -127.0), constant(g, 127.0))
    if unsigned:
        clipped = g.op("Add", clipped, constant(g, 128.0))
    return g.op("Cast", clipped, to_i=T.UINT8 if unsigned else T.INT8), scale


def integer_product(g, left, right):
    # Offset-binary UINT8 encoding of the signed left input improves ORT portability.
    # With a_zero_point=128 this is exactly the same signed INT8 dot product.
    acc = g.op("MatMulInteger", left, right,
               constant(g, 128, torch.uint8), constant(g, 0, torch.int8))
    return g.op("Cast", acc, to_i=T.FLOAT)


def cast_like_source(g, result, source):
    dtype = T.BFLOAT16 if source.type().scalarType() == "BFloat16" else T.FLOAT
    return g.op("Cast", result, to_i=dtype)


class IntegerLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, codes, weight_scale, bias):
        maximum = value.float().abs().amax(-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
        qvalue = quantize_activation(value, scale)
        result = F.linear(qvalue.float(), codes.float()) * scale * weight_scale
        if bias is not None:
            result = result + bias.float()
        return result.to(value.dtype)

    @staticmethod
    def symbolic(g, value, codes, weight_scale, bias):
        qvalue, scale = dynamic_codes(g, value, -1, unsigned=True)
        transposed = g.op("Transpose", codes, perm_i=[1, 0])
        result = integer_product(g, qvalue, transposed)
        result = g.op("Mul", g.op("Mul", result, scale), weight_scale)
        if not _is_none(bias):
            result = g.op("Add", result, g.op("Cast", bias, to_i=T.FLOAT))
        return cast_like_source(g, result, value)


class IntegerAttentionMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        return simulated_int8_matmul(left, right)

    @staticmethod
    def symbolic(g, left, right):
        qleft, left_scale = dynamic_codes(g, left, -1, unsigned=True)
        qright, right_scale = dynamic_codes(g, right, -2)
        result = integer_product(g, qleft, qright)
        result = g.op("Mul", g.op("Mul", result, left_scale), right_scale)
        return cast_like_source(g, result, left)


class AttentionExportMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.matmul:
            return IntegerAttentionMatmul.apply(*args)
        return func(*args, **(kwargs or {}))


class QuantizedLinear(torch.nn.Module):
    def __init__(self, original, codes, scale):
        super().__init__()
        self.register_buffer("codes", codes.to(original.weight.device))
        self.register_buffer("weight_scale", scale.to(original.weight.device))
        self.register_buffer("bias", original.bias)
        self.register_buffer("dtype_marker", torch.empty(0, device=original.weight.device, dtype=original.weight.dtype))

    @property
    def weight(self):
        # The SmolVLA attention implementation reads projection.weight.dtype.
        return self.dtype_marker

    def forward(self, value):
        return IntegerLinear.apply(value, self.codes, self.weight_scale, self.bias)


def replace_linears(policy, tensors, scales, names):
    for name, module in list(policy.named_modules()):
        weight_name = f"{name}.weight"
        if isinstance(module, torch.nn.Linear) and weight_name in names:
            replacement = QuantizedLinear(module, tensors[weight_name], scales[f"{name}.weight_scale"])
            parent_name, _, child = name.rpartition(".")
            parent = policy.get_submodule(parent_name) if parent_name else policy
            setattr(parent, child, replacement)


def specialize_fixed_vision(policy):
    """All pixels/patches are present in this fixed 512x512 camera interface."""
    from transformers.modeling_outputs import BaseModelOutput

    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    side = 512 // vision.patch_size
    if side != vision.embeddings.num_patches_per_side:
        raise ValueError("Fixed vision grid does not match the learned position grid")
    vision.register_buffer("onnx_position_ids", torch.arange(side * side, device=next(vision.parameters()).device)[None])

    def forward(self, pixel_values, patch_attention_mask=None, **kwargs):
        if patch_attention_mask is not None:
            raise ValueError("This export specializes the all-valid patch mask")
        patches = self.embeddings.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        hidden = patches + self.embeddings.position_embedding(self.onnx_position_ids)
        hidden = self.encoder(inputs_embeds=hidden, attention_mask=None).last_hidden_state
        return BaseModelOutput(last_hidden_state=self.post_layernorm(hidden))

    vision.forward = MethodType(forward, vision)


class ActionChunk(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, wrist, front, state, language_tokens, language_mask, noise):
        batch = {
            "observation.images.wrist": wrist,
            "observation.images.front": front,
            "observation.state": state,
            "observation.language.tokens": language_tokens,
            "observation.language.attention_mask": language_mask,
        }
        with AttentionExportMode():
            return self.policy.predict_action_chunk(batch, noise=noise)


def cpu_compatible_bf16(model):
    """Run BF16 operations as FP32 with explicit BF16 rounding at each output.

    BF16 initializers remain BF16 on disk. CPU Cast supports BF16 even where
    CPU arithmetic kernels do not. This preserves rounding boundaries rather
    than silently upgrading the complete floating-point path to FP32.
    """
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    for initializer in model.graph.initializer:
        types[initializer.name] = initializer.data_type
    nodes = []
    for initializer in model.graph.initializer:
        if initializer.data_type == T.BFLOAT16:
            original_name = initializer.name
            initializer.name += "__bf16_storage"
            nodes.append(helper.make_node("Cast", [initializer.name], [original_name], to=T.FLOAT))
    for node in model.graph.node:
        is_bf16_cast = node.op_type == "Cast" and any(a.name == "to" and a.i == T.BFLOAT16 for a in node.attribute)
        if is_bf16_cast:
            original_name = node.output[0]
            node.output[0] = original_name + "__bf16_cast"
            nodes.append(node)
            nodes.append(helper.make_node("Cast", [node.output[0]], [original_name], to=T.FLOAT))
            continue
        converted_constant = False
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == T.BFLOAT16:
                array = numpy_helper.to_array(attribute.t).astype(np.float32)
                attribute.t.CopyFrom(numpy_helper.from_array(array))
                converted_constant = True
        rounded_outputs = []
        for index, name in enumerate(node.output):
            if types.get(name) == T.BFLOAT16 and not (node.op_type == "Constant" and converted_constant):
                node.output[index] = name + "__float_compute"
                rounded_outputs.append((name, node.output[index]))
        nodes.append(node)
        for name, raw in rounded_outputs:
            rounded = name + "__bf16_round"
            nodes.append(helper.make_node("Cast", [raw], [rounded], to=T.BFLOAT16))
            nodes.append(helper.make_node("Cast", [rounded], [name], to=T.FLOAT))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == T.BFLOAT16:
            value.type.tensor_type.elem_type = T.FLOAT
    return model


def repair_integer_reductions(model):
    """Torch promotes bool/small-integer cumulative sums to INT64."""
    model = onnx.shape_inference.infer_shapes(model)
    types = {value.name: value.type.tensor_type.elem_type
             for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
             if value.type.HasField("tensor_type")}
    types.update({value.name: value.data_type for value in model.graph.initializer})
    nodes = []
    promoted = set()
    for node in model.graph.node:
        if node.op_type in ("CumSum", "ReduceSum", "ReduceProd") and types.get(node.input[0]) in (
            T.BOOL, T.INT8, T.UINT8, T.INT16, T.UINT16,
        ):
            original = node.input[0]
            cast_name = node.output[0] + "__int64_input"
            nodes.append(helper.make_node("Cast", [original], [cast_name], to=T.INT64))
            node.input[0] = cast_name
            promoted.update(node.output)
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Discard stale inferred types on downstream Unsqueeze/etc.; infer them again
    # after the promotion instead of retaining the exporter's incorrect BOOL type.
    del model.graph.value_info[:]
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-frames", type=int, default=3)
    parser.add_argument("--resume-export", action="store_true", help="Reuse the already traced native graph")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_export:
        parser.error("Output is nonempty; choose a new directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    started = time.monotonic()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "quantization.json").read_text())
    if manifest["linear_activation_scheme"] != "per_token":
        raise ValueError("This exporter currently requires the per_token W8A8 bundle")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy, reference_mode = load_reference(bundle, args.device)
    pre, _ = make_pre_post_processors(
        policy.config, str(bundle),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset_info = manifest["calibration"]["dataset"]
    dataset = LeRobotDataset(dataset_info["repo_id"], root=dataset_info["root"], video_backend="pyav")
    frames = manifest["validation"]["frame_indices"][:args.validation_frames]
    if not frames:
        raise ValueError("Need at least one validation frame")

    def inputs(index, seed):
        batch = pre(dataset[index])
        noise = torch.randn(1, 50, 32, device=args.device,
                            generator=torch.Generator(device=args.device).manual_seed(seed))
        return (batch["observation.images.wrist"], batch["observation.images.front"],
                batch["observation.state"], batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK], noise)

    examples = [inputs(frame, 2000 + i) for i, frame in enumerate(frames)]
    references = []
    with torch.inference_mode():
        for values in examples:
            batch = {"observation.images.wrist": values[0], "observation.images.front": values[1],
                     "observation.state": values[2], OBS_LANGUAGE_TOKENS: values[3],
                     OBS_LANGUAGE_ATTENTION_MASK: values[4]}
            with reference_mode:
                references.append(policy.predict_action_chunk(batch, noise=values[5]).cpu().numpy())
    tensors = load_file(bundle / "model.int8.safetensors")
    scales = load_file(bundle / "quantization_scales.safetensors")
    replace_linears(policy, tensors, scales, manifest["linear_weights"])
    specialize_fixed_vision(policy)
    del tensors, scales, reference_mode
    wrapper = ActionChunk(policy).eval()
    with torch.inference_mode():
        for values, reference in zip(examples, references, strict=True):
            actual = wrapper(*values).cpu().numpy()
            np.testing.assert_array_equal(actual, reference)
    print("Export wrapper exactly matches the original W8A8 reference", flush=True)
    names = ["wrist", "front", "state", "language_tokens", "language_mask", "noise"]
    native_path = output / "smolvla_w8a8_native_bf16.onnx"
    if not args.resume_export:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, examples[0], str(native_path),
                input_names=names, output_names=["actions"],
                opset_version=18, dynamo=False, do_constant_folding=False,
                external_data=False,
            )
    print("Native BF16 ONNX export completed", flush=True)
    model = repair_integer_reductions(onnx.load(native_path))
    onnx.save_model(model, native_path)
    onnx.checker.check_model(str(native_path))
    model = cpu_compatible_bf16(model)
    model.producer_name = "lerobot-smolvla-w8a8-export"
    helper.set_model_props(model, {
        "quantization": "INT8 weights; dynamic INT8 matrix inputs; INT32 MatMulInteger accumulation",
        "left_operand_encoding": "UINT8(q_signed+128), zero_point=128; mathematically signed INT8",
        "floating_precision": "Original BF16 rounding retained using explicit FP32->BF16->FP32 boundaries",
        "source_bundle": str(bundle), "euler_steps": "10", "chunk_size": "50",
    })
    path = output / "smolvla_w8a8.onnx"
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path))
    counts = dict(collections.Counter(node.op_type for node in model.graph.node))
    int8_initializers = sum(v.data_type == T.INT8 for v in model.graph.initializer)
    if not counts.get("MatMulInteger") or not int8_initializers:
        raise ValueError("Export lost integer matrix multiplication or INT8 weights")
    io = {"inputs": [{"name": n, "shape": list(v.shape), "dtype": str(v.cpu().numpy().dtype)}
                     for n, v in zip(names, examples[0], strict=True)],
          "outputs": [{"name": "actions", "shape": [1, 50, 6], "dtype": "float32"}]}
    del model
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    validation = []
    for i, (frame, values, reference) in enumerate(zip(frames, examples, references, strict=True)):
        feed = {name: value.cpu().numpy() for name, value in zip(names, values, strict=True)}
        actual = session.run(["actions"], feed)[0]
        if not np.isfinite(actual).all():
            raise ValueError("ONNX produced non-finite actions")
        error = np.abs(actual - reference)
        validation.append({"frame": frame, "noise_seed": 2000 + i,
                           "action_mae_vs_w8a8_reference": float(error.mean()),
                           "action_max_abs_error_vs_w8a8_reference": float(error.max())})
        np.savez(output / f"validation_{i}.npz", **feed, reference_actions=reference, onnx_actions=actual)
        print("ONNX Runtime validation", validation[-1], flush=True)
    report = {"source_bundle": str(bundle), "opset": 18, "fixed_shapes": True, **io,
              "int8_weight_initializers": int8_initializers, "operator_counts": counts,
              "validation": validation, "onnx_checker_passed": True,
              "notes": ["Images are float32 RGB [0,1]; image resize/padding and [-1,1] conversion are inside the graph.",
                        "State is normalized with the supplied policy preprocessor; tokenize and pad instructions externally to 48 tokens.",
                        "Noise is external standard Gaussian float32 [1,50,32]. Output is normalized actions [1,50,6].",
                        "Graph includes context encoding, KV reuse, all 10 Euler steps, and action dimension selection.",
                        "Native BF16 graph is also retained; CPU-compatible graph preserves BF16 storage and rounding boundaries.",
                        "MatMulInteger performs exact INT32 accumulation; the previous W8A8 reference simulates products using FP32 GEMM, so small differences are expected.",
                        "No FPGA bitstream or FPGA performance measurement is included."],
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (path, native_path)},
              "elapsed_seconds": time.monotonic() - started}
    (output / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for sidecar in ("tokenizer", "policy_preprocessor.json", "policy_postprocessor.json",
                    "policy_preprocessor_step_5_normalizer_processor.safetensors",
                    "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        source = bundle / sidecar
        if source.is_dir():
            shutil.copytree(source, output / sidecar)
        else:
            shutil.copy2(source, output / sidecar)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("run_smolvla_onnx.py"), output / "run_smolvla_onnx.py")
    (output / "README.md").write_text(
        "# SmolVLA W8A8 ONNX\n\n"
        "`smolvla_w8a8.onnx` is a complete fixed-shape action-chunk graph (ONNX opset 18). "
        "It includes both cameras, context encoding, cached attention, and all ten Euler updates. "
        "Main matrix products use MatMulInteger with INT8 weight tensors and dynamic activation quantization. "
        "The left operand uses equivalent offset-binary UINT8 encoding (zero point 128) for CPU runtime support. "
        "Other computation remains floating-point, with original BF16 rounding boundaries retained. "
        "`smolvla_w8a8_native_bf16.onnx` retains native BF16 arithmetic operators for supporting hardware.\n\n"
        "Inputs: wrist/front RGB float32 [1,3,480,640] in [0,1]; normalized state float32 [1,6]; "
        "language_tokens int64 [1,48]; language_mask bool [1,48]; standard-normal noise float32 [1,50,32]. "
        "Output: normalized actions float32 [1,50,6]. Tokenization/state normalization and action unnormalization "
        "use the supplied sidecars outside the graph. All shapes, keys, checksums and numerical comparisons "
        "are recorded in export_report.json.\n\n"
        "To run a recorded example: create an onnxruntime.InferenceSession for smolvla_w8a8.onnx, load "
        "validation_0.npz, and pass only the six named input arrays to session.run. "
        "For precision verification disable graph optimizations as in the exporter. "
        "The native BF16 file has passed ONNX checker; runtime numerical validation uses the CPU-compatible file. "
        "The model still requires an HLS/PL implementation or a compatible compiler for FPGA deployment.\n"
    )
    print(json.dumps({"output": str(path), "bytes": path.stat().st_size, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
