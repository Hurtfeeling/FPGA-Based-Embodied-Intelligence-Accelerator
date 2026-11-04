#!/usr/bin/env python
"""Export calibrated SmolVLA W8A8 tensors for a custom HLS/PL implementation.

This is an explicit hardware interchange format, not a standard LeRobot checkpoint.
The included TorchFunctionMode is a floating-point simulation of quantized matrix
products for numerical validation; it is not an accelerated INT8 inference backend.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.overrides import TorchFunctionMode


def quantize_weight(weight):
    """Symmetric per-output-channel INT8, nearest-even rounding, zero point 0."""
    weight = weight.detach().float()
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("Expected a finite 2D Linear weight")
    maximum = weight.abs().amax(dim=1)
    scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    codes = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return codes, scale


def quantize_activation(value, scale):
    return torch.round(value.float() / scale).clamp(-127, 127).to(torch.int8)


def simulated_int8_matmul(left, right):
    """Dynamic row/column activation scales; dequantize the INT32 product.

    Both operands are INT8, including Softmax output when it feeds the PV
    product. Softmax itself remains unchanged. Float32 GEMM simulates the
    integer product here; it is not guaranteed bit-exact for large reductions.
    """
    left_max = left.float().abs().amax(dim=-1, keepdim=True)
    right_max = right.float().abs().amax(dim=-2, keepdim=True)
    left_scale = torch.where(left_max > 0, left_max / 127, torch.ones_like(left_max))
    right_scale = torch.where(right_max > 0, right_max / 127, torch.ones_like(right_max))
    qleft = quantize_activation(left, left_scale)
    qright = quantize_activation(right, right_scale)
    product = torch.matmul(qleft.float(), qright.float())
    return (product * left_scale * right_scale).to(left.dtype)


class W8A8Mode(TorchFunctionMode):
    """Observe Linear inputs, or simulate Linear and Attention INT8 products."""

    def __init__(self, policy, weight_names, quantized=None, scales=None, activation_scheme="static"):
        super().__init__()
        self.names = {}
        self.maximum = {}
        self.calls = {}
        self.matmul_calls = 0
        self.quantized = quantized
        self.scales = scales
        self.device_weights = {}
        self.device_scales = {}
        self.clipped = {}
        self.elements = {}
        if activation_scheme not in ("static", "per_token"):
            raise ValueError(f"Unknown activation scheme: {activation_scheme}")
        self.activation_scheme = activation_scheme
        for name, module in policy.named_modules():
            if isinstance(module, torch.nn.Linear) and f"{name}.weight" in weight_names:
                self.names[id(module.weight)] = name
                if quantized is not None:
                    device = module.weight.device
                    self.device_weights[name] = quantized[f"{name}.weight"].to(device).float()
                    self.device_scales[name] = (
                        scales[f"{name}.weight_scale"].to(device),
                        scales[f"{name}.input_scale"].to(device) if f"{name}.input_scale" in scales else None,
                    )
        if len(self.names) != len(weight_names):
            raise ValueError("Some exported weights could not be mapped to nn.Linear modules")

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            value = args[0] if args else kwargs["input"]
            weight = args[1] if len(args) > 1 else kwargs["weight"]
            name = self.names.get(id(weight))
            if name is not None:
                self.calls[name] = self.calls.get(name, 0) + 1
                if self.quantized is None:
                    maximum = value.detach().float().abs().amax()
                    self.maximum[name] = torch.maximum(self.maximum.get(name, maximum), maximum)
                else:
                    weight_scale, input_scale = self.device_scales[name]
                    if input_scale is None:
                        raise RuntimeError(f"Layer {name} was inactive during action calibration")
                    if self.activation_scheme == "per_token":
                        maximum = value.float().abs().amax(dim=-1, keepdim=True)
                        input_scale = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
                    bias = args[2] if len(args) > 2 else kwargs.get("bias")
                    scaled = value.float() / input_scale
                    clipped = (scaled.abs() > 127).sum()
                    self.clipped[name] = self.clipped.get(name, 0) + clipped
                    self.elements[name] = self.elements.get(name, 0) + value.numel()
                    qvalue = quantize_activation(value, input_scale)
                    out = F.linear(qvalue.float(), self.device_weights[name])
                    out = out * input_scale * weight_scale
                    if bias is not None:
                        out = out + bias.float()
                    return out.to(value.dtype)
        elif func is torch.matmul:
            self.matmul_calls += 1
            if self.quantized is not None:
                return simulated_int8_matmul(*args, **kwargs)
        return func(*args, **kwargs)


def force_eager_vision(policy):
    """Expose vision QK/PV products instead of bypassing them through SDPA."""
    count = 0
    for module in policy.modules():
        if type(module).__name__ == "SmolVLMVisionAttention":
            module.config._attn_implementation = "eager"
            count += 1
    if not count:
        raise ValueError("Unsupported vision Attention implementation")


def load_reference(bundle, device="cuda"):
    """Load the self-contained export and return (policy, W8A8 simulation mode).

    Use ``with torch.inference_mode(), mode: policy.predict_action_chunk(...)``.
    No original checkpoint or Hub model weights are needed.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "quantization.json").read_text())
    config = PreTrainedConfig.from_pretrained(bundle)
    config.device = device
    config.load_vlm_weights = False
    config.vlm_model_name = str(bundle / "vlm_assets")
    policy = SmolVLAPolicy(config).eval()
    policy.requires_grad_(False)
    tensors = load_file(bundle / "model.int8.safetensors")
    scales = load_file(bundle / "quantization_scales.safetensors")
    restored = {}
    names = manifest["linear_weights"]
    for name, value in tensors.items():
        if name in names:
            dtype = getattr(torch, names[name]["original_dtype"])
            restored[name] = (value.float() * scales[name.removesuffix(".weight") + ".weight_scale"][:, None]).to(dtype)
        else:
            restored[name] = value
    missing, unexpected = policy.load_state_dict(restored, strict=False, assign=True)
    if unexpected or any(not key.endswith("lm_head.weight") for key in missing):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    policy.to(device)
    force_eager_vision(policy)
    mode = W8A8Mode(policy, names, tensors, scales, manifest.get("linear_activation_scheme", "static"))
    return policy, mode


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_frames(dataset, calibration_count, validation_count):
    """Deterministic frame selection with disjoint calibration/validation episodes."""
    episodes = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    unique = np.unique(episodes)
    if len(unique) < 2:
        raise ValueError("Need at least two episodes for disjoint calibration/validation")
    rng = np.random.default_rng(42)
    unique = rng.permutation(unique)
    split = max(1, min(len(unique) - 1, int(len(unique) * 0.8)))
    cal_candidates = np.flatnonzero(np.isin(episodes, unique[:split]))
    val_candidates = np.flatnonzero(np.isin(episodes, unique[split:]))

    def spread(candidates, count):
        if len(candidates) < count:
            raise ValueError("Requested more unique frames than available")
        return candidates[np.linspace(0, len(candidates) - 1, count, dtype=np.int64)].tolist()

    return spread(cal_candidates, calibration_count), spread(val_candidates, validation_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-frames", type=int, default=64)
    parser.add_argument("--validation-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--linear-activation-scheme", choices=("static", "per_token"), default="per_token")
    args = parser.parse_args()
    if min(args.calibration_frames, args.validation_frames) < 1:
        parser.error("Frame counts must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    started = time.monotonic()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    source_hash = sha256(checkpoint / "model.safetensors")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True).to(args.device).eval()
    policy.requires_grad_(False)
    force_eager_vision(policy)
    pre, post = make_pre_post_processors(
        policy.config, str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    training = json.loads((checkpoint / "train_config.json").read_text())
    dataset_config = training["dataset"]
    dataset = LeRobotDataset(dataset_config["repo_id"], root=dataset_config["root"], video_backend="pyav")
    cal_frames, val_frames = choose_frames(dataset, args.calibration_frames, args.validation_frames)
    tensors = load_file(checkpoint / "model.safetensors")
    linear_names = {
        f"{name}.weight" for name, module in policy.named_modules()
        if isinstance(module, torch.nn.Linear) and f"{name}.weight" in tensors
    }
    observer = W8A8Mode(policy, linear_names)

    def predict(frame_index, seed, mode=None):
        batch = pre(dataset[frame_index])
        generator = torch.Generator(device=args.device).manual_seed(seed)
        noise = torch.randn(
            1, policy.config.chunk_size, policy.config.max_action_dim,
            generator=generator, device=args.device,
        )
        policy.reset()
        if mode is None:
            action = policy.predict_action_chunk(batch, noise=noise)
        else:
            with mode:
                action = policy.predict_action_chunk(batch, noise=noise)
        if not torch.isfinite(action).all():
            raise ValueError(f"Non-finite action at frame {frame_index}")
        return action.detach().cpu(), post(action).detach().cpu()

    with torch.inference_mode():
        for i, frame in enumerate(cal_frames):
            predict(frame, 1000 + i, observer)
            if (i + 1) % 8 == 0 or i + 1 == len(cal_frames):
                print(f"Calibration {i+1}/{len(cal_frames)}", flush=True)
    inactive = sorted({name.removesuffix(".weight") for name in linear_names} - set(observer.maximum))
    # SmolVLA does not call the VLM language-generation head when producing actions.
    if any(not name.endswith(".lm_head") for name in inactive):
        raise ValueError(f"Calibration did not execute required Linear layers: {inactive}")
    if observer.matmul_calls == 0:
        raise ValueError("Attention matrix products were not observed")
    calibration_matmul_calls = observer.matmul_calls

    scales = {}
    metadata = {}
    for name in sorted(linear_names):
        original = tensors[name]
        codes, weight_scale = quantize_weight(original.to(args.device))
        layer = name.removesuffix(".weight")
        scales[f"{layer}.weight_scale"] = weight_scale.cpu()
        if layer not in inactive:
            maximum = observer.maximum[layer].cpu()
            if not torch.isfinite(maximum):
                raise ValueError(f"Non-finite calibration range: {layer}")
            scales[f"{layer}.input_scale"] = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
        metadata[name] = {"shape": list(original.shape), "original_dtype": str(original.dtype).split(".")[-1]}
        tensors[name] = codes.cpu()
    simulation = W8A8Mode(policy, linear_names, tensors, scales, args.linear_activation_scheme)
    normalized_errors = []
    physical_errors = []
    reference_outputs = {}
    with torch.inference_mode():
        for i, frame in enumerate(val_frames):
            baseline, baseline_physical = predict(frame, 2000 + i)
            quantized, quantized_physical = predict(frame, 2000 + i, simulation)
            normalized_errors.append(quantized - baseline)
            physical_errors.append(quantized_physical - baseline_physical)
            reference_outputs[f"frame_{frame}.float_actions"] = baseline
            reference_outputs[f"frame_{frame}.w8a8_actions"] = quantized
            print(f"Validation {i+1}/{len(val_frames)} MAE={(quantized-baseline).abs().mean().item():.6f}", flush=True)
    normalized_errors = torch.cat(normalized_errors)
    physical_errors = torch.cat(physical_errors)
    clipping = {
        name: float(value.item()) / simulation.elements[name]
        for name, value in simulation.clipped.items()
    }
    metrics = {
        "normalized_action_mae": normalized_errors.abs().mean().item(),
        "normalized_action_rmse": normalized_errors.square().mean().sqrt().item(),
        "normalized_action_max_abs_error": normalized_errors.abs().max().item(),
        "physical_action_mae_per_joint": physical_errors.abs().mean(dim=(0, 1)).tolist(),
        "physical_action_max_abs_error_per_joint": physical_errors.abs().amax(dim=(0, 1)).tolist(),
        "linear_input_clipping_fraction_per_layer": clipping,
        "max_linear_input_clipping_fraction": max(clipping.values()),
        "all_outputs_finite": True,
        "validation_type": "offline output comparison, not robot success-rate evaluation",
    }
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    for item in checkpoint.iterdir():
        if item.name == "model.safetensors":
            continue
        if item.is_dir():
            shutil.copytree(item, staging / item.name)
        else:
            shutil.copy2(item, staging / item.name)
    policy.model.vlm_with_expert.vlm.config.save_pretrained(staging / "vlm_assets")
    policy.model.vlm_with_expert.processor.save_pretrained(staging / "vlm_assets")
    save_file(tensors, staging / "model.int8.safetensors", metadata={"format": "smolvla-hls-w8a8-v1"})
    save_file(scales, staging / "quantization_scales.safetensors")
    save_file(reference_outputs, staging / "validation_outputs.safetensors")
    report = {
        "format": "smolvla-hls-w8a8-v1",
        "source_checkpoint": str(checkpoint),
        "source_weights_sha256": source_hash,
        "weight_quantization": "symmetric INT8 [-127,127], per-output-channel FP32 scale, zero_point=0",
        "linear_activation_scheme": args.linear_activation_scheme,
        "linear_activation_quantization": (
            "dynamic per-token INT8; absmax over the last (input feature) dimension / 127; zero rows use scale 1"
            if args.linear_activation_scheme == "per_token"
            else "static per-tensor INT8, calibration absolute maximum / 127"
        ),
        "stored_linear_input_scales": "calibrated static scales; reference only when per_token scheme is selected",
        "attention_activation_quantization": "dynamic INT8; lhs per row, rhs per column; both QK and PV",
        "rounding": "nearest, ties to even; saturate to [-127,127]",
        "integer_accumulation_target": "INT32; apply lhs/rhs scales, then floating-point bias",
        "reference_backend": "FP32 matrix multiplication of INT8 codes, not an accelerated INT8 kernel or a bit-exact INT32 emulator",
        "floating_operations": ["Softmax", "normalization", "RoPE", "time embedding", "activations", "residuals", "Euler integration", "input/output processing"],
        "unquantized_weights": "Embedding tables, patch Conv2d, normalization parameters, biases; original BF16/FP32 storage",
        "linear_weights": metadata,
        "inactive_linear_layers": inactive,
        "calibration": {"dataset": dataset_config, "frame_indices": cal_frames, "seed": 42},
        "validation": {"frame_indices": val_frames, "noise_seed_start": 2000, "metrics": metrics},
        "torch_version": torch.__version__,
    }
    for filename in ("model.int8.safetensors", "quantization_scales.safetensors"):
        report.setdefault("files", {})[filename] = {
            "bytes": (staging / filename).stat().st_size,
            "sha256": sha256(staging / filename),
        }
    (staging / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")

    # Verify serialization against the exact source, not just a plausible file size.
    reread = load_file(staging / "model.int8.safetensors")
    source = load_file(checkpoint / "model.safetensors")
    for name, value in reread.items():
        if name in linear_names:
            if value.dtype != torch.int8 or not torch.equal(value, tensors[name]):
                raise ValueError(f"INT8 round-trip failed: {name}")
        elif not torch.equal(value, source[name]):
            raise ValueError(f"Floating-point tensor changed: {name}")
    if sha256(checkpoint / "model.safetensors") != source_hash:
        raise ValueError("Source checkpoint changed during export")
    del reread, source, simulation, observer

    # Reconstruct from the export's config/assets and serialized INT8 tensors.
    loaded_policy, loaded_mode = load_reference(staging, args.device)
    batch = pre(dataset[val_frames[0]])
    generator = torch.Generator(device=args.device).manual_seed(2000)
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=generator, device=args.device)
    with torch.inference_mode(), loaded_mode:
        reloaded = loaded_policy.predict_action_chunk(batch, noise=noise).cpu()
    expected = reference_outputs[f"frame_{val_frames[0]}.w8a8_actions"]
    torch.testing.assert_close(reloaded, expected, rtol=0, atol=0)
    report["verification"] = {
        "all_float_tensors_unchanged": True,
        "all_int8_tensors_roundtrip_exact": True,
        "source_checkpoint_unchanged": True,
        "standalone_reload_action_max_abs_error": (reloaded - expected).abs().max().item(),
        "calibrated_linear_layers": len(linear_names) - len(inactive),
        "int8_linear_weight_tensors": len(linear_names),
        "observed_attention_matmul_calls_during_calibration": calibration_matmul_calls,
    }
    # Count from a complete inference on reload (QK and PV for vision and action).
    report["verification"]["attention_matmul_calls_per_inference"] = loaded_mode.matmul_calls
    (staging / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(Path(__file__), staging / "quantize_smolvla_w8a8.py")
    (staging / "README.md").write_text(
        "# SmolVLA W8A8 HLS export\n\n"
        "Actual INT8 Linear weights are in `model.int8.safetensors`; FP32 weight/input scales are in "
        "`quantization_scales.safetensors`. Original tensor names and row-major [out,in] layout are retained. "
        "For a Linear: qx=clip(round_even(x/input_scale),-127,127); "
        "acc_int32=qx@qw.T; y=acc_int32*input_scale*weight_scale+bias. "
        "With per_token activation quantization, input_scale is computed at runtime as "
        "absmax(x, last_dimension)/127 (or 1 for a zero row); saved static input scales are "
        "calibration reference values, not the runtime scales. "
        "Attention uses dynamic row/column scales described in quantization.json. "
        "All other operations retain their original floating-point computation.\n\n"
        "This custom bundle cannot be passed to ordinary `SmolVLAPolicy.from_pretrained`: "
        "that loader does not implement W8A8 operators. No FPGA bitstream is included.\n\n"
        "`load_reference(bundle, device)` in the accompanying Python file returns a policy and a "
        "TorchFunctionMode. Run `with torch.inference_mode(), mode:` around "
        "`policy.predict_action_chunk(batch, noise=noise)`. This simulates quantized products in FP32 "
        "for HLS validation; it does not provide GPU INT8 acceleration or bit-exact INT32 accumulation. "
        "The export includes base-model configuration/processor assets and tokenizer, so reloading does "
        "not require the source weights. Install the same LeRobot/Transformers dependencies.\n\n"
        "Calibration/validation use disjoint episodes of the local training dataset, not an independent "
        "test dataset. Metrics compare output actions with the original model under identical noise. "
        "Robot success rate and FPGA latency have not been measured. See quantization.json for metrics, "
        "frame IDs, seeds, precision rules, byte sizes and SHA256 checksums.\n"
    )
    staging.rename(output)
    print(json.dumps({"output": str(output), "weights_and_scales_bytes": sum(v["bytes"] for v in report["files"].values()),
                      "normalized_action_mae": metrics["normalized_action_mae"],
                      "elapsed_seconds": time.monotonic() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
