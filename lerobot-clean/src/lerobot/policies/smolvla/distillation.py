"""S8 initialization and differentiable flow-matching policy distillation."""

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.import_utils import _transformers_available, require_package

from ..common.vla_utils import make_att_2d_masks
from .distillation_distributed import distributed_loss_scales
from .modeling_smolvla import SmolVLAPolicy

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig
else:
    AutoConfig = None


S8_LAYERS = (0, 1, 4, 5, 8, 9, 14, 15)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_assets(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in Path(source).iterdir():
        if item.is_file() and item.suffix in (".json", ".txt", ".model", ".tiktoken"):
            shutil.copy2(item, destination / item.name)


def create_s8_assets(source, destination):
    require_package("transformers", extra="smolvla")
    copy_assets(source, destination)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    text = config.text_config
    text.hidden_size = 576
    text.intermediate_size = 1536
    text.num_hidden_layers = 8
    text.num_attention_heads = 9
    text.num_key_value_heads = 3
    text.head_dim = 64
    for component in (config, config.text_config, config.vision_config):
        component.dtype = torch.float32
    config.save_pretrained(destination)
    # Transformers omits its private attention selector from save_pretrained.
    # Persist the public constructor option so normal policy reloads use the
    # same vision kernel as distillation and the integer ONNX export.
    config_path = Path(destination) / "config.json"
    serialized = json.loads(config_path.read_text())
    serialized["attn_implementation"] = "eager"
    serialized["vision_config"]["attn_implementation"] = "eager"
    config_path.write_text(json.dumps(serialized, indent=2) + "\n")
    return config


def eager_vision(policy):
    for module in policy.modules():
        if type(module).__name__ == "SmolVLMVisionAttention":
            module.config._attn_implementation = "eager"


def load_teacher(checkpoint, assets, device):
    checkpoint = Path(checkpoint).resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.load_vlm_weights = False
    config.vlm_model_name = str(Path(assets).resolve())
    config.device = device
    policy = SmolVLAPolicy(config)
    tensors = load_file(checkpoint / "model.safetensors")
    missing, unexpected = policy.load_state_dict(tensors, strict=False, assign=True)
    if unexpected or any(not name.endswith("lm_head.weight") for name in missing):
        raise ValueError(f"Teacher checkpoint mismatch: {missing=}, {unexpected=}")
    policy.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    eager_vision(policy)
    return policy


def s8_indices():
    groups = torch.tensor([0, 2, 4])
    heads = (groups[:, None] * 3 + torch.arange(3)).flatten()

    def channels(selected, width):
        return (selected[:, None] * width + torch.arange(width)).flatten()

    return {
        "text": channels(heads, 64),
        "query": channels(heads, 64),
        "kv": channels(groups, 64),
        "expert": channels(heads, 48),
        "text_ffn": torch.linspace(0, 2559, 1536).round().long(),
        "expert_ffn": torch.linspace(0, 2047, 1280).round().long(),
    }


def prune_tensor(name, value, target_shape, indices):
    """Apply a consistent residual/head/FFN mapping to one teacher tensor."""
    text, expert = indices["text"], indices["expert"]
    query, kv = indices["query"], indices["kv"]
    if ".vision_model." in name:
        result = value
    elif ".connector." in name:
        result = value.index_select(0, text)
    elif ".text_model.embed_tokens." in name or ".vlm.lm_head." in name:
        result = value.index_select(1, text)
    elif ".layers." in name:
        is_expert = ".lm_expert." in name
        residual = expert if is_expert else text
        ffn = indices["expert_ffn" if is_expert else "text_ffn"]
        if ".q_proj." in name:
            result = value.index_select(0, query).index_select(1, residual)
        elif ".k_proj." in name or ".v_proj." in name:
            source_input = kv if is_expert and value.shape[1] == 320 else residual
            result = value.index_select(0, kv).index_select(1, source_input)
        elif ".o_proj." in name:
            result = value.index_select(0, residual).index_select(1, query)
        elif ".gate_proj." in name or ".up_proj." in name:
            result = value.index_select(0, ffn).index_select(1, residual)
        elif ".down_proj." in name:
            result = value.index_select(0, residual).index_select(1, ffn)
        elif "layernorm.weight" in name:
            result = value.index_select(0, residual)
        elif tuple(value.shape) == tuple(target_shape):
            result = value
        else:
            raise ValueError(f"Unmapped layer tensor: {name}")
    elif ".text_model.norm." in name:
        result = value.index_select(0, text)
    elif ".lm_expert.norm." in name:
        result = value.index_select(0, expert)
    elif ".state_proj." in name:
        result = value.index_select(0, text)
    elif ".action_in_proj." in name:
        result = value.index_select(0, expert)
    elif ".action_out_proj.weight" in name:
        result = value.index_select(1, expert)
    elif ".action_time_mlp_in." in name:
        result = value.index_select(0, expert)
        if value.ndim == 2:
            result = result.index_select(1, torch.cat([expert, expert + 720]))
    elif ".action_time_mlp_out." in name:
        result = value.index_select(0, expert)
        if value.ndim == 2:
            result = result.index_select(1, expert)
    elif tuple(value.shape) == tuple(target_shape):
        result = value
    else:
        raise ValueError(f"Unmapped tensor: {name}, {tuple(value.shape)} -> {tuple(target_shape)}")
    if tuple(result.shape) != tuple(target_shape):
        raise ValueError(f"Bad S8 slice: {name}: {tuple(result.shape)} != {tuple(target_shape)}")
    return result.float().contiguous()


def configure_student_gradients(student):
    student.requires_grad_(True)
    backbone = student.model.vlm_with_expert
    backbone.vlm.model.vision_model.requires_grad_(False)
    backbone.vlm.lm_head.requires_grad_(False)
    backbone.vlm.model.text_model.norm.requires_grad_(False)
    last = backbone.vlm.model.text_model.layers[-1]
    for module in (last.self_attn.q_proj, last.self_attn.o_proj, last.mlp, last.post_attention_layernorm):
        module.requires_grad_(False)
    eager_vision(student)


def initialize_s8(teacher, assets):
    teacher_backbone = teacher.model.vlm_with_expert
    tc = teacher_backbone.config.text_config
    if (
        tc.hidden_size,
        tc.num_attention_heads,
        tc.num_key_value_heads,
        tc.head_dim,
        teacher_backbone.num_vlm_layers,
        teacher_backbone.expert_hidden_size,
    ) != (960, 15, 5, 64, 16, 720):
        raise ValueError("S8 initialization requires the audited 16-layer 960/720 SmolVLA teacher")
    if set(teacher.config.image_features) != {"observation.images.camera2", "observation.images.camera3"}:
        raise ValueError("S8 requires the wrist/beside two-camera teacher")
    if (
        teacher.config.chunk_size,
        teacher.config.max_action_dim,
        teacher.config.n_obs_steps,
        teacher.config.self_attn_every_n_layers,
        teacher_backbone.num_expert_layers,
    ) != (50, 32, 1, 2, 16):
        raise ValueError("Teacher action/history/attention layout is incompatible with S8")
    config = copy.deepcopy(teacher.config)
    config.vlm_model_name = str(Path(assets).resolve())
    config.load_vlm_weights = False
    config.num_vlm_layers = config.num_expert_layers = 8
    config.expert_width_multiplier = 0.75
    config.train_expert_only = False
    config.freeze_vision_encoder = True
    config.train_state_proj = True
    config.use_amp = False
    config.compile_model = False
    # Construct on CPU, avoiding a second random GPU backbone allocation.
    config.device = "cpu"
    student = SmolVLAPolicy(config).float()
    indices = s8_indices()
    source = teacher.state_dict()
    copied = {}
    for name, target in student.state_dict().items():
        source_name = name
        if ".layers." in name and ".vision_model." not in name:
            prefix, rest = name.split(".layers.", 1)
            layer, suffix = rest.split(".", 1)
            source_name = f"{prefix}.layers.{S8_LAYERS[int(layer)]}.{suffix}"
        if source_name not in source:
            raise ValueError(f"Teacher missing {source_name}")
        # Only the current tensor is copied from GPU to CPU at a time.
        copied[name] = prune_tensor(name, source[source_name].detach().cpu(), target.shape, indices)
    student.load_state_dict(copied, strict=True)
    configure_student_gradients(student)
    return student, {
        "architecture": "S8",
        "teacher_layers": list(S8_LAYERS),
        "text_width": 576,
        "expert_width": 432,
        "head_dim": 64,
        "q_heads": 9,
        "kv_heads": 3,
        "indices": {k: v.tolist() for k, v in indices.items()},
        "initialized_tensors": len(copied),
        "random_student_tensors": 0,
        "initialization": "deterministic parity-preserving depth and uniform structured width pruning",
        "importance_optimized": False,
    }


def masked_action_mse(prediction, target, padding_mask, action_dim):
    if (
        prediction.shape != target.shape
        or not 0 < action_dim <= prediction.shape[-1]
        or padding_mask.shape != prediction.shape[:-1]
    ):
        raise ValueError("Action shapes/dimension do not match")
    valid = ~padding_mask.bool()
    if not valid.any():
        raise ValueError("Batch contains no valid action timesteps")
    error = (prediction[..., :action_dim] - target[..., :action_dim]).square()
    return (error * valid[..., None]).sum() / (valid.sum() * action_dim)


def velocity_forward(policy, batch, noisy_actions, time):
    """Differentiable forward matching the existing training attention masks."""
    images, image_masks = policy.prepare_images(batch)
    model = policy.model
    prefix, prefix_pad, prefix_att = model.embed_prefix(
        images,
        image_masks,
        batch["observation.language.tokens"],
        batch["observation.language.attention_mask"],
        state=policy.prepare_state(batch),
    )
    suffix, suffix_pad, suffix_att = model.embed_suffix(noisy_actions, time)
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    att = torch.cat([prefix_att, suffix_att], dim=1)
    (_, output), _ = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad, att),
        position_ids=pad.cumsum(1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
    )
    velocity = model.action_out_proj(output[:, -policy.config.chunk_size :].float())
    return velocity, prefix_pad


class SmolVLADistiller(nn.Module):
    def __init__(self, student, teacher, demo_weight=1.0, kd_weight=1.0, feature_weight=0.1):
        super().__init__()
        if min(demo_weight, kd_weight, feature_weight) < 0 or demo_weight + kd_weight <= 0:
            raise ValueError("Loss weights must be nonnegative with an action supervision term")
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.weights = demo_weight, kd_weight, feature_weight
        self.feature_layers = (1, 3, 7)
        self.adapters = nn.ModuleDict({str(i): nn.Linear(576, 960, bias=False) for i in self.feature_layers})
        indices = s8_indices()["text"]
        with torch.no_grad():
            for adapter in self.adapters.values():
                adapter.weight.zero_()
                adapter.weight[indices, torch.arange(576)] = 1

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.student.model.vlm_with_expert.vlm.model.vision_model.eval()
        return self

    def forward(self, batch, noise=None, time=None, distributed_loss=False):
        actions = self.student.prepare_action(batch)
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = self.student.model.sample_time(actions.shape[0], actions.device)
        if noise.shape != actions.shape or time.shape != (actions.shape[0],):
            raise ValueError("Shared flow noise/time shapes do not match the batch")
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target = noise - actions
        teacher_features, student_features, handles = {}, {}, []

        def capture(storage, key):
            def hook(_module, _inputs, output):
                storage[key] = output

            return hook

        if self.weights[2]:
            for i in self.feature_layers:
                teacher_layer = self.teacher.model.vlm_with_expert.vlm.model.text_model.layers[S8_LAYERS[i]]
                student_layer = self.student.model.vlm_with_expert.vlm.model.text_model.layers[i]
                handles.append(
                    teacher_layer.input_layernorm.register_forward_hook(capture(teacher_features, i))
                )
                handles.append(
                    student_layer.input_layernorm.register_forward_hook(capture(student_features, i))
                )
        try:
            with torch.no_grad():
                teacher_velocity, teacher_mask = velocity_forward(self.teacher, batch, noisy, time)
            student_velocity, prefix_mask = velocity_forward(self.student, batch, noisy, time)
        finally:
            for handle in handles:
                handle.remove()
        if not torch.equal(teacher_mask, prefix_mask):
            raise ValueError("Student and teacher prefix masks differ")
        padding = batch.get(
            "action_is_pad", torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
        )
        action_dim = self.student.config.action_feature.shape[0]
        demo = masked_action_mse(student_velocity, target, padding, action_dim)
        kd = masked_action_mse(student_velocity, teacher_velocity, padding, action_dim)
        teacher_demo = masked_action_mse(teacher_velocity, target, padding, action_dim)
        feature = student_velocity.new_zeros(())
        if self.weights[2]:
            for i in self.feature_layers:
                prediction = self.adapters[str(i)](student_features[i].float())
                cosine = 1 - functional.cosine_similarity(prediction, teacher_features[i].float(), dim=-1)
                feature = feature + (cosine * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)
            feature = feature / len(self.feature_layers)
        action_count, prefix_count = (~padding.bool()).sum(), prefix_mask.sum()
        action_scale, feature_scale = (
            distributed_loss_scales(action_count, prefix_count) if distributed_loss else (1.0, 1.0)
        )
        loss = (
            action_scale * (self.weights[0] * demo + self.weights[1] * kd)
            + feature_scale * self.weights[2] * feature
        )
        return loss, {
            "loss": loss.detach(),
            "demo": demo.detach(),
            "kd": kd.detach(),
            "teacher_demo": teacher_demo.detach(),
            "feature": feature.detach(),
            "action_count": action_count.detach(),
            "prefix_count": prefix_count.detach(),
        }


def save_student(student, destination, original_checkpoint, assets, metadata, published_path=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    copy_assets(assets, staging / "vlm_assets")
    config = copy.deepcopy(student.config)
    published = Path(published_path).resolve() if published_path is not None else destination
    config.vlm_model_name = str(published / "vlm_assets")
    config.load_vlm_weights = False
    config._save_pretrained(staging)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()},
        staging / "model.safetensors",
    )
    for item in Path(original_checkpoint).iterdir():
        if (
            item.name.startswith("policy_pre")
            or item.name.startswith("policy_post")
            or item.name == "tokenizer"
        ):
            if item.is_dir():
                shutil.copytree(item, staging / item.name)
            else:
                shutil.copy2(item, staging / item.name)
    (staging / "distillation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    staging.rename(destination)
