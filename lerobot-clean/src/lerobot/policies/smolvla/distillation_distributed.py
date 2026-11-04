"""Distributed sampling, loss normalization and checkpoint coordination for S8."""

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import Sampler


@dataclass
class DistributedRun:
    rank: int
    world_size: int
    device: torch.device

    @classmethod
    def start(cls, device):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            return cls(0, 1, torch.device(device))
        if not device.startswith("cuda"):
            raise ValueError("Multi-GPU distillation requires a CUDA device")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        distributed.init_process_group(backend="nccl", device_id=device)
        return cls(distributed.get_rank(), world_size, device)

    @property
    def primary(self):
        return self.rank == 0

    def primary_call(self, function):
        """Publish rank-zero results/errors so filesystem failures reach all ranks."""
        result = [None, None]
        if self.primary:
            try:
                result[0] = function()
            except Exception as error:
                result[1] = f"{type(error).__name__}: {error}"
        if self.world_size > 1:
            distributed.broadcast_object_list(result, src=0, device=self.device)
        if result[1] is not None:
            raise RuntimeError(result[1])
        return result[0]

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        result = [None] * self.world_size
        distributed.all_gather_object(result, value)
        return result

    def close(self):
        if distributed.is_initialized():
            distributed.destroy_process_group()


class ReproducibleBatches(Sampler):
    """Slice a reproducible global batch without repeating the RNG stream by rank."""

    def __init__(self, length, batch_size, count, start, seed, rank=0, world_size=1):
        if not 0 <= rank < world_size:
            raise ValueError("Invalid sampler rank/world size")
        self.length, self.batch_size, self.count, self.start, self.seed = (
            length,
            batch_size,
            count,
            start,
            seed,
        )
        self.rank, self.world_size = rank, world_size

    def __iter__(self):
        for micro in range(self.start, self.start + self.count):
            indices = np.random.default_rng(self.seed + micro).integers(
                0, self.length, self.batch_size * self.world_size
            )
            offset = self.rank * self.batch_size
            yield indices[offset : offset + self.batch_size].tolist()

    def __len__(self):
        return self.count


def distributed_loss_scales(action_count, prefix_count):
    """Compensate DDP's rank average for unequal valid-token counts."""
    counts = torch.stack([action_count, prefix_count]).detach().to(torch.float64)
    totals = counts.clone()
    distributed.all_reduce(totals)
    scales = counts * distributed.get_world_size() / totals.clamp_min(1)
    return scales[0].float(), scales[1].float()


def metric_statistics(metrics):
    actions, prefix = metrics["action_count"], metrics["prefix_count"]
    return (
        torch.stack(
            [
                metrics["demo"] * actions,
                metrics["kd"] * actions,
                metrics["teacher_demo"] * actions,
                metrics["feature"] * prefix,
                actions,
                prefix,
            ]
        )
        .detach()
        .to(torch.float64)
    )


def summarize_statistics(statistics, weights):
    totals = statistics.clone()
    if distributed.is_initialized():
        distributed.all_reduce(totals)
    if totals[4] <= 0:
        raise ValueError("No valid actions in metric reduction")
    demo, kd, teacher_demo = (totals[:3] / totals[4]).tolist()
    feature = float(totals[3] / totals[5].clamp_min(1))
    return {
        "loss": weights[0] * demo + weights[1] * kd + weights[2] * feature,
        "demo": demo,
        "kd": kd,
        "teacher_demo": teacher_demo,
        "feature": feature,
    }


def capture_rng(generator, device):
    return {
        "flow": generator.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, generator, device):
    generator.set_state(state["flow"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def state_sha256(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
