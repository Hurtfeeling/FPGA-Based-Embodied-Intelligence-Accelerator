"""Verify rank sharding and DDP gradients against a single-process reference."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lerobot.policies.smolvla.distillation_distributed import (
    DistributedRun,
    ReproducibleBatches,
    capture_rng,
    distributed_loss_scales,
    metric_statistics,
    restore_rng,
    summarize_statistics,
)


class TwoLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, -0.25]))

    def forward(self, actions, prefix):
        return (self.weights[0] * actions).square().mean(), (self.weights[1] * prefix).square().mean()


def inputs(rank, micro):
    actions = torch.tensor([1.0, 2.0]) if rank == 0 else torch.tensor([3.0, 4.0, 5.0])
    prefix = torch.tensor([2.0, 3.0, 4.0]) if rank == 0 else torch.tensor([5.0])
    return actions + micro, prefix + micro


def ddp_worker(rank, rendezvous):
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = TwoLossModel()
        wrapped = DistributedDataParallel(model)
        for micro in range(2):
            actions, prefix = inputs(rank, micro)
            with wrapped.no_sync() if micro == 0 else nullcontext():
                demo, feature = wrapped(actions, prefix)
                scale_a, scale_p = distributed_loss_scales(
                    torch.tensor(actions.numel()), torch.tensor(prefix.numel())
                )
                ((demo * scale_a + 0.1 * feature * scale_p) / 2).backward()
        reference = TwoLossModel()
        for micro in range(2):
            batches = [inputs(r, micro) for r in range(2)]
            demo, feature = reference(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]))
            ((demo + 0.1 * feature) / 2).backward()
        torch.testing.assert_close(model.weights.grad, reference.weights.grad, rtol=1e-6, atol=1e-6)
        # Unequal evaluation shard sizes must be reduced by counts, not rank means.
        metrics = {
            "demo": torch.tensor(1.0 + rank),
            "kd": torch.tensor(2.0 + rank),
            "teacher_demo": torch.tensor(3.0 + rank),
            "feature": torch.tensor(4.0 + rank),
            "action_count": torch.tensor(1 + 2 * rank),
            "prefix_count": torch.tensor(3 - 2 * rank),
        }
        summary = summarize_statistics(metric_statistics(metrics), (1, 1, 0.1))
        assert abs(summary["demo"] - 1.75) < 1e-6
        assert abs(summary["feature"] - 4.25) < 1e-6
        run = DistributedRun(rank, 2, torch.device("cpu"))
        assert run.gather(rank) == [0, 1]

        def failure():
            raise FileExistsError("shared output is nonempty")

        try:
            run.primary_call(failure)
        except RuntimeError as error:
            assert "FileExistsError" in str(error)
        else:
            raise AssertionError("Rank-zero filesystem failure was not propagated")
    finally:
        distributed.destroy_process_group()


class S8DistributedTest(unittest.TestCase):
    def test_rank_batches_reconstruct_global_draw_and_resume_position(self):
        batches = [list(ReproducibleBatches(10000, 2, 5, 0, 42, rank, 2)) for rank in range(2)]
        for micro in range(5):
            expected = np.random.default_rng(42 + micro).integers(0, 10000, 4).tolist()
            self.assertEqual(batches[0][micro] + batches[1][micro], expected)
        for rank in range(2):
            self.assertEqual(list(ReproducibleBatches(10000, 2, 3, 2, 42, rank, 2)), batches[rank][2:])

    def test_sampler_rejects_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            ReproducibleBatches(10, 2, 1, 0, 42, rank=2, world_size=2)

    def test_rng_restore_reproduces_flow_and_dropout_streams(self):
        device = torch.device("cpu")
        generator = torch.Generator().manual_seed(142)
        state = capture_rng(generator, device)
        expected_flow, expected_global = torch.randn(20, generator=generator), torch.randn(20)
        restore_rng(state, generator, device)
        torch.testing.assert_close(torch.randn(20, generator=generator), expected_flow, rtol=0, atol=0)
        torch.testing.assert_close(torch.randn(20), expected_global, rtol=0, atol=0)

    @unittest.skipUnless(distributed.is_gloo_available(), "Gloo unavailable")
    def test_ddp_padding_weighted_gradients_and_accumulation_match_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = (Path(tmp) / "rendezvous").as_uri()
            multiprocessing.spawn(ddp_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
