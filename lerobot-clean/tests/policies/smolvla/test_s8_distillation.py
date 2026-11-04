"""Check S8 tensor mappings and masked policy supervision."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from lerobot.policies.smolvla.distillation import (
    S8_LAYERS,
    create_s8_assets,
    masked_action_mse,
    prune_tensor,
    s8_indices,
)
from lerobot.utils.import_utils import _transformers_available

if _transformers_available:
    from transformers import AutoConfig


class S8DistillationTest(unittest.TestCase):
    @unittest.skipUnless(_transformers_available, "transformers unavailable")
    def test_student_assets_override_teacher_bf16_for_exact_reload(self):
        config = {
            "model_type": "smolvlm",
            "torch_dtype": "bfloat16",
            "text_config": {"model_type": "llama", "torch_dtype": "bfloat16"},
            "vision_config": {"model_type": "smolvlm_vision", "torch_dtype": "bfloat16"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(config))
            destination = Path(tmp) / "student"
            create_s8_assets(source, destination)
            loaded = AutoConfig.from_pretrained(destination, local_files_only=True)
            for component in (loaded, loaded.text_config, loaded.vision_config):
                self.assertEqual(component.dtype, torch.float32)
            self.assertEqual(loaded.vision_config._attn_implementation, "eager")
            self.assertEqual(loaded.text_config.hidden_size, 576)

    def test_depth_selection_preserves_attention_parity(self):
        self.assertEqual(len(set(S8_LAYERS)), 8)
        self.assertEqual(S8_LAYERS[-1], 15)
        for student, teacher in enumerate(S8_LAYERS):
            self.assertEqual(student % 2, teacher % 2)

    def test_indices_keep_complete_query_kv_groups(self):
        indices = s8_indices()
        for key, size in (
            ("text", 576),
            ("query", 576),
            ("kv", 192),
            ("expert", 432),
            ("text_ffn", 1536),
            ("expert_ffn", 1280),
        ):
            self.assertEqual(indices[key].numel(), size)
            self.assertEqual(indices[key].unique().numel(), size)
        self.assertEqual(
            (indices["query"].reshape(9, 64)[:, 0] // 64).tolist(), [0, 1, 2, 6, 7, 8, 12, 13, 14]
        )

    def test_cross_attention_and_time_slices(self):
        indices = s8_indices()
        cross = torch.arange(320 * 320).reshape(320, 320).float()
        actual = prune_tensor(
            "model.vlm_with_expert.lm_expert.layers.1.self_attn.k_proj.weight", cross, (192, 192), indices
        )
        torch.testing.assert_close(actual, cross[indices["kv"]][:, indices["kv"]], rtol=0, atol=0)
        time = torch.arange(720 * 1440).reshape(720, 1440).float()
        actual = prune_tensor("model.action_time_mlp_in.weight", time, (432, 864), indices)
        expected = time[indices["expert"]][:, torch.cat([indices["expert"], indices["expert"] + 720])]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_masked_loss_excludes_padding_and_unused_action_dimensions(self):
        prediction = torch.ones(1, 3, 32, requires_grad=True)
        target = torch.zeros_like(prediction)
        padding = torch.tensor([[False, False, True]])
        loss = masked_action_mse(prediction, target, padding, 6)
        self.assertEqual(float(loss.detach()), 1.0)
        loss.backward()
        self.assertEqual(float(prediction.grad[:, 2:].abs().sum()), 0.0)
        self.assertEqual(float(prediction.grad[..., 6:].abs().sum()), 0.0)
        self.assertGreater(float(prediction.grad[:, :2, :6].abs().sum()), 0)

    def test_empty_valid_actions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no valid"):
            masked_action_mse(
                torch.ones(1, 2, 32), torch.zeros(1, 2, 32), torch.ones(1, 2, dtype=torch.bool), 6
            )

    def test_unmapped_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            prune_tensor("unsupported.weight", torch.ones(4, 4), (2, 2), s8_indices())


if __name__ == "__main__":
    unittest.main()
