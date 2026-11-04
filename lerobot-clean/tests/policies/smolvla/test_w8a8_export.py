"""Numerical tests for the standalone HLS quantization exporter (no robot needed)."""

import unittest

import torch
import torch.nn.functional as F

from scripts.quantize_smolvla_w8a8 import (
    W8A8Mode,
    quantize_activation,
    quantize_weight,
    simulated_int8_matmul,
)


class W8A8ExportTest(unittest.TestCase):
    def test_weight_zero_rows_and_reconstruction_bound(self):
        weight = torch.tensor([[0.0, 0.0, 0.0], [-2.3, 0.7, 1.4]])
        codes, scale = quantize_weight(weight)
        self.assertEqual(codes.dtype, torch.int8)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue(torch.equal(codes[0], torch.zeros(3, dtype=torch.int8)))
        error = (codes.float() * scale[:, None] - weight).abs()
        self.assertTrue((error <= scale[:, None] / 2 + 1e-6).all())

    def test_round_even_and_saturation(self):
        values = torch.tensor([-1000.0, -2.5, -1.5, 0.5, 1.5, 2.5, 1000.0])
        expected = torch.tensor([-127, -2, -2, 0, 2, 2, 127], dtype=torch.int8)
        self.assertTrue(torch.equal(quantize_activation(values, 1.0), expected))

    def test_attention_batched_product_matches_integer_accumulation(self):
        torch.manual_seed(1)
        left = torch.randn(2, 3, 4, 5)
        right = torch.randn(2, 3, 5, 6)
        ls = left.abs().amax(-1, keepdim=True) / 127
        rs = right.abs().amax(-2, keepdim=True) / 127
        qleft = quantize_activation(left, ls).int()
        qright = quantize_activation(right, rs).int()
        expected = torch.matmul(qleft, qright).float() * ls * rs
        torch.testing.assert_close(simulated_int8_matmul(left, right), expected, rtol=0, atol=0)

    def test_linear_mode_matches_integer_accumulation_and_preserves_softmax(self):
        model = torch.nn.Sequential(torch.nn.Linear(5, 3))
        weight = model[0].weight
        values = torch.tensor([[0.1, 0.5, -0.2, 0.8, -0.7]])
        names = {"0.weight"}
        observer = W8A8Mode(model, names)
        with observer:
            original = model(values)
        torch.testing.assert_close(original, model(values), rtol=0, atol=0)
        codes, scale = quantize_weight(weight)
        input_scale = observer.maximum["0"] / 127
        mode = W8A8Mode(model, names, {"0.weight": codes}, {
            "0.weight_scale": scale, "0.input_scale": input_scale,
        })
        with mode:
            actual = model(values)
            softmax = F.softmax(values, dim=-1)
        acc = quantize_activation(values, input_scale).int() @ codes.int().T
        expected = acc.float() * input_scale * scale + model[0].bias
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(softmax, F.softmax(values, dim=-1), rtol=0, atol=0)

    def test_zero_attention_inputs_remain_finite(self):
        result = simulated_int8_matmul(torch.zeros(2, 3), torch.zeros(3, 4))
        self.assertTrue(torch.equal(result, torch.zeros(2, 4)))

    def test_per_token_linear_handles_different_ranges(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2))
        values = torch.tensor([[[0.01, 0.02, -0.03], [10.0, -20.0, 30.0]]])
        codes, scale = quantize_weight(model[0].weight)
        scales = {"0.weight_scale": scale, "0.input_scale": torch.tensor(30.0 / 127)}
        mode = W8A8Mode(model, {"0.weight"}, {"0.weight": codes}, scales, "per_token")
        with mode:
            actual = model(values)
        input_scale = values.abs().amax(-1, keepdim=True) / 127
        acc = quantize_activation(values, input_scale).int() @ codes.int().T
        expected = acc.float() * input_scale * scale + model[0].bias
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            quantize_weight(torch.tensor([[float("nan")]]))


if __name__ == "__main__":
    unittest.main()
