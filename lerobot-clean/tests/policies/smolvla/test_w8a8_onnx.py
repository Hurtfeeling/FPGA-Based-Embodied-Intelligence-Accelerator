"""Check integer ONNX lowering independently of the large policy export."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from scripts.export_smolvla_w8a8_onnx import (
    AttentionExportMode,
    QuantizedLinear,
    cpu_compatible_bf16,
    repair_integer_reductions,
)
from scripts.quantize_smolvla_w8a8 import quantize_weight


class Matmul(torch.nn.Module):
    def forward(self, left, right):
        with AttentionExportMode():
            return torch.matmul(left, right)


class BF16Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        original = torch.nn.Linear(5, 5).to(torch.bfloat16)
        codes, scale = quantize_weight(original.weight)
        self.linear = QuantizedLinear(original, codes, scale)
        self.norm = torch.nn.LayerNorm(5).to(torch.bfloat16)

    def forward(self, value):
        value = value.to(torch.bfloat16)
        return (self.norm(self.linear(value)) + value).float()


class ReductionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        original = torch.nn.Linear(5, 3)
        codes, scale = quantize_weight(original.weight)
        self.linear = QuantizedLinear(original, codes, scale)

    def forward(self, value, mask):
        with AttentionExportMode():
            positions = mask.cumsum(-1).float().unsqueeze(-1)
            return self.linear(value + positions)


class IntegerOnnxTest(unittest.TestCase):
    def check_export(self, model, inputs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            names = [f"input_{i}" for i in range(len(inputs))]
            with torch.inference_mode():
                expected = model(*inputs).float().numpy()
                torch.onnx.export(model, inputs, str(path), dynamo=False, opset_version=18,
                                  input_names=names, output_names=["output"], do_constant_folding=False)
            graph = cpu_compatible_bf16(repair_integer_reductions(onnx.load(path)))
            onnx.save(graph, path)
            onnx.checker.check_model(str(path))
            self.assertTrue(any(n.op_type == "MatMulInteger" for n in graph.graph.node))
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            result = session.run(None, {name: value.numpy() for name, value in zip(names, inputs, strict=True)})[0]
            np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)

    def test_linear_bias_and_zero_token(self):
        torch.manual_seed(3)
        original = torch.nn.Linear(7, 4)
        codes, scale = quantize_weight(original.weight)
        layer = QuantizedLinear(original, codes, scale)
        self.check_export(layer, (torch.cat([torch.randn(1, 2, 7), torch.zeros(1, 1, 7)], dim=1),))

    def test_linear_without_bias(self):
        original = torch.nn.Linear(5, 3, bias=False)
        codes, scale = quantize_weight(original.weight)
        self.check_export(QuantizedLinear(original, codes, scale), (torch.randn(1, 4, 5),))

    def test_batched_attention(self):
        torch.manual_seed(4)
        self.check_export(Matmul(), (torch.randn(1, 2, 4, 5), torch.randn(1, 2, 5, 3)))

    def test_bf16_rounding_norm_and_initializers(self):
        torch.manual_seed(8)
        self.check_export(BF16Block(), (torch.randn(1, 4, 5),))

    def test_boolean_position_cumulative_sum(self):
        self.check_export(ReductionBlock(), (torch.randn(1, 4, 5), torch.tensor([[True, True, False, True]])))


if __name__ == "__main__":
    unittest.main()
