"""Verify exact integer lowering and reject every non-FP32 floating type."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto as T, helper

from scripts.export_smolvla_w8a8_fp32_onnx import (
    ExactQuantizedLinear,
    QuantizedLanguageEmbedding,
    W8A8FP32Mode,
    audit,
    exact_int8_matmul,
)
from scripts.quantize_smolvla_w8a8 import quantize_weight


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        linear = torch.nn.Linear(7, 5)
        codes, scale = quantize_weight(linear.weight)
        self.linear = ExactQuantizedLinear(linear, codes, scale)
        self.norm = torch.nn.LayerNorm(5)

    def forward(self, value, right, mask):
        with W8A8FP32Mode():
            value = value + mask.cumsum(-1).float().unsqueeze(-1)
            value = self.norm(self.linear(value))
            return torch.matmul(torch.softmax(value, -1), right)


class StaticBlock(Block):
    def forward(self, value, right, mask):
        positions = torch.arange(value.shape[1], device=value.device).float()
        return super().forward(value + torch.sin(positions)[None, :, None], right, mask)


class W8A8FP32OnnxTest(unittest.TestCase):
    def test_int8_embedding_gathers_rows_before_dequantization(self):
        torch.manual_seed(20)
        embedding = QuantizedLanguageEmbedding(torch.nn.Embedding(31, 7))
        original = torch.nn.Linear(7, 5)
        codes, scale = quantize_weight(original.weight)
        model = torch.nn.Sequential(embedding, ExactQuantizedLinear(original, codes, scale)).eval()
        tokens = torch.tensor([[0, 3, 30, 0]], dtype=torch.int64)
        with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
            expected = model(tokens).numpy()
            path = Path(directory) / "embedding.onnx"
            torch.onnx.export(
                model,
                (tokens,),
                str(path),
                input_names=["tokens"],
                output_names=["actions"],
                dynamo=False,
                opset_version=18,
                do_constant_folding=False,
            )
            report = audit(path)
            self.assertTrue(report["all_floating_tensors_fp32"])
            graph = onnx.load(path).graph
            self.assertEqual(next(t.data_type for t in graph.initializer if t.name == "0.codes"), T.INT8)
            producers = {output: n for n in graph.node for output in n.output}
            cast = next(
                n
                for n in graph.node
                if n.op_type == "Cast"
                and n.input[0] in producers
                and producers[n.input[0]].op_type == "Gather"
                and producers[n.input[0]].input[0] == "0.codes"
            )
            self.assertEqual(next(a.i for a in cast.attribute if a.name == "to"), T.FLOAT)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            np.testing.assert_allclose(
                session.run(None, {"tokens": tokens.numpy()})[0], expected, atol=1e-6, rtol=1e-6
            )

    def test_ort_integer_accumulation_full_range_has_no_partial_saturation(self):
        left = np.full((4, 4096), 127, dtype=np.int8)
        left[1, ::2] = -127
        left[2] = -127
        left[3] = 0
        right = np.full((4096, 4), 127, dtype=np.int8)
        right[::2, 1] = -127
        right[:, 2] = -127
        graph = helper.make_graph(
            [helper.make_node("MatMulInteger", ["left", "right", "left_zero", "right_zero"], ["acc"])],
            "exact_integer_accumulator",
            [
                helper.make_tensor_value_info("left", T.UINT8, list(left.shape)),
                helper.make_tensor_value_info("right", T.INT8, list(right.shape)),
            ],
            [helper.make_tensor_value_info("acc", T.INT32, [4, 4])],
            [
                helper.make_tensor("left_zero", T.UINT8, [], [128]),
                helper.make_tensor("right_zero", T.INT8, [], [0]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"left": (left.astype(np.int32) + 128).astype(np.uint8), "right": right})[
            0
        ]
        expected = left.astype(np.int64) @ right.astype(np.int64)
        self.assertEqual(actual.dtype, np.int32)
        np.testing.assert_array_equal(actual, expected)

    def test_static_simplification_is_bit_exact_and_keeps_float_sin(self):
        torch.manual_seed(18)
        model = StaticBlock().eval()
        values = (torch.randn(1, 4, 7), torch.randn(1, 5, 3), torch.ones(1, 4, dtype=torch.bool))
        feed = dict(zip(("value", "right", "mask"), (v.numpy() for v in values), strict=True))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            torch.onnx.export(
                model,
                values,
                str(path),
                input_names=list(feed),
                output_names=["actions"],
                dynamo=False,
                opset_version=18,
                do_constant_folding=False,
            )
            raw = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            before = raw.run(None, feed)[0]
            products = [
                n.SerializeToString() for n in onnx.load(path).graph.node if n.op_type == "MatMulInteger"
            ]
            report = audit(path)
            self.assertGreater(report["identical_constants_shared"], 0)
            self.assertFalse(report["floating_arithmetic_constant_folding"])
            self.assertEqual(report["operator_counts"]["Sin"], 1)
            self.assertEqual(
                products,
                [n.SerializeToString() for n in onnx.load(path).graph.node if n.op_type == "MatMulInteger"],
            )
            compact = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            np.testing.assert_array_equal(compact.run(None, feed)[0], before)

    def test_exact_broadcast_integer_products(self):
        torch.manual_seed(13)
        a = torch.randint(-127, 128, (2, 3, 9), dtype=torch.int8)
        b = torch.randint(-127, 128, (1, 9, 5), dtype=torch.int8)
        actual = exact_int8_matmul(a, b)
        self.assertEqual(actual.dtype, torch.int32)
        np.testing.assert_array_equal(actual.numpy(), a.numpy().astype(np.int64) @ b.numpy().astype(np.int64))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_integer_padding_and_accumulation(self):
        torch.manual_seed(14)
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        a = torch.randint(-127, 128, (2, 1, 9), dtype=torch.int8)
        b = torch.randint(-127, 128, (1, 9, 6), dtype=torch.int8)
        actual = exact_int8_matmul(a.to(device), b.to(device)).cpu()
        np.testing.assert_array_equal(actual.numpy(), a.numpy().astype(np.int64) @ b.numpy().astype(np.int64))

    def test_linear_attention_norm_softmax_and_mask_export(self):
        torch.manual_seed(15)
        model = Block().eval()
        values = (torch.randn(1, 4, 7), torch.randn(1, 5, 3), torch.tensor([[True, True, False, True]]))
        # Include a zero row in the direct attention test to exercise scale fallback.
        values[1][:, :, 0] = 0
        with torch.inference_mode():
            expected = model(*values).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            torch.onnx.export(
                model,
                values,
                str(path),
                input_names=["value", "right", "mask"],
                output_names=["actions"],
                dynamo=False,
                opset_version=18,
                do_constant_folding=False,
            )
            report = audit(path)
            self.assertTrue(report["all_floating_tensors_fp32"])
            self.assertEqual(report["operator_counts"]["MatMulInteger"], 2)
            self.assertEqual(report["floating_matmul_operators"], 0)
            self.assertTrue(report["shape_information_persisted"])
            self.assertGreater(len(onnx.load(path).graph.value_info), 0)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = session.run(
                None, dict(zip(("value", "right", "mask"), (v.numpy() for v in values), strict=True))
            )[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_audit_rejects_forbidden_float_types(self):
        for dtype in (T.FLOAT16, T.BFLOAT16, T.FLOAT8E4M3FN, T.DOUBLE):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                constant = helper.make_tensor("bad", dtype, [1], [1.0])
                graph = helper.make_graph(
                    [], "bad", [], [helper.make_tensor_value_info("bad", dtype, [1])], [constant]
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
                path = Path(directory) / "bad.onnx"
                onnx.save(model, path)
                with self.assertRaisesRegex(ValueError, "Forbidden"):
                    audit(path)

    def test_cuda_export_uses_symbolic_integer_operator(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        original = torch.nn.Linear(7, 5).to(device)
        codes, scale = quantize_weight(original.weight.cpu())
        model = ExactQuantizedLinear(original, codes, scale).eval()
        value = torch.randn(1, 2, 7, device=device)
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            reference = model(value).cpu().numpy()
            path = Path(directory) / "cuda.onnx"
            torch.onnx.export(
                model,
                (value,),
                str(path),
                input_names=["value"],
                output_names=["actions"],
                dynamo=False,
                opset_version=18,
                do_constant_folding=False,
            )
            self.assertEqual(audit(path)["operator_counts"]["MatMulInteger"], 1)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            np.testing.assert_allclose(
                session.run(None, {"value": value.cpu().numpy()})[0], reference, atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
