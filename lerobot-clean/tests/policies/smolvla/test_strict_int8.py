"""Check INT8 storage, integer kernels and the custom ONNX contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto as T, helper, numpy_helper
import onnx

from scripts.export_smolvla_strict_int8 import audit, encode, export
from scripts.strict_int8_runtime import DOMAIN, IntegerSession, Q8, Scale32, add, multiply, pack, quantized_operation, rms_normalize


def decode(value):
    return value.codes.astype(np.float64) * np.exp2(value.exponent)


class StrictInt8Test(unittest.TestCase):
    def test_every_parameter_code_is_int8_with_integer_scale(self):
        values = np.array([[0.001, -0.002, 0.003], [100, -20, 40]], np.float32)
        value = encode(values, per_channel=True)
        self.assertEqual(value.codes.dtype, np.int8)
        self.assertEqual(value.exponent.dtype, np.int32)
        self.assertEqual(value.codes.nbytes, values.size)
        np.testing.assert_allclose(decode(value), values, atol=0.5, rtol=0.01)

    def test_mask_does_not_destroy_finite_attention_scores(self):
        scores = encode(np.array([[0, -1, 3]], np.float32))
        mask = encode(np.array([[0, 0, -3e38]], np.float32))
        result = add(scores, mask)
        np.testing.assert_array_equal(result.mask, [[False, False, True]])
        np.testing.assert_allclose(decode(result)[:, :2], [[0, -1]], atol=0.02)

    def test_integer_matmul_requantizes_to_int8(self):
        left, right = encode(np.array([[1, 2, -3]], np.float32)), encode(np.array([[2], [3], [4]], np.float32))
        result = quantized_operation("MatMul", [left, right], {}, {})[0]
        self.assertEqual(result.codes.dtype, np.int8)
        np.testing.assert_array_equal(decode(result), [[-4]])

    def test_matrix_scale_metadata_keeps_int32_precision(self):
        value = encode(np.array([[1, 2, -3]], np.float32))
        weights = np.array([[2, 1], [3, 2], [4, 3]], np.int8)
        coefficients = np.array([0.0000137, 0.00213])
        exponent = -38
        scale = Scale32(np.rint(coefficients / (2.0 ** exponent)).astype(np.int32), np.array(exponent, np.int32))
        actual = quantized_operation("MatMulWeights", [value, weights, scale], {}, {})[0]
        expected = (decode(value) @ weights.astype(np.float64)) * coefficients
        np.testing.assert_allclose(decode(actual), expected, atol=1e-4)

    def test_negative_products_and_large_accumulator(self):
        left, right = encode(np.array([-2, 3], np.float32)), encode(np.array([4, -5], np.float32))
        np.testing.assert_allclose(decode(multiply(left, right)), [-8, -15], atol=0.1)
        value = pack(np.array([100_000_000, -100_000_000], np.int64), -20)
        self.assertEqual(value.codes.dtype, np.int8)
        np.testing.assert_allclose(decode(value), np.array([100_000_000, -100_000_000]) / (1 << 20), rtol=0.01)

    def test_custom_export_runtime_norm_and_softmax(self):
        nodes = [helper.make_node("LayerNormalization", ["state", "scale", "bias"], ["norm"], epsilon=1e-5),
                 helper.make_node("Softmax", ["norm"], ["actions"], axis=-1)]
        initializers = [numpy_helper.from_array(np.ones(4, np.float32), "scale"),
                        numpy_helper.from_array(np.zeros(4, np.float32), "bias")]
        graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("state", T.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("actions", T.FLOAT, [1, 4])], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        model = onnx.shape_inference.infer_shapes(model)
        result, _ = export(model)
        audit(result)
        self.assertTrue(all(n.domain == DOMAIN for n in result.graph.node))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            onnx.save(result, path)
            value = encode(np.array([[-1, 0, 1, 2]], np.float32))
            actual = IntegerSession(path).run({"state": value.codes, "state__exponent": value.exponent})
        actions = Q8(actual["actions"], actual["actions__exponent"])
        self.assertEqual(actions.codes.dtype, np.int8)
        x = np.array([-1, 0, 1, 2], np.float64)
        normalized = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        expected = np.exp(normalized) / np.exp(normalized).sum()
        np.testing.assert_allclose(decode(actions)[0], expected, atol=0.015)

    def test_audit_rejects_floating_initializer(self):
        graph = helper.make_graph([], "bad", [], [], [numpy_helper.from_array(np.ones(1, np.float32), "bad")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        with self.assertRaisesRegex(ValueError, "Non-integer initializer"):
            audit(model)

    def test_rms_norm_keeps_small_and_zero_rows_finite(self):
        values = encode(np.array([[1, 2, -3, 4], [0, 0, 0, 0], [0.01, 0.02, 0, 0]], np.float32))
        scale = encode(np.ones(4, np.float32))
        actual = rms_normalize(values, scale, {"epsilon_q20": 1})
        decoded = decode(values)
        expected = decoded / np.sqrt((decoded * decoded).mean(-1, keepdims=True) + 1e-6)
        self.assertEqual(actual.codes.dtype, np.int8)
        np.testing.assert_allclose(decode(actual), expected, atol=0.035)

    def test_shared_parameters_are_not_serialized_twice(self):
        matrix = np.arange(64, dtype=np.float32).reshape(16, 4) / 32
        nodes = [helper.make_node("Gather", ["a", "indices"], ["left"]),
                 helper.make_node("Gather", ["b", "indices"], ["right"]),
                 helper.make_node("Add", ["left", "right"], ["actions"])]
        graph = helper.make_graph(nodes, "shared", [helper.make_tensor_value_info("indices", T.INT64, [1])],
                                  [helper.make_tensor_value_info("actions", T.FLOAT, [1, 4])],
                                  [numpy_helper.from_array(matrix, "a"), numpy_helper.from_array(matrix, "b")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        result, _ = export(onnx.shape_inference.infer_shapes(model))
        actual = [t for t in result.graph.initializer if list(t.dims) == [16, 4] and t.data_type == T.INT8]
        self.assertEqual(len(actual), 1)

    def test_rotary_angles_use_integer_positions_and_integer_tables(self):
        half = 32
        positions = np.array([[0, 1, 50, 550]], np.int64)
        frequencies = np.zeros((257, 256), np.int32)
        frequencies[half, :half] = np.rint(10000 ** (-np.arange(half) / half) * (1 << 30)).astype(np.int32)
        angles = np.arange(16384) * (2 * np.pi / 16384)
        tables = {"__lut_rope_frequencies": frequencies,
                  "__lut_sin": np.rint(np.sin(angles) * 64).astype(np.int8)}
        actual = quantized_operation("RotarySin", [positions, np.array(half, np.int64)],
                                     {"period_q20": round(2 * np.pi * (1 << 20))}, tables)[0]
        expected = np.sin(positions[..., None] * 10000 ** (-np.arange(half) / half))
        np.testing.assert_allclose(decode(actual)[..., 0, :], expected, atol=0.01)
        expanded = quantized_operation("RotarySin", [positions[..., None], np.array(half, np.int64)],
                                       {"period_q20": round(2 * np.pi * (1 << 20))}, tables)[0]
        np.testing.assert_array_equal(actual.codes, expanded.codes)


if __name__ == "__main__":
    unittest.main()
