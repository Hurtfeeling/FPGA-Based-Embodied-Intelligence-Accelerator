"""Check FP8 encoding, retained integer products and mixed graph execution."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

from scripts.export_smolvla_w8a8_fp8 import audit, convert
from scripts.w8a8_fp8_runtime import DOMAIN, FP8_DTYPE, FP8Session, Float8, encode, integer_matmul


class W8A8FP8Test(unittest.TestCase):
    def test_real_fp8_serialization_and_scaled_range(self):
        values = np.array([0, 1e-6, 1, 1e4], np.float32)
        encoded = encode(values, per_element=True)
        tensor = numpy_helper.from_array(encoded.data)
        self.assertEqual(tensor.data_type, T.FLOAT8E4M3FN)
        self.assertEqual(len(tensor.raw_data), values.size)
        np.testing.assert_allclose(encoded.decode(), values, rtol=0.063, atol=1e-8)

    def test_attention_mask_does_not_swallow_finite_values(self):
        encoded = encode(np.array([0.1, 1, -3e38], np.float32))
        self.assertEqual(int(encoded.data.view(np.uint8)[2]), 255)
        self.assertLess(encoded.decode()[2], -1e30)
        np.testing.assert_allclose(encoded.decode()[:2], [0.1, 1], atol=0.005)

    def test_offset_binary_integer_product_stays_exact(self):
        left = np.array([[128, 129, 127], [255, 1, 128]], np.uint8)
        right = np.array([[127, -127], [-5, 8], [9, -11]], np.int8)
        expected = (left.astype(np.int32) - 128) @ right.astype(np.int32)
        actual = integer_matmul(left, right, np.uint8(128), np.int8(0))
        self.assertEqual(actual.dtype, np.int32)
        np.testing.assert_array_equal(actual, expected)

    def test_end_to_end_original_quantizer_and_matmul_are_preserved(self):
        arrays = {"axes": np.array([-1], np.int64), "zero": np.array(0, np.float32),
                  "one": np.array(1, np.float32), "limit": np.array(127, np.float32),
                  "lower": np.array(-127, np.float32), "offset": np.array(128, np.float32),
                  "left_zp": np.array(128, np.uint8), "right_zp": np.array(0, np.int8),
                  "weights": np.arange(12, dtype=np.int8).reshape(4, 3),
                  "weight_scale": np.array([0.01, 0.02, 0.03], np.float32)}
        nodes = [helper.make_node("Abs", ["state"], ["abs"]),
                 helper.make_node("ReduceMax", ["abs", "axes"], ["maximum"], keepdims=1),
                 helper.make_node("Greater", ["maximum", "zero"], ["positive"]),
                 helper.make_node("Div", ["maximum", "limit"], ["scale_div"]),
                 helper.make_node("Where", ["positive", "scale_div", "one"], ["scale"]),
                 helper.make_node("Div", ["state", "scale"], ["div"]),
                 helper.make_node("Round", ["div"], ["round"]),
                 helper.make_node("Clip", ["round", "lower", "limit"], ["clip"]),
                 helper.make_node("Add", ["clip", "offset"], ["offset_codes"]),
                 helper.make_node("Cast", ["offset_codes"], ["codes"], to=T.UINT8),
                 helper.make_node("MatMulInteger", ["codes", "weights", "left_zp", "right_zp"], ["acc"], name="retained_product"),
                 helper.make_node("Cast", ["acc"], ["acc_float"], to=T.FLOAT),
                 helper.make_node("Mul", ["acc_float", "scale"], ["first_scale"]),
                 helper.make_node("Mul", ["first_scale", "weight_scale"], ["actions"])]
        graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("state", T.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("actions", T.FLOAT, [1, 3])],
                                  [numpy_helper.from_array(v, k) for k, v in arrays.items()])
        source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        converted, report = convert(source)
        self.assertEqual(report["matmul_integer_nodes_identical"], 1)
        self.assertEqual(report["fp16_bf16_fp32_fp64_tensors"], 0)
        audit(converted, source)
        value = encode(np.array([[1, -1, 0.5, 0]], np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            onnx.save(converted, path)
            actual = FP8Session(path).run({"state": value.data, "state__fp8_exponent": value.exponent})
        name = converted.graph.output[0].name
        result = Float8(actual[name], actual[name + "__fp8_exponent"]).decode()
        expected = np.array([[1, -1, 0.5, 0]], np.float32) @ arrays["weights"].astype(np.float32)
        expected *= arrays["weight_scale"]
        np.testing.assert_allclose(result, expected, atol=0.006)
        self.assertEqual(actual[name].dtype, FP8_DTYPE)

    def test_shared_fp8_codes_keep_distinct_scale_exponents(self):
        arrays = {"a": np.ones(4, np.float32), "b": np.full(4, 2, np.float32)}
        nodes = [helper.make_node("Mul", ["state", "a"], ["a_out"]),
                 helper.make_node("Mul", ["state", "b"], ["b_out"]),
                 helper.make_node("Add", ["a_out", "b_out"], ["actions"])]
        graph = helper.make_graph(nodes, "scales", [helper.make_tensor_value_info("state", T.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("actions", T.FLOAT, [1, 4])],
                                  [numpy_helper.from_array(v, k) for k, v in arrays.items()])
        source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        converted, _ = convert(source)
        self.assertEqual(sum(t.data_type == T.FLOAT8E4M3FN for t in converted.graph.initializer), 1)
        value = encode(np.array([[0.25, 0.5, 1, 2]], np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save(converted, path)
            result = FP8Session(path).run({"state": value.data, "state__fp8_exponent": value.exponent})
        np.testing.assert_array_equal(Float8(result["actions"], result["actions__fp8_exponent"]).decode(),
                                      value.decode() * 3)

    def test_quantizer_zero_rows_and_tiny_scales_do_not_wrap(self):
        node = helper.make_node("F8QuantizeINT8", ["state", "state_exp"], ["codes", "scale", "scale_exp"],
                                domain=DOMAIN, input_kinds=[1], output_kinds=[0, 1], axis=-1, unsigned=1)
        graph = helper.make_graph([node], "tiny", [helper.make_tensor_value_info("state", T.FLOAT8E4M3FN, [2, 4]),
                                                   helper.make_tensor_value_info("state_exp", T.INT32, [])],
                                  [helper.make_tensor_value_info("codes", T.UINT8, [2, 4]),
                                   helper.make_tensor_value_info("scale", T.FLOAT8E4M3FN, [2, 1]),
                                   helper.make_tensor_value_info("scale_exp", T.INT32, [2, 1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid(DOMAIN, 1)], ir_version=9)
        value = encode(np.array([[0, 0, 0, 0], [1e-6, -1e-6, 0, 0]], np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            onnx.save(model, path)
            result = FP8Session(path).run({"state": value.data, "state_exp": value.exponent})
        self.assertTrue(np.all(Float8(result["scale"], result["scale_exp"]).decode() > 0))
        np.testing.assert_array_equal(result["codes"][0], [128, 128, 128, 128])
        self.assertEqual(int(result["codes"][1, 0]), 255)
        self.assertEqual(int(result["codes"][1, 1]), 1)


if __name__ == "__main__":
    unittest.main()
