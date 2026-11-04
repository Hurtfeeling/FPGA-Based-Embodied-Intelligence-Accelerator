"""Check standard FP32 export and integer mask handling without the policy."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto as T, helper, numpy_helper

from scripts.export_smolvla_fp32_onnx import FP32ExecutionMode, audit, fp32_time_embedding


class ReductionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)

    def forward(self, value, mask):
        with FP32ExecutionMode():
            positions = mask.cumsum(-1).float().unsqueeze(-1)
            count = mask.sum(-1, keepdim=True).float().unsqueeze(-1)
            return self.linear(value + positions + count)


class FP32OnnxTest(unittest.TestCase):
    def test_fp64_forward_outputs_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "Non-FP32"):
            with FP32ExecutionMode():
                torch.ones(2, dtype=torch.float64)

    def test_time_embedding_is_fp32_and_close_to_fp64_formula(self):
        time = torch.tensor([0.0, 0.123, 0.987], dtype=torch.float32)
        with FP32ExecutionMode():
            actual = fp32_time_embedding(time, 720, 0.004, 4.0, device=torch.device("cpu"))
        self.assertEqual(actual.dtype, torch.float32)
        fraction = np.linspace(0, 1, 360, dtype=np.float64)
        radians = time.numpy().astype(np.float64)[:, None] * (2 * np.pi / (0.004 * 1000 ** fraction))
        expected = np.concatenate([np.sin(radians), np.cos(radians)], -1)
        np.testing.assert_allclose(actual.numpy(), expected, atol=0.0003, rtol=0)

    def test_standard_fp32_graph_with_boolean_reductions(self):
        torch.manual_seed(4)
        model = ReductionBlock().eval()
        value = torch.randn(1, 4, 5)
        mask = torch.tensor([[True, True, False, True]])
        with torch.inference_mode():
            reference = model(value, mask).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            torch.onnx.export(model, (value, mask), str(path), input_names=["value", "mask"],
                              output_names=["actions"], dynamo=False, opset_version=18,
                              do_constant_folding=False)
            report = audit(path)
            self.assertTrue(report["all_floating_tensors_fp32"])
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"value": value.numpy(), "mask": mask.numpy()})[0]
            inputs = Path(directory) / "inputs.npz"
            output = Path(directory) / "actions.npz"
            np.savez(inputs, value=value.numpy(), mask=mask.numpy(), reference_actions=reference)
            runner = Path(__file__).resolve().parents[3] / "scripts" / "run_smolvla_fp32_onnx.py"
            completed = subprocess.run([sys.executable, str(runner), "--model", str(path),
                                        "--inputs", str(inputs), "--output", str(output)],
                                       check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(completed.stdout)["dtype"], "float32")
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["actions"], actual)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)

    def test_audit_rejects_fp16_weights(self):
        tensor = numpy_helper.from_array(np.ones(2, np.float16), "weights")
        graph = helper.make_graph([], "bad", [], [helper.make_tensor_value_info("weights", T.FLOAT16, [2])], [tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.onnx"
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, "Non-FP32"):
                audit(path)


if __name__ == "__main__":
    unittest.main()
