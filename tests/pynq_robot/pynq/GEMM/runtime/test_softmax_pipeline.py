"""Local protocol checks; does not require PYNQ or a board."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'softmax_pipeline_under_test', str(Path(__file__).with_name('softmax_pipeline.py')))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run_softmax_pipeline = module.run_softmax_pipeline


class Buffer(np.ndarray):
    def __new__(cls, size):
        owner = np.empty(size + 16, dtype=np.float32)
        offset = (-int(owner.ctypes.data) % 64) // 4
        result = owner[offset:offset + size].view(cls)
        result.control = None
        return result

    def __array_finalize__(self, original):
        self.control = getattr(original, 'control', None)

    @property
    def device_address(self):
        return int(self.ctypes.data)

    def _check(self, output):
        control = self.control
        if control and control.pending:
            current = control.pending.destination if output else control.pending.source
            if np.shares_memory(current, self):
                raise AssertionError('Cache maintenance overlaps an in-flight PL range')
            if output:
                control.overlapped_drains += 1

    def sync_to_device(self):
        self._check(False)

    def sync_from_device(self):
        self._check(True)


class Task:
    def __init__(self, control, source, destination, k):
        self.control, self.source, self.destination, self.k = control, source, destination, k
        self.saved = source.copy()

    def wait(self, timeout_s):
        np.testing.assert_array_equal(self.source, self.saved)
        x = self.source[:, :self.k]
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.destination[:, :self.k] = e / e.sum(axis=1, keepdims=True)
        self.control.pending = None


class Control:
    def __init__(self, source, destination):
        self.source, self.destination = source, destination
        source.control = destination.control = self
        self.pending, self.overlapped_drains, self.submits = None, 0, 0

    def submit_softmax(self, src_addr, dst_addr, rows, k):
        if self.pending:
            raise AssertionError('One Softmax IP accepts only one active task')
        stride = (k + 3) // 4 * 4
        first = (src_addr - self.source.device_address) // 4
        last = first + rows * stride
        si = self.source[first:last].reshape(rows, stride)
        first = (dst_addr - self.destination.device_address) // 4
        so = self.destination[first:first + rows * stride].reshape(rows, stride)
        self.pending = Task(self, si, so, k)
        self.submits += 1
        return self.pending


class PipelineTest(unittest.TestCase):
    def run_case(self, rows, k, capacity, threshold=None):
        source, destination = Buffer(capacity), Buffer(capacity)
        source.fill(-111)
        destination.fill(-222)
        x = np.random.RandomState(2).normal(size=(rows, k)).astype(np.float32)
        y = np.empty_like(x)
        control = Control(source, destination)
        result = run_softmax_pipeline(control, source, destination, x, y,
                                      minimum_elements=threshold)
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        np.testing.assert_array_equal(y, e / e.sum(axis=-1, keepdims=True))
        self.assertIsNone(control.pending)
        self.assertEqual(result['tasks'], control.submits)
        return control, result

    def test_padded_k_and_tail(self):
        control, result = self.run_case(7, 177, 1024)
        self.assertTrue(result['pipelined'])
        self.assertEqual(result['rows_per_task'], 2)
        self.assertEqual(result['tasks'], 4)
        self.assertEqual(control.overlapped_drains, 3)

    def test_exact_k_and_repeated_slots(self):
        control, result = self.run_case(5, 1024, 2048)
        self.assertTrue(result['pipelined'])
        self.assertEqual(result['tasks'], 5)
        self.assertEqual(control.overlapped_drains, 4)

    def test_auto_keeps_small_tensor_one_task(self):
        control, result = self.run_case(2, 177, 1024)
        self.assertFalse(result['pipelined'])
        self.assertEqual(result['tasks'], 1)
        self.assertEqual(control.overlapped_drains, 0)

    def test_forced_and_disabled_modes(self):
        self.assertTrue(self.run_case(3, 227, 2048, threshold=0)[1]['pipelined'])
        self.assertFalse(self.run_case(20, 227, 2048, threshold=10000)[1]['pipelined'])

    def test_reject_host_alias(self):
        a, b = Buffer(2048), Buffer(2048)
        x = np.zeros((2, 227), dtype=np.float32)
        with self.assertRaises(ValueError):
            run_softmax_pipeline(Control(a, b), a, b, x, x)


if __name__ == '__main__':
    unittest.main()
