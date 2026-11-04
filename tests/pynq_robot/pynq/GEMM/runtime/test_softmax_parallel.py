"""Local row partition/concurrency checks without board or ORT binaries."""
import importlib.util
from pathlib import Path
import sys
import threading
import time
import types
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
package = types.ModuleType('isolated_softmax_runtime')
package.__path__ = [str(HERE)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    package.__name__ + '.softmax_parallel', str(HERE / 'softmax_parallel.py'))
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)
spec = importlib.util.spec_from_file_location('pipeline_test_helpers', str(HERE / 'test_softmax_pipeline.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def stable(x, out):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    out[:] = e / e.sum(axis=-1, keepdims=True)


class Cpu:
    def __init__(self, barrier=None):
        self.barrier = barrier

    def run(self, x, out):
        if self.barrier:
            self.barrier.wait(timeout=2)
            time.sleep(0.01)
        stable(x, out)


class Ops:
    def __init__(self, barrier=None):
        self._scratch_lock = threading.RLock()
        self.barrier = barrier
        self.metrics = dict(softmax_calls=0, softmax_tasks=0, softmax_seconds=0)
        self.activations = 0
        self.sin = helpers.Buffer(1024)
        self.sout = helpers.Buffer(1024)
        self.control = helpers.Control(self.sin, self.sout)

    def softmax(self, x, out):
        if self.barrier:
            self.barrier.wait(timeout=2)
            time.sleep(0.01)
        stable(x, out)
        self.metrics['softmax_calls'] += 1
        self.metrics['softmax_tasks'] += 1

    def _activate_softmax_scratch(self):
        self.activations += 1


class ParallelTest(unittest.TestCase):
    def inputs(self, rows=20):
        x = np.random.RandomState(3).normal(size=(rows, 177)).astype(np.float32)
        return x, np.empty_like(x)

    def test_mixed_runs_simultaneously_and_fills_disjoint_rows(self):
        barrier = threading.Barrier(2)
        ops, cpu = Ops(barrier), Cpu(barrier)
        router = parallel.SoftmaxRouter(ops, cpu, cpu_fraction=0.25, minimum_split_elements=0)
        x, y = self.inputs()
        record = router(x, y)
        ref = np.empty_like(x)
        stable(x, ref)
        np.testing.assert_array_equal(y, ref)
        self.assertEqual((record['cpu_rows'], record['pl_rows']), (5, 15))
        self.assertGreater(record['overlap_seconds'], 0.005)
        self.assertEqual(router.metrics['mixed_calls'], 1)
        self.assertEqual(router.metrics['shape_metrics'][str(x.shape)]['cpu_rows'], 5)
        router.close()

    def test_mixed_small_tensor_uses_cpu_once(self):
        router = parallel.SoftmaxRouter(Ops(), Cpu(), cpu_fraction=0.25)
        x, y = self.inputs()
        record = router(x, y)
        self.assertEqual(record['pl_rows'], 0)
        self.assertEqual(router.ops.metrics['softmax_calls'], 0)
        router.close()

    def test_full_pl_pipeline_and_shared_activation(self):
        router = parallel.SoftmaxRouter(Ops(), None, cpu_fraction=0, pipeline=True)
        x, y = self.inputs()
        record = router(x, y)
        ref = np.empty_like(x)
        stable(x, ref)
        np.testing.assert_array_equal(y, ref)
        self.assertTrue(record['pipelined'])
        self.assertEqual(router.ops.activations, 1)
        self.assertEqual(router.ops.metrics['softmax_tasks'], 10)
        self.assertEqual(router.metrics['pipeline_calls'], 1)
        router.reset_metrics()
        self.assertEqual(router.metrics['calls'], 0)
        router.close()

    def test_cpu_failure_awaited(self):
        class FailedCpu:
            def run(self, x, out):
                time.sleep(0.01)
                raise RuntimeError('CPU failed')
        router = parallel.SoftmaxRouter(Ops(), FailedCpu(), cpu_fraction=0.5,
                                        minimum_split_elements=0)
        x, y = self.inputs()
        with self.assertRaisesRegex(RuntimeError, 'CPU failed'):
            router(x, y)
        router.close()


if __name__ == '__main__':
    unittest.main()
