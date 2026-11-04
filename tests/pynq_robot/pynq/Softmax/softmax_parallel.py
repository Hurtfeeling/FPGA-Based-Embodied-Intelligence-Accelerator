"""Opt-in row-disjoint CPU ORT and PL Softmax execution.

Only the CPU partition runs on a persistent worker. ctypes releases the GIL
while ORT runs, so it can overlap the main thread's complete PL copy/execute
path. The overlap metric below measures those call intervals; it is not an
IP cycle-counter measurement.
"""
import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import math
import os
import time

import numpy as np

from softmax_pipeline import run_softmax_pipeline


def _tensors(x, out):
    if (x.dtype != np.float32 or out.dtype != np.float32 or x.shape != out.shape or
            x.ndim < 1 or not x.shape[-1] or not x.flags.c_contiguous or
            not out.flags.c_contiguous):
        raise ValueError('Softmax requires matching contiguous nonempty FP32 tensors')
    if np.shares_memory(x, out):
        raise ValueError('Softmax input/output must be disjoint')
    k = int(x.shape[-1])
    return x.reshape(-1, k), out.reshape(-1, k), k


class CpuSoftmax:
    """Own one reusable tiny ORT session through the board C++ bridge."""
    def __init__(self, library_path, model_path, threads=4):
        if int(threads) != threads or not 1 <= int(threads) <= 64:
            raise ValueError('CPU Softmax thread count must be in [1,64]')
        self.lib = ctypes.CDLL(os.fspath(library_path))
        self.lib.smol_softmax_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.smol_softmax_create.restype = ctypes.c_void_p
        self.lib.smol_softmax_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.smol_softmax_run.restype = ctypes.c_int
        self.lib.smol_softmax_destroy.argtypes = [ctypes.c_void_p]
        self.lib.smol_softmax_destroy.restype = None
        self.lib.smol_softmax_error.argtypes = []
        self.lib.smol_softmax_error.restype = ctypes.c_char_p
        self.threads = int(threads)
        self.handle = self.lib.smol_softmax_create(os.fsencode(model_path), self.threads)
        if not self.handle:
            raise RuntimeError(self._error())

    def _error(self):
        message = self.lib.smol_softmax_error()
        return message.decode('utf-8', 'replace') if message else 'CPU Softmax bridge failed'

    def run(self, x, out):
        if not self.handle:
            raise RuntimeError('CPU Softmax session is closed')
        xx, yy, k = _tensors(x, out)
        if self.lib.smol_softmax_run(self.handle, int(xx.ctypes.data), int(yy.ctypes.data),
                                     int(len(xx)), k):
            raise RuntimeError(self._error())

    def close(self):
        if self.handle:
            self.lib.smol_softmax_destroy(self.handle)
            self.handle = None


class SoftmaxRouter:
    """Choose CPU, PL, or simultaneous row partitions for each Softmax call.

    For 0 < cpu_fraction < 1 only tensors larger than
    minimum_split_elements are split. Smaller tensors use CPU once, avoiding
    a second dispatch and CMA copies. Full-CPU/full-PL settings bypass that
    size rule. The caller must serialize calls to one router; close() waits
    for the worker but does not close its borrowed CPU session or operators.
    """
    def __init__(self, ops, cpu, cpu_fraction=1.0, pipeline=False,
                 minimum_split_elements=2097152):
        self.cpu_fraction = float(cpu_fraction)
        if not math.isfinite(self.cpu_fraction) or not 0 <= self.cpu_fraction <= 1:
            raise ValueError('cpu_fraction must be in [0,1]')
        self.minimum_split_elements = int(minimum_split_elements)
        if self.minimum_split_elements < 0:
            raise ValueError('minimum_split_elements must be nonnegative')
        self.ops, self.cpu, self.pipeline = ops, cpu, bool(pipeline)
        if self.cpu_fraction > 0 and self.cpu is None:
            raise ValueError('A CPU Softmax session is required')
        self.executor = None
        self.closed = False
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = dict(calls=0, cpu_calls=0, pl_calls=0, mixed_calls=0,
                            cpu_rows=0, pl_rows=0, cpu_seconds=0.0, pl_seconds=0.0,
                            overlap_seconds=0.0, wall_seconds=0.0,
                            pipeline_calls=0, pipeline_tasks=0,
                            shape_metrics={}, interval_records=[])

    @contextmanager
    def _scratch(self, needed):
        lock = getattr(self.ops, '_scratch_lock', None) if needed else None
        if lock is None:
            yield
        else:
            with lock:
                yield

    def _run_pl(self, x, out):
        begin = time.perf_counter()
        details = None
        if self.pipeline:
            activate = getattr(self.ops, '_activate_softmax_scratch', None)
            if activate is not None:
                activate()
            elif getattr(self.ops, 'shared_scratch', False):
                raise RuntimeError('Shared scratch requires its ownership hook')
            details = run_softmax_pipeline(self.ops.control, self.ops.sin, self.ops.sout, x, out)
            self.ops.metrics['softmax_calls'] += 1
            self.ops.metrics['softmax_tasks'] += details['tasks']
            self.ops.metrics['softmax_seconds'] += time.perf_counter() - begin
        else:
            self.ops.softmax(x, out)
        return begin, time.perf_counter(), details

    def _run_cpu(self, x, out):
        begin = time.perf_counter()
        self.cpu.run(x, out)
        return begin, time.perf_counter()

    def __call__(self, x, out):
        if self.closed:
            raise RuntimeError('Softmax router is closed')
        xx, yy, k = _tensors(x, out)
        rows = len(xx)
        f = self.cpu_fraction
        if f == 0:
            cpu_rows = 0
        elif f == 1 or x.size <= self.minimum_split_elements or rows < 2:
            cpu_rows = rows
        else:
            cpu_rows = max(1, min(rows - 1, int(round(rows * f))))
        pl_rows = rows - cpu_rows
        cpu_interval = pl_interval = None
        details = None
        begin = time.perf_counter()
        # The complete callback retains shared scratch until both partitions
        # finish. Only the PL partition touches CMA; CPU outputs are disjoint.
        with self._scratch(pl_rows > 0):
            if cpu_rows and pl_rows:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1)
                future = self.executor.submit(self._run_cpu, xx[:cpu_rows], yy[:cpu_rows])
                try:
                    a, b, details = self._run_pl(xx[cpu_rows:], yy[cpu_rows:])
                    pl_interval = (a, b)
                finally:
                    # Await CPU even if PL fails: no background writes may
                    # outlive the callback's host tensor lifetime.
                    cpu_interval = future.result()
            elif pl_rows:
                a, b, details = self._run_pl(xx, yy)
                pl_interval = (a, b)
            elif cpu_rows:
                cpu_interval = self._run_cpu(xx, yy)
        end = time.perf_counter()
        cpu_s = 0.0 if cpu_interval is None else cpu_interval[1] - cpu_interval[0]
        pl_s = 0.0 if pl_interval is None else pl_interval[1] - pl_interval[0]
        overlap = (max(0.0, min(cpu_interval[1], pl_interval[1]) -
                       max(cpu_interval[0], pl_interval[0]))
                   if cpu_interval and pl_interval else 0.0)
        record = dict(shape=list(x.shape), cpu_rows=cpu_rows, pl_rows=pl_rows,
                      cpu_interval=list(cpu_interval) if cpu_interval else None,
                      pl_path_interval=list(pl_interval) if pl_interval else None,
                      cpu_seconds=cpu_s, pl_seconds=pl_s, overlap_seconds=overlap,
                      wall_seconds=end - begin,
                      pipelined=bool(details and details['pipelined']),
                      pl_tasks=details['tasks'] if details else None)
        metrics = self.metrics
        metrics['calls'] += 1
        metrics['cpu_calls'] += int(cpu_rows > 0)
        metrics['pl_calls'] += int(pl_rows > 0)
        metrics['mixed_calls'] += int(cpu_rows > 0 and pl_rows > 0)
        for name, value in (('cpu_rows', cpu_rows), ('pl_rows', pl_rows),
                            ('cpu_seconds', cpu_s), ('pl_seconds', pl_s),
                            ('overlap_seconds', overlap), ('wall_seconds', end - begin)):
            metrics[name] += value
        if details:
            metrics['pipeline_calls'] += int(details['pipelined'])
            metrics['pipeline_tasks'] += details['tasks']
        shape_metric = metrics['shape_metrics'].setdefault(str(tuple(x.shape)),
            dict(calls=0, cpu_rows=0, pl_rows=0, cpu_seconds=0.0, pl_seconds=0.0,
                 overlap_seconds=0.0, wall_seconds=0.0))
        shape_metric['calls'] += 1
        for name in ('cpu_rows', 'pl_rows', 'cpu_seconds', 'pl_seconds',
                     'overlap_seconds', 'wall_seconds'):
            shape_metric[name] += record[name]
        metrics['interval_records'].append(record)
        return record

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.closed = True
