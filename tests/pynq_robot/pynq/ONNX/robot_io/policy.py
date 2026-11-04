"""One persistent instance of the existing CPU/PL C++ ONNX bridge."""
import ctypes
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from infer_board import CALLBACK, INPUT_NAMES, SCHEMA, U, tensor


class OnnxPolicy(object):
    def __init__(self, model, mode='hybrid', library=None, runtime_root=None,
                 runtime_config=None, ort_directory=None):
        if mode not in ('cpu', 'hybrid'):
            raise ValueError('mode must be cpu or hybrid')
        self.model = Path(model).resolve()
        self.mode = mode
        self.library = Path(library or BASE / 'libsmolvla_ort_sweep.so').resolve()
        self.runtime_root = Path(runtime_root or BASE.parent / 'GEMM').resolve()
        self.ort_directory = Path(ort_directory or BASE / 'onnxruntime-linux-aarch64-1.16.3')
        if isinstance(runtime_config, (str, Path)):
            runtime_config = json.loads(Path(runtime_config).read_text(encoding='utf-8'))
        self.config = dict(runtime_config or {'resident_cache_mib': 80, 'shared_scratch': True})
        if self.config.get('softmax'):
            raise ValueError('Robot entry uses the native CPU Softmax graph/configuration')
        self.handle = self.ops = self.lib = self.ort = self.callback = None
        self.errors = []
        self.lock = threading.Lock()
        self.callback_lock = threading.Lock()
        self.metrics = {}

    def _error(self):
        message = self.lib.smol_error()
        return message.decode('utf-8', 'replace') if message else 'ONNX bridge failed'

    def load(self):
        if self.handle is not None:
            return self
        for path in (self.model, self.library):
            if not path.is_file():
                raise FileNotFoundError(str(path))
        environment = dict(SMOL_ORT_INTRA_OP_THREADS='4', SMOL_ORT_INTER_OP_THREADS='1',
            SMOL_ORT_EXECUTION_MODE='sequential', SMOL_ORT_GRAPH_OPT='disable',
            SMOL_ORT_CPU_ARENA='0', SMOL_ORT_MEMORY_PATTERN='0',
            SMOL_ORT_ALLOW_SPINNING='1', OMP_WAIT_POLICY='PASSIVE')
        for name, value in environment.items():
            os.environ[name] = value
        if hasattr(os, 'sched_setaffinity'):
            if not {0, 1, 2, 3}.issubset(os.sched_getaffinity(0)):
                raise RuntimeError('Four CPU cores must be available')
            os.sched_setaffinity(0, {0, 1, 2, 3})
        try:
            if self.mode == 'hybrid':
                path = self.model.with_suffix('.manifest.json')
                if not path.is_file():
                    raise FileNotFoundError('Matching model manifest required: ' + str(path))
                manifest = json.loads(path.read_text(encoding='utf-8'))
                if manifest.get('pl_softmax_enabled', False):
                    raise ValueError('Select the GEMM-only resident graph; robot entry keeps native CPU Softmax')
                budget = int(self.config.get('resident_cache_mib', 80)) * 1024**2
                slots = [slot for slot in manifest.get('resident_cache_slots', [])
                         if int(slot['offset_bytes']) + int(slot['size_bytes']) <= budget]
                sys.path.insert(0, str(self.runtime_root))
                from runtime.operators import PynqOperators
                self.ops = PynqOperators(cpu_threshold_ops=0,
                    resident_cache_slots=slots, resident_cache_bytes=budget,
                    shared_scratch=bool(self.config.get('shared_scratch', True)),
                    softmax_buffer_elements=manifest.get('softmax_cma_buffer_elements', 1024**2))

            def dispatch(kind, constant_b, cache_id, ap, sa, ra, bp, sb, rb, op, so, ro):
                with self.callback_lock:
                    try:
                        if kind != 0 or self.ops is None:
                            raise RuntimeError('Only selected GEMM nodes are supported by this robot entry')
                        ashape = tuple(sa[i] for i in range(ra))
                        bshape = tuple(sb[i] for i in range(rb))
                        oshape = tuple(so[i] for i in range(ro))
                        self.ops.matmul_integer(tensor(ap, ashape, np.uint8),
                            tensor(bp, bshape, np.int8), tensor(op, oshape, np.int32),
                            int(constant_b), int(cache_id))
                        return 0
                    except BaseException:
                        self.errors.append(traceback.format_exc())
                        return 1

            self.callback = CALLBACK(dispatch)  # Keep alive until session destruction.
            ort_path = self.ort_directory / 'lib/libonnxruntime.so.1.16.3'
            self.ort = ctypes.CDLL(str(ort_path.resolve()), mode=ctypes.RTLD_GLOBAL)
            self.lib = ctypes.CDLL(str(self.library))
            self.lib.smol_error.restype = ctypes.c_char_p
            self.lib.smol_runtime_version.restype = ctypes.c_char_p
            self.lib.smol_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self.lib.smol_create.restype = U
            self.lib.smol_create_hybrid.argtypes = [ctypes.c_char_p, ctypes.c_int, CALLBACK]
            self.lib.smol_create_hybrid.restype = U
            self.lib.smol_run.argtypes = [U] * 8
            self.lib.smol_run.restype = ctypes.c_int
            self.lib.smol_destroy.argtypes = [U]
            self.lib.smol_destroy.restype = None
            print('SESSION_LOADING', self.mode, str(self.model), flush=True)
            started = time.monotonic()
            self.handle = (self.lib.smol_create_hybrid(os.fsencode(str(self.model)), 4, self.callback)
                           if self.mode == 'hybrid' else
                           self.lib.smol_create(os.fsencode(str(self.model)), 4))
            if not self.handle:
                raise RuntimeError(self._error())
            self.metrics['session_load_s'] = time.monotonic() - started
            self.metrics['ort_version'] = self.lib.smol_runtime_version().decode()
            print('SESSION_READY', json.dumps(self.metrics), flush=True)
            return self
        except BaseException:
            self.close()
            raise

    def predict(self, data):
        arrays = {}
        for name, (shape, dtype) in SCHEMA.items():
            value = np.asarray(data[name])
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError('{} requires finite {} {}'.format(name, shape, dtype))
            arrays[name] = np.ascontiguousarray(value)
        if not arrays['language_mask'].any() or (arrays['language_tokens'] < 0).any():
            raise ValueError('Invalid language tokens/mask')
        with self.lock:
            if not self.handle:
                raise RuntimeError('Load ONNX session first')
            self.errors[:] = []
            if self.ops is not None:
                self.ops.reset_metrics()
            output = np.empty((1, 50, 6), dtype=np.float32)
            pointers = [U(arrays[name].ctypes.data) for name in INPUT_NAMES]
            started = time.monotonic()
            status = self.lib.smol_run(self.handle, *pointers, U(output.ctypes.data))
            self.metrics['inference_s'] = time.monotonic() - started
            if status or self.errors:
                raise RuntimeError(self._error() + '\n' + '\n'.join(self.errors))
            if not np.isfinite(output).all():
                raise RuntimeError('Model output contains nonfinite actions')
            self.metrics['operators'] = dict(self.ops.metrics) if self.ops else {}
            return output

    def close(self):
        with self.lock:
            if self.handle is not None:
                self.lib.smol_destroy(self.handle)
                self.handle = None
            if self.ops is not None:
                self.ops.close()
                self.ops = None
            self.callback = None

    def __enter__(self):
        return self.load()

    def __exit__(self, *unused):
        self.close()
