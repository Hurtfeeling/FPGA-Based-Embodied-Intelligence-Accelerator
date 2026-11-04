"""Fused four-core NEON UINT8->signed INT8 packing directly into CMA."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np

SOURCE = Path(__file__).with_name('pack_a_neon.cpp')
LIBRARY = SOURCE.with_name('libpack_a_neon.so')


class NeonAPacker:
    def __init__(self):
        if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
            temporary = LIBRARY.with_suffix('.so.tmp')
            subprocess.check_call(['g++', '-O3', '-mcpu=cortex-a53', '-std=c++14',
                                   '-fPIC', '-shared', '-fopenmp', str(SOURCE),
                                   '-o', str(temporary)])
            temporary.replace(LIBRARY)
        self.lib = ctypes.CDLL(str(LIBRARY))
        self.lib.pack_a_u8_to_s8.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.pack_a_u8_to_s8.restype = ctypes.c_int
        self.cpu_ids = (ctypes.c_int * 4)(0, 1, 2, 3)

    def __call__(self, src, dst, rows, k, padded_rows, padded_k):
        src, dst = np.asarray(src), np.asarray(dst)
        if src.dtype != np.uint8 or dst.dtype != np.int8:
            raise ValueError('Expected UINT8 source and INT8 destination')
        if src.ndim != 2 or dst.shape != (padded_rows, padded_k):
            raise ValueError('Invalid A pack shapes')
        if src.shape[0] < rows or src.shape[1] < k or src.strides[1] != 1:
            raise ValueError('A source must be row-contiguous and cover requested rows')
        threads = 4 if padded_rows * padded_k >= 262144 else 1
        rc = self.lib.pack_a_u8_to_s8(src.ctypes.data, dst.ctypes.data,
            rows, k, padded_rows, padded_k, src.strides[0], threads, self.cpu_ids)
        if rc:
            raise RuntimeError('NEON A packing failed: {}'.format(rc))
