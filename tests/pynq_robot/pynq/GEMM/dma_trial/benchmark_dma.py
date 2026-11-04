"""Run as root after loading ps_dma_trial. Never touches GEMM buffers."""
import fcntl
import json
import mmap
import os
from pathlib import Path
import statistics
import struct
import sys
import time

import numpy as np

SIZE = 4 * 1024 * 1024
REGISTER, TO_DMA, FROM_DMA = 0xD101, 0xD102, 0xD103


def dma_copy(fd, command):
    result = bytearray(40)
    wall, cpu = time.perf_counter(), time.process_time()
    fcntl.ioctl(fd, command, result, True)
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    values = struct.unpack('=5Q', result)
    return dict(wall_ms=wall * 1000, cpu_ms=cpu * 1000,
                kernel_ms=values[0] / 1e6, sync_ms=values[1] / 1e6,
                transfer_wait_ms=values[2] / 1e6,
                dma_segments=values[3], bytes=values[4])


def main():
    os.sched_setaffinity(0, {0, 1, 2, 3})
    fd = os.open('/dev/ps_dma_trial', os.O_RDWR)
    mapping = None
    cma = None
    use_cma = '--cma' in sys.argv
    try:
        if use_cma:
            from pynq import allocate
            cma = allocate(shape=(SIZE,), dtype=np.uint8, cacheable=1)
            device = np.asarray(cma)
            cma_start = time.perf_counter()
            fcntl.ioctl(fd, 0xD104, struct.pack('=2Q', cma.ctypes.data, SIZE))
            cma_register_ms = (time.perf_counter() - cma_start) * 1000
        else:
            mapping = mmap.mmap(fd, SIZE, flags=mmap.MAP_SHARED,
                                prot=mmap.PROT_READ | mmap.PROT_WRITE)
            device = np.ndarray((SIZE,), dtype=np.uint8, buffer=mapping)
            cma_register_ms = None
        host = np.empty(SIZE, dtype=np.uint8)
        rng = np.random.RandomState(1)
        expected = rng.randint(0, 256, SIZE, dtype=np.uint8)
        np.copyto(host, expected)
        register_start = time.perf_counter()
        fcntl.ioctl(fd, REGISTER, struct.pack('=2Q', host.ctypes.data, SIZE))
        register_ms = (time.perf_counter() - register_start) * 1000
        warmups = [dma_copy(fd, TO_DMA)]
        assert np.array_equal(device, expected), 'DMA host -> coherent mismatch'
        host.fill(0)
        warmups.append(dma_copy(fd, FROM_DMA))
        assert np.array_equal(host, expected), 'DMA coherent -> host mismatch'
        records = []
        for repeat in range(8):
            # Alter source contents every round to rule out stale cache success.
            expected ^= np.uint8(17 + repeat)
            np.copyto(host, expected)
            methods = ['CPU', 'DMA'] if repeat % 2 == 0 else ['DMA', 'CPU']
            for method in methods:
                device.fill(0)
                if method == 'DMA':
                    result = dma_copy(fd, TO_DMA)
                else:
                    wall, cpu = time.perf_counter(), time.process_time()
                    np.copyto(device, host)
                    result = dict(wall_ms=(time.perf_counter()-wall)*1000,
                                  cpu_ms=(time.process_time()-cpu)*1000)
                assert np.array_equal(device, expected), (method, 'to_dma', repeat)
                records.append(dict(method=method, direction='host_to_dma',
                                    repeat=repeat, **result))
                host.fill(0)
                if method == 'DMA':
                    result = dma_copy(fd, FROM_DMA)
                else:
                    wall, cpu = time.perf_counter(), time.process_time()
                    np.copyto(host, device)
                    result = dict(wall_ms=(time.perf_counter()-wall)*1000,
                                  cpu_ms=(time.process_time()-cpu)*1000)
                assert np.array_equal(host, expected), (method, 'from_dma', repeat)
                records.append(dict(method=method, direction='dma_to_host',
                                    repeat=repeat, **result))
        summary = {}
        for method in ['CPU', 'DMA']:
            for direction in ['host_to_dma', 'dma_to_host']:
                rows = [r for r in records if r['method'] == method and r['direction'] == direction]
                summary[method + '_' + direction] = {
                    'median_' + key: statistics.median(r[key] for r in rows)
                    for key in rows[0] if key not in ['method', 'direction', 'repeat']}
        result = dict(bytes=SIZE, register_ms=register_ms,
                      cma_register_ms=cma_register_ms, warmups=warmups,
                      summary=summary, records=records, all_verified=True,
                      scope=('ordinary NumPy host <-> PYNQ allocate(cacheable=1) CMA; independent owned buffer, NOT GEMM fixed windows'
                             if use_cma else 'ordinary NumPy host <-> driver-owned dma_alloc_coherent mmap; NOT GEMM fixed CMA windows'))
        Path('dma_cma_results.json' if use_cma else 'dma_results.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)
    finally:
        if mapping is not None:
            del device
            mapping.close()
        os.close(fd)
        if cma is not None:
            del device
            cma.freebuffer()


if __name__ == '__main__':
    main()
