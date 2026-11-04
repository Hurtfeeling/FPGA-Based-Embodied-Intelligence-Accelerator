"""Directional DMA mapping, real cacheable PYNQ CMA, verified each round."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import struct
import time

import numpy as np
from pynq import allocate
from benchmark_dma import dma_copy, SIZE, REGISTER, TO_DMA, FROM_DMA


def main():
    os.sched_setaffinity(0, {0, 1, 2, 3})
    records, registrations = [], []
    for direction, command in [('host_to_cma', TO_DMA), ('cma_to_host', FROM_DMA)]:
        cma = allocate(shape=(SIZE,), dtype=np.uint8, cacheable=1)
        target = np.asarray(cma)
        host = np.empty(SIZE, dtype=np.uint8)
        expected = np.random.RandomState(1).randint(0, 256, SIZE, dtype=np.uint8)
        np.copyto(host, expected)
        fd = os.open('/dev/ps_dma_trial', os.O_RDWR)
        try:
            fcntl.ioctl(fd, 0xD105, int(command == TO_DMA))
            start = time.perf_counter()
            fcntl.ioctl(fd, 0xD104, struct.pack('=2Q', cma.ctypes.data, SIZE))
            fcntl.ioctl(fd, REGISTER, struct.pack('=2Q', host.ctypes.data, SIZE))
            registrations.append(dict(direction=direction, both_register_ms=(time.perf_counter()-start)*1000))
            for repeat in range(9):
                expected ^= np.uint8(31 + repeat)
                methods = ['CPU', 'DMA'] if repeat % 2 else ['DMA', 'CPU']
                for method in methods:
                    if command == TO_DMA:
                        np.copyto(host, expected)
                        target.fill(0)
                        source, dest = host, target
                    else:
                        np.copyto(target, expected)
                        cma.sync_to_device()  # setup; source ready as if PL wrote it
                        host.fill(0)
                        source, dest = target, host
                    if method == 'DMA':
                        result = dma_copy(fd, command)
                    else:
                        wall, cpu = time.perf_counter(), time.process_time()
                        # Full CPU path includes the CMA sync needed by GEMM.
                        if command == FROM_DMA:
                            cma.sync_from_device()
                        np.copyto(dest, source)
                        copy_end = time.perf_counter()
                        if command == TO_DMA:
                            cma.sync_to_device()
                        result = dict(wall_ms=(time.perf_counter()-wall)*1000,
                                      cpu_ms=(time.process_time()-cpu)*1000,
                                      copy_and_presync_ms=(copy_end-wall)*1000)
                    assert np.array_equal(dest, expected), (method, direction, repeat)
                    if repeat:
                        records.append(dict(direction=direction, method=method, repeat=repeat, **result))
        finally:
            os.close(fd)  # releases DMA maps / page refs BEFORE freeing XLNK owner
            del target
            cma.freebuffer()
    summary = {}
    for direction in ['host_to_cma', 'cma_to_host']:
        for method in ['CPU', 'DMA']:
            rows = [r for r in records if r['direction']==direction and r['method']==method]
            summary[method+'_'+direction] = {
                'median_'+key:statistics.median(r[key] for r in rows)
                for key in rows[0] if key not in ['direction','method','repeat']}
    result = dict(bytes=SIZE, registrations=registrations, summary=summary,
                  records=records, all_verified=True,
                  scope='ordinary NumPy <-> PYNQ allocate(cacheable=1); directional mapping, full required cache sync included; independent buffers, not full GEMM')
    Path('dma_directional_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    main()
