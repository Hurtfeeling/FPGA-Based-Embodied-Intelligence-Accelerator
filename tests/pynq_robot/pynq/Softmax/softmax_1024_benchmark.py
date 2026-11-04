"""Test only the loaded GEMM_V2 FP32 Softmax on a 1024x1024 matrix."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
import time

import numpy as np


def stable_softmax(x):
    peak = np.max(x, axis=1, keepdims=True)
    exp = np.exp(x - peak).astype(np.float32, copy=False)
    denom = np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
    return exp / denom


def main(repeats=8, warmups=2):
    base = Path(__file__).resolve().parent
    board_dir = Path(__file__).resolve().parent.parent / 'GEMM'
    sys.path.insert(0, str(board_dir))
    from accelerator_driver import AcceleratorDriver
    from pynq import Clocks, PL, allocate

    bitfile = str(Path(board_dir / 'hardware/design_1.bit').resolve())
    if str(Path(PL.bitfile_name).resolve()) != bitfile:
        raise RuntimeError('GEMM_V2 bitstream is not currently loaded; refusing to reload/interrupt PL')
    m = k = 1024
    stride_bytes = ((k + 3) // 4) * 16
    assert stride_bytes == k * 4

    rng = np.random.RandomState(20261102)
    host_input = rng.normal(0.0, 3.0, (m, k)).astype(np.float32)
    reference = stable_softmax(host_input)
    result_dir = base / 'benchmark_results'
    result_dir.mkdir(exist_ok=True)
    output_json = result_dir / ('softmax_1024_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
    src = allocate(shape=(m, k), dtype=np.float32, cacheable=1)
    dst = allocate(shape=(m, k), dtype=np.float32, cacheable=1)
    driver = None
    records = []
    try:
        for name, buf in [('src', src), ('dst', dst)]:
            addr = int(buf.device_address)
            if addr % 16 or addr + buf.nbytes > 0x80000000:
                raise RuntimeError('{} CMA span is not aligned/in V2 HP0 DDR range: {:#x}+{}'.format(
                    name, addr, buf.nbytes))
        if np.shares_memory(src, dst):
            raise RuntimeError('Softmax input and output CMA buffers overlap')
        driver = AcceleratorDriver.from_hwh(board_dir / 'hardware/design_1.hwh', poll_interval=0.0001)

        def run_pl(repeat, warmup=False):
            t0 = time.perf_counter()
            np.copyto(src, host_input, casting='no')
            t1 = time.perf_counter()
            src.sync_to_device()
            dst.sync_to_device()
            t2 = time.perf_counter()
            task_result = driver.run_softmax(int(src.device_address), int(dst.device_address), m, k,
                                             timeout_s=10.0)
            t3 = time.perf_counter()
            dst.sync_from_device()
            t4 = time.perf_counter()
            actual = np.array(dst, dtype=np.float32, copy=True, subok=False)
            t5 = time.perf_counter()
            np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=2e-6)
            row_sums = actual.sum(axis=1, dtype=np.float64)
            if not np.isfinite(actual).all() or np.max(np.abs(row_sums - 1.0)) > 2e-5:
                raise AssertionError('Softmax output contains non-finite values or invalid row sums')
            record = dict(mode='PL_SOFTMAX', repeat=repeat,
                input_copy_s=t1-t0, input_cache_sync_s=t2-t1,
                submit_and_wait_s=t3-t2, output_cache_sync_s=t4-t3,
                output_copy_s=t5-t4, total_s=t5-t0,
                driver_elapsed_s=task_result['software_elapsed_s'],
                max_abs_error=float(np.max(np.abs(actual-reference))),
                max_row_sum_error=float(np.max(np.abs(row_sums-1.0))),
                src_addr=int(src.device_address), dst_addr=int(dst.device_address),
                src_coherent=bool(src.coherent), dst_coherent=bool(dst.coherent),
                verified=True)
            if not warmup:
                records.append(record)
            print(('WARMUP' if warmup else 'MEASURE'), repeat,
                  'total_ms={:.3f}'.format(record['total_s']*1000),
                  'PL_call_ms={:.3f}'.format(record['submit_and_wait_s']*1000),
                  'max_err={:.3g}'.format(record['max_abs_error']), flush=True)
            return record

        for i in range(warmups):
            run_pl(i+1, warmup=True)

        cpu_records = []
        # Alternate CPU and PL order to reduce warm-cache/order bias.
        for i in range(1, repeats+1):
            if i % 2:
                run_pl(i)
                start = time.perf_counter()
                cpu_output = stable_softmax(host_input)
                cpu_s = time.perf_counter()-start
            else:
                start = time.perf_counter()
                cpu_output = stable_softmax(host_input)
                cpu_s = time.perf_counter()-start
                run_pl(i)
            np.testing.assert_allclose(cpu_output, reference, rtol=1e-6, atol=1e-7)
            cpu_records.append(dict(mode='CPU_NUMPY_FP32', repeat=i, compute_s=cpu_s,
                                    max_abs_error=float(np.max(np.abs(cpu_output-reference))),
                                    verified=True))

        pl_fields = ['input_copy_s','input_cache_sync_s','submit_and_wait_s',
                     'output_cache_sync_s','output_copy_s','total_s','driver_elapsed_s']
        pl_summary = {key:dict(median=statistics.median(r[key] for r in records),
                               mean=statistics.mean(r[key] for r in records),
                               minimum=min(r[key] for r in records),
                               maximum=max(r[key] for r in records)) for key in pl_fields}
        cpu_times=[r['compute_s'] for r in cpu_records]
        summary=dict(shape=[m,k],dtype='FP32',stride_bytes=stride_bytes,
            elements=m*k,pl_mhz=float(Clocks.fclk0_mhz),
            cpu_mhz=float(Clocks.cpu_mhz),warmups=warmups,repeats=repeats,
            input_sha256=hashlib.sha256(host_input.tobytes()).hexdigest(),
            cpu=dict(median_s=statistics.median(cpu_times),mean_s=statistics.mean(cpu_times),
                     min_s=min(cpu_times),max_s=max(cpu_times),
                     median_melements_per_s=m*k/statistics.median(cpu_times)/1e6),
            pl={**pl_summary,
                'median_end_to_end_melements_per_s':m*k/pl_summary['total_s']['median']/1e6,
                'median_submit_wait_melements_per_s':m*k/pl_summary['submit_and_wait_s']['median']/1e6},
            max_abs_error=max(r['max_abs_error'] for r in records),
            max_row_sum_error=max(r['max_row_sum_error'] for r in records),
            all_verified=True,
            timing_scope='CPU NumPy FP32 stable softmax; PL total includes input copy to CMA, input/output sync, accelerator submit+wait, output copy to host; no PL cycle counter exposed by this driver')
        output_json.write_text(json.dumps(dict(summary=summary,pl_records=records,cpu_records=cpu_records),indent=2))
        print('SUMMARY', json.dumps(summary), flush=True)
        print('RESULT', str(output_json), flush=True)
        return summary
    finally:
        if driver is not None:
            if driver.pending:
                driver.wait_all(timeout_s=10.0)
            driver.close()
        dst.freebuffer()
        src.freebuffer()


if __name__ == '__main__':
    main()
