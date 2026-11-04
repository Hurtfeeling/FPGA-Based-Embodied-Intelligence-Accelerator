"""Board-only Softmax host-copy/PL pipeline microbenchmark.

Run from the isolated runtime root with:
    sudo python3 -m runtime.benchmark_softmax_pipeline --output softmax_pipeline.json
This does not load an overlay or change CPU/PL clocks.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'GEMM'))
from softmax_pipeline import run_softmax_pipeline


SHAPES = [(1, 12, 1024, 1024), (1, 15, 177, 177),
          (1, 15, 50, 177), (1, 15, 50, 227)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--buffer-elements', type=int, default=2 * 1024**2)
    parser.add_argument('--hwh', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'GEMM/hardware/design_1.hwh')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error('positive repeats and nonnegative warmups required')
    from accelerator_driver import AcceleratorDriver
    from pynq import allocate, Clocks

    source = destination = control = None
    report = dict(clock=dict(cpu_mhz=float(Clocks.cpu_mhz), pl_mhz=float(Clocks.fclk0_mhz)),
                  buffer_elements=args.buffer_elements, shapes=[],
                  scope='Host-to-CMA copy, cache sync, Softmax submit/wait, output copy; '
                        'no session loading, input generation or correctness comparison')
    try:
        source = allocate((args.buffer_elements,), dtype=np.float32, cacheable=True)
        destination = allocate((args.buffer_elements,), dtype=np.float32, cacheable=True)
        control = AcceleratorDriver.from_hwh(args.hwh, poll_interval=0)
        rng = np.random.RandomState(20261104)
        for shape in SHAPES:
            x = rng.normal(0, 3, shape).astype(np.float32)
            baseline = np.empty_like(x)
            candidate = np.empty_like(x)
            row = dict(shape=list(shape), records=[], all_bitwise_equal=True)
            modes = [('sequential', int(x.size) + source.size), ('pipeline_auto', None),
                     ('pipeline_forced', 0)]
            # Establish the exact PL result independently of task partitioning.
            run_softmax_pipeline(control, source, destination, x, baseline,
                                 minimum_elements=int(x.size) + source.size)
            sample = x.reshape(-1, shape[-1])[:16]
            exp = np.exp(sample - np.max(sample, axis=1, keepdims=True))
            reference = exp / np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
            np.testing.assert_allclose(baseline.reshape(-1, shape[-1])[:16], reference,
                                       rtol=3e-5, atol=2e-6)
            for repeat in range(args.warmups + args.repeats):
                # Rotate test order to reduce cache/order bias.
                order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
                for name, threshold in order:
                    result = run_softmax_pipeline(control, source, destination, x, candidate,
                                                   minimum_elements=threshold)
                    np.testing.assert_array_equal(candidate, baseline)
                    if repeat >= args.warmups:
                        result.update(mode=name, repeat=repeat - args.warmups, bitwise_equal=True)
                        row['records'].append(result)
                        print('SOFTMAX_PIPELINE', shape, name, result['seconds'],
                              'tasks', result['tasks'], flush=True)
            row['median_seconds'] = {
                name: statistics.median(r['seconds'] for r in row['records'] if r['mode'] == name)
                for name, threshold in modes}
            report['shapes'].append(row)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        report['complete'] = True
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('SOFTMAX_PIPELINE_SUMMARY', json.dumps(report['shapes'][-1]['median_seconds']), flush=True)
    finally:
        if control is not None:
            # Never release CMA while an asynchronous PL task still uses it.
            while control.pending:
                try:
                    control.wait_all(timeout_s=30)
                except TimeoutError:
                    time.sleep(0.01)
            control.close()
        if destination is not None:
            destination.freebuffer()
        if source is not None:
            source.freebuffer()


if __name__ == '__main__':
    main()
