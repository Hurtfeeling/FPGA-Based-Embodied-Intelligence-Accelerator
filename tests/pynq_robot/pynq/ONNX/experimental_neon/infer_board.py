#!/usr/bin/python3
"""Measure the two-camera W8A8 model on PYNQ CPU or selected PL GEMM nodes.

Only nodes explicitly converted to domain=pynq invoke the accelerator. All
other MatMulInteger nodes and every Softmax remain in ONNX Runtime on CPU.
Compatible with the board's Python 3.6 and NumPy.
"""
from __future__ import print_function
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

import numpy as np

BASE = Path(__file__).resolve().parent
INPUT_NAMES = ('camera2', 'camera3', 'state', 'language_tokens', 'language_mask', 'noise')
SCHEMA = {
    'camera2': ((1, 3, 480, 640), np.dtype('float32')),
    'camera3': ((1, 3, 480, 640), np.dtype('float32')),
    'state': ((1, 6), np.dtype('float32')),
    'language_tokens': ((1, 48), np.dtype('int64')),
    'language_mask': ((1, 48), np.dtype('bool')),
    'noise': ((1, 50, 32), np.dtype('float32')),
}
U = ctypes.c_void_p
I = ctypes.POINTER(ctypes.c_int64)
S = ctypes.c_size_t
CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           U, I, S, U, I, S, U, I, S)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(path):
    with np.load(str(path), allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in INPUT_NAMES}
        references = {name: arrays[name].copy() for name in
                      ('reference_actions', 'fp32_actions', 'onnx_actions') if name in arrays}
    for name, (shape, dtype) in SCHEMA.items():
        value = data[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError('{} expected finite {} {}, got {} {}'.format(
                name, shape, dtype, value.shape, value.dtype))
        data[name] = np.ascontiguousarray(value)
    if np.any(data['language_tokens'] < 0) or not data['language_mask'].any():
        raise ValueError('Invalid language tokens or empty language mask')
    for name, value in references.items():
        if value.shape != (1, 50, 6) or not np.isfinite(value).all():
            raise ValueError('Invalid reference tensor ' + name)
    return data, references


def tensor(pointer, dimensions, dtype):
    shape = tuple(int(dimensions[i]) for i in range(len(dimensions)))
    if not shape or min(shape) <= 0 or not pointer:
        raise ValueError('Invalid callback tensor pointer/shape')
    count = int(np.prod(shape))
    element = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
    return np.ctypeslib.as_array(ctypes.cast(pointer, ctypes.POINTER(element)),
                                shape=(count,)).reshape(shape)


def frequency_info():
    result = {}
    try:
        from pynq import Clocks
        result['cpu_mhz'] = float(Clocks.cpu_mhz)
        result['pl_mhz'] = float(Clocks.fclk0_mhz)
    except BaseException as error:
        result['clock_read_error'] = str(error)
    policy = Path('/sys/devices/system/cpu/cpu0/cpufreq')
    for name in ('scaling_min_freq', 'scaling_max_freq', 'scaling_cur_freq', 'cpuinfo_cur_freq'):
        try:
            result[name + '_khz'] = int((policy / name).read_text().strip())
        except (OSError, ValueError):
            result[name + '_khz'] = None
    return result


def errors_against(output, reference):
    delta = np.abs(output.astype(np.float64) - reference.astype(np.float64))
    return {'mae': float(delta.mean()), 'max_abs': float(delta.max())}


def stable_softmax_diagnostic(inputs, outputs, row_chunk=256):
    """Compare one PL Softmax result to stable FP32 NumPy in bounded memory."""
    k = inputs.shape[-1]
    x = inputs.reshape(-1, k)
    y = outputs.reshape(-1, k)
    rows = x.shape[0]
    total_abs = 0.0
    max_abs = 0.0
    max_row_sum_error = 0.0
    neg_inf = pos_inf = nan_count = fully_masked = 0
    input_finite = output_finite = True
    for first in range(0, rows, row_chunk):
        last = min(first + row_chunk, rows)
        block_x = x[first:last]
        block_y = y[first:last]
        nan_count += int(np.isnan(block_x).sum())
        pos_inf += int(np.isposinf(block_x).sum())
        neg_inf += int(np.isneginf(block_x).sum())
        input_finite = input_finite and bool(np.isfinite(block_x).all())
        output_finite = output_finite and bool(np.isfinite(block_y).all())
        fully_masked += int(np.isneginf(block_x).all(axis=1).sum())
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            peak = np.max(block_x, axis=1, keepdims=True)
            ex = np.exp(block_x - peak).astype(np.float32, copy=False)
            ex[~np.isfinite(ex)] = 0.0
            denom = np.sum(ex, axis=1, keepdims=True, dtype=np.float32)
            ref = np.divide(ex, denom, out=np.zeros_like(ex), where=denom != 0)
        diff = np.abs(block_y.astype(np.float64) - ref.astype(np.float64))
        total_abs += float(diff.sum(dtype=np.float64))
        if diff.size:
            max_abs = max(max_abs, float(diff.max()))
        row_sums = block_y.sum(axis=1, dtype=np.float64)
        if row_sums.size:
            max_row_sum_error = max(max_row_sum_error,
                                    float(np.max(np.abs(row_sums - 1.0))))
    return dict(shape=list(inputs.shape), elements=int(inputs.size), input_finite=input_finite,
                input_neg_inf=neg_inf, input_pos_inf=pos_inf, input_nan=nan_count,
                fully_masked_rows=fully_masked, output_finite=output_finite,
                max_abs_error=max_abs, mean_abs_error=total_abs / max(1, int(inputs.size)),
                max_row_sum_error=max_row_sum_error)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(str(temporary), str(path))


def ort_config_from_env():
    def flag(name, default):
        value = os.environ.get(name, default).strip().lower()
        if value in ('1', 'true', 'yes', 'on'):
            return True
        if value in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError('{} must be a boolean value'.format(name))

    intra = int(os.environ.get('SMOL_ORT_INTRA_OP_THREADS', '4'))
    inter = int(os.environ.get('SMOL_ORT_INTER_OP_THREADS', '1'))
    if not 1 <= intra <= 64 or not 1 <= inter <= 64:
        raise ValueError('ORT intra/inter-op thread settings must be in [1,64]')
    execution = os.environ.get('SMOL_ORT_EXECUTION_MODE', 'sequential').strip().lower()
    optimization = os.environ.get('SMOL_ORT_GRAPH_OPT', 'disable').strip().lower()
    if execution not in ('sequential', 'parallel'):
        raise ValueError('SMOL_ORT_EXECUTION_MODE must be sequential or parallel')
    if optimization not in ('disable', 'disabled', 'basic', 'extended', 'all'):
        raise ValueError('SMOL_ORT_GRAPH_OPT must be disable/basic/extended/all')
    return dict(execution_mode=execution, intra_op_threads=intra, inter_op_threads=inter,
                graph_optimization=optimization,
                cpu_mem_arena=flag('SMOL_ORT_CPU_ARENA', '0'),
                memory_pattern=flag('SMOL_ORT_MEMORY_PATTERN', '0'))


def load_runtime_config(path):
    """Validate optional CMA and CPU/PL Softmax policies before board setup."""
    if path is None:
        return {}
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Runtime config must be a JSON object')
    budget = config.get('resident_cache_mib')
    if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget < 0):
        raise ValueError('resident_cache_mib must be a nonnegative integer')
    if 'shared_scratch' in config and not isinstance(config['shared_scratch'], bool):
        raise ValueError('shared_scratch must be a boolean')
    softmax = config.get('softmax', {})
    if not isinstance(softmax, dict):
        raise ValueError('softmax must be a JSON object')
    shapes = softmax.get('shapes', {})
    if not isinstance(shapes, dict):
        raise ValueError('softmax.shapes must map shape strings to policies')
    policies = list(shapes.items())
    if 'default' in softmax:
        policies.append(('default', softmax['default']))
    for name, policy in policies:
        if not isinstance(name, str) or not isinstance(policy, dict):
            raise ValueError('Softmax route names and policies must be strings and objects')
        fraction = policy.get('fraction', 1.0)
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
                not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError('Softmax fraction must be a finite CPU fraction in [0,1]')
        threads = policy.get('threads', 4)
        if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 4:
            raise ValueError('Softmax threads must be an integer in [1,4]')
        if 'pipeline' in policy and not isinstance(policy['pipeline'], bool):
            raise ValueError('Softmax pipeline must be a boolean')
    for name in ('softmax_library', 'softmax_model'):
        if name in config and (not isinstance(config[name], str) or not config[name]):
            raise ValueError('{} must be a nonempty path string'.format(name))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'hybrid'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=BASE / 'validation_0.npz')
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--runtime-root', type=Path,
                        help='GEMM_V2 directory containing runtime/, fast_gemm/, hardware/')
    parser.add_argument('--runtime-config', type=Path,
                        help='Optional JSON with resident CMA budget and CPU/PL Softmax routes')
    parser.add_argument('--library', type=Path, default=BASE / 'libsmolvla_ort_cpu.so')
    parser.add_argument('--cpu-baseline', type=Path,
                        help='NPY output from CPU mode using this input; required for hybrid')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--progress-every', type=int, default=50)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.progress_every < 1:
        parser.error('warmups must be nonnegative, repeats/progress-every must be positive')
    if args.mode == 'hybrid' and (args.runtime_root is None or args.cpu_baseline is None):
        parser.error('hybrid requires --runtime-root and --cpu-baseline')
    if args.runtime_config is not None and args.mode != 'hybrid':
        parser.error('--runtime-config requires hybrid mode')
    runtime_config = load_runtime_config(args.runtime_config)
    args.model = args.model.resolve()
    args.inputs = args.inputs.resolve()
    ort_config = ort_config_from_env()
    prefix = (args.output_prefix or BASE / 'results' /
              (args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(str(prefix) + '.json')
    if report_path.exists():
        parser.error('Output report already exists; choose a new --output-prefix')
    cpu_ids = {0, 1, 2, 3}
    available = set(os.sched_getaffinity(0))
    if not cpu_ids.issubset(available):
        raise RuntimeError('All four CPU cores must be available to this process')
    os.sched_setaffinity(0, cpu_ids)
    started = time.perf_counter()
    data, references = load_inputs(args.inputs)
    input_digest = sha256(args.inputs)
    model_digest = sha256(args.model)
    baseline = None
    baseline_info = None
    if args.cpu_baseline:
        baseline = np.load(str(args.cpu_baseline), allow_pickle=False)
        if baseline.shape != (1, 50, 6) or baseline.dtype != np.float32 or not np.isfinite(baseline).all():
            raise ValueError('Expected a finite FLOAT32 CPU baseline shaped (1,50,6)')
        sidecar = args.cpu_baseline.with_suffix('.json')
        if sidecar.exists():
            baseline_info = json.loads(sidecar.read_text())
            previous_digest = baseline_info.get('inputs_sha256')
            if previous_digest and previous_digest != input_digest:
                raise ValueError('CPU baseline was generated with a different input archive')
    report = dict(mode=args.mode, status='starting', model=str(args.model),
                  model_sha256=model_digest, model_bytes=args.model.stat().st_size,
                  bridge_library=str(args.library.resolve()),
                  inputs=str(args.inputs), inputs_sha256=input_digest,
                  cpu_baseline=str(args.cpu_baseline.resolve()) if args.cpu_baseline else None,
                  cpu_baseline_info=baseline_info, cpu_threads=ort_config['intra_op_threads'],
                  intra_op_threads=ort_config['intra_op_threads'],
                  inter_op_threads=ort_config['inter_op_threads'],
                  process_cpu_affinity=sorted(os.sched_getaffinity(0)),
                  graph_optimization=ort_config['graph_optimization'],
                  cpu_mem_arena=ort_config['cpu_mem_arena'],
                  memory_pattern=ort_config['memory_pattern'], ort_config=ort_config,
                  warmups=args.warmups, requested_repeats=args.repeats, records=[],
                  runtime_root=str(args.runtime_root.resolve()) if args.runtime_root else None,
                  clock_at_start=frequency_info(), ps_dma_used=False,
                  timing_scope='smol_run full graph; copies, callbacks, dispatch and PL included; '
                               'session initialization and report/output file writes excluded',
                  hybrid_scope='Only explicitly converted GEMM nodes use PL; Softmax placement '
                               'follows the selected graph manifest (CPU or PL)',
                  preparation_seconds=time.perf_counter() - started)
    if args.runtime_config is not None:
        report['runtime_config_file'] = str(args.runtime_config.resolve())
        report['runtime_config'] = runtime_config
    report['cpu_mhz'] = report['clock_at_start'].get('cpu_mhz')
    report['pl_mhz'] = report['clock_at_start'].get('pl_mhz')
    write_json(report_path, report)
    ops = None
    handle = None
    softmax_routers = {}
    softmax_cpus = {}
    callback_errors = []
    round_state = {'calls': 0, 'gemm_calls': 0, 'softmax_calls': 0,
                   'seconds': 0.0, 'shapes': {}, 'operator_intervals': [],
                   'softmax_diagnostics': {}, 'softmax_shape_metrics': {}}
    validate_softmax = os.environ.get('SMOL_SOFTMAX_VALIDATE', '').strip().lower() in (
        '1', 'true', 'yes', 'on')
    # GEMM and Softmax each have private CMA buffers and independent IPs, so
    # callbacks of different kinds may overlap. Same-kind callbacks remain
    # serialized because their staging buffers are shared.
    callback_locks = {0: threading.Lock(), 1: threading.Lock()}
    callback_state_lock = threading.Lock()

    def callback(kind, constant_b, cache_id, ap, sa, ra, bp, sb, rb, op, so, ro):
        begin = time.perf_counter()
        if kind not in callback_locks:
            message = 'Unsupported PynqOp kind {}'.format(kind)
            with callback_state_lock:
                callback_errors.append(message)
            return 1
        with callback_locks[kind]:
            try:
                if kind == 0:
                    ashape = tuple(sa[i] for i in range(ra))
                    bshape = tuple(sb[i] for i in range(rb))
                    oshape = tuple(so[i] for i in range(ro))
                    operation_start = time.perf_counter()
                    ops.matmul_integer(tensor(ap, ashape, np.uint8), tensor(bp, bshape, np.int8),
                                       tensor(op, oshape, np.int32), int(constant_b), int(cache_id))
                    key = 'GEMM A{} B{} constant_b={}'.format(ashape, bshape, int(constant_b))
                else:
                    xshape = tuple(sa[i] for i in range(ra))
                    oshape = tuple(so[i] for i in range(ro))
                    if xshape != oshape:
                        raise ValueError('Softmax input/output shape mismatch')
                    operation_start = time.perf_counter()
                    softmax_input = tensor(ap, xshape, np.float32)
                    softmax_output = tensor(op, oshape, np.float32)
                    tasks_before = ops.metrics['softmax_tasks']
                    router = softmax_routers.get(str(xshape), softmax_routers.get('default'))
                    if router is None:
                        ops.softmax(softmax_input, softmax_output)
                    else:
                        router(softmax_input, softmax_output)
                    if validate_softmax:
                        # Compare every callback to stable NumPy; accumulate by
                        # shape so model attention distributions are all covered.
                        diagnostic_key = str(xshape)
                        diagnostic_start = time.perf_counter()
                        diag = stable_softmax_diagnostic(softmax_input, softmax_output)
                        diag_seconds = time.perf_counter() - diagnostic_start
                        with callback_state_lock:
                            summary = round_state['softmax_diagnostics'].setdefault(
                                diagnostic_key, dict(shape=list(xshape), calls=0, elements=0,
                                    sum_abs_error=0.0, max_abs_error=0.0,
                                    max_row_sum_error=0.0, input_finite=True,
                                    input_neg_inf=0, input_pos_inf=0, input_nan=0,
                                    fully_masked_rows=0, output_finite=True,
                                    validation_seconds=0.0))
                            summary['calls'] += 1
                            summary['elements'] += diag['elements']
                            summary['sum_abs_error'] += diag['mean_abs_error'] * diag['elements']
                            summary['max_abs_error'] = max(summary['max_abs_error'],
                                                           diag['max_abs_error'])
                            summary['max_row_sum_error'] = max(summary['max_row_sum_error'],
                                                               diag['max_row_sum_error'])
                            summary['input_finite'] = summary['input_finite'] and diag['input_finite']
                            summary['output_finite'] = summary['output_finite'] and diag['output_finite']
                            for name in ('input_neg_inf', 'input_pos_inf', 'input_nan',
                                         'fully_masked_rows'):
                                summary[name] += diag[name]
                            summary['validation_seconds'] += diag_seconds
                    key = 'Softmax {}'.format(xshape)
                operation_end = time.perf_counter()
                with callback_state_lock:
                    round_state['calls'] += 1
                    round_state['gemm_calls' if kind == 0 else 'softmax_calls'] += 1
                    round_state['shapes'][key] = round_state['shapes'].get(key, 0) + 1
                    round_state['operator_intervals'].append((operation_start, operation_end))
                    if kind == 1:
                        shape_metric = round_state['softmax_shape_metrics'].setdefault(
                            str(xshape), dict(calls=0, tasks=0, seconds=0.0))
                        shape_metric['calls'] += 1
                        shape_metric['tasks'] += ops.metrics['softmax_tasks'] - tasks_before
                        shape_metric['seconds'] += operation_end - operation_start
                    call_count = round_state['calls']
                if call_count % args.progress_every == 0:
                    print('PL_PROGRESS', json.dumps(dict(round=round_state.get('round'),
                          callbacks=call_count, metrics=dict(ops.metrics))), flush=True)
                return 0
            except BaseException:
                message = traceback.format_exc()
                with callback_state_lock:
                    callback_errors.append(message)
                print(message, flush=True)
                return 1
            finally:
                with callback_state_lock:
                    round_state['seconds'] += time.perf_counter() - begin

    cb = CALLBACK(callback)

    def occupied_operator_seconds(intervals):
        """Union duration, so overlapping GEMM/Softmax is not double-counted."""
        if not intervals:
            return 0.0
        ordered = sorted(intervals)
        total = 0.0
        left, right = ordered[0]
        for start, end in ordered[1:]:
            if start <= right:
                right = max(right, end)
            else:
                total += right - left
                left, right = start, end
        return total + right - left

    try:
        if args.mode == 'hybrid':
            setup_begin = time.perf_counter()
            sys.path.insert(0, str(args.runtime_root.resolve()))
            from runtime.operators import PynqOperators
            # The graph converter already filters by measured dimensions and
            # records the optional resident static-B CMA slots in the manifest.
            # Avoid routing selected nodes back through a separate CPU heuristic.
            manifest_path = args.model.with_suffix('.manifest.json')
            if not manifest_path.exists():
                manifest_path = args.model.with_name('selective_graph_manifest.json')
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            resident_slots = manifest.get('resident_cache_slots', [])
            resident_bytes = manifest.get('resident_cache_capacity_bytes', 64*1024**2)
            if 'resident_cache_mib' in runtime_config:
                resident_bytes = runtime_config['resident_cache_mib'] * 1024**2
                resident_slots = [slot for slot in resident_slots
                                  if int(slot['offset_bytes']) + int(slot['size_bytes']) <= resident_bytes]
            operator_options = dict(cpu_threshold_ops=0,
                                    resident_cache_slots=resident_slots,
                                    resident_cache_bytes=resident_bytes,
                                    softmax_buffer_elements=manifest.get('softmax_cma_buffer_elements', 1024**2))
            if 'shared_scratch' in runtime_config:
                operator_options['shared_scratch'] = runtime_config['shared_scratch']
            ops = PynqOperators(**operator_options)
            softmax_config = runtime_config.get('softmax', {})
            policies = dict(softmax_config.get('shapes', {}))
            if 'default' in softmax_config:
                policies['default'] = softmax_config['default']
            if policies:
                from runtime.softmax_parallel import CpuSoftmax, SoftmaxRouter
                def softmax_path(name, default):
                    value = Path(runtime_config.get(name, default))
                    return (value if value.is_absolute() else BASE / value).resolve()
                cpu_library = softmax_path('softmax_library', 'libsmolvla_softmax_cpu.so')
                cpu_model = softmax_path('softmax_model', 'softmax_rows.onnx')
                resolved_policies = {}
                for name, policy in policies.items():
                    fraction = float(policy.get('fraction', 1.0))
                    threads = int(policy.get('threads', 4))
                    pipeline = bool(policy.get('pipeline', False))
                    if fraction > 0 and threads not in softmax_cpus:
                        softmax_cpus[threads] = CpuSoftmax(cpu_library, cpu_model, threads=threads)
                    softmax_routers[name] = SoftmaxRouter(
                        ops, softmax_cpus.get(threads), cpu_fraction=fraction, pipeline=pipeline)
                    resolved_policies[name] = dict(fraction=fraction, threads=threads, pipeline=pipeline)
                report['softmax_cpu_library'] = str(cpu_library)
                report['softmax_cpu_model'] = str(cpu_model)
                report['softmax_route_policies'] = resolved_policies
                report['softmax_route_timing_scope'] = (
                    'CPU ORT and complete PL path call intervals; overlap is wall time, not IP cycles')
            if args.runtime_config is not None:
                report['resident_cache_capacity_bytes'] = resident_bytes
                report['resident_cache_active_ids'] = [int(slot['cache_id']) for slot in resident_slots]
                report['shared_scratch'] = bool(getattr(ops, 'shared_scratch', False))
            report['graph_manifest'] = str(manifest_path) if manifest_path.exists() else None
            report['pl_softmax_enabled'] = bool(manifest.get('pl_softmax_enabled', False))
            report['softmax_cma_buffer_elements'] = ops.softmax_buffer_elements
            report['accelerator_setup_seconds'] = time.perf_counter() - setup_begin
        lib = ctypes.CDLL(str(args.library.resolve()))
        lib.smol_error.restype = ctypes.c_char_p
        lib.smol_runtime_version.restype = ctypes.c_char_p
        lib.smol_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.smol_create.restype = U
        lib.smol_create_hybrid.argtypes = [ctypes.c_char_p, ctypes.c_int, CALLBACK]
        lib.smol_create_hybrid.restype = U
        lib.smol_run.argtypes = [U] * 8
        lib.smol_run.restype = ctypes.c_int
        lib.smol_destroy.argtypes = [U]
        report['onnxruntime_version'] = lib.smol_runtime_version().decode()
        report['status'] = 'loading_session'
        write_json(report_path, report)
        print('SESSION_START', args.mode, str(args.model), flush=True)
        begin = time.perf_counter()
        handle = (lib.smol_create_hybrid(os.fsencode(str(args.model)), 4, cb)
                  if args.mode == 'hybrid' else lib.smol_create(os.fsencode(str(args.model)), 4))
        report['session_load_seconds'] = time.perf_counter() - begin
        if not handle:
            raise RuntimeError(lib.smol_error().decode('utf-8', 'replace'))
        report['status'] = 'running'
        write_json(report_path, report)
        print('SESSION_READY', args.mode, report['session_load_seconds'], flush=True)
        pointers = [U(data[name].ctypes.data) for name in INPUT_NAMES]
        output = np.empty((1, 50, 6), np.float32)
        for repeat in range(args.warmups + args.repeats):
            phase = 'warmup' if repeat < args.warmups else 'measure'
            with callback_state_lock:
                round_state.update(calls=0, gemm_calls=0, softmax_calls=0,
                                   seconds=0.0, shapes={}, operator_intervals=[],
                                   softmax_diagnostics={}, softmax_shape_metrics={}, round=repeat)
            if ops is not None:
                ops.reset_metrics()
            for router in softmax_routers.values():
                router.reset_metrics()
            print('ROUND_START', repeat, phase, flush=True)
            begin = time.perf_counter()
            code = lib.smol_run(handle, *pointers, U(output.ctypes.data))
            elapsed = time.perf_counter() - begin
            if code:
                raise RuntimeError(lib.smol_error().decode('utf-8', 'replace') + '\n' +
                                   '\n'.join(callback_errors))
            if not np.isfinite(output).all():
                raise RuntimeError('Inference produced a nonfinite output')
            if args.mode == 'hybrid' and round_state['calls'] == 0:
                raise RuntimeError('Hybrid graph executed zero accelerator callbacks')
            with callback_state_lock:
                callback_calls = round_state['calls']
                gemm_callback_calls = round_state['gemm_calls']
                softmax_callback_calls = round_state['softmax_calls']
                callback_seconds = round_state['seconds']
                callback_shapes = dict(round_state['shapes'])
                operator_intervals = list(round_state['operator_intervals'])
                softmax_diagnostics = {}
                for key, summary in round_state['softmax_diagnostics'].items():
                    summary = dict(summary)
                    summary['mean_abs_error'] = (summary.pop('sum_abs_error') /
                                                 max(1, summary['elements']))
                    softmax_diagnostics[key] = summary
                softmax_shape_metrics = dict(round_state['softmax_shape_metrics'])
            operator_busy = occupied_operator_seconds(operator_intervals)
            operator_sum = ((ops.metrics['gemm_seconds'] + ops.metrics['softmax_seconds'])
                            if ops is not None else 0.0)
            record = dict(repeat=repeat, phase=phase, seconds=elapsed,
                          callback_calls=callback_calls,
                          gemm_callback_calls=gemm_callback_calls,
                          softmax_callback_calls=softmax_callback_calls,
                          callback_seconds=callback_seconds,
                          callback_shapes=callback_shapes,
                          softmax_diagnostics=softmax_diagnostics,
                          softmax_shape_metrics=softmax_shape_metrics,
                          accelerator_operator_seconds=operator_sum,
                          accelerator_operator_busy_seconds=operator_busy,
                          accelerator_operator_overlap_seconds=max(0.0, operator_sum-operator_busy),
                          operators=dict(ops.metrics) if ops is not None else {},
                          clock=frequency_info(), output_finite=True,
                          output_min=float(output.min()), output_max=float(output.max()))
            if softmax_routers:
                record['softmax_routes'] = {name: dict(router.metrics)
                                            for name, router in softmax_routers.items()}
            if baseline is not None:
                record['vs_board_cpu'] = errors_against(output, baseline)
            record['vs_export_references'] = {name: errors_against(output, value)
                                              for name, value in references.items()}
            if ops is not None:
                record['other_cpu_and_graph_seconds'] = elapsed - operator_busy
            round_output = Path(str(prefix) + '.round_{:02d}_{}.npy'.format(repeat, phase))
            np.save(str(round_output), output, allow_pickle=False)
            record['output_file'] = str(round_output)
            round_report = Path(str(prefix) + '.round_{:02d}_{}.json'.format(repeat, phase))
            write_json(round_report, dict(mode=args.mode, model_sha256=model_digest,
                       inputs_sha256=input_digest, cpu_threads=4,
                       process_cpu_affinity=sorted(os.sched_getaffinity(0)), record=record))
            record['report_file'] = str(round_report)
            report['records'].append(record)
            measured = [r['seconds'] for r in report['records'] if r['phase'] == 'measure']
            report['completed_measured_repeats'] = len(measured)
            report['median_inference_seconds'] = float(np.median(measured)) if measured else None
            report['actual_pl_execution'] = any((r['operators'].get('gemm_tasks', 0) > 0 or
                                                 r['operators'].get('softmax_tasks', 0) > 0)
                                                 for r in report['records'])
            report['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report['output_shape'] = list(output.shape)
            report['output_dtype'] = str(output.dtype)
            report['output_file'] = str(prefix) + '.npy'
            np.save(report['output_file'], output, allow_pickle=False)
            write_json(report_path, report)
            print('MEASURE', json.dumps(record), flush=True)
        report['status'] = 'complete'
        report['clock_at_end'] = frequency_info()
        report['process_cpu_affinity_at_end'] = sorted(os.sched_getaffinity(0))
        write_json(report_path, report)
        print('SUMMARY', json.dumps(dict(report_file=str(report_path),
              median_inference_seconds=report['median_inference_seconds'],
              session_load_seconds=report['session_load_seconds'],
              actual_pl_execution=report['actual_pl_execution'])), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        write_json(report_path, report)
        raise
    finally:
        if handle is not None:
            lib.smol_destroy(handle)
        for router in softmax_routers.values():
            router.close()
        for cpu in softmax_cpus.values():
            cpu.close()
        if ops is not None:
            ops.close()


if __name__ == '__main__':
    main()
