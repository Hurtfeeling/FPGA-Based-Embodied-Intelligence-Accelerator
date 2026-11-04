#!/usr/bin/python3
"""Compare CMA/cache/Softmax policies with one loaded ONNX session.

Every configuration gets its own warmup and measured full graph rounds.
CPU Softmax uses the same ORT CPU kernel as the original graph, with external
input/output buffers. Timing includes callback, packing, synchronization and PL.
"""
from __future__ import print_function
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import signal
from collections import defaultdict

import numpy as np
from infer_board import (CALLBACK, U, INPUT_NAMES, load_inputs, tensor,
                         errors_against, frequency_info, write_json, sha256)


def cma_info():
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('CmaTotal:', 'CmaFree:')):
            key, value, _ = line.split()
            result[key[:-1] + '_kib'] = int(value)
    return result


def main():
    # Legacy XLNK allocations outlive a killed process. Route termination
    # through the existing finally block so CMA is drained and released.
    def terminate(signum, frame):
        raise KeyboardInterrupt('Sweep terminated by signal {}'.format(signum))
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--cpu-baseline', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--softmax-library', type=Path, required=True)
    p.add_argument('--softmax-model', type=Path, required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--micro-repeats', type=int, default=3)
    p.add_argument('--configurations', type=Path)
    p.add_argument('--micro-results', type=Path,
                   help='Reuse measured per-shape policies and skip random microbenchmarks')
    p.add_argument('--native-softmax', action='store_true',
                   help='Use a GEMM-only graph with ordinary in-graph CPU Softmax')
    args = p.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.micro_repeats < 1:
        p.error('Invalid repeat count')
    prefix = args.output_prefix.resolve()
    report_path = Path(str(prefix) + '.json')
    if report_path.exists():
        p.error('Refusing to overwrite an existing report')
    os.sched_setaffinity(0, {0, 1, 2, 3})
    # Avoid idle ORT pools consuming cores while the other processor works.
    os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
    sys.path.insert(0, str(args.runtime_root.resolve()))
    from runtime.operators import PynqOperators
    from runtime.softmax_parallel import CpuSoftmax, SoftmaxRouter
    manifest = json.loads(args.manifest.read_text())
    data, references = load_inputs(args.inputs)
    baseline = np.load(str(args.cpu_baseline), allow_pickle=False)
    if baseline.shape != (1, 50, 6) or baseline.dtype != np.float32:
        raise ValueError('Invalid CPU baseline')
    base_sidecar = args.cpu_baseline.with_suffix('.json')
    if base_sidecar.exists():
        prev = json.loads(base_sidecar.read_text())
        if prev.get('inputs_sha256') != sha256(args.inputs):
            raise ValueError('CPU baseline inputs differ')
    report = dict(status='starting', model=str(args.model),
                  model_sha256=sha256(args.model), inputs_sha256=sha256(args.inputs),
                  cpu_baseline=str(args.cpu_baseline), clock=frequency_info(),
                  idle_cma=cma_info(), microbenchmark=[], records=[], summaries=[],
                  native_softmax=args.native_softmax,
                  timing_scope='full smol_run including copies/cache/dispatch/CPU/PL; '
                               'session load, setup and artifact writes excluded',
                  ort=dict(intra=4, inter=1, graph_opt='disable', execution='sequential',
                           allow_spinning=os.environ.get('SMOL_ORT_ALLOW_SPINNING','1')))
    write_json(report_path, report)
    cpus = {}
    routers = {}
    ops = handle = None
    callback_errors = []
    state = dict(softmax_calls=0, gemm_calls=0, softmax_seconds=0.0,
                 shapes={}, round_start=0.0, last_progress=0.0)

    def cpu(threads):
        if threads not in cpus:
            cpus[threads] = CpuSoftmax(args.softmax_library, args.softmax_model, threads)
        return cpus[threads]

    def make_ops(cache_mib, shared):
        limit = int(cache_mib) * 1024**2
        slots = [r for r in manifest['resident_cache_slots']
                 if int(r['offset_bytes']) + int(r['size_bytes']) <= limit]
        return PynqOperators(cpu_threshold_ops=0, resident_cache_slots=slots,
                             resident_cache_bytes=limit, softmax_buffer_elements=2*1024**2,
                             shared_scratch=shared)

    def close_routers():
        for router in routers.values():
            router.close()
        routers.clear()

    def callback(kind, constant_b, cache_id, ap, sa, ra, bp, sb, rb, op, so, ro):
        try:
            ashape = tuple(sa[i] for i in range(ra))
            oshape = tuple(so[i] for i in range(ro))
            if kind == 0:
                bshape = tuple(sb[i] for i in range(rb))
                ops.matmul_integer(tensor(ap, ashape, np.uint8),
                    tensor(bp, bshape, np.int8), tensor(op, oshape, np.int32),
                    int(constant_b), int(cache_id))
                state['gemm_calls'] += 1
            elif kind == 1:
                begin = time.perf_counter()
                router = routers.get(ashape, routers.get('default'))
                router(tensor(ap, ashape, np.float32), tensor(op, oshape, np.float32))
                elapsed = time.perf_counter() - begin
                state['softmax_calls'] += 1
                state['softmax_seconds'] += elapsed
                item = state['shapes'].setdefault(str(ashape), dict(calls=0, seconds=0.0))
                item['calls'] += 1
                item['seconds'] += elapsed
            else:
                raise ValueError('Unknown callback kind')
            now = time.perf_counter()
            if now - state['last_progress'] >= 20:
                print('PROGRESS', json.dumps(dict(config=state.get('config'),
                    gemm_calls=state['gemm_calls'], softmax_calls=state['softmax_calls'],
                    round_seconds=now-state['round_start'])), flush=True)
                state['last_progress'] = now
            return 0
        except BaseException as e:
            import traceback
            callback_errors.append(traceback.format_exc())
            print('CALLBACK_ERROR', callback_errors[-1], flush=True)
            return 1

    cb = CALLBACK(callback)
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.smol_create_hybrid.argtypes = [ctypes.c_char_p, ctypes.c_int, CALLBACK]
    lib.smol_create_hybrid.restype = U
    lib.smol_run.argtypes = [U] * 8
    lib.smol_run.restype = ctypes.c_int
    lib.smol_destroy.argtypes = [U]
    lib.smol_error.restype = ctypes.c_char_p
    try:
        # Small Softmax-only tuning avoids reloading the large ONNX graph.
        if args.micro_results is None:
            ops = make_ops(80, True)
            report['micro_cma'] = cma_info()
        shapes = [(1,12,1024,1024), (1,15,177,177), (1,15,50,177), (1,15,50,227)]
        policies = [dict(name='cpu4', fraction=1.0, threads=4, pipeline=False),
                    dict(name='pl_seq', fraction=0.0, threads=1, pipeline=False),
                    dict(name='pl_pipeline', fraction=0.0, threads=1, pipeline=True)]
        for threads in (2, 3):
            for fraction in (0.25, 0.5, 0.75):
                policies.append(dict(name='split{}_t{}'.format(fraction,threads),
                                     fraction=fraction, threads=threads, pipeline=True))
        winners = {}
        rng = np.random.RandomState(20261104)
        for shape in (shapes if args.micro_results is None else []):
            x = rng.normal(0,3,shape).astype(np.float32)
            out = np.empty_like(x)
            ref = np.empty_like(x)
            cpu(4).run(x,ref)
            candidates = []
            for policy in policies:
                if np.prod(shape) <= 2*1024**2 and policy['fraction'] not in (0.0,1.0):
                    continue
                router = SoftmaxRouter(ops,cpu(policy['threads']),
                    cpu_fraction=policy['fraction'],pipeline=policy['pipeline'])
                try:
                    router(x,out)
                    times=[]
                    router.reset_metrics()
                    for repeat in range(args.micro_repeats):
                        begin=time.perf_counter();router(x,out)
                        times.append(time.perf_counter()-begin)
                    err=errors_against(out,ref)
                    item=dict(shape=list(shape),policy=policy,times=times,
                        median_seconds=float(np.median(times)),vs_ort_cpu=err,
                        router_metrics=dict(router.metrics))
                    report['microbenchmark'].append(item)
                    candidates.append(item)
                    print('MICRO',json.dumps(item),flush=True)
                finally:
                    router.close()
            winners[shape]=min(candidates,key=lambda q:q['median_seconds'])['policy']
            write_json(report_path,report)
        if args.micro_results is None:
            del x,out,ref,router
            close_routers();ops.close();ops=None
        else:
            import ast
            prior=json.loads(args.micro_results.read_text())
            winners={tuple(ast.literal_eval(k)):v for k,v in prior['micro_selected_policies'].items()}
            report['microbenchmark']=prior['microbenchmark']
            report['micro_results_source']=str(args.micro_results)
        import gc
        gc.collect()
        report['after_micro_cma']=cma_info()
        report['micro_selected_policies']={str(k):v for k,v in winners.items()}
        configs = (json.loads(args.configurations.read_text()) if args.configurations else [
            dict(name='separate64_cpu4',cache_mib=64,shared=False,softmax='cpu4'),
            dict(name='shared80_cpu4',cache_mib=80,shared=True,softmax='cpu4'),
            dict(name='shared88_cpu4',cache_mib=88,shared=True,softmax='cpu4'),
            dict(name='shared88_pl_seq',cache_mib=88,shared=True,softmax='pl_seq'),
            dict(name='shared88_pl_pipeline',cache_mib=88,shared=True,softmax='pl_pipeline'),
            dict(name='shared88_shape_auto',cache_mib=88,shared=True,softmax='auto')])
        report['configurations']=configs
        write_json(report_path,report)
        print('SESSION_START',flush=True)
        begin=time.perf_counter()
        handle=lib.smol_create_hybrid(os.fsencode(str(args.model)),4,cb)
        report['session_load_seconds']=time.perf_counter()-begin
        if not handle:raise RuntimeError(lib.smol_error().decode())
        print('SESSION_READY',report['session_load_seconds'],flush=True)
        previous=None
        output=np.empty((1,50,6),np.float32)
        pointers=[U(data[name].ctypes.data) for name in INPUT_NAMES]
        for config in configs:
            close_routers()
            key=(config['cache_mib'],config['shared'])
            if key!=previous:
                if ops is not None:ops.close()
                ops=make_ops(*key);previous=key
            ops.a_pack_mode=config.get('a_pack_mode','full')
            if ops.a_pack_mode not in ('full','edges','neon','adaptive'):
                raise ValueError('Unknown A pack mode '+str(ops.a_pack_mode))
            chosen={p['name']:p for p in policies}
            if args.native_softmax:
                pass
            elif config['softmax']=='auto':
                for shape,policy in winners.items():
                    routers[shape]=SoftmaxRouter(ops,cpu(policy['threads']),
                        cpu_fraction=policy['fraction'],pipeline=policy['pipeline'])
            else:
                policy=chosen[config['softmax']]
                routers['default']=SoftmaxRouter(ops,cpu(policy['threads']),
                    cpu_fraction=policy['fraction'],pipeline=policy['pipeline'])
            print('CONFIG_START',json.dumps(config),json.dumps(cma_info()),flush=True)
            for repeat in range(args.warmups+args.repeats):
                phase='warmup' if repeat<args.warmups else 'measure'
                ops.reset_metrics()
                for router in routers.values():router.reset_metrics()
                state.update(config=config['name'],gemm_calls=0,softmax_calls=0,
                    softmax_seconds=0.0,shapes={},round_start=time.perf_counter(),
                    last_progress=time.perf_counter())
                callback_errors[:]=[]
                begin=time.perf_counter()
                code=lib.smol_run(handle,*pointers,U(output.ctypes.data))
                elapsed=time.perf_counter()-begin
                if code:raise RuntimeError(lib.smol_error().decode()+'\n'+'\n'.join(callback_errors))
                if not np.isfinite(output).all():raise RuntimeError('Non-finite model output')
                if state['gemm_calls']!=1121 or state['softmax_calls']!=(0 if args.native_softmax else 199):
                    raise RuntimeError('Unexpected graph callback counts')
                item=dict(config=config['name'],repeat=repeat,phase=phase,seconds=elapsed,
                    clock=frequency_info(),cma=cma_info(),operators=dict(ops.metrics),
                    gemm_callbacks=state['gemm_calls'],softmax_callbacks=state['softmax_calls'],
                    softmax_seconds=state['softmax_seconds'],softmax_shapes=state['shapes'],
                    routers={str(k):dict(v.metrics) for k,v in routers.items()},
                    vs_board_cpu=errors_against(output,baseline),
                    vs_export_references={n:errors_against(output,v) for n,v in references.items()})
                path=Path(str(prefix)+'.{}_round{:02d}_{}.npy'.format(config['name'],repeat,phase))
                np.save(str(path),output,allow_pickle=False);item['output_file']=str(path)
                report['records'].append(item)
                write_json(report_path,report)
                print('ROUND_RESULT',json.dumps(item),flush=True)
            records=[r for r in report['records'] if r['config']==config['name'] and r['phase']=='measure']
            summary=dict(config=config,median_seconds=float(np.median([r['seconds'] for r in records])),
                seconds=[r['seconds'] for r in records],
                softmax_median_seconds=float(np.median([r['softmax_seconds'] for r in records])),
                max_output_error=max(r['vs_board_cpu']['max_abs'] for r in records))
            report['summaries'].append(summary)
            write_json(report_path,report)
            print('CONFIG_RESULT',json.dumps(summary),flush=True)
        report['status']='complete'
        report['fastest']=min(report['summaries'],key=lambda r:r['median_seconds'])
        exact=[r for r in report['summaries'] if r['max_output_error']==0]
        report['fastest_exact']=min(exact,key=lambda r:r['median_seconds']) if exact else None
        write_json(report_path,report)
        print('SWEEP_COMPLETE',json.dumps(report['fastest']),flush=True)
    except BaseException:
        import traceback
        report['status']='failed';report['error']=traceback.format_exc()
        write_json(report_path,report)
        raise
    finally:
        close_routers()
        if handle:lib.smol_destroy(handle)
        if ops is not None:ops.close()
        for instance in cpus.values():instance.close()


if __name__=='__main__':main()
