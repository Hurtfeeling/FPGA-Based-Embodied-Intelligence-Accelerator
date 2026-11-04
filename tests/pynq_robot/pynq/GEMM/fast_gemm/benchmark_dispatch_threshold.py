"""Measure four-core NEON versus complete PL paths for model-sized GEMMs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from pynq import Clocks,PL

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
from runtime.operators import PynqOperators
from runtime.neon_runtime import highest_cpu_environment


def matrices(shapes,seed):
    rng=np.random.RandomState(seed)
    result=[]
    for m,n,k in shapes:
        a=rng.randint(0,256,(m,k)).astype(np.uint8)
        b=rng.randint(-8,8,(k,n)).astype(np.int8)
        c=np.empty((m,n),np.int32)
        result.append((a,b,c))
    return result


def run_benchmark(repeats):
    bitfile=ROOT/'hardware/design_1.bit'
    if not PL.bitfile_name or Path(PL.bitfile_name).resolve()!=bitfile.resolve():
        raise RuntimeError('Load the matching GEMM_V2 bitstream first')
    # Exact model shapes plus small padded shapes around the CPU/PL crossover.
    cases=[('32x32x32',[(32,32,32)]),
           ('50x64x64',[(50,64,64)]),
           ('50x50x64',[(50,50,64)]),
           ('64x64x64',[(64,64,64)]),
           ('50x720x320',[(50,320,720)]),
           ('50x720x960',[(50,960,720)]),
           ('50x720x2048',[(50,2048,720)]),
           ('50x2048x720',[(50,720,2048)]),
           ('64x768x768',[(64,768,768)]),
           ('128x768x768',[(128,768,768)]),
           ('256x768x768',[(256,768,768)]),
           ('1024x768x768',[(1024,768,768)]),
           ('1024x3072x768',[(1024,3072,768)]),
           ('1024x768x3072',[(1024,768,3072)]),
           ('group3_attention',[(50,64,64),(50,50,64),(50,320,64)])]
    result=dict(repeats=repeats,pl_mhz=float(Clocks.fclk0_mhz),
                cpu_mhz=float(Clocks.cpu_mhz),cpu_threads=4,
                cpu_kernel='native four-thread Cortex-A53 NEON',
                pl_kernel='GEMM_V2 native AXI-lite path; all staging/sync/copy included',
                records=[],threshold_selection='operations are based on dimensions padded to 32')
    ops=PynqOperators(cpu_threshold_ops=0)
    try:
        for ci,(name,shapes) in enumerate(cases):
            data=matrices(shapes,220+ci)
            expected=[]
            for a,b,c in data:
                ops._cpu_gemm(a,b,c)
                expected.append(c.copy())
            def cpu_run():
                for a,b,c in data:ops._cpu_gemm(a,b,c)
            def pl_run():
                if len(data)==1:
                    a,b,c=data[0];ops._gemm_matrix(a,b,c)
                else:
                    if not ops._gemm_group(data):raise RuntimeError('group does not fit CMA slots')
            cpu_run();pl_run()
            for (_,_,c),ref in zip(data,expected):np.testing.assert_array_equal(c,ref)
            cpu_values=[];pl_values=[]
            for i in range(repeats+1):
                begin=time.perf_counter();cpu_run();cpu_s=time.perf_counter()-begin
                for (_,_,c),ref in zip(data,expected):np.testing.assert_array_equal(c,ref)
                begin=time.perf_counter();pl_run();pl_s=time.perf_counter()-begin
                for (_,_,c),ref in zip(data,expected):np.testing.assert_array_equal(c,ref)
                if i:
                    cpu_values.append(cpu_s);pl_values.append(pl_s)
                print('WARMUP' if i==0 else 'MEASURE',name,'cpu_ms={:.3f}'.format(cpu_s*1000),
                      'pl_ms={:.3f}'.format(pl_s*1000),flush=True)
            logical_ops=sum(2*a.shape[0]*b.shape[1]*a.shape[1] for a,b,_ in data)
            padded_job_ops=[2*((a.shape[0]+31)//32*32)*((b.shape[1]+31)//32*32)*
                            ((a.shape[1]+31)//32*32) for a,b,_ in data]
            row=dict(name=name,shapes=[list(a.shape[:1])+[b.shape[1],a.shape[1]] for a,b,_ in data],
                     padded_ops=sum(padded_job_ops),padded_job_ops=padded_job_ops,
                     logical_ops=logical_ops,cpu_median_ms=statistics.median(cpu_values)*1000,
                     pl_median_ms=statistics.median(pl_values)*1000,
                     speedup=statistics.median(cpu_values)/statistics.median(pl_values),
                     cpu_samples_ms=[x*1000 for x in cpu_values],
                     pl_samples_ms=[x*1000 for x in pl_values],
                     driver_groups=ops.metrics['gemm_groups'],driver_core_jobs=ops.metrics['gemm_core_jobs'])
            result['records'].append(row)
            print('SUMMARY',json.dumps(row),flush=True)
        # Conservative threshold: highest measured padded-op size where 4T NEON
        # still wins. Keep grouped work on PL only if all three jobs fit CMA.
        winners=[max(r['padded_job_ops']) for r in result['records']
                 if r['cpu_median_ms']<=r['pl_median_ms']]
        result['cpu_threshold_ops']=max(winners) if winners else 0
        result['result_file']=str(BASE/'dispatch_threshold_results.json')
        Path(result['result_file']).write_text(json.dumps(result,indent=2))
        print('RESULT',json.dumps(result),flush=True)
    finally:ops.close()


def main(repeats):
    with highest_cpu_environment(keep_highest=True):
        os.sched_setaffinity(0,{0,1,2,3})
        run_benchmark(repeats)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=7)
    main(parser.parse_args().repeats)
