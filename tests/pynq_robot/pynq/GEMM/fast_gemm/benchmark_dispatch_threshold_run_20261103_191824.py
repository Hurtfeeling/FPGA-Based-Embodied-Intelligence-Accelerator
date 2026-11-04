"""Measure four-core NEON versus full GEMM_V2 paths on model-shaped inputs."""
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


def matrices(shapes,seed,shared_b=False):
    rng=np.random.RandomState(seed)
    shared_weights={}
    result=[]
    for m,n,k in shapes:
        a=rng.randint(0,256,(m,k)).astype(np.uint8)
        key=(k,n)
        if shared_b and key in shared_weights:
            b=shared_weights[key]
        else:
            b=rng.randint(-8,8,(k,n)).astype(np.int8)
            if shared_b:shared_weights[key]=b
        result.append((a,b,np.empty((m,n),np.int32)))
    return result


def padded_ops(shape):
    m,n,k=shape
    up=lambda x:(x+31)//32*32
    return 2*up(m)*up(n)*up(k)


def run_benchmark(repeats,output_path=None):
    bitfile=ROOT/'hardware/design_1.bit'
    if not PL.bitfile_name or Path(PL.bitfile_name).resolve()!=bitfile.resolve():
        raise RuntimeError('Load the matching GEMM_V2 bitstream first')
    cases=[
        ('tiny_32',[(32,32,32)],False),
        ('small_50_64_64',[(50,64,64)],False),
        ('small_50_50_64',[(50,50,64)],False),
        ('small_64_64_64',[(64,64,64)],False),
        ('model_50_720_32',[(50,720,32)],False),
        ('model_50_32_720',[(50,32,720)],False),
        ('model_50_320_720',[(50,320,720)],False),
        ('model_50_960_720',[(50,960,720)],False),
        ('model_50_2048_720',[(50,2048,720)],False),
        ('model_50_720_2048',[(50,720,2048)],False),
        ('model_50_720_1440',[(50,720,1440)],False),
        ('model_50_720_720',[(50,720,720)],False),
        ('model_1_960_32',[(1,960,32)],False),
        ('model_1024_768_768',[(1024,768,768)],False),
        ('model_1024_3072_768',[(1024,3072,768)],False),
        ('model_1024_768_3072',[(1024,768,3072)],False),
        ('text_attention_qk',[(241,241,64)],False),
        ('text_attention_pv',[(241,64,241)],False),
        ('action_qk_241_3',[(50,241,64)]*3,True),
        ('action_pv_241_3',[(50,64,241)]*3,True),
        ('action_qk_291_3',[(50,291,64)]*3,True),
        ('action_pv_291_3',[(50,64,291)]*3,True),
        ('vision_attention_qk',[(1024,1024,64)],False),
        ('vision_attention_pv',[(1024,64,1024)],False),
        ('dynamic_text_projection_241_960_960',[(241,960,960)],False),
        ('dynamic_text_projection_241_2560_960',[(241,2560,960)],False),
        ('dynamic_text_projection_241_960_2560',[(241,960,2560)],False),
        ('dynamic_text_projection_241_320_960',[(241,320,960)],False),
        ('dynamic_text_projection_241_320_320',[(241,320,320)],False),
        ('dynamic_vision_projection_64_960_12288',[(64,960,12288)],False),
        ('attention_group3_small',[(50,64,64),(50,50,64),(50,320,64)],False),
        ('attention_group3_projection',[(50,2048,720)]*3,True),
        ('attention_group3_qk_241',[(50,241,64)]*3,True),
        ('attention_group3_pv_241',[(50,64,241)]*3,True),
    ]
    result=dict(repeats=repeats,pl_mhz=float(Clocks.fclk0_mhz),
                cpu_mhz=float(Clocks.cpu_mhz),cpu_threads=4,
                cpu_kernel='native four-thread Cortex-A53 NEON; input packing included',
                pl_kernel='GEMM_V2 up to three cores; staging, cache sync, wait/clear and host output copy included',
                weights='B treated as constant; first transpose/sync recorded separately, steady-state samples reuse cached BT',
                records=[],threshold_rule='largest padded-op count in a contiguous low-op range where CPU is at least 10% faster; otherwise use PL')
    result_file=Path(output_path) if output_path else BASE/'dispatch_threshold_results.json'
    ops=PynqOperators(cpu_threshold_ops=0)
    try:
        for ci,(name,shapes,shared_b) in enumerate(cases):
            data=matrices(shapes,220+ci,shared_b)
            def cpu_run():
                for a,b,c in data:ops._cpu_gemm(a,b,c)
            def pl_run():
                if len(data)==1:
                    a,b,c=data[0]
                    ops._gemm_matrix(a,b,c,constant_b=True)
                elif not ops._gemm_group(data,constant_b=True):
                    raise RuntimeError(name+' does not fit the CMA slots')

            cpu_run()
            expected=[c.copy() for _,_,c in data]
            begin=time.perf_counter();pl_run();cold_pl_ms=(time.perf_counter()-begin)*1000
            for (_,_,c),ref in zip(data,expected):np.testing.assert_array_equal(c,ref)
            cpu_run();pl_run()
            for (_,_,c),ref in zip(data,expected):np.testing.assert_array_equal(c,ref)

            cpu_values=[];pl_values=[]
            for i in range(repeats):
                if i%2:
                    begin=time.perf_counter();pl_run();pl_s=time.perf_counter()-begin
                    begin=time.perf_counter();cpu_run();cpu_s=time.perf_counter()-begin
                else:
                    begin=time.perf_counter();cpu_run();cpu_s=time.perf_counter()-begin
                    begin=time.perf_counter();pl_run();pl_s=time.perf_counter()-begin
                for (_,_,c),ref in zip(data,expected):np.testing.assert_array_equal(c,ref)
                cpu_values.append(cpu_s*1000);pl_values.append(pl_s*1000)
                print('MEASURE',name,'cpu_ms={:.3f}'.format(cpu_s*1000),
                      'pl_ms={:.3f}'.format(pl_s*1000),flush=True)

            shapes_mnk=[[a.shape[0],b.shape[1],a.shape[1]] for a,b,_ in data]
            padded_job_ops=[padded_ops(s) for s in shapes_mnk]
            row=dict(name=name,shapes_mnk=shapes_mnk,shared_b=shared_b,
                     padded_job_ops=padded_job_ops,padded_ops=sum(padded_job_ops),
                     logical_ops=sum(2*a.shape[0]*b.shape[1]*a.shape[1] for a,b,_ in data),
                     cpu_median_ms=statistics.median(cpu_values),
                     pl_warm_median_ms=statistics.median(pl_values),cold_pl_ms=cold_pl_ms,
                     speedup=statistics.median(cpu_values)/statistics.median(pl_values),
                     cpu_samples_ms=cpu_values,pl_warm_samples_ms=pl_values,
                     eligible_for_single_matrix_threshold=len(data)==1)
            result['records'].append(row)
            result_file.write_text(json.dumps(result,indent=2))
            print('SUMMARY',json.dumps(row),flush=True)

        singles=sorted((r for r in result['records'] if r['eligible_for_single_matrix_threshold']),
                       key=lambda r:r['padded_ops'])
        threshold=0
        for row in singles:
            same_size=[r for r in singles if r['padded_ops']==row['padded_ops']]
            if not all(r['cpu_median_ms']<=0.9*r['pl_warm_median_ms'] for r in same_size):
                break
            threshold=row['padded_ops']
        result['cpu_threshold_ops']=threshold
        result_file.write_text(json.dumps(result,indent=2))
        print('RESULT',json.dumps(result),flush=True)
    finally:ops.close()


def main(repeats,output_path=None):
    with highest_cpu_environment(keep_highest=True) as environment:
        os.sched_setaffinity(0,{0,1,2,3})
        run_benchmark(repeats,output_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output')
    args=parser.parse_args()
    main(args.repeats,args.output)
