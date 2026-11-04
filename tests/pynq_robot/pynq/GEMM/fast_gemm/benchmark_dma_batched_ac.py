"""Benchmark batched PS DMA A input and optional DMA C output with NEON B pack."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from pynq import Clocks,PL,allocate

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'dma_batch'))
sys.path.insert(0, str(ROOT/'runtime'))
from fast_gemm import FastGemm
from dma_batch import DmaCopy,aligned_empty
from neon_runtime import FastCpu,FastPack,highest_cpu_environment


def main(repeats):
    bitfile=ROOT/'hardware/design_1.bit'
    if not PL.bitfile_name or Path(PL.bitfile_name).resolve()!=bitfile.resolve():
        raise RuntimeError('Load paired GEMM_V2 overlay first')
    rng=np.random.RandomState(87)
    host_a=aligned_empty((2048,2048),np.int8)
    host_c=aligned_empty((2048,2048),np.int32)
    base_a=rng.randint(-8,8,host_a.shape).astype(np.int8)
    alt_a=(base_a+np.int8(1)).astype(np.int8)
    host_b=rng.randint(-8,8,(2048,2048)).astype(np.int8)
    result=dict(shape_mnk=[2048]*3,records=[],profiles=[],
        pl_mhz=float(Clocks.fclk0_mhz),
        full_path='Includes A source copy/sync or DMA, four-thread NEON transpose, '
            'all input cache sync, GEMM submit/wait/CLEAR, C sync and independent ordinary host output copy. '
            'Allocation, DMA registration and reference are outside timing. '
            'All variants use same buffers, rotate order, alternate A values, and fully validate C.')
    buffers=[];a_dma=c_dma=driver=executor=None;future=None
    try:
        with highest_cpu_environment(keep_highest=True):
            import os
            os.sched_setaffinity(0,{0,1,2,3})
            refs=[]
            for src in (base_a,alt_a):
                host_a[:]=src
                ref=np.empty((2048,2048),np.int32)
                FastCpu(host_a,host_b,ref).run(None,dict(
                    variant=2,mb=32,nb=128,pad=0,mode=1,block=32,pack_threads=4))
                refs.append(ref)
            for dtype in (np.int8,np.int8,np.int32):
                buffers.append(allocate((2048,2048),dtype=dtype,cacheable=1))
            a,bt,c=buffers
            for b in buffers:b.fill(0);b.sync_to_device()
            reg_start=time.perf_counter()
            a_dma=DmaCopy(host_a,a,True,0)
            c_dma=DmaCopy(host_c,c,False,1)
            result['dma_registration_ms']=(time.perf_counter()-reg_start)*1000
            result['a_dma_registration_ms']=a_dma.registration_ms
            result['c_dma_registration_ms']=c_dma.registration_ms
            result['a_dma_segments']=a_dma.copy(True)['segments']
            result['c_dma_segments']=c_dma.copy(True)['segments']
            # Restore the buffers after descriptor bring-up copies.
            host_a[:]=base_a;a.fill(-11);a.sync_to_device();c.fill(0);c.sync_to_device()
            driver=FastGemm(ROOT/'hardware/design_1.hwh',backend='native')
            plans={n:driver.prepare(a,bt,c,tuple(range(n))) for n in (1,2,3)}
            pack=FastPack(host_b,np.asarray(bt))
            executor=ThreadPoolExecutor(max_workers=1)
            executor.submit(lambda:None).result()
            variants=[('cpuA_cpuC','cpu','cpu',False),
                      ('dmaA_batched_cpuC','serial','cpu',True),
                      ('dmaA_batched_dmaC','serial','dma',True),
                      ('dmaA_batched_dmaC_nosync','serial','dma_nosync',True),
                      ('dmaA_overlap_cpuC','parallel','cpu',True),
                      ('dmaA_overlap_dmaC','parallel','dma',True)]
            for cores in (1,2,3):
                for repeat in range(repeats+1):
                    variant=repeat%2
                    host_a[:]=(base_a,alt_a)[variant]
                    order=variants[repeat%len(variants):]+variants[:repeat%len(variants)]
                    for name,a_method,c_method,batched in order:
                        a.fill(-31);a.sync_to_device();c.fill(-111);c.sync_to_device()
                        future=None
                        t0=time.perf_counter()
                        if a_method=='cpu':
                            np.copyto(a,host_a);a.sync_to_device()
                            a_copy=None
                        elif a_method=='serial':
                            a_copy=a_dma.copy(batched)
                        else:
                            future=executor.submit(a_dma.copy,batched)
                            a_copy=None
                        t1=time.perf_counter()
                        pack.run(None,dict(mode=1,block=64,threads=4))
                        t2=time.perf_counter()
                        bt.sync_to_device();c.sync_to_device()
                        t3=time.perf_counter()
                        if future is not None:a_copy=future.result();future=None
                        t4=time.perf_counter()
                        task=plans[cores].submit();t5=time.perf_counter()
                        task.wait(timeout_s=20);t6=time.perf_counter()
                        if c_method!='dma_nosync':c.sync_from_device()
                        t7=time.perf_counter()
                        if c_method=='cpu':
                            np.copyto(host_c,c)
                            c_copy=None
                        else:
                            c_copy=c_dma.copy(True)
                        t8=time.perf_counter()
                        np.testing.assert_array_equal(host_c,refs[variant])
                        a.sync_from_device()
                        np.testing.assert_array_equal(a,host_a)
                        row=dict(method=name,cores=cores,repeat=repeat,input_variant=variant,
                            a_copy_or_enqueue_ms=(t1-t0)*1000,transpose_ms=(t2-t1)*1000,
                            bt_c_sync_ms=(t3-t2)*1000,a_dma_wait_ms=(t4-t3)*1000,
                            gemm_submit_ms=(t5-t4)*1000,gemm_wait_cleanup_ms=(t6-t5)*1000,
                            c_sync_ms=(t7-t6)*1000,c_copy_ms=(t8-t7)*1000,
                            total_ms=(t8-t0)*1000,
                            gops=2*2048**3/((t8-t0)*1e9),
                            a_dma=a_copy,c_dma=c_copy,
                            cpu_mhz=float(Clocks.cpu_mhz),pl_mhz=float(Clocks.fclk0_mhz),
                            verified=True)
                        if repeat:result['records'].append(row)
                        print('MEASURE' if repeat else 'WARMUP',json.dumps(row),flush=True)
                # Profiles exclude timing and output validation.
                for name,a_method,c_method,batched in variants:
                    host_a[:]=(base_a,alt_a)[variant]
                    if a_method=='cpu':np.copyto(a,host_a);a.sync_to_device()
                    elif a_method=='serial':a_dma.copy(batched)
                    else:future=executor.submit(a_dma.copy,batched)
                    pack.run(None,dict(mode=1,block=64,threads=4))
                    bt.sync_to_device();c.sync_to_device()
                    if future is not None:future.result();future=None
                    p=plans[cores].submit().wait_with_counters(timeout_s=20)
                    result['profiles'].append(dict(method=name,cores=cores,**p))
            result['summary']={str(cores):{name:{key:statistics.median(x[key] for x in result['records']
                if x['cores']==cores and x['method']==name) for key in
                ('a_copy_or_enqueue_ms','transpose_ms','bt_c_sync_ms','a_dma_wait_ms',
                 'gemm_submit_ms','gemm_wait_cleanup_ms','c_sync_ms','c_copy_ms','total_ms','gops')}
                for name,_,_,_ in variants} for cores in (1,2,3)}
            result['all_verified']=True
            path=BASE/('dma_batched_ac_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
            path.write_text(json.dumps(result,indent=2))
            print('RESULT',path,flush=True)
            print('SUMMARY',json.dumps(result['summary']),flush=True)
    finally:
        try:
            if future is not None:future.result()
        finally:
            if executor:executor.shutdown(wait=True)
            if driver:
                while driver.active:
                    try:driver.recover(timeout_s=30)
                    except TimeoutError:time.sleep(.01)
                driver.close()
            if a_dma:a_dma.close()
            if c_dma:c_dma.close()
            for b in reversed(buffers):b.freebuffer()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=8)
    main(parser.parse_args().repeats)
