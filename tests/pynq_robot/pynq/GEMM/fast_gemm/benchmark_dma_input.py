"""Compare CPU A copy, serial PS DMA and PS DMA concurrent with 4T NEON B pack.

Uses the previously verified root-only ps_dma_trial module (4 MiB, XLNK).
No overlay/PL clock change. DMA fd must close before the CMA owner is freed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

import numpy as np
from pynq import Clocks, PL, allocate

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT/'runtime'))
from fast_gemm import FastGemm
from neon_runtime import FastCpu, FastPack, highest_cpu_environment

SIZE=4*1024*1024


def copy_a_dma(fd):
    raw=bytearray(40)
    start=time.perf_counter()
    fcntl.ioctl(fd,0xD102,raw,True)  # releases GIL while kernel waits for DMA
    end=time.perf_counter()
    values=struct.unpack('=5Q',raw)
    if values[4]!=SIZE:
        raise RuntimeError('Incomplete A DMA copy')
    return dict(wall_ms=(end-start)*1000,kernel_ms=values[0]/1e6,
                sync_ms=values[1]/1e6,transfer_wait_ms=values[2]/1e6,
                segments=values[3],bytes=values[4])


def main(repeats):
    bitfile=ROOT/'hardware/design_1.bit'
    if not PL.bitfile_name or Path(PL.bitfile_name).resolve()!=bitfile.resolve():
        raise RuntimeError('Load paired GEMM_V2 overlay first')
    result=dict(shape_mnk=[2048,2048,2048],records=[],profiles=[],
        pl_mhz=float(Clocks.fclk0_mhz),dma_channel='dma1chan0',
        reference_threads=4,transpose_threads=4,
        full_path='Ordinary host A/B -> independent ordinary host INT32 C. '
            'Includes input copy/transpose, cache sync, DMA ioctl/wait, GEMM submit/wait/CLEAR, output sync/copy. '
            'Pin/map registration and allocation excluded and reported separately. '
            'All methods share identical A/BT/C addresses and rotate execution order. '
            'A contents alternate between two inputs; every output fully checked.')
    target=BASE/('dma_input_results_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    buffers=[]
    driver=None
    fd=None
    executor=None
    future=None
    try:
        with highest_cpu_environment(keep_highest=True):
            os.sched_setaffinity(0,{0,1,2,3})
            rng=np.random.RandomState(55)
            base_a=rng.randint(-8,8,(2048,2048)).astype(np.int8)
            alternate_a=(base_a+np.int8(1)).astype(np.int8)
            host_b=rng.randint(-8,8,(2048,2048)).astype(np.int8)
            host_a=np.empty_like(base_a)
            refs=[]
            for source in (base_a,alternate_a):
                host_a[:]=source
                ref=np.empty((2048,2048),np.int32)
                cpu=FastCpu(host_a,host_b,ref)
                cpu.run(None,dict(variant=2,mb=32,nb=128,pad=0,mode=1,block=32,pack_threads=4))
                refs.append(ref)
                del cpu
            for dtype in (np.int8,np.int8,np.int32):
                buffers.append(allocate((2048,2048),dtype=dtype,cacheable=1))
            a,bt,c=buffers
            a.fill(0);bt.fill(0);c.fill(0)
            for buffer in buffers:buffer.sync_to_device()
            fd=os.open('/dev/ps_dma_trial',os.O_RDWR)
            registration_start=time.perf_counter()
            fcntl.ioctl(fd,0xD105,1)  # source TO_DEVICE; CMA destination FROM_DEVICE
            fcntl.ioctl(fd,0xD104,struct.pack('=2Q',a.ctypes.data,SIZE))
            fcntl.ioctl(fd,0xD101,struct.pack('=2Q',host_a.ctypes.data,SIZE))
            result['dma_registration_ms']=(time.perf_counter()-registration_start)*1000
            result['addresses']=dict(a=int(a.device_address),bt=int(bt.device_address),c=int(c.device_address))
            init_start=time.perf_counter()
            driver=FastGemm(ROOT/'hardware/design_1.hwh',backend='native')
            plans={count:driver.prepare(a,bt,c,tuple(range(count))) for count in (1,2,3)}
            packer=FastPack(host_b,np.asarray(bt))
            executor=ThreadPoolExecutor(max_workers=1)
            executor.submit(lambda:None).result()  # worker startup excluded
            result['driver_and_plan_setup_ms']=(time.perf_counter()-init_start)*1000
            methods=['cpu_serial','dma_serial','dma_parallel']
            for count in (1,2,3):
                for repeat in range(repeats+1):
                    variant=repeat%2
                    host_a[:]=(base_a,alternate_a)[variant]  # upstream producer, outside all methods
                    order=methods[repeat%3:]+methods[:repeat%3]
                    for method in order:
                        # Dest poison outside timer; sync included so stale A cannot pass.
                        a.fill(-77)
                        a.sync_to_device()
                        c.fill(-1234567)
                        dma=None
                        t0=time.perf_counter()
                        if method=='cpu_serial':
                            np.copyto(a,host_a)
                            a.sync_to_device()
                        elif method=='dma_serial':
                            dma=copy_a_dma(fd)
                        else:
                            future=executor.submit(copy_a_dma,fd)
                        t1=time.perf_counter()
                        packer.run(None,dict(mode=1,block=64,threads=4))
                        t2=time.perf_counter()
                        bt.sync_to_device();c.sync_to_device()
                        t3=time.perf_counter()
                        if future is not None:
                            dma=future.result()
                            future=None
                        t4=time.perf_counter()
                        task=plans[count].submit()
                        t5=time.perf_counter()
                        task.wait(timeout_s=20)
                        t6=time.perf_counter()
                        c.sync_from_device()
                        output=np.array(c,copy=True,subok=False)
                        t7=time.perf_counter()
                        np.testing.assert_array_equal(output,refs[variant])
                        np.testing.assert_array_equal(a,host_a)
                        row=dict(method=method,cores=count,repeat=repeat,input_variant=variant,
                            a_copy_or_enqueue_ms=(t1-t0)*1000,transpose_ms=(t2-t1)*1000,
                            bt_c_sync_ms=(t3-t2)*1000,dma_remaining_wait_ms=(t4-t3)*1000,
                            input_prepare_ms=(t4-t0)*1000,submit_ms=(t5-t4)*1000,
                            wait_cleanup_ms=(t6-t5)*1000,control_ms=(t6-t4)*1000,
                            output_sync_copy_ms=(t7-t6)*1000,total_ms=(t7-t0)*1000,
                            gops=2*2048**3/((t7-t0)*1e9),dma=dma,
                            cpu_mhz=float(Clocks.cpu_mhz),pl_mhz=float(Clocks.fclk0_mhz),verified=True)
                        if repeat:result['records'].append(row)
                        target.write_text(json.dumps(result,indent=2))
                        print('MEASURE' if repeat else 'WARMUP',json.dumps(row),flush=True)
                # Same data, independent performance run, excluded from latency samples.
                for method in methods:
                    if method=='cpu_serial':
                        np.copyto(a,host_a);a.sync_to_device()
                    elif method=='dma_serial':
                        copy_a_dma(fd)
                    else:
                        future=executor.submit(copy_a_dma,fd)
                    packer.run(None,dict(mode=1,block=64,threads=4))
                    bt.sync_to_device();c.sync_to_device()
                    if future is not None:
                        future.result();future=None
                    profile=plans[count].submit().wait_with_counters(timeout_s=20)
                    c.sync_from_device()
                    np.testing.assert_array_equal(c,refs[variant])
                    result['profiles'].append(dict(method=method,cores=count,**profile))
            result['summary']={str(count):{method:{key:statistics.median(r[key] for r in result['records']
                if r['cores']==count and r['method']==method) for key in
                ('a_copy_or_enqueue_ms','transpose_ms','bt_c_sync_ms','dma_remaining_wait_ms',
                 'input_prepare_ms','submit_ms','wait_cleanup_ms','control_ms',
                 'output_sync_copy_ms','total_ms','gops')} for method in methods} for count in (1,2,3)}
            result['all_verified']=True
            target.write_text(json.dumps(result,indent=2))
            print('RESULT',str(target),flush=True)
            print('SUMMARY',json.dumps(result['summary']),flush=True)
    finally:
        # A worker may still own the registered source/destination on an exception.
        try:
            if future is not None:future.result()
        finally:
            if executor:executor.shutdown(wait=True)
            if driver:
                while driver.active:
                    try:driver.recover(timeout_s=30)
                    except TimeoutError:time.sleep(0.01)
                driver.close()
            if fd is not None:os.close(fd)
            for buffer in reversed(buffers):buffer.freebuffer()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=8)
    main(parser.parse_args().repeats)
