"""On-board comparison; no fabricated timing if board/bitstream unavailable.

Run from GEMM_V2/fast_gemm with matching parent accelerator_driver.py and the
included runtime/neon_runtime.py / NEON library. No clock register writes.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from pynq import Clocks, PL, MMIO, allocate

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runtime'))
from accelerator_driver import AcceleratorDriver
from fast_gemm import FastGemm
from neon_runtime import FastCpu, FastPack, highest_cpu_environment


def main(repeats=8):
    bitfile = ROOT/'hardware/design_1.bit'
    if not PL.bitfile_name or Path(PL.bitfile_name).resolve() != bitfile.resolve():
        raise RuntimeError('Load the paired GEMM_V2 bitstream first; this test never reloads or changes PL clock')
    for base in (0x80000000,0x80001000,0x80002000):
        if MMIO(base,0x200).read(0x18)&65:
            raise RuntimeError('GEMM busy; do not start another benchmark')
    hwh = bitfile.with_suffix('.hwh')
    result_file = BASE/('fast_submit_results_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    result = dict(shape=[2048,2048,2048],records=[],profiles=[],
                  pl_mhz=float(Clocks.fclk0_mhz),hardware_measured=True,
                  timing='Full host-to-host total; separate submit, wait+cleanup, prepare. '
                         'Fast wait reads no counters. One explicit profile run per configuration is excluded from timing. '
                         'Addresses alternate between two live A/C allocations. No PL reload/clock changes.')
    buffers, active_driver = [], None
    try:
        with highest_cpu_environment(keep_highest=True):
            rng=np.random.RandomState(1)
            host_a=rng.randint(-8,8,(2048,2048)).astype(np.int8)
            host_b=rng.randint(-8,8,(2048,2048)).astype(np.int8)
            ref=np.empty((2048,2048),np.int32)
            cpu=FastCpu(host_a,host_b,ref)
            cpu.run(None,dict(variant=2,mb=32,nb=128,pad=0,mode=1,block=32,pack_threads=4))
            # Five allocations total 44 MiB; every buffer owner remains live.
            for dtype in [np.int8,np.int8,np.int8,np.int32,np.int32]:
                buffers.append(allocate(shape=(2048,2048),dtype=dtype,cacheable=1))
            a0,a1,bt,c0,c1=buffers
            packer=FastPack(host_b,np.asarray(bt))
            for count in [1,2,3]:
                cores=tuple(range(count))
                versions=['legacy','fast_python','fast_native']
                if count%2==0:versions.reverse()
                for version in versions:
                    init_start=time.perf_counter()
                    if version=='legacy':
                        driver=AcceleratorDriver.from_hwh(hwh,poll_interval=0)
                    else:
                        driver=FastGemm(hwh,backend=version.split('_',1)[1])
                    active_driver=driver
                    init_ms=(time.perf_counter()-init_start)*1000
                    plans=[]
                    prepare_start=time.perf_counter()
                    if version!='legacy':
                        plans=[driver.prepare(a0,bt,c0,cores),driver.prepare(a1,bt,c1,cores)]
                    prepare_ms=(time.perf_counter()-prepare_start)*1000
                    for repeat in range(repeats+1):
                        slot=repeat%2
                        aa,cc=[(a0,c0),(a1,c1)][slot]
                        t0=time.perf_counter()
                        np.copyto(aa,host_a)
                        packer.run(None,dict(mode=1,block=64,threads=4))
                        aa.sync_to_device();bt.sync_to_device();cc.sync_to_device()
                        t1=time.perf_counter()
                        if version=='legacy':
                            task=driver.submit_gemm_rows(int(aa.device_address),int(bt.device_address),
                                int(cc.device_address),2048,2048,2048,cores)
                        else:
                            task=plans[slot].submit()
                        t2=time.perf_counter()
                        task.wait(timeout_s=20)
                        t3=time.perf_counter()
                        cc.sync_from_device()
                        output=np.array(cc,copy=True,subok=False)
                        t4=time.perf_counter()
                        np.testing.assert_array_equal(output,ref)
                        row=dict(version=version,cores=count,repeat=repeat,
                            prepare_two_plans_ms=prepare_ms,init_ms=init_ms,
                            input_prepare_ms=(t1-t0)*1000,submit_ms=(t2-t1)*1000,
                            wait_cleanup_ms=(t3-t2)*1000,
                            submit_through_cleanup_ms=(t3-t1)*1000,
                            output_sync_copy_ms=(t4-t3)*1000,total_ms=(t4-t0)*1000,
                            a_address=int(aa.device_address),c_address=int(cc.device_address),
                            pl_mhz=float(Clocks.fclk0_mhz),cpu_mhz=float(Clocks.cpu_mhz),verified=True)
                        if repeat:result['records'].append(row)
                        result_file.write_text(json.dumps(result,indent=2))
                        print('WARMUP' if not repeat else 'MEASURE',json.dumps(row),flush=True)
                    if version!='legacy':
                        # Explicit profile run: not part of the fast-path latency samples.
                        profiled=plans[slot].submit().wait_with_counters(timeout_s=20)
                        cc.sync_from_device()
                        np.testing.assert_array_equal(np.asarray(cc),ref)
                        result['profiles'].append(dict(version=version,cores=count,
                            counters=profiled['counters'],rows=profiled['rows'],verified=True))
                    driver.close()
                    active_driver=None
            summary={}
            for count in [1,2,3]:
                summary[str(count)]={}
                for version in ['legacy','fast_python','fast_native']:
                    rows=[r for r in result['records'] if r['cores']==count and r['version']==version]
                    summary[str(count)][version]={key:statistics.median(r[key] for r in rows)
                        for key in ['submit_ms','wait_cleanup_ms','submit_through_cleanup_ms',
                                    'total_ms','prepare_two_plans_ms','init_ms']}
            result.update(summary=summary,all_verified=True)
            result_file.write_text(json.dumps(result,indent=2))
            print('RESULT',str(result_file),flush=True)
            print('SUMMARY',json.dumps(summary),flush=True)
            return result
    finally:
        if active_driver is not None:
            if isinstance(active_driver,FastGemm):
                while active_driver.active is not None:
                    try:active_driver.recover(timeout_s=30)
                    except TimeoutError:time.sleep(.1)
            else:
                while active_driver.pending:
                    try:active_driver.wait_all(timeout_s=30)
                    except Exception:
                        if active_driver.pending:time.sleep(.1)
            active_driver.close()
        for buf in reversed(buffers):buf.freebuffer()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=8)
    main(parser.parse_args().repeats)
