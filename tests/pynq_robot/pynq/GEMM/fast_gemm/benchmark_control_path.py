"""Isolate submission, polling and CLEAR on the same buffers, round-robin."""
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
from pynq import allocate, Clocks

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0, str(ROOT/'runtime'))
from fast_gemm.fast_gemm import FastGemm, PythonOps
from neon_runtime import highest_cpu_environment


class MixedOps:
    def __init__(self, submitter, poller, clearer):
        self.submitter,self.poller,self.clearer=submitter,poller,clearer

    def submit(self,plan):
        t=time.perf_counter()
        self.submitter.submit(plan)
        self.raw_submit_ms=(time.perf_counter()-t)*1000

    def poll(self,task):
        return self.poller.poll(task)

    def clear(self,cores):
        t=time.perf_counter()
        result=self.clearer.clear(cores)
        self.raw_clear_ms=(time.perf_counter()-t)*1000
        return result


def main():
    buffers=[]
    driver=None
    result=dict(records=[],profiles=[])
    try:
        with highest_cpu_environment(keep_highest=True):
            rng=np.random.RandomState(47)
            a_host=rng.randint(-8,8,(2048,2048)).astype(np.int8)
            bt_host=rng.randint(-8,8,(2048,2048)).astype(np.int8)
            for dtype in (np.int8,np.int8,np.int32):
                buffers.append(allocate((2048,2048),dtype=dtype,cacheable=1))
            a,bt,c=buffers
            a[:]=a_host;bt[:]=bt_host;c.fill(0)
            for b in buffers:b.sync_to_device()
            driver=FastGemm(ROOT/'hardware/design_1.hwh',backend='native')
            native=driver.ops
            python=PythonOps(driver.mmios)
            plan=driver.prepare(a,bt,c,(0,1,2))
            options={
                'native_native':MixedOps(native,native,native),
                'native_python':MixedOps(native,python,native),
                'python_native':MixedOps(python,native,native),
                'python_python':MixedOps(python,python,native),
            }
            for repeat in range(9):
                names=list(options)
                names=names[repeat%4:]+names[:repeat%4]
                for name in names:
                    ops=options[name]
                    driver.ops=ops
                    t0=time.perf_counter()
                    task=plan.submit()
                    t1=time.perf_counter()
                    task.wait()
                    t2=time.perf_counter()
                    row=dict(name=name,repeat=repeat,submit_ms=(t1-t0)*1000,
                        wait_ms=(t2-t1)*1000,total_ms=(t2-t0)*1000,
                        raw_submit_ms=ops.raw_submit_ms,clear_ms=ops.raw_clear_ms)
                    if repeat:result['records'].append(row)
            # Independent profile runs. Full CPU reference already tested in main benchmark;
            # spot-check this experiment against exact CPU dot products.
            for name,ops in options.items():
                driver.ops=ops
                row=plan.submit().wait_with_counters()
                c.sync_from_device()
                for i,j in ((0,0),(703,511),(704,512),(1376,1024),(2047,2047)):
                    assert c[i,j]==np.dot(a_host[i].astype(np.int32),bt_host[j].astype(np.int32))
                result['profiles'].append(dict(name=name,**row))
            result['summary']={name:{key:statistics.median(r[key] for r in result['records'] if r['name']==name)
                for key in ('submit_ms','wait_ms','total_ms','raw_submit_ms','clear_ms')} for name in options}
            result.update(pl_mhz=float(Clocks.fclk0_mhz),cpu_mhz=float(Clocks.cpu_mhz))
            target=BASE/'control_path_results.json'
            target.write_text(json.dumps(result,indent=2))
            print(json.dumps(result['summary'],indent=2),flush=True)
    finally:
        if driver:
            if driver.active:driver.recover()
            driver.close()
        for b in reversed(buffers):b.freebuffer()


if __name__=='__main__':main()
