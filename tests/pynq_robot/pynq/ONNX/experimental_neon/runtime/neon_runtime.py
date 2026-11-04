"""Exact INT8 GEMM and packing on four pinned Cortex-A53 cores.
Every run packs the original B inside timing. No cached input shortcut.
"""
import ctypes
from functools import lru_cache
import sys
import warnings
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import threading
import time
import numpy as np

os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'gemm_benchmark_neon.cpp'
LIBRARY=ROOT/'libgemm_benchmark_neon.so'

@lru_cache(maxsize=1)
def load_library():
    if not LIBRARY.exists() or SOURCE.stat().st_mtime>LIBRARY.stat().st_mtime:
        subprocess.check_call(['g++','-O3','-mcpu=cortex-a53','-std=c++14','-fPIC','-shared','-fopenmp',str(SOURCE),'-o',str(LIBRARY)])
    lib=ctypes.CDLL(str(LIBRARY))
    lib.fast_pack.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*7
    lib.fast_pack.restype=ctypes.c_int
    lib.fast_gemm.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*9
    lib.fast_gemm.restype=ctypes.c_int
    lib.fast_pack_parallel.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*6+[ctypes.c_void_p]*2
    lib.fast_pack_parallel.restype=ctypes.c_int
    lib.fast_cpu_parallel.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*10+[ctypes.c_void_p]*3
    lib.fast_cpu_parallel.restype=ctypes.c_int
    return lib

def thread_time():
    r=resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_utime+r.ru_stime

def pin(core):
    os.sched_setaffinity(0,{core})
    if set(os.sched_getaffinity(0))!={core}: raise RuntimeError('Worker pin failed')

class FastPack:
    def __init__(self,b,bt,cpu_ids=(0,1,2,3)):
        self.b,self.bt=map(np.asarray,(b,bt))
        if self.b.ndim!=2 or self.bt.ndim!=2 or self.b.dtype!=np.int8 or self.bt.dtype!=np.int8:
            raise ValueError('2-D INT8 required')
        self.k,self.n=self.b.shape
        if (self.bt.shape[0]<self.n or self.bt.shape[1]<self.k or
            not self.b.flags.c_contiguous or self.bt.strides[1]!=1 or
            self.bt.strides[0]<self.k or self.bt.strides[0]%8):
            raise ValueError('Expected contiguous B[K,N] and BT[>=N,>=K] with 8-byte-aligned row stride')
        if self.k%8 or self.n%8 or min(self.k,self.n)<8 or np.shares_memory(self.b,self.bt):
            raise ValueError('Unsupported dimensions or overlapping buffers')
        if len(cpu_ids)!=4 or len(set(cpu_ids))!=4: raise ValueError('Four distinct cores required')
        self.cpu_ids=tuple(cpu_ids)
        self.lib=load_library()
        self.bp=ctypes.c_void_p(self.b.ctypes.data)
        self.tp=ctypes.c_void_p(self.bt.ctypes.data)
        self.bs=self.bt.strides[0]
        self.plan={'mode':1,'block':32,'threads':4}
        self.candidates=[]

    def _part(self,part,parts,plan):
        first=part*(self.n//8)//parts*8
        last=(part+1)*(self.n//8)//parts*8
        if plan['mode']==2:
            np.copyto(self.bt[first:last,:self.k],self.b[:,first:last].T)
        else:
            if first==last:return
            rc=self.lib.fast_pack(self.bp,self.tp,self.k,self.n,self.bs,first,last,plan['mode'],plan['block'])
            if rc: raise RuntimeError('Packing failed: '+str(rc))

    def run(self,pool,plan=None):
        plan=dict(self.plan if plan is None else plan)
        if plan['mode']==2:
            # NumPy candidate retains the original worker implementation.
            parts=plan['threads'];ready=threading.Barrier(parts)
            def worker(index):
                try:
                    pin(self.cpu_ids[index]);ready.wait(timeout=30)
                except BaseException:
                    ready.abort();raise
                self._part(index,parts,plan)
            begin=time.perf_counter()
            futures=[pool.submit(worker,i) for i in range(parts)]
            for f in futures: f.result()
            elapsed=time.perf_counter()-begin
            self.last_profile=dict(seconds=elapsed,transpose_s=elapsed,software_preprocess_s=0.0,postprocess_s=0.0)
            return elapsed
        ids=(ctypes.c_int*4)(*self.cpu_ids)
        metrics=(ctypes.c_double*3)()
        begin=time.perf_counter()
        rc=self.lib.fast_pack_parallel(self.bp,self.tp,self.k,self.n,self.bs,plan['mode'],plan['block'],plan['threads'],ids,metrics)
        if rc: raise RuntimeError('Native parallel pack failed')
        elapsed=time.perf_counter()-begin
        self.last_profile=dict(seconds=elapsed,transpose_s=float(metrics[0]),
             native_startup_s=float(metrics[1]),python_wrapper_s=elapsed-sum(metrics),
             software_preprocess_s=elapsed-metrics[0]-metrics[2],postprocess_s=float(metrics[2]))
        return elapsed

    def tune(self,pool,repeats=7):
        candidates=[]
        for mode in (2,0,1):
            for block in ((32,) if mode==2 else (16,32,64)):
                for threads in (1,4):
                    plan=dict(mode=mode,block=block,threads=threads)
                    self.run(pool,plan)
                    values=[self.run(pool,plan) for _ in range(repeats)]
                    np.testing.assert_array_equal(self.bt[:self.n,:self.k],self.b.T)
                    candidates.append(dict(plan=plan,seconds=float(np.median(values))))
        self.candidates=candidates
        self.plan=dict(min(candidates,key=lambda r:r['seconds'])['plan'])
        return self.plan

    @staticmethod
    def label(plan):
        return '{} block={} threads={}'.format(('C blocked','NEON 8x8','NumPy')[plan['mode']],plan['block'],plan['threads'])

class FastCpu:
    def __init__(self,a,b,out,cpu_ids=(0,1,2,3),workspace_pads=(0,32,64,128)):
        self.a,self.b,self.out=map(np.asarray,(a,b,out))
        if self.a.ndim!=2 or self.b.ndim!=2 or self.out.ndim!=2: raise ValueError('2-D required')
        self.m,self.k=self.a.shape
        kb,self.n=self.b.shape
        if kb!=self.k or self.out.shape!=(self.m,self.n): raise ValueError('Shape mismatch')
        if self.a.dtype!=np.int8 or self.b.dtype!=np.int8 or self.out.dtype!=np.int32: raise ValueError('INT8 inputs, INT32 output required')
        if any(not x.flags.c_contiguous for x in (self.a,self.b,self.out)): raise ValueError('Contiguous matrices required')
        if any(np.shares_memory(x,self.out) for x in (self.a,self.b)): raise ValueError('Output overlap')
        if min(self.m,self.n,self.k)<32 or any(v%32 for v in (self.m,self.n,self.k)) or self.k>16352: raise ValueError('Unsupported dimensions')
        if len(cpu_ids)!=4 or len(set(cpu_ids))!=4: raise ValueError('Four distinct cores required')
        self.cpu_ids=tuple(cpu_ids)
        self.lib=load_library()
        self.ap=ctypes.c_void_p(self.a.ctypes.data);self.cp=ctypes.c_void_p(self.out.ctypes.data)
        self.workspaces={}
        for pad in workspace_pads:
            if pad<0 or pad%32:raise ValueError('Workspace padding must be a nonnegative multiple of 32')
            buf=np.zeros((self.n,self.k+pad),np.int8)
            self.workspaces[pad]=(buf,FastPack(self.b,buf[:,:self.k],cpu_ids))
        self.plan=dict(variant=2,mb=32,nb=128,pad=0,mode=1,block=32,pack_threads=4)
        self.candidates=[];self.setup_seconds=0
        self.source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    def run(self,pool,plan=None,progress=False):
        plan=dict(self.plan if plan is None else plan)
        buf,packer=self.workspaces[plan['pad']]
        if plan['mode']==2:
            raise ValueError('CPU native packing requires C or NEON mode')
        ids=(ctypes.c_int*4)(*self.cpu_ids)
        metrics=(ctypes.c_double*7)()
        begin=time.perf_counter()
        rc=self.lib.fast_cpu_parallel(self.ap,ctypes.c_void_p(buf.ctypes.data),self.cp,
             self.m,self.n,self.k,buf.shape[1],plan['variant'],plan['mb'],plan['nb'],plan['mode'],plan['block'],plan['pack_threads'],
             ids,metrics,ctypes.c_void_p(self.b.ctypes.data))
        end=time.perf_counter()
        if rc: raise RuntimeError('Native four-core GEMM failed')
        packing=float(metrics[0]);compute=float(metrics[4])
        return dict(seconds=end-begin,packing_seconds=packing,compute_seconds=compute,
                    transpose_seconds=float(metrics[3]),postprocess_seconds=float(metrics[5]),
                    software_preprocess_seconds=end-begin-metrics[3]-metrics[4]-metrics[5],
                    parallelism=metrics[2]/compute,plan=plan,kernel=self.label(plan),gops=2*self.m*self.n*self.k/(end-begin)/1e9)

    @staticmethod
    def label(p):
        return 'NEON {} MB={} NB={} BTpad={} pack={}x{}'.format(('4x4','2x8','8x2')[p['variant']],p['mb'],p['nb'],p['pad'],p['mode'],p['pack_threads'])

    def tune(self,pool,reference,verbose=True):
        start=time.perf_counter()
        packing={}
        for pad,(_,packer) in sorted(self.workspaces.items()):
            packer.tune(pool)
            packer.plan=dict(min((x for x in packer.candidates if x['plan']['mode']!=2),key=lambda x:x['seconds'])['plan'])
            packing[pad]=packer.plan
            if verbose: print('CPU BT pad={} packing: {}'.format(pad,packer.label(packer.plan)),flush=True)
        candidates=[]
        # All existing tile shapes plus cache-stride alternatives for the 8x2 tile.
        for pad in (0,32,64,128):
            for variant in ((0,1,2) if pad==0 else (2,)):
                for mb in (16,32,64):
                    for nb in (32,64,128):
                        pp=packing[pad]
                        p=dict(variant=variant,mb=mb,nb=nb,pad=pad,mode=pp['mode'],block=pp['block'],pack_threads=pp['threads'])
                        row=self.run(pool,p)
                        np.testing.assert_array_equal(self.out,reference)
                        candidates.append(dict(plan=p,seconds=row['seconds']))
            if verbose: print('CPU BT pad={} full-output checks passed'.format(pad),flush=True)
        finalists=sorted(candidates,key=lambda x:x['seconds'])[:3]
        for entry in finalists:
            vals=[]
            for _ in range(5):
                row=self.run(pool,entry['plan']);np.testing.assert_array_equal(self.out,reference);vals.append(row['seconds'])
            entry['seconds']=float(np.median(vals));entry['repeats']=vals
        self.plan=dict(min(finalists,key=lambda r:r['seconds'])['plan'])
        self.candidates=candidates
        self.setup_seconds=time.perf_counter()-start
        if verbose: print('CPU selected: '+self.label(self.plan),flush=True)
        return self.plan


CPU_IDS=(0,1,2,3)
CPU_SYSFS_ROOT=Path("/sys/devices/system/cpu")
FREQUENCY_TOLERANCE=0.01

def read_text(path):
    return path.read_text().strip()


def write_value(path, value):
    path.write_text(str(value) + "\n")


def set_limits(policy, minimum, maximum):
    # 先放宽上限，避免写入时出现 min > max。
    if minimum > int(read_text(policy / "scaling_max_freq")):
        write_value(policy / "scaling_max_freq", maximum)
    write_value(policy / "scaling_min_freq", minimum)
    write_value(policy / "scaling_max_freq", maximum)


def frequency_report(environment):
    reports = []
    for item in environment["policies"]:
        p = item["path"]
        try:
            frequency = int(read_text(p / "cpuinfo_cur_freq"))
            source = "cpuinfo_cur_freq"
        except (OSError, ValueError):
            from pynq import Clocks
            frequency = int(round(Clocks.cpu_mhz * 1000))
            source = "pynq.Clocks.cpu_mhz"
        if frequency <= 0:
            raise RuntimeError(f"无法读取有效的CPU硬件频率：{p}")
        if int(read_text(p / "scaling_min_freq")) != item["selected_khz"] or int(read_text(p / "scaling_max_freq")) != item["selected_khz"]:
            raise RuntimeError("CPU最高档位限制在测速期间被改变")
        baseline = item.get("baseline_khz")
        if baseline is not None and abs(frequency - baseline) > baseline * FREQUENCY_TOLERANCE:
            raise RuntimeError(f"CPU实际频率在测速期间发生变化：{frequency} kHz，初始为 {baseline} kHz；请检查系统负载或调频状态")
        reports.append({"policy": p.name, "selected_khz": item["selected_khz"],
                        "reported_khz": frequency, "source": source})
    return reports


@contextmanager
def highest_cpu_environment(keep_highest=False):
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("四核绑定和CPU调频需要在PYNQ Linux上运行")
    old_affinity = set(os.sched_getaffinity(0))
    if not set(CPU_IDS).issubset(old_affinity):
        raise RuntimeError(f"当前任务不能使用全部CPU{CPU_IDS}，可用：{sorted(old_affinity)}")
    policies = {}
    for cpu in CPU_IDS:
        directory = CPU_SYSFS_ROOT / f"cpu{cpu}"
        online = directory / "online"
        if online.exists() and read_text(online) != "1":
            raise RuntimeError(f"CPU{cpu}未在线，请先启用四核")
        try:
            p = (directory / "cpufreq").resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"CPU{cpu}没有可用的CPUFreq策略") from exc
        if p in policies:
            continue  # A53多个核心通常共享同一个频率策略。
        minimum = int(read_text(p / "cpuinfo_min_freq"))
        maximum = int(read_text(p / "cpuinfo_max_freq"))
        table = p / "scaling_available_frequencies"
        frequencies = [int(x) for x in read_text(table).split()] if table.exists() else []
        supported = [f for f in frequencies if minimum <= f <= maximum]
        selected = max(supported) if supported else maximum
        if selected <= 0:
            raise RuntimeError(f"CPUFreq没有有效的最高频率：{p}")
        governors = read_text(p / "scaling_available_governors").split()
        governor = "userspace" if "userspace" in governors else "performance"
        if governor not in governors:
            raise RuntimeError(f"无法固定CPU频率，可用策略：{governors}")
        speed = p / "scaling_setspeed"
        old_governor = read_text(p / "scaling_governor")
        old_speed = int(read_text(speed)) if old_governor == "userspace" else None
        policies[p] = {"path": p, "selected_khz": selected, "governor": governor,
                       "old_governor": old_governor, "old_speed": old_speed,
                       "old_min": int(read_text(p / "scaling_min_freq")),
                       "old_max": int(read_text(p / "scaling_max_freq"))}
    environment = {"policies": list(policies.values()), "restored": False}
    changed = []
    configured = False
    try:
        os.sched_setaffinity(0, set(CPU_IDS))
        for item in environment["policies"]:
            changed.append(item)
            p = item["path"]
            write_value(p / "scaling_governor", item["governor"])
            set_limits(p, item["selected_khz"], item["selected_khz"])
            if item["governor"] == "userspace":
                write_value(p / "scaling_setspeed", item["selected_khz"])
        time.sleep(0.02)
        environment["initial_readback"] = frequency_report(environment)
        for item, report in zip(environment["policies"], environment["initial_readback"]):
            item["baseline_khz"] = report["reported_khz"]
        configured = True
        print("CPU最高支持档位及硬件实际频率:", environment["initial_readback"])
        print("使用最高支持档位；后续按实际读数检查频率稳定性，不要求达到固定GHz数值。")
        yield environment
    finally:
        errors = []
        for item in (reversed(changed) if not keep_highest or not configured else ()):
            try:
                p = item["path"]
                set_limits(p, item["old_min"], item["old_max"])
                write_value(p / "scaling_governor", item["old_governor"])
                if item["old_speed"] is not None:
                    write_value(p / "scaling_setspeed", item["old_speed"])
            except OSError as exc:
                errors.append(str(exc))
        try:
            os.sched_setaffinity(0, old_affinity)
        except OSError as exc:
            errors.append(str(exc))
        environment["restored"] = not errors and (not keep_highest or not configured)
        environment["affinity_restored"] = not errors
        environment["kept_highest"] = bool(keep_highest and configured)
        if errors:
            warnings.warn("CPU策略/亲和性恢复未完成：" + "; ".join(errors))


def configure_highest_cpu():
    """Keep CPUFreq at the highest supported setting; report actual clock separately."""
    with highest_cpu_environment(keep_highest=True) as environment:
        return environment
