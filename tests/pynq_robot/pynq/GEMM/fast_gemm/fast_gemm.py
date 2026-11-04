"""GEMM_V2 fast path: six writes/core; no performance reads at normal finish.

One driver exclusively owns GEMM0/1/2. One task group at a time. Buffers must
remain allocated until wait succeeds or recover drains the outstanding task.
No overlay loading, clock changes, data copies or cache sync are performed here.
"""
import ctypes
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import monotonic, sleep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from accelerator_driver import GemmJob, _gemm_job, discover_hardware

BUSY, ERROR, WRITE_BUSY, WRITE_DONE = 1, 4, 64, 128
CLEAR_MASK = BUSY | ERROR | WRITE_BUSY | WRITE_DONE
_inflight = set()  # Keep MMIO mappings and buffers alive even if caller drops task.


class PythonOps:
    def __init__(self, mmios):
        self.mmios = mmios

    def submit(self, plan):
        for core, values in zip(plan.cores, plan.values):
            mm = self.mmios[core]
            for offset, value in zip((4, 8, 12, 16, 20), values):
                mm.write(offset, value)
            mm.write(0, 0x101)

    def poll(self, task):
        for i, (core, job) in enumerate(zip(task.plan.cores, task.jobs)):
            if task.states[i] in (1, 2):
                continue
            mm = self.mmios[core]
            flags = mm.read(0x18)
            if flags & (BUSY | WRITE_BUSY):
                continue
            if flags & ERROR:
                task.states[i] = 2
            elif flags & WRITE_DONE:
                task.states[i] = (1 if mm.read(0x28) == job and mm.read(0x20) == job else 3)
        return list(task.states)

    def clear(self, cores):
        result = []
        for core in cores:
            mm = self.mmios[core]
            mm.write(0, 0x201)
            result.append((int(mm.read(0x18)), int(mm.read(0x20))))
        return result


class NativeOps(PythonOps):
    def __init__(self, mmios):
        super().__init__(mmios)
        if platform.machine().lower() not in ('aarch64', 'arm64') or sys.platform != 'linux':
            raise RuntimeError('Native MMIO helper requires PYNQ ARM64 Linux')
        source = Path(__file__).with_name('fast_axilite.c')
        library = source.with_name('libfast_axilite.so')
        if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
            temporary = library.with_suffix('.so.tmp')
            subprocess.check_call(['gcc', '-O3', '-shared', '-fPIC', '-std=c11',
                                   str(source), '-o', str(temporary)])
            temporary.replace(library)
        self.lib = ctypes.CDLL(str(library))
        self.uptr = ctypes.POINTER(ctypes.c_size_t)
        self.u32ptr = ctypes.POINTER(ctypes.c_uint32)
        self.lib.fg_submit.argtypes = [self.uptr, self.u32ptr, ctypes.c_uint]
        self.lib.fg_submit.restype = None
        self.lib.fg_poll.argtypes = [self.uptr, self.u32ptr, ctypes.c_uint,
                                    ctypes.c_uint32, self.u32ptr]
        self.lib.fg_poll.restype = None
        self.lib.fg_clear.argtypes = [self.uptr, ctypes.c_uint, self.u32ptr, self.u32ptr]
        self.lib.fg_clear.restype = None
        # Only genuine local NumPy mappings; remote/tinynumpy arrays must use PythonOps.
        import numpy as np
        for mm in mmios:
            if type(mm.array) is not np.ndarray or mm.array.dtype != np.uint32 or not mm.array.flags.c_contiguous:
                raise TypeError('Native path needs local contiguous uint32 PYNQ MMIO arrays')
        self.addresses = tuple(int(mm.array.ctypes.data) for mm in mmios)
        self.clear_arrays = {}

    def _clear_arrays(self, cores):
        key = tuple(cores)
        if key not in self.clear_arrays:
            count = len(key)
            self.clear_arrays[key] = (
                (ctypes.c_size_t * count)(*(self.addresses[c] for c in key)),
                (ctypes.c_uint32 * count)(), (ctypes.c_uint32 * count)())
        return self.clear_arrays[key]

    def attach(self, plan):
        plan.native_bases = (ctypes.c_size_t * len(plan.cores))(
            *(self.addresses[c] for c in plan.cores))
        plan.native_values = (ctypes.c_uint32 * (5 * len(plan.cores)))(
            *(v for values in plan.values for v in values))
        self._clear_arrays(plan.cores)

    def submit(self, plan):
        self.lib.fg_submit(plan.native_bases, plan.native_values, len(plan.cores))

    def poll(self, task):
        skip = sum(1 << i for i, state in enumerate(task.states) if state in (1, 2))
        self.lib.fg_poll(task.plan.native_bases, task.native_jobs,
                         len(task.plan.cores), skip, task.states)
        return list(task.states)

    def clear(self, cores):
        count = len(cores)
        bases, flags, jobs = self._clear_arrays(cores)
        self.lib.fg_clear(bases, count, flags, jobs)
        return list(zip(flags, jobs))


class GemmPlan:
    def __init__(self, driver, cores, values, rows, buffers):
        self.driver, self.cores = driver, tuple(cores)
        self.values, self.rows, self.buffers = tuple(values), tuple(rows), tuple(buffers)
        if isinstance(driver.ops, NativeOps):
            driver.ops.attach(self)

    def submit(self):
        return self.driver._submit(self)

    def run(self, timeout_s=10.0):
        return self.submit().wait(timeout_s)


class FastTask:
    def __init__(self, driver, plan, jobs):
        self.driver, self.plan, self.jobs = driver, plan, tuple(jobs)
        self.states = (ctypes.c_uint32 * len(jobs))()
        self.native_jobs = (ctypes.c_uint32 * len(jobs))(*jobs)
        self.result = None
        self.error = None
        self.submission_failed = False

    def wait(self, timeout_s=10.0):
        """Fast completion: does not read any performance counter registers."""
        return self.driver._wait(self, timeout_s, profile=False)

    def wait_with_counters(self, timeout_s=10.0):
        """Explicit profile interface: capture counters BEFORE clearing state."""
        return self.driver._wait(self, timeout_s, profile=True)


class FastGemm:
    def __init__(self, hwh_path=None, backend='native', mmio_factory=None,
                 poll_interval=0.0):
        if backend not in ('native', 'python'):
            raise ValueError('backend must be native or python')
        if poll_interval < 0:
            raise ValueError('poll_interval must be nonnegative')
        if mmio_factory is None:
            from pynq import MMIO
            mmio_factory = MMIO
        self.hardware = discover_hardware(hwh_path) if hwh_path is not None else None
        self.bases = tuple(d['base'] for d in self.hardware['gemm']) if self.hardware else (
            0x80000000, 0x80001000, 0x80002000)
        self.mmios = tuple(mmio_factory(base, 0x200) for base in self.bases)
        self.lock = threading.RLock()
        self.poll_interval = poll_interval
        self.closed, self.active = False, None
        self.ready = [False] * 3
        self.previous = [0] * 3
        self._last_counters = None
        # Check all cores before clearing any of them. Do not take over busy work.
        for mm in self.mmios:
            if int(mm.read(0x50)) != 0x0A080020:
                raise RuntimeError('GEMM ABI signature mismatch')
            if int(mm.read(0x18)) & (BUSY | WRITE_BUSY):
                raise RuntimeError('GEMM core is busy; cannot initialize')
        self.ops = NativeOps(self.mmios) if backend == 'native' else PythonOps(self.mmios)
        self._clear((0, 1, 2))

    def _clear(self, cores):
        for core, (flags, accepted) in zip(cores, self.ops.clear(cores)):
            if flags & CLEAR_MASK:
                self.ready[core] = False
                raise RuntimeError('GEMM{} status clear not confirmed'.format(core))
            self.previous[core] = accepted
            self.ready[core] = True

    def prepare(self, a, bt, c, cores=(0, 1, 2)):
        """Validate live PYNQ buffers and prepare reusable row partitions.

        A INT8[M,K], already transposed BT INT8[N,K], C INT32[M,N]. This method
        performs no MMIO, no copying, no cache sync. Inputs can change per submit.
        """
        import numpy as np
        with self.lock:
            if self.closed:
                raise RuntimeError('Driver is closed')
            cores = tuple(cores)
            if not cores or any(type(i) is not int or i not in (0, 1, 2) for i in cores) or len(set(cores)) != len(cores):
                raise ValueError('cores must contain unique indices 0/1/2')
            for buffer, dtype in [(a, np.int8), (bt, np.int8), (c, np.int32)]:
                if not isinstance(buffer, np.ndarray) or buffer.ndim != 2 or buffer.dtype != np.dtype(dtype) or not buffer.flags.c_contiguous:
                    raise ValueError('Contiguous 2-D INT8/INT8/INT32 PYNQ buffers required')
                if not hasattr(buffer, 'device_address') or getattr(buffer, 'freed', False):
                    raise ValueError('Live PYNQ buffers required')
            m, k = map(int, a.shape)
            n, bk = map(int, bt.shape)
            if bk != k or tuple(c.shape) != (m, n):
                raise ValueError('Expected A[M,K], BT[N,K], C[M,N]')
            aa, bb, cc = (int(v.device_address) for v in (a, bt, c))
            _, reads, writes = _gemm_job(GemmJob(0, aa, bb, cc, m, n, k))
            cores = cores[:min(len(cores), m // 32)]
            if self.hardware:
                for core in cores:
                    for lo, hi in reads + writes:
                        if not any(start <= lo and hi <= end for start, end in self.hardware['gemm'][core]['ddr_ranges']):
                            raise ValueError('Buffer span outside HWH DDR range')
            values, rows, first = [], [], 0
            blocks, extra = divmod(m // 32, len(cores))
            for i, core in enumerate(cores):
                count = (blocks + (i < extra)) * 32
                values.append((aa + first * k, bb, cc + first * n * 4,
                               (count // 32) | ((n // 32) << 16), k // 32))
                rows.append((core, first, count))
                first += count
            return GemmPlan(self, cores, values, rows, (a, bt, c))

    def prepare_jobs(self, jobs):
        """Prepare one independent GEMM per core for a single grouped submit.

        Each entry is ``(core, a, bt, c)`` with live CMA tensors A[M,K],
        BT[N,K], and C[M,N]. Shapes may differ across cores. Read-only BT
        ranges may be shared; writable C ranges may not overlap any other job.
        """
        import numpy as np
        with self.lock:
            if self.closed:
                raise RuntimeError('Driver is closed')
            jobs = tuple(jobs)
            if not 1 <= len(jobs) <= 3:
                raise ValueError('One to three independent GEMM jobs required')
            cores = tuple(item[0] for item in jobs)
            if any(type(i) is not int or i not in (0,1,2) for i in cores) or len(set(cores)) != len(cores):
                raise ValueError('Jobs require unique GEMM core indices 0/1/2')
            values, rows, buffers, spans = [], [], [], []
            for core, a, bt, c in jobs:
                for buffer, dtype in ((a,np.int8),(bt,np.int8),(c,np.int32)):
                    if (not isinstance(buffer,np.ndarray) or buffer.ndim!=2 or
                        buffer.dtype!=np.dtype(dtype) or not buffer.flags.c_contiguous or
                        not hasattr(buffer,'device_address') or getattr(buffer,'freed',False)):
                        raise ValueError('Each job requires live contiguous 2-D CMA A/BT/C buffers')
                m,k=map(int,a.shape);n,bk=map(int,bt.shape)
                if bk!=k or tuple(c.shape)!=(m,n):
                    raise ValueError('Expected matching A[M,K], BT[N,K], C[M,N]')
                aa,bb,cc=(int(v.device_address) for v in (a,bt,c))
                job,reads,writes=_gemm_job(GemmJob(core,aa,bb,cc,m,n,k))
                if self.hardware:
                    for lo,hi in reads+writes:
                        if not any(start<=lo and hi<=end for start,end in self.hardware['gemm'][core]['ddr_ranges']):
                            raise ValueError('Buffer span outside HWH DDR range')
                values.append((aa,bb,cc,(m//32)|((n//32)<<16),k//32))
                rows.append((core,0,m))
                buffers.extend((a,bt,c));spans.append((reads,writes))
            for i,(_,writes) in enumerate(spans):
                for j,(reads_j,writes_j) in enumerate(spans):
                    if i!=j and any(lo<end and start<hi for lo,hi in writes for start,end in reads_j+writes_j):
                        raise ValueError('Independent jobs must not overlap a C output with another job')
            return GemmPlan(self,cores,values,rows,tuple(buffers))

    def _submit(self, plan):
        with self.lock:
            if self.closed or plan.driver is not self:
                raise RuntimeError('Closed or incorrect driver')
            if self.active is not None or not all(self.ready[c] for c in plan.cores):
                raise RuntimeError('A task is pending or state needs recovery')
            if any(getattr(buf, 'freed', False) for buf in plan.buffers):
                raise ValueError('Prepared buffer has been freed')
            task = FastTask(self, plan, [(self.previous[c] + 1) & 0xFFFFFFFF for c in plan.cores])
            # Hold the plan/buffers and mark every selected core unavailable BEFORE
            # entering native code; partial submission must never release them.
            self.active = task
            _inflight.add(self)
            self._last_counters = None
            for c in plan.cores:
                self.ready[c] = False
            try:
                self.ops.submit(plan)
            except BaseException:
                task.submission_failed = True
                raise
            return task

    def _capture_counters(self, task):
        result = {}
        for core in task.plan.cores:
            mm = self.mmios[core]
            fields = ['cycles_lo', 'cycles_hi', 'compute_cycles', 'load_cycles',
                      'store_cycles', 'read_bytes', 'write_bytes']
            values = [int(mm.read(off)) for off in range(0x34, 0x50, 4)]
            row = dict(zip(fields, values))
            row['cycles'] = row.pop('cycles_lo') | (row.pop('cycles_hi') << 32)
            row['saturated'] = bool(mm.read(0x18) & 16)
            result[core] = row
        return result

    def _wait(self, task, timeout_s, profile):
        if not isinstance(timeout_s, (float, int)) or not 0 < timeout_s < float('inf'):
            raise ValueError('timeout_s must be finite and positive')
        with self.lock:
            if task.result is not None:
                if task.error:
                    raise task.error
                if profile and task.result['counters'] is None:
                    raise RuntimeError('Counters already cleared; use wait_with_counters before fast wait')
                return task.result
            if self.active is not task:
                raise RuntimeError('Task does not belong to this outstanding submission')
            if task.submission_failed:
                raise RuntimeError('Partial submission: retain buffers and call recover()')
            deadline = monotonic() + timeout_s
            while True:
                states = self.ops.poll(task)
                if any(s == 3 for s in states):
                    raise RuntimeError('Unexpected job ID; possible external reset/writer; retain buffers')
                if all(s in (1, 2) for s in states):
                    break
                if monotonic() >= deadline:
                    raise TimeoutError('Task still pending; keep buffers and wait again')
                if self.poll_interval:
                    sleep(self.poll_interval)
            counters = self._capture_counters(task) if profile else None
            # Clear only after all cores' complete DDR write responses were observed.
            # Re-read ACCEPTED after CLEAR: works whether CLEAR preserves or resets ID.
            self._clear(task.plan.cores)
            self._last_counters = counters
            result = dict(rows=task.plan.rows, job_ids=task.jobs, counters=counters,
                          hardware_error_cores=[c for c, s in zip(task.plan.cores, states) if s == 2])
            task.result, self.active = result, None
            _inflight.discard(self)
            if result['hardware_error_cores']:
                task.error = RuntimeError('GEMM hardware error; drained/cleared, output invalid: '+str(result['hardware_error_cores']))
                raise task.error
            return result

    def read_counters(self):
        """Get the last explicitly captured snapshot; never silently reads cleared counters."""
        with self.lock:
            if self._last_counters is None:
                raise RuntimeError('Use task.wait_with_counters() for a profiling run')
            return json.loads(json.dumps(self._last_counters))

    def recover(self, timeout_s=10.0):
        """Drain partial submissions/errors; never cancel DMA or report output valid."""
        with self.lock:
            if self.active is None:
                return
            deadline = monotonic() + timeout_s
            cores = self.active.plan.cores
            while any(self.mmios[c].read(0x18) & (BUSY | WRITE_BUSY) for c in cores):
                if monotonic() >= deadline:
                    raise TimeoutError('Hardware still owns buffers; recovery not complete')
                sleep(0.0001)
            self._clear(cores)
            task = self.active
            task.error = RuntimeError('Task recovered after failure; output is invalid')
            task.result = dict(recovered=True, counters=None)
            self.active = None
            _inflight.discard(self)

    def close(self):
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Task pending; wait/recover before closing')
            self.closed = True
