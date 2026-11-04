"""三个 GEMM + 一个 FP32 Softmax 的纯物理地址驱动，兼容 Python 3.6+。

连接已加载的硬件，只访问控制寄存器，不分配/映射矩阵内存，不复制、转置、
量化或同步 CPU cache，也不下载 bitstream/修改时钟。调用方持有 DDR 内存。
GEMM: INT8 A[M,K] @ INT8 BT[N,K].T -> INT32 C[M,N]。
Softmax: FP32 [M,K] -> FP32 [M,K]，每行按 16 字节补齐。
"""

from collections import namedtuple
from operator import index
from time import monotonic, sleep
import math
import threading
import xml.etree.ElementTree as ET

GEMM_BASES = (0x80000000, 0x80001000, 0x80002000)
SOFTMAX_BASE = 0x90000000
CTRL, A_BASE, BT_BASE, C_BASE, DIM_MN, DIM_K = 0, 4, 8, 12, 16, 20
STATUS, ACCEPTED_JOB, WRITE_DONE_JOB = 0x18, 0x20, 0x28
CYCLES_LO, CYCLES_HI, COMPUTE, LOAD, STORE = 0x34, 0x38, 0x3C, 0x40, 0x44
READ_BYTES, WRITE_BYTES, CAP_CONFIG = 0x48, 0x4C, 0x50
CAP_SIGNATURE = 0x0A080020
ENABLE, START, CLEAR = 1, 1 << 8, 1 << 9
G_BUSY, G_ERROR, G_SAT, G_WRITE_BUSY, G_WRITE_DONE = 1, 4, 16, 64, 128
S_BUSY, S_DONE, S_ERROR = 2, 4, 8
SOFTMAX_ERRORS = {
    1: "invalid dimensions/address/overlap/enable",
    2: "NaN or +inf input", 3: "fully masked row",
    4: "AXI read error", 5: "AXI write error", 6: "AXI read length mismatch",
}

GemmJob = namedtuple("GemmJob", "core a_addr bt_addr c_addr m n k")
SoftmaxJob = namedtuple("SoftmaxJob", "src_addr dst_addr m k")


class HardwareError(RuntimeError):
    def __init__(self, device, status, code=None):
        self.device, self.status, self.code = device, status, code
        detail = SOFTMAX_ERRORS.get(code, "unknown error") if code is not None else "GEMM ERROR"
        super().__init__("{}: {} (status=0x{:08x}); DMA drained, output invalid".format(
            device, detail, status))


class BatchError(RuntimeError):
    """所有所选任务都已结束，其中部分失败；results 保留成功任务结果。"""
    def __init__(self, errors, results):
        self.errors, self.results = tuple(errors), tuple(results)
        super().__init__("{} accelerator task(s) failed; selected tasks drained".format(len(errors)))


class SubmissionError(RuntimeError):
    """START 写入异常，tasks 中的任务可能已经启动；须保留 DDR 并继续 wait。"""
    def __init__(self, tasks):
        self.tasks = tuple(tasks)
        super().__init__("START write failed; {} task(s) may be running; use wait_all(error.tasks)".format(
            len(tasks)))


def _integer(value, name):
    if isinstance(value, bool):
        raise TypeError("{} must be an integer, not bool".format(name))
    try:
        return int(index(value))
    except TypeError:
        raise TypeError("{} must be an integer".format(name))


def _seconds(value, name, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError("{} must be finite and positive".format(name))
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("{} must be finite and {}".format(name, "nonnegative" if allow_zero else "positive"))
    return value


def _span(addr, size, alignment, name):
    addr = _integer(addr, name)
    if addr < 0 or addr % alignment or size <= 0 or addr + size > 1 << 32:
        raise ValueError("{} must be {}-byte aligned and its full span below 4 GiB".format(name, alignment))
    return addr, addr + size


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _gemm_job(job):
    if not isinstance(job, GemmJob):
        raise TypeError("Expected GemmJob(core, a_addr, bt_addr, c_addr, m, n, k)")
    job = GemmJob(*(_integer(v, name) for name, v in zip(GemmJob._fields, job)))
    if job.core not in (0, 1, 2):
        raise ValueError("core must be 0, 1 or 2")
    if any(v < 32 or v % 32 for v in (job.m, job.n, job.k)):
        raise ValueError("GEMM M/N/K must be positive multiples of 32")
    if max(job.m, job.n) > 65535 * 32 or job.k > 16352:
        raise ValueError("GEMM requires M/N<=2097120 and K<=16352")
    a = _span(job.a_addr, job.m * job.k, 32, "A")
    bt = _span(job.bt_addr, job.n * job.k, 32, "BT")
    c = _span(job.c_addr, 4 * job.m * job.n, 32, "C")
    if _overlap(c, a) or _overlap(c, bt):
        raise ValueError("GEMM C must not overlap A or BT")
    return job, (a, bt), (c,)


def _softmax_job(job):
    job = SoftmaxJob(*(_integer(v, name) for name, v in zip(SoftmaxJob._fields, job)))
    if not 1 <= job.m <= 65536 or not 1 <= job.k <= 4096:
        raise ValueError("Softmax requires 1<=M<=65536 and 1<=K<=4096")
    stride = ((job.k + 3) // 4) * 16
    src = _span(job.src_addr, job.m * stride, 16, "Softmax source")
    dst = _span(job.dst_addr, job.m * stride, 16, "Softmax destination")
    if _overlap(src, dst):
        raise ValueError("Softmax source and destination must not overlap")
    return job, (src,), (dst,)


def discover_hardware(hwh_path):
    """只解析 HWH，要求恰好三个 ABI10 GEMM 模块和一个 Softmax 模块。"""
    root = ET.parse(str(hwh_path)).getroot()
    devices = []
    for module in root.iter("MODULE"):
        kind = {"gemm_accelerator_top": "gemm", "softmax_accelerator_top": "softmax"}.get(
            module.get("MODTYPE"))
        if kind is None:
            continue
        name = module.get("INSTANCE")
        aliases = {name, module.get("FULLNAME", "").lstrip("/")}
        controls = {(int(r.get("BASEVALUE"), 0), int(r.get("HIGHVALUE"), 0) + 1)
                    for r in root.iter("MEMRANGE")
                    if r.get("INSTANCE") in aliases and r.get("SLAVEBUSINTERFACE") == "S_AXI"}
        if len(controls) != 1:
            raise ValueError("{}: missing or ambiguous AXI-Lite address map".format(name))
        base, end = controls.pop()
        parameters = {p.get("NAME"): p.get("VALUE") for p in module.iter("PARAMETER")}
        encoded_base = parameters.get("C_S_AXI_BASEADDR")
        if encoded_base is not None and int(encoded_base, 0) != base:
            raise ValueError("{}: RTL C_S_AXI_BASEADDR differs from Address Editor".format(name))
        minimum = 0x54 if kind == "gemm" else 0x10
        if base % 4 or end - base < minimum:
            raise ValueError("{}: invalid control window".format(name))
        ddr = tuple(sorted({(int(r.get("BASEVALUE"), 0), int(r.get("HIGHVALUE"), 0) + 1)
                            for r in module.iter("MEMRANGE")
                            if r.get("MASTERBUSINTERFACE") == "M_AXI"
                            and "DDR" in r.get("ADDRESSBLOCK", "")}))
        if not ddr:
            raise ValueError("{}: expected a single M_AXI port with a DDR map".format(name))
        devices.append(dict(kind=kind, name=name, base=base, length=end-base, ddr_ranges=ddr))
    gemm = sorted((d for d in devices if d["kind"] == "gemm"), key=lambda d: d["base"])
    softmax = [d for d in devices if d["kind"] == "softmax"]
    if len(gemm) != 3 or len(softmax) != 1:
        raise ValueError("Expected 3 GEMM + 1 Softmax; HWH contains {} GEMM + {} Softmax".format(
            len(gemm), len(softmax)))
    return {"gemm": gemm, "softmax": softmax[0]}


class Task:
    """异步任务句柄。完成表示全部硬件写响应已收尾，不代表 CPU cache 已同步。"""
    def __init__(self, driver, device, job, reads, writes, previous=None, row_start=0):
        self._driver, self.device, self.job = driver, device, job
        self.read_ranges, self.write_ranges = reads, writes
        self.previous = previous
        self.job_id = None if previous is None else (previous + 1) & 0xFFFFFFFF
        self.row_start = row_start
        self.done = False
        self.result = None
        self.error = None
        self._fault_status = None
        self._accepted = False
        self.started_at = None
        self.start_write_failed = False

    def poll(self):
        """单次查询；未完成返回 None，完成返回结果字典，错误排空后抛 HardwareError。"""
        return self._driver._poll(self)

    def wait(self, timeout_s=10.0):
        return self._driver._wait((self,), timeout_s, single=True)[0]


class TaskGroup:
    def __init__(self, driver, tasks, shape=None, output_addr=None):
        self._driver, self.tasks = driver, tuple(tasks)
        self.shape, self.output_addr = shape, output_addr

    @property
    def done(self):
        return all(task.done for task in self.tasks)

    def poll(self):
        return self._driver._poll_many(self.tasks)

    def wait(self, timeout_s=10.0):
        return self._driver._wait(self.tasks, timeout_s)


class AcceleratorDriver:
    """连接已加载的三核 GEMM + Softmax。一个实例独占四个控制窗口。

    默认地址对应当前 BD；也可用 from_hwh() 验证实际导出的地址和 DDR 范围。
    mmio_factory(base, length) 默认是 pynq.MMIO，可注入以测试驱动协议。
    """
    def __init__(self, gemm_bases=GEMM_BASES, softmax_base=SOFTMAX_BASE,
                 poll_interval=0.0001, mmio_factory=None):
        self.poll_interval = _seconds(poll_interval, "poll_interval", allow_zero=True)
        bases = tuple(_integer(v, "GEMM base") for v in gemm_bases)
        softmax_base = _integer(softmax_base, "Softmax base")
        if len(bases) != 3:
            raise ValueError("Exactly three GEMM control addresses are required")
        windows = [_span(base, 0x200, 4, "control base") for base in bases + (softmax_base,)]
        if any(_overlap(a, b) for i, a in enumerate(windows) for b in windows[i+1:]):
            raise ValueError("Control windows overlap")
        if mmio_factory is None:
            from pynq import MMIO
            mmio_factory = MMIO
        self._lock = threading.RLock()
        self._closed = False
        self._active = {}
        self._devices = {}
        for i, base in enumerate(bases):
            self._devices["gemm{}".format(i)] = dict(
                mmio=mmio_factory(base, 0x200), base=base, ddr_ranges=None)
        self._devices["softmax"] = dict(
            mmio=mmio_factory(softmax_base, 0x200), base=softmax_base, ddr_ranges=None)
        for i in range(3):
            device = "gemm{}".format(i)
            if self._read(device, CAP_CONFIG) != CAP_SIGNATURE:
                raise ValueError("{}: ABI mismatch; expected 0x0A080020".format(device))

    @classmethod
    def from_hwh(cls, hwh_path, **kwargs):
        hardware = discover_hardware(hwh_path)
        driver = cls(gemm_bases=tuple(d["base"] for d in hardware["gemm"]),
                     softmax_base=hardware["softmax"]["base"], **kwargs)
        for i, desc in enumerate(hardware["gemm"]):
            driver._devices["gemm{}".format(i)]["ddr_ranges"] = desc["ddr_ranges"]
        driver._devices["softmax"]["ddr_ranges"] = hardware["softmax"]["ddr_ranges"]
        return driver

    @property
    def pending(self):
        with self._lock:
            return tuple(self._active.values())

    @property
    def addresses(self):
        return {name: desc["base"] for name, desc in self._devices.items()}

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Driver is closed")

    def _read(self, device, offset):
        return int(self._devices[device]["mmio"].read(offset)) & 0xFFFFFFFF

    def _write(self, device, offset, value):
        self._devices[device]["mmio"].write(offset, int(value))

    def _idle(self, device):
        if device in self._active:
            raise RuntimeError("{} has a pending task; poll/wait before reuse".format(device))
        flags = self._read(device, CTRL if device == "softmax" else STATUS)
        mask = S_BUSY if device == "softmax" else G_BUSY | G_WRITE_BUSY
        if flags & mask:
            raise RuntimeError("{} is already busy".format(device))

    def _check_hazards(self, descriptions):
        # Read-only sharing (e.g. BT) is allowed; any overlap with a writer is not.
        existing = [(t.read_ranges, t.write_ranges) for t in self._active.values()]
        for device, job, reads, writes, row in descriptions:
            regions = self._devices[device]["ddr_ranges"]
            if regions is not None:
                for span in reads + writes:
                    if not any(lo <= span[0] and span[1] <= hi for lo, hi in regions):
                        raise ValueError("{}: matrix span is outside its HWH DDR map".format(device))
            for old_reads, old_writes in existing:
                if (any(_overlap(a, b) for a in writes for b in old_reads + old_writes)
                        or any(_overlap(a, b) for a in reads for b in old_writes)):
                    raise ValueError("Concurrent tasks have overlapping read/write DDR ranges; wait for the producer")
            existing.append((reads, writes))

    def _submit(self, descriptions):
        with self._lock:
            self._require_open()
            if not descriptions:
                raise ValueError("Submit at least one job")
            names = [d[0] for d in descriptions]
            if len(set(names)) != len(names):
                raise ValueError("At most one job per core may be submitted in a batch")
            self._check_hazards(descriptions)
            for name in names:
                self._idle(name)
            prepared = []
            # Configure every core before issuing any START, including clearing
            # stale DONE/ERROR and cumulative counters from the previous task.
            for device, job, reads, writes, row in descriptions:
                self._write(device, CTRL, ENABLE | CLEAR)
                flags = self._read(device, CTRL if device == "softmax" else STATUS)
                mask = S_BUSY | S_DONE | S_ERROR if device == "softmax" else G_BUSY | G_WRITE_BUSY | G_ERROR | G_WRITE_DONE
                if flags & mask:
                    raise RuntimeError("{}: status did not clear; no new START issued".format(device))
                if device == "softmax":
                    previous = None
                    pairs = ((4, job.src_addr), (8, job.dst_addr),
                             (12, (job.m - 1) | ((job.k - 1) << 16)))
                else:
                    previous = self._read(device, ACCEPTED_JOB)
                    pairs = ((A_BASE, job.a_addr), (BT_BASE, job.bt_addr), (C_BASE, job.c_addr),
                             (DIM_MN, (job.m // 32) | ((job.n // 32) << 16)), (DIM_K, job.k // 32))
                for offset, value in pairs:
                    self._write(device, offset, value)
                prepared.append(Task(self, device, job, reads, writes, previous, row))
            started = []
            for task in prepared:
                # Retain the task before MMIO: a failed write may have reached PL.
                self._active[task.device] = task
                started.append(task)
                task.started_at = monotonic()
                try:
                    self._write(task.device, CTRL, ENABLE | START)
                except BaseException as exc:
                    task.start_write_failed = True
                    raise SubmissionError(started) from exc
            return tuple(started)

    def submit_gemm(self, core, a_addr, bt_addr, c_addr, m, n, k):
        job, reads, writes = _gemm_job(GemmJob(core, a_addr, bt_addr, c_addr, m, n, k))
        return self._submit([("gemm{}".format(job.core), job, reads, writes, 0)])[0]

    def submit_gemm_batch(self, jobs):
        """不同 GEMM 核各自运行独立任务；先全部配置，再连续发出 START。"""
        descriptions = []
        for job in jobs:
            job, reads, writes = _gemm_job(job)
            descriptions.append(("gemm{}".format(job.core), job, reads, writes, 0))
        return TaskGroup(self, self._submit(descriptions))

    def submit_gemm_rows(self, a_addr, bt_addr, c_addr, m, n, k, cores=(0, 1, 2)):
        """按32行块均分M；共享BT，各核直接写C的互不重叠行，无复制或拼接。"""
        full, _, _ = _gemm_job(GemmJob(0, a_addr, bt_addr, c_addr, m, n, k))
        cores = tuple(_integer(c, "core") for c in cores)
        if not cores or len(set(cores)) != len(cores) or any(c not in (0, 1, 2) for c in cores):
            raise ValueError("cores must contain unique indices from 0, 1, 2")
        cores = cores[:min(len(cores), full.m // 32)]
        blocks, extra = divmod(full.m // 32, len(cores))
        descriptions, row = [], 0
        for i, core in enumerate(cores):
            rows = (blocks + (i < extra)) * 32
            job, reads, writes = _gemm_job(GemmJob(core, full.a_addr + row * full.k,
                full.bt_addr, full.c_addr + row * full.n * 4, rows, full.n, full.k))
            descriptions.append(("gemm{}".format(core), job, reads, writes, row))
            row += rows
        return TaskGroup(self, self._submit(descriptions), (full.m, full.n), full.c_addr)

    def submit_softmax(self, src_addr, dst_addr, m, k):
        job, reads, writes = _softmax_job(SoftmaxJob(src_addr, dst_addr, m, k))
        return self._submit([("softmax", job, reads, writes, 0)])[0]

    def _finish(self, task, flags):
        if task._fault_status is not None:
            task.error = HardwareError(task.device, task._fault_status,
                (task._fault_status >> 4) & 15 if task.device == "softmax" else None)
        else:
            job = task.job
            if task.device == "softmax":
                result = dict(output_addr=job.dst_addr, shape=(job.m, job.k), dtype="float32",
                              row_stride_bytes=((job.k + 3) // 4) * 16)
            else:
                result = dict(output_addr=job.c_addr, shape=(job.m, job.n), dtype="int32",
                              row_stride_bytes=job.n * 4, row_start=task.row_start, job_id=task.job_id,
                              counters=self._counters(task.device, flags))
            result.update(device=task.device, status=flags, software_elapsed_s=monotonic()-task.started_at)
            task.result = result
        task.done = True
        del self._active[task.device]
        if task.error is not None:
            raise task.error
        return task.result

    def _poll(self, task):
        with self._lock:
            self._require_open()
            if not isinstance(task, Task) or task._driver is not self:
                raise ValueError("Task belongs to a different driver")
            if task.done:
                if task.error is not None:
                    raise task.error
                return task.result
            if self._active.get(task.device) is not task:
                raise RuntimeError("Task is not registered as pending")
            if task.device == "softmax":
                flags = self._read(task.device, CTRL)
                if flags & S_ERROR:
                    task._fault_status = flags
                if flags & S_BUSY:
                    return None
                if task._fault_status is not None or flags & S_DONE:
                    return self._finish(task, flags)
                # Softmax has no job ID. A quick task may finish before software
                # observes BUSY; stale DONE was explicitly cleared before START.
                return None
            flags = self._read(task.device, STATUS)
            if flags & G_ERROR:
                task._fault_status = flags
            accepted = self._read(task.device, ACCEPTED_JOB)
            if accepted == task.job_id:
                task._accepted = True
            elif accepted != task.previous or task._accepted:
                raise RuntimeError("{}: unexpected job ID; external reset/writer, task retained".format(task.device))
            if flags & (G_BUSY | G_WRITE_BUSY):
                return None
            if task._fault_status is not None:
                return self._finish(task, flags)
            if task._accepted and flags & G_WRITE_DONE and self._read(task.device, WRITE_DONE_JOB) == task.job_id:
                return self._finish(task, flags)
            return None

    def _poll_many(self, tasks, single=False):
        results, errors, complete = [], [], True
        for task in tasks:
            try:
                result = task.poll()
                complete = complete and result is not None
                results.append(result)
            except HardwareError as exc:
                errors.append(exc)
                results.append(None)
        if not complete:
            return None
        if errors:
            if single:
                raise errors[0]
            raise BatchError(errors, results)
        return tuple(results)

    def _wait(self, tasks, timeout_s, single=False):
        deadline = monotonic() + _seconds(timeout_s, "timeout_s")
        while True:
            result = self._poll_many(tasks, single=single)
            if result is not None:
                return result
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Accelerator wait timed out; tasks remain pending, keep DDR and call wait again")
            if self.poll_interval:
                sleep(min(self.poll_interval, remaining))

    def wait_all(self, tasks=None, timeout_s=10.0):
        """等待指定句柄或当前全部在途任务；错误也等待所选任务排空后统一报告。"""
        tasks = self.pending if tasks is None else tuple(tasks)
        if any(not isinstance(t, Task) or t._driver is not self for t in tasks):
            raise ValueError("All tasks must belong to this driver")
        return self._wait(tasks, timeout_s)

    def run_gemm(self, core, a_addr, bt_addr, c_addr, m, n, k, timeout_s=10.0):
        _seconds(timeout_s, "timeout_s")
        return self.submit_gemm(core, a_addr, bt_addr, c_addr, m, n, k).wait(timeout_s)

    def run_gemm_rows(self, a_addr, bt_addr, c_addr, m, n, k, cores=(0, 1, 2), timeout_s=10.0):
        _seconds(timeout_s, "timeout_s")
        return self.submit_gemm_rows(a_addr, bt_addr, c_addr, m, n, k, cores).wait(timeout_s)

    def run_softmax(self, src_addr, dst_addr, m, k, timeout_s=10.0):
        _seconds(timeout_s, "timeout_s")
        return self.submit_softmax(src_addr, dst_addr, m, k).wait(timeout_s)

    def _counters(self, device, flags):
        hi = self._read(device, CYCLES_HI)
        lo = self._read(device, CYCLES_LO)
        return dict(cycles=(hi << 32) | lo, compute_cycles=self._read(device, COMPUTE),
                    load_cycles=self._read(device, LOAD), store_cycles=self._read(device, STORE),
                    read_bytes=self._read(device, READ_BYTES), write_bytes=self._read(device, WRITE_BYTES),
                    saturated=bool(flags & G_SAT))

    def read_counters(self, core):
        core = _integer(core, "core")
        if core not in (0, 1, 2):
            raise ValueError("core must be 0, 1 or 2")
        with self._lock:
            self._require_open()
            device = "gemm{}".format(core)
            self._idle(device)
            return self._counters(device, self._read(device, STATUS))

    def close(self):
        """仅关闭软件控制对象，不释放调用方内存、不复位硬件或卸载 Overlay。"""
        with self._lock:
            if self._active:
                raise RuntimeError("Tasks are pending; wait before closing")
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Preserve the original timeout/submission exception and the pending
        # handles so callers can continue waiting outside the with block.
        if not self.pending:
            self.close()
        return False
