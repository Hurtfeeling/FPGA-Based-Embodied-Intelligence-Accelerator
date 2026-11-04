"""Host-only register protocol tests; no PYNQ allocation or RTL simulation."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("accelerator_driver", str(ROOT / "tests/accelerator_driver.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class FakeMMIO:
    def __init__(self, board, base):
        self.board, self.base = board, base
        self.softmax = base == d.SOFTMAX_BASE
        self.regs = {d.CAP_CONFIG: d.CAP_SIGNATURE, d.ACCEPTED_JOB: 0}
        self.hold = False
        self.fail_start = None
        self.fail_config = False
        self.stale_done = False
        self.keep_stale_on_clear = False
        self.started = 0
        self.done_id = None
        self.on_status = None

    def read(self, offset):
        self.board.events.append(("read", self.base, offset))
        if offset == (d.CTRL if self.softmax else d.STATUS) and self.on_status:
            self.on_status()
        return self.regs.get(offset, 0)

    def write(self, offset, value):
        self.board.events.append(("write", self.base, offset, value))
        if self.fail_config and offset == d.DIM_MN:
            raise OSError("configuration write failed")
        if offset != d.CTRL:
            self.regs[offset] = value
            return
        if value & d.CLEAR:
            self.regs[d.CTRL] = d.ENABLE
            self.regs[d.STATUS] = 0
            if self.keep_stale_on_clear:
                self.regs[d.CTRL if self.softmax else d.STATUS] |= d.S_DONE if self.softmax else d.G_WRITE_DONE
            for reg in (d.CYCLES_LO, d.CYCLES_HI, d.COMPUTE, d.LOAD, d.STORE, d.READ_BYTES, d.WRITE_BYTES):
                self.regs[reg] = 0
        if value & d.START:
            if self.fail_start == "before":
                raise OSError("uncertain START write, not delivered in this mock")
            self.started += 1
            if self.softmax:
                self.regs[d.CTRL] = d.ENABLE | d.S_BUSY
            else:
                self.regs[d.ACCEPTED_JOB] = (self.regs[d.ACCEPTED_JOB] + 1) & 0xFFFFFFFF
                self.regs[d.STATUS] = d.G_BUSY | d.G_WRITE_BUSY
            if not self.hold:
                self.finish()
            if self.fail_start == "after":
                raise OSError("START reached hardware before exception")

    def finish(self, error=None):
        if self.softmax:
            self.regs[d.CTRL] = d.ENABLE | (d.S_DONE if error is None else d.S_ERROR | error << 4)
        else:
            self.regs[d.STATUS] = (2 | d.G_WRITE_DONE) if error is None else d.G_ERROR
            self.regs[d.WRITE_DONE_JOB] = self.regs[d.ACCEPTED_JOB] if self.done_id is None else self.done_id
            self.regs.update({d.CYCLES_LO: 1000, d.CYCLES_HI: 1, d.COMPUTE: 800,
                              d.LOAD: 400, d.STORE: 900, d.READ_BYTES: 4096, d.WRITE_BYTES: 8192})


class Board:
    def __init__(self):
        self.events = []
        self.mmios = {}

    def factory(self, base, length):
        assert length == 0x200
        self.mmios[base] = FakeMMIO(self, base)
        return self.mmios[base]

    def core(self, i):
        return self.mmios[d.GEMM_BASES[i]]


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.board = Board()
        self.driver = d.AcceleratorDriver(mmio_factory=self.board.factory, poll_interval=0)

    def submit(self, core=0, **kwargs):
        params = dict(core=core, a_addr=0x100000, bt_addr=0x200000, c_addr=0x300000, m=32, n=64, k=64)
        params.update(kwargs)
        return self.driver.submit_gemm(**params)

    def test_default_addresses_and_no_import_pynq(self):
        self.assertEqual(self.driver.addresses, dict(gemm0=0x80000000, gemm1=0x80001000,
                                                   gemm2=0x80002000, softmax=0x90000000))
        self.assertNotIn("pynq", sys.modules)

    def test_three_way_rows_and_configure_before_all_starts(self):
        group = self.driver.submit_gemm_rows(0x10000000, 0x20000000, 0x30000000,
                                            m=2048, n=512, k=2048)
        self.assertEqual([t.job.m for t in group.tasks], [704, 672, 672])
        self.assertEqual([t.row_start for t in group.tasks], [0, 704, 1376])
        for i, task in enumerate(group.tasks):
            r = self.board.core(i).regs
            self.assertEqual(r[d.A_BASE], 0x10000000 + task.row_start * 2048)
            self.assertEqual(r[d.BT_BASE], 0x20000000)
            self.assertEqual(r[d.C_BASE], 0x30000000 + task.row_start * 512 * 4)
            self.assertEqual(r[d.DIM_MN], task.job.m // 32 | (512 // 32) << 16)
            self.assertEqual(r[d.DIM_K], 64)
        events = self.board.events
        starts = [i for i, e in enumerate(events) if e[0] == "write" and e[2:] == (0, 0x101)]
        configs = [i for i, e in enumerate(events) if e[0] == "write" and e[2] in (4, 8, 12, 16, 20)]
        self.assertLess(max(configs), min(starts))
        self.assertEqual(starts, list(range(starts[0], starts[0]+3)))
        results = group.wait()
        self.assertEqual(len(results), 3)
        self.assertTrue(group.done)
        self.assertEqual(group.shape, (2048, 512))
        self.assertEqual(self.driver.pending, ())

    def test_small_m_uses_only_needed_cores_and_custom_order(self):
        group = self.driver.submit_gemm_rows(0x100000, 0x200000, 0x300000,
                                            64, 32, 32, cores=(2, 0, 1))
        self.assertEqual([t.job.core for t in group.tasks], [2, 0])
        self.assertEqual(self.board.core(1).started, 0)
        group.wait()

    def test_job_id_wrap_and_counters(self):
        self.board.core(0).regs[d.ACCEPTED_JOB] = 0xFFFFFFFF
        task = self.submit()
        self.assertEqual(task.job_id, 0)
        result = task.wait()
        self.assertEqual(result["counters"]["cycles"], (1 << 32) + 1000)
        self.assertEqual(result["counters"]["compute_cycles"], 800)
        self.assertEqual(self.driver.read_counters(0), result["counters"])
        self.assertIs(task.poll(), result)

    def test_completion_requires_write_done_job_and_write_drain(self):
        core = self.board.core(0)
        core.hold = True
        task = self.submit()
        core.regs[d.STATUS] = 2 | d.G_WRITE_DONE | d.G_WRITE_BUSY
        core.regs[d.WRITE_DONE_JOB] = task.job_id
        self.assertIsNone(task.poll())
        core.done_id = 99
        core.finish()
        self.assertIsNone(task.poll())
        core.regs[d.WRITE_DONE_JOB] = task.job_id
        self.assertIsNotNone(task.poll())

    def test_timeout_retains_task_and_can_resume(self):
        self.board.core(0).hold = True
        task = self.submit()
        with self.assertRaises(TimeoutError):
            task.wait(0.00001)
        self.assertEqual(self.driver.pending, (task,))
        with self.assertRaises(RuntimeError):
            self.driver.close()
        with self.assertRaises(RuntimeError):
            self.submit(a_addr=0x400000, c_addr=0x500000)
        self.board.core(0).finish()
        task.wait()
        self.driver.close()

    def test_softmax_tail_stride_and_max_dimensions(self):
        task = self.driver.submit_softmax(0x10000000, 0x60000000, 65536, 4096)
        self.assertEqual(self.board.mmios[d.SOFTMAX_BASE].regs[12], 0x0FFFFFFF)
        task.wait()
        task = self.driver.submit_softmax(0x100000, 0x200000, 3, 5)
        result = task.wait()
        self.assertEqual(result["row_stride_bytes"], 32)
        self.assertEqual(task.read_ranges, ((0x100000, 0x100060),))
        self.assertEqual(result["shape"], (3, 5))
        self.assertNotIn("counters", result)

    def test_softmax_fast_done_is_not_missed_and_stale_done_is_cleared(self):
        core = self.board.mmios[d.SOFTMAX_BASE]
        core.regs[0] = d.ENABLE | d.S_DONE
        task = self.driver.submit_softmax(0x100000, 0x200000, 1, 1)
        self.assertIsNotNone(task.poll())
        core.keep_stale_on_clear = True
        with self.assertRaises(RuntimeError):
            self.driver.submit_softmax(0x100000, 0x200000, 1, 1)
        self.assertEqual(core.started, 1)

    def test_all_four_devices_can_run_concurrently(self):
        for core in self.board.mmios.values():
            core.hold = True
        jobs = [d.GemmJob(i, 0x100000 + i*0x10000, 0x200000, 0x300000 + i*0x10000, 32, 32, 32)
                for i in range(3)]
        group = self.driver.submit_gemm_batch(jobs)
        soft = self.driver.submit_softmax(0x400000, 0x500000, 4, 1024)
        self.assertEqual(len(self.driver.pending), 4)
        for core in self.board.mmios.values():
            core.finish()
        self.assertEqual(len(self.driver.wait_all(group.tasks + (soft,))), 4)

    def test_read_write_dependencies_rejected_shared_reads_allowed(self):
        self.board.core(0).hold = True
        task = self.submit()
        # A producer/consumer dependency cannot run concurrently.
        with self.assertRaises(ValueError):
            self.driver.submit_softmax(0x300000, 0x400000, 32, 64)
        with self.assertRaises(ValueError):
            self.submit(core=1, a_addr=0x500000, c_addr=0x100000)
        other = self.submit(core=1, c_addr=0x600000)
        other.wait()
        self.board.core(0).finish()
        task.wait()

    def test_batch_validation_before_any_write(self):
        jobs = [d.GemmJob(0, 0x100000, 0x200000, 0x300000, 32, 32, 32),
                d.GemmJob(1, 0x400000, 0x200000, 0x300000, 32, 32, 32)]
        with self.assertRaises(ValueError):
            self.driver.submit_gemm_batch(jobs)
        self.assertFalse(any(e[0] == "write" for e in self.board.events))

    def test_configuration_failure_does_not_start_any_core(self):
        self.board.core(1).fail_config = True
        with self.assertRaises(OSError):
            self.driver.submit_gemm_rows(0x100000, 0x200000, 0x300000, 96, 64, 64)
        self.assertTrue(all(c.started == 0 for c in self.board.mmios.values()))
        self.assertFalse(self.driver.pending)

    def test_partial_start_failure_preserves_handles_and_does_not_start_third(self):
        self.board.core(1).fail_start = "after"
        with self.assertRaises(d.SubmissionError) as caught:
            self.driver.submit_gemm_rows(0x100000, 0x200000, 0x300000, 96, 64, 64)
        self.assertEqual(len(caught.exception.tasks), 2)
        self.assertEqual(self.board.core(2).started, 0)
        self.assertTrue(caught.exception.tasks[1].start_write_failed)
        self.assertEqual(len(self.driver.wait_all(caught.exception.tasks)), 2)

    def test_uncertain_start_without_acceptance_not_reported_as_success(self):
        self.board.core(0).fail_start = "before"
        with self.assertRaises(d.SubmissionError) as caught:
            self.submit()
        task = caught.exception.tasks[0]
        self.assertIsNone(task.poll())
        with self.assertRaises(TimeoutError):
            task.wait(0.00001)
        self.assertFalse(task.done)

    def test_error_waits_for_all_selected_cores_to_drain(self):
        for i in range(3):
            self.board.core(i).hold = True
        group = self.driver.submit_gemm_rows(0x100000, 0x200000, 0x300000, 96, 32, 32)
        self.board.core(0).regs[d.STATUS] |= d.G_ERROR
        self.assertIsNone(group.poll())
        self.board.core(0).finish(error=1)
        self.assertIsNone(group.poll())
        self.assertTrue(group.tasks[0].done)
        self.board.core(1).finish()
        self.board.core(2).finish()
        with self.assertRaises(d.BatchError) as caught:
            group.wait()
        self.assertEqual(len(caught.exception.errors), 1)
        self.assertIsNone(caught.exception.results[0])
        self.assertIsNotNone(caught.exception.results[1])
        self.assertFalse(self.driver.pending)
        # The next submission clears the completed error safely.
        self.board.core(0).hold = False
        self.submit().wait()

    def test_softmax_error_code_and_drain(self):
        core = self.board.mmios[d.SOFTMAX_BASE]
        core.hold = True
        task = self.driver.submit_softmax(0x100000, 0x200000, 1, 1024)
        core.regs[0] |= d.S_ERROR | 3 << 4
        self.assertIsNone(task.poll())
        core.finish(error=3)
        with self.assertRaises(d.HardwareError) as caught:
            task.wait()
        self.assertEqual(caught.exception.code, 3)
        self.assertTrue(task.done)

    def test_invalid_arguments_have_no_mmio_writes(self):
        invalid = [dict(k=16384), dict(m=0), dict(n=31), dict(core=True), dict(m=32.0),
                   dict(a_addr=0x100010), dict(c_addr=0x100000), dict(a_addr=0xFFFFFFE0),
                   dict(n=65536*32), dict(core=3)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.submit(**kwargs)
        for params in ((0x100000, 0x200000, 65537, 5), (0x100000, 0x200000, 1, 4097),
                       (0x100000, 0x100010, 1, 5), (0x100004, 0x200000, 1, 4)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.driver.submit_softmax(*params)
        with self.assertRaises(ValueError):
            self.driver.run_gemm(0, 0x100000, 0x200000, 0x300000, 32, 32, 32, timeout_s=0)
        self.assertFalse(any(e[0] == "write" for e in self.board.events))

    def test_abi_and_control_window_checks(self):
        def bad_factory(base, length):
            mmio = self.board.factory(base, length)
            if base == d.GEMM_BASES[2]:
                mmio.regs[d.CAP_CONFIG] = 0
            return mmio
        with self.assertRaises(ValueError):
            d.AcceleratorDriver(mmio_factory=bad_factory)
        with self.assertRaises(ValueError):
            d.AcceleratorDriver(gemm_bases=(0, 0, 0), mmio_factory=self.board.factory)

    def test_real_current_hwh_and_reject_old_two_core_hwh(self):
        path = ROOT / "three_gemm_softmax/hardware/design_1.hwh"
        info = d.discover_hardware(path)
        self.assertEqual(tuple(c["base"] for c in info["gemm"]), d.GEMM_BASES)
        self.assertEqual(info["softmax"]["base"], d.SOFTMAX_BASE)
        driver = d.AcceleratorDriver.from_hwh(path, mmio_factory=Board().factory)
        with self.assertRaises(ValueError):
            driver.submit_softmax(0x90000000, 0x10000000, 1, 1024)
        with self.assertRaises(ValueError):
            d.discover_hardware(ROOT / "tests/fixtures/old_two_core.hwh")

    def test_context_timeout_keeps_driver_open(self):
        self.board.core(0).hold = True
        with self.assertRaises(TimeoutError):
            with self.driver:
                self.submit().wait(0.00001)
        self.board.core(0).finish()
        self.driver.wait_all()
        self.driver.close()

    def test_no_allocation_or_cache_or_overlay_calls(self):
        source = (ROOT / "tests/accelerator_driver.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        prohibited = {"allocate", "Overlay", "freebuffer", "sync_to_device", "sync_from_device", "mmap"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                self.assertNotIn(name, prohibited)


if __name__ == "__main__":
    unittest.main()



