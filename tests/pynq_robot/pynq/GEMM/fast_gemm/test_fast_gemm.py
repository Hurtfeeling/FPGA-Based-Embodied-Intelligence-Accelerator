"""Protocol tests, not a hardware speed measurement."""
import unittest
import numpy as np
from fast_gemm import FastGemm, _inflight


class Buffer(np.ndarray):
    def __new__(cls, shape, dtype, address):
        obj = np.zeros(shape, dtype=dtype).view(cls)
        obj.device_address, obj.freed = address, False
        return obj


class RegisterModel:
    def __init__(self, reset_id=False):
        self.events, self.devices = [], {}
        self.reset_id = reset_id
        self.fail = None
        self.busy = False
        self.error_core = None
    def factory(self, base, length):
        model = self
        class MMIO:
            def __init__(self):
                self.base = base
                self.values = {0x50:0x0A080020, 0x18:0, 0x20:0, 0x28:0}
                self.running = False
                self.reads_pending = 0
            def read(self, offset):
                model.events.append(('R', base, offset))
                if offset == 0x18 and self.running and not model.busy:
                    # First poll observes BUSY to test the wait/retry path.
                    self.reads_pending += 1
                    if self.reads_pending > 1:
                        self.running = False
                        self.values[0x18] = 4 if model.error_core == base else 128
                        self.values[0x28] = self.values[0x20]
                return self.values.get(offset, 0)
            def write(self, offset, value):
                model.events.append(('W', base, offset, value))
                if model.fail == (base, offset):
                    raise OSError('injected write failure')
                if offset == 0 and value == 0x201:
                    if self.running: raise AssertionError('CLEAR while running')
                    self.values[0x18] = 0
                    if model.reset_id:self.values[0x20] = 0
                    for off in range(0x34,0x50,4):self.values[off] = 0
                elif offset == 0 and value == 0x101:
                    self.values[0x20] = (self.values[0x20]+1)&0xFFFFFFFF
                    self.values[0x18] = 65
                    self.reads_pending, self.running = 0, True
                    self.values[0x34] = 1000
                    self.values[0x3C] = 900
                else:self.values[offset] = value
        mm = MMIO()
        self.devices[base] = mm
        return mm


class Tests(unittest.TestCase):
    def setup_driver(self, reset_id=False):
        model = RegisterModel(reset_id)
        driver = FastGemm(backend='python',mmio_factory=model.factory)
        a = Buffer((2048,32),np.int8,0x1000000)
        bt = Buffer((32,32),np.int8,0x2000000)
        c = Buffer((2048,32),np.int32,0x3000000)
        plan = driver.prepare(a,bt,c)
        model.events.clear()
        self.addCleanup(self.cleanup_driver,driver,model)
        return driver,model,plan
    def cleanup_driver(self,driver,model):
        model.fail,model.busy = None,False
        if driver.active:
            for mm in model.devices.values():mm.running=False;mm.values[0x18]=0
            driver.recover()
        driver.close()
    def test_exact_18_writes_and_order(self):
        driver,model,plan=self.setup_driver()
        task=plan.submit()
        self.assertEqual(len(model.events),18)
        self.assertTrue(all(e[0]=='W' for e in model.events))
        for i,base in enumerate(driver.bases):
            self.assertEqual([e[2] for e in model.events[i*6:(i+1)*6]],[4,8,12,16,20,0])
            self.assertEqual(model.events[i*6+5],('W',base,0,0x101))
        self.assertEqual(plan.rows,((0,0,704),(1,704,672),(2,1376,672)))
        task.wait()
    def test_fast_finish_no_performance_reads(self):
        driver,model,plan=self.setup_driver()
        plan.run()
        self.assertFalse(any(e[0]=='R' and 0x34<=e[2]<=0x4C for e in model.events))
        self.assertTrue(all(driver.ready))
        with self.assertRaises(RuntimeError):driver.read_counters()
    def test_explicit_profile_before_clear(self):
        driver,model,plan=self.setup_driver()
        result=plan.submit().wait_with_counters()
        self.assertEqual(result['counters'][0]['cycles'],1000)
        self.assertEqual(driver.read_counters()['0']['compute_cycles'],900)
        for base in driver.bases:
            counter=model.events.index(('R',base,0x34))
            clear=model.events.index(('W',base,0,0x201))
            self.assertLess(counter,clear)
    def test_clear_id_preserve_or_reset(self):
        for reset in [False,True]:
            driver,model,plan=self.setup_driver(reset)
            first=plan.submit();first.wait()
            second=plan.submit();second.wait()
            self.assertEqual(second.jobs,(1,1,1) if reset else (2,2,2))
    def test_pending_rejects_second_submit_and_close(self):
        driver,model,plan=self.setup_driver()
        task=plan.submit()
        with self.assertRaises(RuntimeError):plan.submit()
        with self.assertRaises(RuntimeError):driver.close()
        self.assertIn(driver,_inflight)
        task.wait();self.assertNotIn(driver,_inflight)
    def test_timeout_keeps_task_and_buffers(self):
        driver,model,plan=self.setup_driver()
        model.busy=True
        task=plan.submit()
        with self.assertRaises(TimeoutError):task.wait(.001)
        self.assertIs(driver.active,task)
        self.assertFalse(any(driver.ready))
        model.busy=False;task.wait()
    def test_later_configuration_failure_retains_active(self):
        driver,model,plan=self.setup_driver()
        model.fail=(0x80001000,4)
        with self.assertRaises(OSError):plan.submit()
        self.assertTrue(driver.active.submission_failed)
        self.assertIn(driver,_inflight)
        self.assertTrue(model.devices[0x80000000].running)
        self.assertNotIn(('W',0x80002000,0,0x101),model.events)
        model.fail=None
        for mm in model.devices.values():mm.running=False;mm.values[0x18]=0
        driver.recover()
        self.assertTrue(all(driver.ready))
    def test_new_addresses_are_written(self):
        driver,model,plan=self.setup_driver()
        plan.run()
        new_a=Buffer((2048,32),np.int8,0x1100000)
        new_c=Buffer((2048,32),np.int32,0x3100000)
        changed=driver.prepare(new_a,plan.buffers[1],new_c)
        model.events.clear()
        changed.submit().wait()
        self.assertIn(('W',0x80000000,4,0x1100000),model.events)
        self.assertIn(('W',0x80000000,12,0x3100000),model.events)
    def test_hardware_error_drains_other_cores(self):
        driver,model,plan=self.setup_driver()
        model.error_core=0x80000000
        with self.assertRaises(RuntimeError):plan.run()
        self.assertIsNone(driver.active)
        self.assertTrue(all(driver.ready))
        self.assertTrue(all(not mm.running for mm in model.devices.values()))
    def test_freed_buffer_rejected_before_any_mmio(self):
        driver,model,plan=self.setup_driver()
        plan.buffers[0].freed=True
        with self.assertRaises(ValueError):plan.submit()
        self.assertFalse(model.events)

    def test_independent_jobs_share_one_ordered_three_core_submit(self):
        driver,model,_=self.setup_driver()
        jobs=[]
        for core,(m,n,k) in enumerate(((64,96,64),(32,64,96),(96,32,32))):
            base=0x10000000+core*0x1000000
            a=Buffer((m,k),np.int8,base)
            bt=Buffer((n,k),np.int8,base+0x200000)
            c=Buffer((m,n),np.int32,base+0x400000)
            jobs.append((core,a,bt,c))
        plan=driver.prepare_jobs(jobs)
        self.assertEqual(plan.cores,(0,1,2))
        self.assertEqual(plan.rows,((0,0,64),(1,0,32),(2,0,96)))
        self.assertEqual(plan.values[0][3:],((2|(3<<16)),2))
        self.assertEqual(plan.values[1][3:],((1|(2<<16)),3))
        model.events.clear()
        plan.run()
        starts=[e for e in model.events if e[0]=='W' and e[2]==0 and e[3]==0x101]
        self.assertEqual(tuple(e[1] for e in starts),driver.bases)

    def test_independent_jobs_reject_cross_job_output_overlap(self):
        driver,model,_=self.setup_driver()
        a0=Buffer((32,32),np.int8,0x10000000)
        b0=Buffer((32,32),np.int8,0x11000000)
        c0=Buffer((32,32),np.int32,0x12000000)
        a1=Buffer((32,32),np.int8,0x13000000)
        b1=Buffer((32,32),np.int8,0x14000000)
        c1=Buffer((32,32),np.int32,0x10000000)
        with self.assertRaises(ValueError):
            driver.prepare_jobs(((0,a0,b0,c0),(1,a1,b1,c1)))
        self.assertFalse(model.events)


if __name__=='__main__':unittest.main()
