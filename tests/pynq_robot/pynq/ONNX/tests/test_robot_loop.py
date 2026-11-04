"""Offline checks for camera layout and final motor-dispatch guards."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_robot
from robot_io import cameras
try:
    import resource
except ImportError:
    sys.modules['resource'] = types.ModuleType('resource')
from robot_io.policy import OnnxPolicy, SCHEMA


class Clock(object):
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Motors(object):
    def __init__(self, state=None, final=None):
        self.state = np.zeros(6, np.float32) if state is None else np.asarray(state, np.float32)
        self.final = final
        self.read_hook = lambda: None
        self.sent = []
        self.enabled = False

    def read_state(self):
        self.read_hook()
        return self.state.copy()

    def enable(self, before_write=None):
        if before_write:
            before_write()
        self.enabled = True

    def write_action(self, value, before_write=None):
        self.read_hook()  # Match the driver's fresh read before the final hook.
        actual = value.copy() if self.final is None else np.asarray(self.final, np.float32)
        if before_write:
            before_write(actual)
        self.sent.append(actual.copy())
        return actual


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stop = Path(self.tmp.name) / 'STOP'
        self.clock = Clock()
        self.control = dict(chunk_steps=1, command_hz=10,
            min=[-5]*6, max=[5]*6, max_step=[2]*6, max_velocity=[10]*6,
            max_state_drift=[2]*6, max_observation_age_s=60, max_command_lateness_s=.2)
        self.actions = np.ones((1,50,6), np.float32)
        self.log = io.StringIO()
        self.patch = mock.patch.object(run_robot, 'time', self.clock)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def play(self, motor, health=None):
        return run_robot.play_chunk(motor, self.actions, self.control, self.stop,
            100.0, motor.state.copy(), self.log, health_check=health)

    def test_stop_created_during_initial_read_blocks_enable(self):
        motor = Motors()
        motor.read_hook = lambda: self.stop.touch()
        with self.assertRaisesRegex(RuntimeError, 'Stop file'):
            self.play(motor)
        self.assertFalse(motor.enabled)
        self.assertEqual(motor.sent, [])

    def test_stale_after_serial_read_blocks_enable(self):
        motor = Motors()
        motor.read_hook = lambda: setattr(self.clock, 'now', 200.0)
        with self.assertRaisesRegex(RuntimeError, 'too old'):
            self.play(motor)
        self.assertFalse(motor.enabled)

    def test_serial_delay_inside_driver_blocks_final_tx(self):
        motor = Motors()
        def delay():
            if motor.enabled:
                self.clock.now += .15
        motor.read_hook = delay
        with self.assertRaisesRegex(RuntimeError, 'deadline'):
            self.play(motor)
        self.assertEqual(motor.sent, [])

    def test_internal_reclamp_cannot_escape_outer_velocity(self):
        motor = Motors(final=[4]*6)
        with self.assertRaisesRegex(RuntimeError, 'velocity'):
            self.play(motor)
        self.assertEqual(motor.sent, [])

    def test_internal_reclamp_cannot_escape_outer_position(self):
        motor = Motors(state=[4.99]*6, final=[5.5]*6)
        with self.assertRaisesRegex(RuntimeError, 'position limits'):
            self.play(motor)
        self.assertEqual(motor.sent, [])

    def test_logs_actual_quantized_target(self):
        motor = Motors(final=[.5]*6)
        self.play(motor)
        record = json.loads(self.log.getvalue())
        self.assertEqual(record['sent'], [.5]*6)
        self.assertEqual(record['requested'], [1.0]*6)

    def test_camera_failure_blocks_enable(self):
        motor = Motors()
        def disconnected():
            raise cameras.CameraDisconnectedError('unplugged')
        with self.assertRaises(cameras.CameraDisconnectedError):
            self.play(motor, disconnected)
        self.assertFalse(motor.enabled)
        self.assertEqual(motor.sent, [])

    def test_invalid_command_envelope_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'envelope'):
            run_robot.guarded_target(np.zeros(6), np.ones(6)*10, np.zeros(6),
                [-5]*6, [5]*6, [1]*6, [1]*6, .1)


class CameraTests(unittest.TestCase):
    def test_bgr_to_rgb_and_independent_contiguous_input(self):
        cap = cameras.DualCameraCapture(dict(device='/dev/fake0'), dict(device='/dev/fake2'))
        cap._np = np
        cap._cv2 = types.SimpleNamespace()
        frame = np.zeros((480,640,3), np.uint8)
        frame[:] = [10,20,255]
        result = cap._to_rgb_nchw(frame, cap.configs['camera2'])
        self.assertEqual(result.shape, (1,3,480,640))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result[0,:,0,0], [1,20/255.,10/255.], atol=1e-7)
        result.fill(0)
        np.testing.assert_array_equal(frame[0,0], [10,20,255])

    def test_size_mismatch_is_not_silently_resized(self):
        cap = cameras.DualCameraCapture(dict(device=0), dict(device=2))
        cap._np = np
        with self.assertRaises(cameras.CameraFrameError):
            cap._to_rgb_nchw(np.zeros((240,320,3),np.uint8), cap.configs['camera2'])


class SessionTests(unittest.TestCase):
    def test_close_waits_until_native_run_returns(self):
        started, finish, closed = threading.Event(), threading.Event(), threading.Event()
        class Lib(object):
            def smol_run(self, handle, *pointers):
                started.set()
                if not finish.wait(3):
                    raise RuntimeError('test timed out')
                import ctypes
                values = np.ctypeslib.as_array(ctypes.cast(pointers[-1], ctypes.POINTER(ctypes.c_float)), shape=(300,))
                values.fill(.25)
                return 0
            def smol_destroy(self, handle):
                closed.set()
        policy = OnnxPolicy('unused', mode='cpu')
        policy.handle = 1
        policy.lib = Lib()
        data = {name:np.zeros(shape,dtype) for name,(shape,dtype) in SCHEMA.items()}
        data['language_mask'][:] = True
        errors = []
        def predict():
            try:
                np.testing.assert_array_equal(policy.predict(data), np.full((1,50,6),.25,np.float32))
            except BaseException as error:
                errors.append(error)
        runner = threading.Thread(target=predict)
        closer = threading.Thread(target=policy.close)
        runner.start()
        self.assertTrue(started.wait(2))
        closer.start()
        self.assertFalse(closed.wait(.05))
        finish.set()
        runner.join(3); closer.join(3)
        self.assertFalse(errors)
        self.assertTrue(closed.is_set())
        with self.assertRaisesRegex(RuntimeError,'Load ONNX'):
            policy.predict(data)


if __name__ == '__main__':
    unittest.main()
