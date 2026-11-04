"""Pure mock checks: no pyserial dependency, SDK import, or physical device.

Run from any directory: python path/to/ONNX/tests/test_motors.py -v
Python 3.6 compatible; NumPy is the only non-stdlib dependency used here.
"""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_io import motors  # noqa: E402


class FakeBus(object):
    """Emulates the pinned SDK's API, including isAvailable's tuple result."""

    def __init__(self):
        self.log = []
        self.offline = set()
        self.missing_sync_id = None
        self.fail_enable_id = None
        self.state = {
            motor_id: dict(position=2000, goal=1000, torque=0, mode=0,
                           offset=-30, lo=100, hi=3900, status=0, model=777)
            for motor_id in range(1, 7)
        }
        bus = self

        class Port(object):
            def __init__(self, name):
                self.is_open = False
                self.baudrate = None

            def openPort(self):
                self.is_open = True
                bus.log.append(("open", self.baudrate))
                return True

            def closePort(self):
                self.is_open = False
                bus.log.append(("close",))

        class Packet(object):
            def __init__(self, port):
                pass

            def ping(self, motor_id):
                bus.log.append(("ping", motor_id))
                if motor_id in bus.offline:
                    return 0, -3001, 0
                return bus.state[motor_id]["model"], 0, 0

            def read1ByteTxRx(self, motor_id, address):
                bus.log.append(("read", motor_id, address))
                key = {40: "torque", 33: "mode"}[address]
                return bus.state[motor_id][key], 0, 0

            def read2ByteTxRx(self, motor_id, address):
                bus.log.append(("read", motor_id, address))
                key = {9: "lo", 11: "hi", 31: "offset", 42: "goal", 56: "position"}[address]
                value = bus.state[motor_id][key]
                if address == 31 and value < 0:
                    value = (-value) | (1 << 11)
                return value, 0, 0

            def write1ByteTxRx(self, motor_id, address, value):
                if address != 40:
                    raise AssertionError("Adapter attempted an unexpected register write")
                bus.log.append(("write", motor_id, address, value))
                # A servo may receive a command even when its ACK is lost.
                bus.state[motor_id]["torque"] = value
                if value == 1 and motor_id == bus.fail_enable_id:
                    return -3001, 0
                return 0, 0

        class Reader(object):
            def __init__(self, packet, address, length):
                if (address, length) != (56, 15):
                    raise AssertionError("Unexpected telemetry block")
                self.ids = []

            def addParam(self, motor_id):
                self.ids.append(motor_id)
                return True

            def txRxPacket(self):
                bus.log.append(("sync_read", tuple(self.ids)))
                return 0

            def isAvailable(self, motor_id, address, length):
                return motor_id != bus.missing_sync_id, 0

            def getData(self, motor_id, address, length):
                return bus.state[motor_id].get({56: "position", 65: "status"}.get(address), 0)

            def clearParam(self):
                self.ids = []

        class Writer(object):
            def __init__(self, packet, address, length):
                if (address, length) != (42, 2):
                    raise AssertionError("Adapter attempted an unexpected sync write")
                self.items = []

            def addParam(self, motor_id, value):
                self.items.append((motor_id, value[0] + (value[1] << 8)))
                return True

            def txPacket(self):
                bus.log.append(("goals", tuple(self.items)))
                for motor_id, position in self.items:
                    bus.state[motor_id]["goal"] = position
                return 0

            def clearParam(self):
                self.items = []

        self.sdk = types.SimpleNamespace(
            PortHandler=Port, sms_sts=Packet, GroupSyncRead=Reader,
            GroupSyncWrite=Writer, COMM_SUCCESS=0)

    def writes(self):
        return [entry for entry in self.log if entry[0] in ("write", "goals")]

    def goals(self):
        return [entry for entry in self.log if entry[0] == "goals"]


class MotorTests(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.sdk_patch = mock.patch.object(motors, "_load_sdk", return_value=self.bus.sdk)
        self.sdk_patch.start()
        self.addCleanup(self.sdk_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calibration_path = os.path.join(self.temp.name, "calibration.json")
        self.config = {
            "port": "MOCK_ONLY", "baudrate": 1000000,
            "joint_order": list(motors.JOINTS), "ids": list(range(1, 7)),
            "norm_modes": ["range_m100_100"] * 5 + ["range_0_100"],
            "min": [-80] * 5 + [0], "max": [80] * 5 + [90],
            "max_step": [2] * 5 + [1], "release_on_close": True,
        }
        self.calibration = {
            name: dict(id=motor_id, drive_mode=motor_id % 2,
                       homing_offset=-30, range_min=100, range_max=3900)
            for name, motor_id in zip(motors.JOINTS, range(1, 7))
        }
        with open(self.calibration_path, "w", encoding="utf-8") as stream:
            json.dump(self.calibration, stream)
        self.devices = []
        self.addCleanup(self.close_devices)

    def close_devices(self):
        for device in reversed(self.devices):
            device.close()

    def device(self, execute=False, config=None):
        device = motors.So101Motors(
            self.config if config is None else config,
            self.calibration_path, execute=execute)
        self.devices.append(device)
        return device

    def enabled_device(self):
        device = self.device(execute=True).open()
        device.enable()
        self.bus.log[:] = []
        return device

    def test_read_only_lifecycle_has_no_register_writes(self):
        device = self.device().open()
        state = device.read_state()
        self.assertEqual(state.shape, (6,))
        self.assertEqual(state.dtype, np.float32)
        device.read_raw_status()
        device.check_calibration()
        with self.assertRaises(motors.MotorError):
            device.enable()
        with self.assertRaises(motors.MotorError):
            device.write_action(np.zeros(6))
        device.close()
        device.close()
        self.assertEqual(self.bus.writes(), [])

    def test_homing_is_decoded_but_not_applied_twice(self):
        device = self.device().open()
        raw = device.read_raw_status()
        self.assertEqual(raw["shoulder_pan"]["homing_offset"], -30)
        self.assertEqual(raw["shoulder_pan"]["present_position"], 2000)
        np.testing.assert_allclose(device.read_state(), [0, 0, 0, 0, 0, 50])

    def test_range_modes_apply_software_drive_direction(self):
        device = self.device().open()
        for entry in self.bus.state.values():
            entry["position"] = 1050  # One quarter of the calibrated range.
        np.testing.assert_allclose(device.read_state(), [50, -50, 50, -50, 50, 25])

    def test_degrees_matches_official_semantics_without_drive_inversion(self):
        config = copy.deepcopy(self.config)
        config["norm_modes"][:5] = ["degrees"] * 5
        device = self.device(config=config).open()
        for entry in self.bus.state.values():
            entry["position"] = 2100
        np.testing.assert_allclose(device.read_state()[:5], [100 * 360.0 / 4095.0] * 5)

    def test_explicit_joint_order_controls_vector_order(self):
        config = copy.deepcopy(self.config)
        for key in ("joint_order", "ids", "norm_modes", "min", "max", "max_step"):
            config[key].reverse()
        device = self.device(config=config).open()
        np.testing.assert_allclose(device.read_state(), [50, 0, 0, 0, 0, 0])

    def test_wrong_model_mode_or_calibration_fails_closed_without_writes(self):
        for field, wrong in (("model", 778), ("mode", 1), ("offset", -31), ("hi", 3899)):
            with self.subTest(field=field):
                previous = self.bus.state[1][field]
                self.bus.state[1][field] = wrong
                self.bus.log[:] = []
                with self.assertRaises(motors.MotorError):
                    self.device().open()
                self.assertEqual(self.bus.log[-1], ("close",))
                self.assertEqual(self.bus.writes(), [])
                self.bus.state[1][field] = previous

    def test_enable_preloads_and_confirms_goals_before_torque(self):
        device = self.device(execute=True).open()
        device.enable()
        goal_index = next(i for i, entry in enumerate(self.bus.log) if entry[0] == "goals")
        first_torque = next(i for i, entry in enumerate(self.bus.log) if entry[0] == "write")
        self.assertLess(goal_index, first_torque)
        readbacks = {entry[1] for entry in self.bus.log[goal_index:first_torque]
                     if entry[0] == "read" and entry[2] == 42}
        self.assertEqual(readbacks, set(range(1, 7)))
        self.assertTrue(device.enabled)
        self.assertTrue(all(entry["goal"] == entry["position"] for entry in self.bus.state.values()))

    def test_enable_refuses_to_take_over_existing_torque(self):
        self.bus.state[1]["torque"] = 1
        device = self.device(execute=True).open()
        with self.assertRaises(motors.MotorError):
            device.enable()
        device.close()
        self.assertEqual(self.bus.writes(), [])
        self.assertEqual(self.bus.state[1]["torque"], 1)

    def test_partial_enable_lost_ack_rolls_back_only_attempted_ids(self):
        self.bus.fail_enable_id = 3
        device = self.device(execute=True).open()
        with self.assertRaises(motors.MotorError):
            device.enable()
        enabled_ids = {entry[1] for entry in self.bus.writes() if entry[0] == "write" and entry[3] == 1}
        disabled_ids = {entry[1] for entry in self.bus.writes() if entry[0] == "write" and entry[3] == 0}
        self.assertEqual(enabled_ids, {1, 2, 3})
        self.assertEqual(disabled_ids, {1, 2, 3})
        self.assertFalse(device.enabled)
        self.assertFalse(any(entry["torque"] for entry in self.bus.state.values()))

    def test_action_clips_range_and_step_including_inverted_joints(self):
        device = self.enabled_device()
        current = device.read_state()
        submitted = device.write_action(np.asarray([100] * 6, dtype=np.float32))
        self.assertEqual(submitted.dtype, np.float32)
        self.assertTrue(np.all(np.abs(submitted - current) <= np.asarray(self.config["max_step"]) + 1e-4))
        self.assertTrue(np.all(submitted >= np.asarray(self.config["min"]) - 1e-5))
        self.assertTrue(np.all(submitted <= np.asarray(self.config["max"]) + 1e-5))
        for index, motor_id in enumerate(device.ids):
            raw = self.bus.state[motor_id]["goal"]
            self.assertAlmostEqual(device._normalize(raw, index), float(submitted[index]), places=4)

    def test_action_uses_fresh_state_after_external_motion(self):
        device = self.enabled_device()
        old_state = device.read_state()
        for entry in self.bus.state.values():
            entry["position"] = 2200
        current = device.read_state()
        submitted = device.write_action(np.zeros(6, dtype=np.float32))
        self.assertTrue(np.all(np.abs(submitted - current) <= np.asarray(self.config["max_step"]) + 1e-4))
        self.assertGreater(float(np.max(np.abs(submitted - old_state))), 2)

    def test_invalid_action_never_transmits_goal(self):
        device = self.enabled_device()
        for action in (np.ones((1, 6)), np.full(6, np.nan), np.full(6, np.inf),
                       np.ones(6, dtype=np.complex64), ["a"] * 6):
            with self.subTest(action_type=str(np.asarray(action).dtype)):
                with self.assertRaises(ValueError):
                    device.write_action(action)
                self.assertEqual(self.bus.goals(), [])

    def test_missing_sync_response_or_servo_fault_cancels_action(self):
        device = self.enabled_device()
        self.bus.missing_sync_id = 3
        with self.assertRaises(motors.MotorError):
            device.write_action(np.zeros(6))
        self.assertEqual(self.bus.goals(), [])
        self.bus.missing_sync_id = None
        self.bus.state[1]["status"] = 4
        with self.assertRaises(motors.MotorError):
            device.write_action(np.zeros(6))
        self.assertEqual(self.bus.goals(), [])

    def test_callback_runs_after_fresh_read_and_before_goal_tx(self):
        device = self.enabled_device()
        received = []

        def guard(actual):
            self.assertEqual(actual.shape, (6,))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(any(entry[0] == "sync_read" for entry in self.bus.log))
            self.assertEqual(self.bus.goals(), [])
            received.append(actual.copy())
            self.bus.log.append(("guard",))

        submitted = device.write_action(np.ones(6), before_write=guard)
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0], submitted)
        guard_index = next(i for i, entry in enumerate(self.bus.log) if entry[0] == "guard")
        goal_index = next(i for i, entry in enumerate(self.bus.log) if entry[0] == "goals")
        self.assertLess(guard_index, goal_index)

    def test_callback_exception_cancels_goal_tx(self):
        device = self.enabled_device()
        previous = {motor_id: entry["goal"] for motor_id, entry in self.bus.state.items()}

        class StaleObservation(RuntimeError):
            pass

        def reject_stale_frame(actual):
            self.assertTrue(any(entry[0] == "sync_read" for entry in self.bus.log))
            self.assertEqual(self.bus.goals(), [])
            raise StaleObservation("Observation expired during the serial read")

        with self.assertRaises(StaleObservation):
            device.write_action(np.ones(6), before_write=reject_stale_frame)
        self.assertEqual(self.bus.goals(), [])
        self.assertEqual(previous, {motor_id: entry["goal"] for motor_id, entry in self.bus.state.items()})

    def test_close_releases_owned_torque_once(self):
        device = self.enabled_device()
        device.close()
        writes = list(self.bus.writes())
        self.assertEqual(len(writes), 6)
        self.assertTrue(all(entry[0] == "write" and entry[3] == 0 for entry in writes))
        device.close()
        self.assertEqual(self.bus.writes(), writes)
        self.assertFalse(any(entry["torque"] for entry in self.bus.state.values()))

    def test_hold_on_close_keeps_owned_torque_without_extra_writes(self):
        config = copy.deepcopy(self.config)
        config["release_on_close"] = False
        device = self.device(execute=True, config=config).open()
        device.enable()
        self.bus.log[:] = []
        device.close()
        self.assertEqual(self.bus.writes(), [])
        self.assertTrue(all(entry["torque"] for entry in self.bus.state.values()))

    def test_probe_does_not_need_calibration_and_never_writes(self):
        self.bus.offline.add(3)
        records = motors.probe("MOCK_ONLY")
        self.assertEqual(len(records), 6)
        self.assertEqual(records[0], dict(id=1, model_number=777, present_position=2000))
        self.assertIn("error", records[2])
        self.assertEqual(records[3]["id"], 4)
        self.assertEqual(self.bus.writes(), [])
        self.assertEqual(self.bus.log[-1], ("close",))


if __name__ == "__main__":
    unittest.main()
