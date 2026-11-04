"""Small, Python 3.6 compatible SO101 follower adapter.

``open`` and all reads send only ping/read instructions. Only ``execute=True``
permits Goal_Position/Torque_Enable writes. No path writes EEPROM, calibration,
baud rate, ID, mode, gains, speed, or acceleration. Setting the host serial baud
rate does not modify the servos' Baud_Rate register.

Required config: port, joint_order[6], ids[6], norm_modes[6], min[6], max[6],
max_step[6]. Optional: baudrate=1000000, release_on_close=False. Names and units
must match the trained policy. Calibration is the official per-joint JSON with
id, drive_mode, homing_offset, range_min, range_max. Homing offset is already
applied inside each servo; software must not subtract it again.

read_state and write_action use configured joint order and return float32[6].
write_action clips to both configured limits and max_step from a fresh measured
position. Its result is the submitted target, not a claim that motion finished.
Sync-write is broadcast and has no individual servo acknowledgement.
"""

import json
import math
import numbers


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NORM_MODES = ("range_m100_100", "range_0_100", "degrees")
MODEL_NUMBER = 777


class MotorError(RuntimeError):
    """Configuration, serial transport, calibration, or servo failure."""


def _load_sdk():
    try:
        from .vendor import scservo_sdk
    except ImportError as exc:
        raise MotorError("Cannot load vendored Feetech SDK; install pyserial==3.5: {}".format(exc))
    return scservo_sdk


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("{} must be an integer".format(label))
    value = int(value)
    if not low <= value <= high:
        raise ValueError("{} must be in [{}, {}]".format(label, low, high))
    return value


def _vector(value, label):
    if isinstance(value, (str, bytes)):
        raise ValueError("{} must contain six numbers".format(label))
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError("{} must contain six numbers".format(label))
    if len(result) != 6 or not all(math.isfinite(v) for v in result):
        raise ValueError("{} must contain six finite numbers".format(label))
    return result


def _signed(raw, bit):
    """Feetech sign-magnitude encoding, not two's complement."""
    raw = int(raw)
    return -(raw & ~(1 << bit)) if raw & (1 << bit) else raw


def _connect(port, baudrate):
    sdk = _load_sdk()
    handler = sdk.PortHandler(port)
    try:
        # openPort sets only the USB/serial adapter speed, never a motor register.
        handler.baudrate = baudrate
        if not handler.openPort():
            raise MotorError("Cannot open motor port {} at {} baud".format(port, baudrate))
        return sdk, handler, sdk.sms_sts(handler)
    except Exception:
        if getattr(handler, "is_open", False):
            handler.closePort()
        raise


def probe(port, baudrate=1000000, ids=None):
    """Read-only ID/model/position discovery; no calibration or torque writes.

    Returns one dict per requested ID, including an error string when it does
    not answer. This only changes host serial settings. It does not configure
    motors, enable torque, or write targets.
    """
    baudrate = _integer(baudrate, "baudrate", 1, 1000000)
    ids = list(range(1, 7)) if ids is None else list(ids)
    ids = [_integer(v, "motor ID", 1, 253) for v in ids]
    sdk, handler, packet = _connect(port, baudrate)
    result = []
    try:
        for motor_id in ids:
            record = {"id": motor_id}
            try:
                model, comm, error = packet.ping(motor_id)
                if comm != sdk.COMM_SUCCESS or error:
                    raise MotorError("ping: communication={}, servo_error={}".format(comm, error))
                pos, comm, error = packet.read2ByteTxRx(motor_id, 56)
                if comm != sdk.COMM_SUCCESS or error:
                    raise MotorError("position: communication={}, servo_error={}".format(comm, error))
                record.update(model_number=int(model), present_position=_signed(pos, 15))
            except Exception as exc:
                record["error"] = str(exc)
            result.append(record)
    finally:
        handler.closePort()
    return result


class So101Motors(object):
    def __init__(self, config, calibration_file, execute=False):
        if not isinstance(config, dict):
            raise ValueError("motors config must be a dictionary")
        if not isinstance(execute, bool):
            raise ValueError("execute must be a bool")
        self.execute = execute
        self.port = config.get("port")
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("motors.port is required")
        self.baudrate = _integer(config.get("baudrate", 1000000), "baudrate", 1, 1000000)
        self.joint_order = tuple(config.get("joint_order", ()))
        if len(self.joint_order) != 6 or set(self.joint_order) != set(JOINTS):
            raise ValueError("joint_order must list the six SO101 joint names exactly once")
        self.ids = tuple(_integer(v, "motor ID", 1, 253) for v in config.get("ids", ()))
        if len(self.ids) != 6 or len(set(self.ids)) != 6:
            raise ValueError("ids must contain six unique motor IDs")
        self.norm_modes = tuple(config.get("norm_modes") or ())
        if len(self.norm_modes) != 6 or any(m not in NORM_MODES for m in self.norm_modes):
            raise ValueError("norm_modes must explicitly specify six supported modes")
        self.minimum = _vector(config.get("min"), "min")
        self.maximum = _vector(config.get("max"), "max")
        self.max_step = _vector(config.get("max_step"), "max_step")
        self.release_on_close = config.get("release_on_close", False)
        if not isinstance(self.release_on_close, bool):
            raise ValueError("release_on_close must be a bool")
        for i in range(6):
            if self.minimum[i] >= self.maximum[i] or self.max_step[i] <= 0:
                raise ValueError("Each joint needs min < max and a positive max_step")
        if not calibration_file:
            raise ValueError("A real follower calibration JSON is required; use probe for discovery")
        with open(calibration_file, "r", encoding="utf-8") as stream:
            calibration = json.load(stream)
        if not isinstance(calibration, dict):
            raise ValueError("Calibration must be a per-joint dictionary")
        self.calibration = []
        for i, name in enumerate(self.joint_order):
            source = calibration.get(name)
            if not isinstance(source, dict):
                raise ValueError("Missing calibration for {}".format(name))
            cal = {
                "id": _integer(source.get("id"), name + ".id", 1, 253),
                "drive_mode": _integer(source.get("drive_mode"), name + ".drive_mode", 0, 1),
                "homing_offset": _integer(source.get("homing_offset"), name + ".homing_offset", -2047, 2047),
                "range_min": _integer(source.get("range_min"), name + ".range_min", 0, 4095),
                "range_max": _integer(source.get("range_max"), name + ".range_max", 0, 4095),
            }
            if cal["id"] != self.ids[i] or cal["range_min"] >= cal["range_max"]:
                raise ValueError("Calibration ID or range invalid for {}".format(name))
            self.calibration.append(cal)
            mode = self.norm_modes[i]
            if mode == "range_m100_100":
                allowed_min, allowed_max = -100.0, 100.0
            elif mode == "range_0_100":
                allowed_min, allowed_max = 0.0, 100.0
            else:
                half_range = (cal["range_max"] - cal["range_min"]) * 180.0 / 4095.0
                allowed_min, allowed_max = -half_range, half_range
            if self.minimum[i] < allowed_min or self.maximum[i] > allowed_max:
                raise ValueError("Configured limits exceed calibrated {} range for {}".format(mode, name))
        self._sdk = self._handler = self._packet = None
        self._opened = False
        self._enabled = False
        self._owned_ids = set()
        self._enable_failed = False

    @property
    def enabled(self):
        return self._enabled

    def _require_open(self):
        if not self._opened:
            raise MotorError("Motor port is not open")

    def _check(self, comm, error, label):
        if comm != self._sdk.COMM_SUCCESS or error:
            raise MotorError("{}: communication={}, servo_error={}".format(label, comm, error))

    def _read(self, motor_id, address, size):
        func = self._packet.read1ByteTxRx if size == 1 else self._packet.read2ByteTxRx
        value, comm, error = func(motor_id, address)
        self._check(comm, error, "Read ID {} address {}".format(motor_id, address))
        return int(value)

    def open(self):
        """Only ping/read; fail on a wrong model, mode, or calibration."""
        if self._opened:
            return self
        self._sdk, self._handler, self._packet = _connect(self.port, self.baudrate)
        self._opened = True
        try:
            for motor_id in self.ids:
                model, comm, error = self._packet.ping(motor_id)
                self._check(comm, error, "Ping ID {}".format(motor_id))
                if model != MODEL_NUMBER:
                    raise MotorError("ID {} model is {}, expected STS3215 model 777".format(motor_id, model))
            self.check_calibration()
            return self
        except Exception:
            self.close()
            raise

    def _telemetry(self):
        self._require_open()
        # A fresh reader prevents stale values after a partial/failed response.
        reader = self._sdk.GroupSyncRead(self._packet, 56, 15)
        try:
            for motor_id in self.ids:
                if not reader.addParam(motor_id):
                    raise MotorError("Cannot add ID {} to sync read".format(motor_id))
            self._check(reader.txRxPacket(), 0, "Sync read")
            records = {}
            for i, motor_id in enumerate(self.ids):
                available, error = reader.isAvailable(motor_id, 56, 15)
                if not available:
                    raise MotorError("No complete sync response from ID {}".format(motor_id))
                self._check(self._sdk.COMM_SUCCESS, error, "Sync read ID {}".format(motor_id))
                status = int(reader.getData(motor_id, 65, 1))
                if status:
                    raise MotorError("ID {} reports Status register fault 0x{:02x}".format(motor_id, status))
                records[self.joint_order[i]] = {
                    "id": motor_id, "model_number": MODEL_NUMBER,
                    "present_position": _signed(reader.getData(motor_id, 56, 2), 15),
                    "present_velocity": _signed(reader.getData(motor_id, 58, 2), 15),
                    "present_load": _signed(reader.getData(motor_id, 60, 2), 10),
                    "present_voltage": int(reader.getData(motor_id, 62, 1)),
                    "present_temperature": int(reader.getData(motor_id, 63, 1)),
                    "status": status, "moving": int(reader.getData(motor_id, 66, 1)),
                    "present_current": int(reader.getData(motor_id, 69, 2)),
                }
            return records
        finally:
            reader.clearParam()

    def read_raw_status(self):
        """Read hardware registers; positions retain hardware-applied homing.

        Voltage/current retain raw register units. No register is modified.
        """
        records = self._telemetry()
        for name, motor_id in zip(self.joint_order, self.ids):
            records[name].update(
                torque_enable=self._read(motor_id, 40, 1),
                operating_mode=self._read(motor_id, 33, 1),
                homing_offset=_signed(self._read(motor_id, 31, 2), 11),
                range_min=self._read(motor_id, 9, 2),
                range_max=self._read(motor_id, 11, 2),
            )
        return records

    def check_calibration(self):
        records = self.read_raw_status()
        for i, name in enumerate(self.joint_order):
            record = records[name]
            if record["operating_mode"] != 0:
                raise MotorError("{} must already be in position mode (0); no mode change was sent".format(name))
            for key in ("homing_offset", "range_min", "range_max"):
                if record[key] != self.calibration[i][key]:
                    raise MotorError("{} {}: hardware {}, calibration {}; no calibration was written".format(
                        name, key, record[key], self.calibration[i][key]))
        return records

    def _normalize(self, position, index):
        cal = self.calibration[index]
        lo, hi = cal["range_min"], cal["range_max"]
        mode = self.norm_modes[index]
        if mode == "degrees":
            # Official LeRobot degrees mode does not apply software drive_mode.
            return (position - (lo + hi) / 2.0) * 360.0 / 4095.0
        position = max(lo, min(hi, position))
        fraction = (position - lo) / float(hi - lo)
        if mode == "range_m100_100":
            value = fraction * 200.0 - 100.0
            return -value if cal["drive_mode"] else value
        value = fraction * 100.0
        return 100.0 - value if cal["drive_mode"] else value

    def _unnormalize(self, value, index):
        cal = self.calibration[index]
        lo, hi = cal["range_min"], cal["range_max"]
        mode = self.norm_modes[index]
        if mode == "degrees":
            raw = int(value * 4095.0 / 360.0 + (lo + hi) / 2.0)
        elif mode == "range_m100_100":
            value = -value if cal["drive_mode"] else value
            raw = int((max(-100.0, min(100.0, value)) + 100.0) / 200.0 * (hi - lo) + lo)
        else:
            value = 100.0 - value if cal["drive_mode"] else value
            raw = int(max(0.0, min(100.0, value)) / 100.0 * (hi - lo) + lo)
        return max(lo, min(hi, raw))

    def read_state(self):
        import numpy as np
        records = self._telemetry()
        return np.asarray([self._normalize(records[name]["present_position"], i)
                           for i, name in enumerate(self.joint_order)], dtype=np.float32)

    def _write_byte(self, motor_id, address, value):
        if not self.execute:
            raise MotorError("Register writes require execute=True")
        comm, error = self._packet.write1ByteTxRx(motor_id, address, value)
        self._check(comm, error, "Write ID {} address {}".format(motor_id, address))

    def _write_goals(self, raw_positions):
        if not self.execute:
            raise MotorError("Goal writes require execute=True")
        writer = self._sdk.GroupSyncWrite(self._packet, 42, 2)
        try:
            for motor_id, position in zip(self.ids, raw_positions):
                if not 0 <= position <= 4095:
                    raise MotorError("Invalid single-turn goal for ID {}".format(motor_id))
                if not writer.addParam(motor_id, [position & 255, (position >> 8) & 255]):
                    raise MotorError("Cannot add ID {} to sync write".format(motor_id))
            self._check(writer.txPacket(), 0, "Goal sync write")
        finally:
            writer.clearParam()

    def enable(self, before_write=None):
        """Preload measured positions, then enable torque; optional guard may abort TX."""
        self._require_open()
        if not self.execute:
            raise MotorError("Torque enable requires execute=True")
        if self._enabled:
            return
        records = self.check_calibration()
        if any(records[name]["torque_enable"] != 0 for name in self.joint_order):
            raise MotorError("Torque is already enabled; refusing to take ownership of another controller")
        raw = [records[name]["present_position"] for name in self.joint_order]
        if before_write is not None:
            before_write()
        self._write_goals(raw)
        # A broadcast success only confirms transmission. Verify preload before torque.
        for motor_id, position in zip(self.ids, raw):
            if self._read(motor_id, 42, 2) != position:
                raise MotorError("Goal preload was not confirmed for ID {}".format(motor_id))
        try:
            for motor_id in self.ids:
                if before_write is not None:
                    before_write()
                # Include an attempted write: a missing ACK may still mean it took effect.
                self._owned_ids.add(motor_id)
                self._write_byte(motor_id, 40, 1)
                if self._read(motor_id, 40, 1) != 1:
                    raise MotorError("Torque enable was not confirmed for ID {}".format(motor_id))
            self._enabled = True
            self._enable_failed = False
        except Exception:
            self._enable_failed = True
            self._release_owned()
            raise

    def write_action(self, action, before_write=None):
        """Send a bounded target; optional callback validates final ticks just before TX.

        ``before_write(actual_float32)`` runs after the fresh serial read and
        quantization, before any goal write. Raising from it cancels the write.
        """
        import numpy as np
        self._require_open()
        if not self.execute or not self._enabled:
            raise MotorError("Action writes require execute=True and successful enable()")
        proposed = np.asarray(action)
        if proposed.shape != (6,) or proposed.dtype.kind not in "fiu":
            raise ValueError("Action must be a numeric vector with shape [6]")
        target = _vector(proposed, "action")
        current = self.read_state()
        raw, actual = [], []
        for i in range(6):
            low = max(self.minimum[i], float(current[i]) - self.max_step[i])
            high = min(self.maximum[i], float(current[i]) + self.max_step[i])
            if low > high:
                raise MotorError("{} is too far outside configured limits; no action was sent".format(self.joint_order[i]))
            target[i] = max(low, min(high, target[i]))
            # Choose an encoder tick inside BOTH intervals. A plain int() near a
            # boundary can round outside one interval, especially when inverted.
            candidates = set()
            for value in (target[i], low, high):
                center = self._unnormalize(value, i)
                candidates.update(range(center - 2, center + 3))
            cal = self.calibration[i]
            valid = [(tick, self._normalize(tick, i)) for tick in candidates
                     if cal["range_min"] <= tick <= cal["range_max"]]
            valid = [(tick, value) for tick, value in valid
                     if low - 1e-5 <= value <= high + 1e-5]
            if not valid:
                raise MotorError("No encoder tick within configured limits/step for {}; no action was sent".format(self.joint_order[i]))
            tick, value = min(valid, key=lambda item: (abs(item[1] - target[i]), item[0]))
            raw.append(tick)
            actual.append(value)
        actual = np.asarray(actual, dtype=np.float32)
        if before_write is not None:
            before_write(actual)
        self._write_goals(raw)
        return actual

    def _release_owned(self):
        failures = []
        for motor_id in sorted(self._owned_ids):
            try:
                self._write_byte(motor_id, 40, 0)
                if self._read(motor_id, 40, 1) != 0:
                    raise MotorError("Torque disable was not confirmed")
                self._owned_ids.remove(motor_id)
            except Exception as exc:
                failures.append("ID {}: {}".format(motor_id, exc))
        self._enabled = False
        return failures

    def close(self):
        """Optionally release only torque enabled here, then close the host port.

        By default enabled motors retain their last target/torque. Set
        release_on_close=True to release them; this may let the arm fall.
        """
        if not self._opened:
            return
        failures = []
        if self.execute and (self.release_on_close or self._enable_failed):
            failures = self._release_owned()
        try:
            self._handler.closePort()
        finally:
            self._opened = self._enabled = False
            self._handler = self._packet = self._sdk = None
            self._owned_ids.clear()
        if failures:
            raise MotorError("Closed serial port, but could not confirm torque release: " + "; ".join(failures))

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
