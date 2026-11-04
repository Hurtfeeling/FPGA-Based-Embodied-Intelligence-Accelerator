"""Latest-frame V4L2 capture for the two SmolVLA image inputs.

Python 3.6 compatible. Importing this module, or requesting its help, does not
import OpenCV/NumPy and does not open a device. ``start()`` is the first operation
that accesses cameras. Device-to-role mapping must be configured explicitly:
``camera2`` is the wrist camera and ``camera3`` is the side camera.

Example::

    cameras = DualCameraCapture(
        camera2=CameraConfig(device="/dev/video0"),
        camera3=CameraConfig(device="/dev/video2"),
    )  # Verify the physical roles before using these example device names.
    with cameras:
        pair = cameras.snapshot(timeout=2.0, max_age=0.5, max_skew=0.1)
        inputs = pair.as_inputs()  # camera2/camera3 -> RGB float32 NCHW

Each worker owns its VideoCapture and retains only its newest BGR frame. Frame
timestamps use time.monotonic() immediately AFTER read() returns; they measure
host arrival, not sensor exposure time or hardware synchronization. At the
default resolution each output is contiguous float32 [1, 3, 480, 640] in [0, 1].
Rotation is clockwise, followed by flip. The resulting dimensions must match
the configured width/height. No resize occurs unless resize=True is explicit.
In particular, a 90/270 degree rotation of a landscape frame changes its shape.

V4L2 drivers may ignore OpenCV timeout/buffer properties. Waiting for startup or
a snapshot is bounded in Python. A driver stuck inside open/read cannot safely
be killed as a Python thread: close() reports a timeout instead of pretending
that its worker has released the device. Normal shutdown releases each capture
in its owning worker; close() can be called again to finish a delayed shutdown.
"""

import argparse
import importlib
import math
import numbers
import threading
import time


__all__ = [
    "CameraError", "CameraOpenError", "CameraDisconnectedError",
    "CameraTimeoutError", "CameraFrameError", "CameraSyncError",
    "CameraConfig", "CameraPairSnapshot", "DualCameraCapture",
]


class CameraError(RuntimeError):
    """A camera dependency, device, capture, or synchronization failure."""


class CameraOpenError(CameraError):
    """The V4L2 device could not be opened or configured."""


class CameraDisconnectedError(CameraError):
    """An opened camera stopped returning frames."""


class CameraTimeoutError(CameraError):
    """Startup, capture, snapshot, or shutdown exceeded its time limit."""


class CameraFrameError(CameraError):
    """A returned frame has an unsupported type or dimensions."""


class CameraSyncError(CameraTimeoutError):
    """No sufficiently recent pair with acceptable timestamp skew arrived."""


def _seconds(value, label, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("{} must be {}finite seconds".format(
            label, "nonnegative " if allow_zero else "positive "))
    return value


class CameraConfig(object):
    """Configuration for one USB camera.

    device: nonnegative index or nonempty V4L2 device string.
    width/height: requested capture and required final output dimensions.
    fps: requested frame rate; actual camera rate is available in diagnostics().
    mjpg: request MJPG when True; False leaves the device pixel format unchanged.
    rotation: clockwise 0, 90, 180, or 270 degrees.
    flip: None, 'horizontal', 'vertical', 'both', or OpenCV code 1, 0, -1.
    resize: explicitly permit resizing after rotation/flip (default False).
    read_timeout: maximum tolerated time within an individual read, in seconds.

    CameraConfig(**config_dict) is supported. Unsupported camera settings are
    diagnosed from returned frame dimensions rather than relying on set()'s
    return value, since V4L2 drivers differ in how they acknowledge settings.
    """

    def __init__(self, device, width=640, height=480, fps=30.0, mjpg=True,
                 rotation=0, flip=None, resize=False, read_timeout=2.0):
        if isinstance(device, bool) or not isinstance(device, (str, numbers.Integral)):
            raise ValueError("device must be a V4L2 string or nonnegative index")
        if isinstance(device, str):
            if not device.strip():
                raise ValueError("device string must not be empty")
            self.device = device
        else:
            if device < 0:
                raise ValueError("camera index must be nonnegative")
            self.device = int(device)
        for value, label in ((width, "width"), (height, "height")):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError("{} must be a positive integer".format(label))
        if isinstance(rotation, bool) or rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be clockwise 0, 90, 180, or 270")
        if not isinstance(mjpg, bool) or not isinstance(resize, bool):
            raise ValueError("mjpg and resize must be booleans")
        flips = {"horizontal": 1, "vertical": 0, "both": -1}
        if isinstance(flip, str):
            if flip not in flips:
                raise ValueError("flip must be horizontal, vertical, both, or None")
            flip = flips[flip]
        if isinstance(flip, bool) or (flip is not None and flip not in (-1, 0, 1)):
            raise ValueError("flip must be None or OpenCV flip code -1, 0, or 1")
        self.width = int(width)
        self.height = int(height)
        self.fps = _seconds(fps, "fps")
        self.mjpg = mjpg
        self.rotation = int(rotation)
        self.flip = flip
        self.resize = resize
        self.read_timeout = _seconds(read_timeout, "read_timeout")

    def as_dict(self):
        return dict(device=self.device, width=self.width, height=self.height,
                    fps=self.fps, mjpg=self.mjpg, rotation=self.rotation,
                    flip=self.flip, resize=self.resize,
                    read_timeout=self.read_timeout)


class CameraPairSnapshot(object):
    """Independent model inputs and their host-arrival timing metadata.

    camera2/camera3 are RGB float32 contiguous NCHW arrays. timestamps and
    sequences are dictionaries with those same keys. ages are measured at
    captured_at (after preprocessing); skew is the absolute arrival-time gap.
    The arrays do not share storage with the capture workers.
    """

    def __init__(self, camera2, camera3, timestamps, sequences, captured_at):
        self.camera2 = camera2
        self.camera3 = camera3
        self.timestamps = dict(timestamps)
        self.sequences = dict(sequences)
        self.captured_at = captured_at
        self.ages = dict((name, captured_at - stamp)
                         for name, stamp in self.timestamps.items())
        self.skew = abs(self.timestamps["camera2"] - self.timestamps["camera3"])

    def as_inputs(self):
        """Return {'camera2': array, 'camera3': array} without another copy."""
        return {"camera2": self.camera2, "camera3": self.camera3}


def _load_dependencies():
    try:
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise CameraError("Camera capture requires OpenCV (cv2) and NumPy: {}".format(exc)) from exc
    if not hasattr(cv2, "CAP_V4L2"):
        raise CameraOpenError("This OpenCV build does not expose the V4L2 backend")
    return cv2, np


class _Reader(object):
    def __init__(self, name, config, condition, stop, cv2, np):
        self.name = name
        self.config = config
        self.condition = condition
        self.stop = stop
        self.cv2 = cv2
        self.np = np
        self.latest = None  # (BGR array, arrival time, sequence number)
        self.error = None
        self.actual = {}
        self.read_started = None
        self.thread = threading.Thread(target=self._run, name="v4l2-" + name)
        self.thread.daemon = True

    def _validate(self, frame):
        if (not isinstance(frame, self.np.ndarray) or frame.dtype != self.np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or frame.shape[0] == 0 or frame.shape[1] == 0):
            raise CameraFrameError("{} ({!r}) returned an empty or non-uint8 BGR frame".format(
                self.name, self.config.device))
        height, width = frame.shape[:2]
        if self.config.rotation in (90, 270):
            height, width = width, height
        if not self.config.resize and (width, height) != (self.config.width, self.config.height):
            raise CameraFrameError(
                "{} ({!r}) frame after rotation would be {}x{}; expected {}x{}. "
                "Configure the camera/rotation correctly or explicitly set resize=True.".format(
                    self.name, self.config.device, width, height,
                    self.config.width, self.config.height))

    def _run(self):
        cap = None
        try:
            # The capture is opened, used, and released only by this worker.
            cap = self.cv2.VideoCapture()
            if not cap.open(self.config.device, self.cv2.CAP_V4L2) or not cap.isOpened():
                raise CameraOpenError("{} could not open V4L2 device {!r}".format(
                    self.name, self.config.device))
            if self.config.mjpg:
                cap.set(self.cv2.CAP_PROP_FOURCC,
                        self.cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(self.cv2.CAP_PROP_FPS, self.config.fps)
            cap.set(self.cv2.CAP_PROP_CONVERT_RGB, 1)
            # Best effort: some V4L2 drivers ignore both of these properties.
            if hasattr(self.cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(self.cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(self.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        self.config.read_timeout * 1000.0)
            actual = dict(width=cap.get(self.cv2.CAP_PROP_FRAME_WIDTH),
                          height=cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT),
                          fps=cap.get(self.cv2.CAP_PROP_FPS),
                          fourcc=cap.get(self.cv2.CAP_PROP_FOURCC))
            with self.condition:
                self.actual = actual
                self.condition.notify_all()
            sequence = 0
            while not self.stop.is_set():
                with self.condition:
                    self.read_started = time.monotonic()
                    self.condition.notify_all()
                ok, frame = cap.read()
                arrived = time.monotonic()
                if self.stop.is_set():
                    break
                if arrived - self.read_started > self.config.read_timeout:
                    raise CameraTimeoutError("{} ({!r}) read took {:.3f}s, limit {:.3f}s".format(
                        self.name, self.config.device, arrived - self.read_started,
                        self.config.read_timeout))
                if not ok or frame is None:
                    raise CameraDisconnectedError(
                        "{} ({!r}) stopped returning frames (disconnected or V4L2 read failure)".format(
                            self.name, self.config.device))
                self._validate(frame)
                sequence += 1
                with self.condition:
                    self.latest = (frame, arrived, sequence)
                    self.read_started = None
                    self.condition.notify_all()
        except Exception as exc:
            if not self.stop.is_set():
                if not isinstance(exc, CameraError):
                    exc = CameraError("{} ({!r}) capture failed: {}".format(
                        self.name, self.config.device, exc))
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as exc:
                    with self.condition:
                        if self.error is None:
                            self.error = CameraError("{} device release failed: {}".format(self.name, exc))
            with self.condition:
                self.read_started = None
                self.condition.notify_all()


class DualCameraCapture(object):
    """Two independent latest-frame workers with checked paired snapshots.

    Pass CameraConfig objects or dictionaries for both explicit device roles.
    start(timeout=10.0) waits for the first valid frame from both devices and
    returns self. snapshot(timeout=2.0, max_age=0.5, max_skew=0.1) waits for a
    recent acceptable pair and returns CameraPairSnapshot. Repeated snapshots
    may reuse a still-recent frame; no frame queue is retained.
    """

    def __init__(self, camera2, camera3):
        self.configs = {}
        for name, config in (("camera2", camera2), ("camera3", camera3)):
            if isinstance(config, dict):
                config = CameraConfig(**config)
            if not isinstance(config, CameraConfig):
                raise TypeError("{} must be CameraConfig or a configuration dict".format(name))
            self.configs[name] = config
        if self.configs["camera2"].device == self.configs["camera3"].device:
            raise ValueError("camera2 and camera3 must refer to different camera devices")
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._readers = {}
        self._started = False
        self._closed = False
        self._cv2 = None
        self._np = None

    def _check_locked(self, now):
        if self._closed:
            raise CameraError("Camera capture is closed")
        if not self._started:
            raise CameraError("Call start() before taking a camera snapshot")
        for reader in self._readers.values():
            if reader.error is not None:
                raise reader.error
            if (reader.read_started is not None
                    and now - reader.read_started > reader.config.read_timeout):
                raise CameraTimeoutError("{} ({!r}) has not completed a read for {:.3f}s".format(
                    reader.name, reader.config.device, now - reader.read_started))

    def start(self, timeout=10.0):
        """Open both devices and wait for their first frames; return self."""
        timeout = _seconds(timeout, "timeout")
        deadline = time.monotonic() + timeout
        try:
            with self._condition:
                if self._closed:
                    raise CameraError("A closed DualCameraCapture cannot be restarted")
                if not self._started:
                    self._cv2, self._np = _load_dependencies()
                    self._readers = dict((name, _Reader(name, config, self._condition,
                                                       self._stop, self._cv2, self._np))
                                         for name, config in self.configs.items())
                    self._started = True
                    for reader in self._readers.values():
                        reader.thread.start()
            with self._condition:
                while True:
                    now = time.monotonic()
                    self._check_locked(now)
                    if all(reader.latest is not None for reader in self._readers.values()):
                        return self
                    remaining = deadline - now
                    if remaining <= 0:
                        missing = [name for name, reader in self._readers.items()
                                   if reader.latest is None]
                        raise CameraTimeoutError("No first frame from {} within {:.3f}s".format(
                            ", ".join(missing), timeout))
                    self._condition.wait(min(remaining, 0.1))
        except BaseException:
            # Preserve the startup failure even if a driver also stalls during
            # cleanup; close() can be retried by the owner if necessary.
            try:
                self.close()
            except CameraError:
                pass
            raise

    def _to_rgb_nchw(self, frame, config):
        if config.rotation == 90:
            frame = self._np.rot90(frame, k=3)
        elif config.rotation == 180:
            frame = self._np.rot90(frame, k=2)
        elif config.rotation == 270:
            frame = self._np.rot90(frame, k=1)
        if config.flip is not None:
            frame = self._cv2.flip(frame, config.flip)
        if frame.shape[:2] != (config.height, config.width):
            if not config.resize:
                raise CameraFrameError("Unexpected frame dimensions during preprocessing")
            frame = self._cv2.resize(frame, (config.width, config.height),
                                     interpolation=self._cv2.INTER_LINEAR)
        # ascontiguousarray allocates independent float32 storage, fixes the
        # negative RGB stride, and avoids mutating the worker's BGR frame.
        output = self._np.ascontiguousarray(
            frame[:, :, ::-1].transpose(2, 0, 1)[None, :, :, :],
            dtype=self._np.float32)
        output *= self._np.float32(1.0 / 255.0)
        return output

    def snapshot(self, timeout=2.0, max_age=0.5, max_skew=0.1):
        """Return a checked pair, including preprocessing in the age check.

        max_age/max_skew are seconds. A disconnect/capture failure is raised
        immediately. If no pair meets both constraints before timeout,
        CameraSyncError reports the most recent ages and timestamp skew.
        """
        timeout = _seconds(timeout, "timeout")
        max_age = _seconds(max_age, "max_age")
        max_skew = _seconds(max_skew, "max_skew", allow_zero=True)
        deadline = time.monotonic() + timeout
        last_reason = "waiting for both first frames"
        last_rejected = None
        while True:
            with self._condition:
                while True:
                    now = time.monotonic()
                    self._check_locked(now)
                    records = dict((name, reader.latest)
                                   for name, reader in self._readers.items())
                    remaining = deadline - now
                    if remaining <= 0:
                        raise CameraSyncError("No acceptable camera pair within {:.3f}s: {}".format(
                            timeout, last_reason))
                    if all(record is not None for record in records.values()):
                        ages = dict((name, now - record[1]) for name, record in records.items())
                        skew = abs(records["camera2"][1] - records["camera3"][1])
                        sequences = tuple(records[name][2] for name in ("camera2", "camera3"))
                        last_reason = ("ages camera2={:.3f}s, camera3={:.3f}s "
                                       "(limit {:.3f}s), skew={:.3f}s (limit {:.3f}s)").format(
                                           ages["camera2"], ages["camera3"],
                                           max_age, skew, max_skew)
                        if (max(ages.values()) <= max_age and skew <= max_skew
                                and sequences != last_rejected):
                            break
                    self._condition.wait(min(remaining, 0.1))
            # Workers only replace their latest reference; they never mutate
            # previously published arrays, so preprocessing needs no long lock.
            try:
                arrays = dict((name, self._to_rgb_nchw(record[0], self.configs[name]))
                              for name, record in records.items())
            except CameraError:
                raise
            except Exception as exc:
                raise CameraFrameError("Camera input preprocessing failed: {}".format(exc)) from exc
            finished = time.monotonic()
            with self._condition:
                self._check_locked(finished)
            timestamps = dict((name, record[1]) for name, record in records.items())
            if (finished <= deadline
                    and max(finished - stamp for stamp in timestamps.values()) <= max_age):
                return CameraPairSnapshot(arrays["camera2"], arrays["camera3"],
                                          timestamps, dict((name, record[2])
                                                           for name, record in records.items()),
                                          finished)
            last_rejected = sequences
            last_reason = "the selected pair exceeded max_age or timeout during preprocessing"

    def diagnostics(self):
        """Return requested/negotiated settings and live timing without reads."""
        with self._condition:
            now = time.monotonic()
            result = dict(started=self._started, closed=self._closed, cameras={})
            for name, config in self.configs.items():
                reader = self._readers.get(name)
                latest = reader.latest if reader is not None else None
                result["cameras"][name] = dict(
                    requested=config.as_dict(),
                    negotiated=dict(reader.actual) if reader is not None else {},
                    alive=reader.thread.is_alive() if reader is not None else False,
                    sequence=latest[2] if latest is not None else 0,
                    timestamp=latest[1] if latest is not None else None,
                    age=now - latest[1] if latest is not None else None,
                    raw_shape=list(latest[0].shape) if latest is not None else None,
                    error=str(reader.error) if reader is not None and reader.error else None)
            return result

    def close(self, timeout=3.0):
        """Stop workers and release their devices; safe to call repeatedly.

        timeout bounds the total join time across both workers. If a V4L2 call
        remains blocked, raise CameraTimeoutError and keep its thread handle so
        a subsequent close() can retry. A closed instance cannot be restarted.
        """
        timeout = _seconds(timeout, "timeout")
        with self._condition:
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
            readers = list(self._readers.values())
        deadline = time.monotonic() + timeout
        for reader in readers:
            if reader.thread.ident is not None:
                reader.thread.join(max(0.0, deadline - time.monotonic()))
        alive = [reader.name for reader in readers if reader.thread.is_alive()]
        if alive:
            raise CameraTimeoutError(
                "Camera shutdown timed out after {:.3f}s; V4L2 worker(s) {} are still blocked. "
                "Devices will be released when their calls return; close() may be retried.".format(
                    timeout, ", ".join(alive)))
        failures = [str(reader.error) for reader in readers
                    if reader.error is not None and "device release failed" in str(reader.error)]
        if failures:
            raise CameraError("; ".join(failures))

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except CameraError:
            if exc_type is None:
                raise
        return False


def _main():
    parser = argparse.ArgumentParser(
        description="SmolVLA dual-camera capture module (help only; opens no cameras).",
        epilog="Import CameraConfig and DualCameraCapture from this module to capture frames.")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    _main()
