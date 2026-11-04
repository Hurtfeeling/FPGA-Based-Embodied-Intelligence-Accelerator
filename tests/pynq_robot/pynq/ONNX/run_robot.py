#!/usr/bin/python3
"""Run camera/state -> persistent SmolVLA ONNX -> calibrated motor targets.

Without --execute, live mode only reads motors and logs predicted targets.
Replay mode never opens cameras or a serial port. Python 3.6 compatible.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import time


BASE = Path(__file__).resolve().parent


def resolve(base, value):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')
    return value


def vector(value, name, positive_only=False):
    import numpy as np
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (6,) or not np.isfinite(arr).all():
        raise ValueError(name + ' must contain six finite values')
    if positive_only and not (arr > 0).all():
        raise ValueError(name + ' must contain positive values')
    return arr


def guarded_target(target, current, previous, lower, upper, max_step, max_velocity, elapsed):
    """Bound against measured position AND the previous command's speed limit."""
    import numpy as np
    target = vector(target, 'action')
    current = vector(current, 'current position')
    previous = vector(previous, 'previous command')
    max_step = vector(max_step, 'max_step', True)
    max_velocity = vector(max_velocity, 'max_velocity', True)
    lower, upper = vector(lower, 'minimum'), vector(upper, 'maximum')
    if not (lower < upper).all():
        raise ValueError('Every joint minimum must be below maximum')
    elapsed = positive(elapsed, 'command interval')
    lo = np.maximum(lower, np.maximum(current - max_step, previous - max_velocity * elapsed))
    hi = np.minimum(upper, np.minimum(current + max_step, previous + max_velocity * elapsed))
    if (lo > hi).any():
        raise RuntimeError('Measured position no longer overlaps the permitted command envelope')
    return np.ascontiguousarray(np.minimum(hi, np.maximum(lo, target)), dtype=np.float32)


def play_chunk(motors, actions, control, stop_file, observation_time, observed_state, log, health_check=None):
    """Synchronous bounded playback; no action queue grows while ONNX runs."""
    import numpy as np
    if stop_file.exists():
        raise RuntimeError('Stop file is present: ' + str(stop_file))
    age = time.monotonic() - observation_time
    if age > positive(control['max_observation_age_s'], 'max_observation_age_s'):
        raise RuntimeError('Prediction is too old to execute ({:.3f}s)'.format(age))
    current = motors.read_state()
    drift = vector(control['max_state_drift'], 'max_state_drift', True)
    if (np.abs(current - observed_state) > drift).any():
        raise RuntimeError('Robot state changed too far during inference; discard this chunk')
    count = int(control['chunk_steps'])
    if count < 1 or count > 50:
        raise ValueError('chunk_steps must be in [1,50]')
    period = 1.0 / positive(control['command_hz'], 'command_hz')
    previous = current.copy()
    def before_enable():
        if stop_file.exists():
            raise RuntimeError('Stop file detected before motor enable')
        if time.monotonic()-observation_time > control['max_observation_age_s']:
            raise RuntimeError('Prediction became too old before motor enable')
    before_enable()
    if health_check is not None:
        health_check()
    motors.enable(before_write=before_enable)
    deadline = time.monotonic()
    last = deadline - period
    for step, target in enumerate(actions[0, :count]):
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        now = time.monotonic()
        if stop_file.exists():
            raise RuntimeError('Stop file detected')
        if now - observation_time > control['max_observation_age_s']:
            raise RuntimeError('Observation age exceeded during playback')
        # Never catch up late commands in a burst after a scheduling stall.
        if now - deadline > positive(control['max_command_lateness_s'], 'max_command_lateness_s'):
            raise RuntimeError('Control loop missed its deadline')
        current = motors.read_state()
        now = time.monotonic()
        def check_time():
            at = time.monotonic()
            if stop_file.exists():
                raise RuntimeError('Stop file detected before command transmission')
            if at - observation_time > control['max_observation_age_s']:
                raise RuntimeError('Prediction became too old before command transmission')
            if at - deadline > control['max_command_lateness_s']:
                raise RuntimeError('Serial/camera work exceeded the control deadline')
            return at
        check_time()
        if health_check is not None:
            health_check()
        now = check_time()
        sent = guarded_target(target, current, previous, control['min'], control['max'],
                              control['max_step'], control['max_velocity'], min(now-last, period*2))
        def before_write(actual):
            at = check_time()
            lo = vector(control['min'], 'minimum')
            hi = vector(control['max'], 'maximum')
            if ((actual < lo-1e-5) | (actual > hi+1e-5)).any():
                raise RuntimeError('Final motor target exceeds control position limits')
            permitted = vector(control['max_velocity'], 'max_velocity', True) * min(at-last, period*2)
            if (np.abs(actual-previous) > permitted + 1e-5).any():
                raise RuntimeError('Encoder quantization exceeds command velocity limit')
        applied = motors.write_action(sent, before_write=before_write)
        now = time.monotonic()
        log.write(json.dumps(dict(event='command', step=step, t=now,
                                  measured=current.tolist(), requested=target.tolist(),
                                  sent=applied.tolist())) + '\n')
        log.flush()
        previous, last = applied, now
        deadline = now + period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'robot_config.json')
    parser.add_argument('--mode', choices=('cpu', 'hybrid'))
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--execute', action='store_true', help='Explicitly enable calibrated motor playback')
    parser.add_argument('--replay-inputs', type=Path, help='Offline NPZ input; no camera or motor access')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('cycles must be positive')
    if args.execute and args.replay_inputs:
        parser.error('replay inputs cannot be sent to motors')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config_base = args.config.resolve().parent
    mode = args.mode or config.get('mode', 'hybrid')
    if args.execute:
        required = ('camera_mapping_confirmed', 'training_mapping_confirmed')
        for flag in required:
            if config.get(flag) is not True:
                parser.error(flag + ' must be true after verifying training/hardware mapping')
        for key in ('min', 'max', 'max_step', 'max_velocity', 'max_state_drift'):
            vector(config['control'].get(key), 'control.'+key, key.startswith('max_'))
    import numpy as np
    from robot_io.preprocessing import StateActionNormalizer, read_language_assets
    from robot_io.policy import OnnxPolicy
    normalizer = StateActionNormalizer(resolve(config_base, config['normalization']))
    replay = None
    if args.replay_inputs:
        from infer_board import load_inputs
        replay, unused = load_inputs(args.replay_inputs)
    else:
        if not resolve(config_base, config.get('calibration', 'follower_calibration.json')).is_file():
            parser.error('Live mode needs the training-compatible follower calibration JSON; probe/capture works without it')
        language = read_language_assets(resolve(config_base, config['language']))
        # Validate all motor parameters before an expensive ONNX session load.
        from robot_io.motors import So101Motors
        motors = So101Motors(config['motors'], resolve(config_base, config['calibration']), execute=args.execute)
    out = args.output_dir or BASE / 'robot_runs' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + mode)
    out.mkdir(parents=True, exist_ok=False)
    stop_file = resolve(config_base, config.get('stop_file', 'STOP_ROBOT'))
    rng = np.random.RandomState(int(config.get('noise_seed', 20261104)))
    policy = cameras = None
    if replay is not None:
        motors = None
    def stop(unused_signal, unused_frame):
        raise KeyboardInterrupt('Robot loop stopped')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        policy = OnnxPolicy(resolve(config_base, config['models'][mode]), mode=mode,
            library=resolve(config_base, config.get('library', 'libsmolvla_ort_sweep.so')),
            runtime_root=resolve(config_base, config.get('runtime_root', '../GEMM')),
            runtime_config=resolve(config_base, config.get('runtime_config', 'runtime_cma80_native_softmax.json')))
        policy.load()  # Load once, before opening devices or enabling any torque.
        if replay is None:
            from robot_io.cameras import DualCameraCapture
            motors.open()
            cameras = DualCameraCapture(**config['cameras'])
            cameras.start()
        with (out / 'events.jsonl').open('w') as log:
            for cycle in range(args.cycles):
                if stop_file.exists():
                    raise RuntimeError('Stop file is present: ' + str(stop_file))
                if replay is not None:
                    data = dict(replay)
                    observed_state = None
                    observed_at = time.monotonic()
                    timing = dict(source='offline_replay')
                else:
                    pair = cameras.snapshot(timeout=3, max_age=0.5, max_skew=0.1)
                    observed_state = motors.read_state()
                    observed_at = min(pair.timestamps.values())
                    data = pair.as_inputs()
                    data.update(language)
                    data['state'] = normalizer.normalize_state(observed_state)
                    data['noise'] = rng.standard_normal((1,50,32)).astype(np.float32)
                    timing = dict(source='live', camera_age_s=pair.ages, camera_skew_s=pair.skew,
                                  observed_state=observed_state.tolist())
                np.savez(str(out / ('inputs_{:03d}.npz'.format(cycle))), **data)
                normalized = policy.predict(data)
                actions = normalizer.denormalize_actions(normalized)
                np.save(str(out / ('normalized_actions_{:03d}.npy'.format(cycle))), normalized)
                np.save(str(out / ('actions_{:03d}.npy'.format(cycle))), actions)
                event = dict(event='prediction', cycle=cycle, execute=args.execute,
                             observation_age_s=time.monotonic()-observed_at,
                             runtime=dict(policy.metrics), first_action=actions[0,0].tolist(), **timing)
                log.write(json.dumps(event) + '\n'); log.flush()
                print('PREDICTION', json.dumps(event), flush=True)
                if args.execute:
                    def check_cameras():
                        cameras.snapshot(timeout=0.05, max_age=0.8, max_skew=0.15)
                    play_chunk(motors, actions, config['control'], stop_file,
                               observed_at, observed_state, log, health_check=check_cameras)
        print('ROBOT_RESULTS', str(out), flush=True)
    finally:
        # Motor adapter releases torque only if configured, and only for torque
        # this program enabled. It never disables a pre-existing hold by default.
        active_error = sys.exc_info()[0] is not None
        cleanup_errors = []
        for name, component in (('motors', motors), ('cameras', cameras), ('policy', policy)):
            if component is not None:
                try:
                    component.close()
                except BaseException as error:
                    cleanup_errors.append(name + ': ' + str(error))
        if cleanup_errors:
            print('CLEANUP', '; '.join(cleanup_errors), file=sys.stderr)
            if not active_error:
                raise RuntimeError('Resource cleanup failed: ' + '; '.join(cleanup_errors))


if __name__ == '__main__':
    main()
