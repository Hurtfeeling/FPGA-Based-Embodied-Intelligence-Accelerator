#!/usr/bin/python3
"""Capture camera inputs or read servo identity/status; never write motor registers."""
import argparse
from datetime import datetime
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'robot_config.json')
    parser.add_argument('--cameras', action='store_true')
    parser.add_argument('--motors', action='store_true', help='Read model and current registers for IDs 1..6')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.cameras and not args.motors:
        parser.error('Select --cameras and/or --motors')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    output = args.output_dir or BASE / 'robot_capture' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    if args.cameras:
        from robot_io.cameras import DualCameraCapture
        import cv2
        import numpy as np
        with DualCameraCapture(**config['cameras']) as cameras:
            pair = cameras.snapshot(timeout=5, max_age=0.8, max_skew=0.15)
            for name, tensor in pair.as_inputs().items():
                rgb = np.rint(tensor[0].transpose(1,2,0) * 255).astype(np.uint8)
                if not cv2.imwrite(str(output / (name + '.jpg')), rgb[:,:,::-1]):
                    raise RuntimeError('Could not save ' + name)
            report['cameras'] = dict(diagnostics=cameras.diagnostics(),
                                     timestamps=pair.timestamps, skew_s=pair.skew)
    if args.motors:
        from robot_io.motors import probe
        motor_config = config['motors']
        report['motors'] = probe(motor_config['port'], motor_config.get('baudrate',1000000),
                                 motor_config.get('ids', [1,2,3,4,5,6]))
    (output / 'inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('SAVED', str(output))


if __name__ == '__main__':
    main()
