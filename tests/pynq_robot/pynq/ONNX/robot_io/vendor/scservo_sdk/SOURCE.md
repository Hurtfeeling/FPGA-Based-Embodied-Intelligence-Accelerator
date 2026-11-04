# Vendored Feetech SDK

- Official source: https://github.com/ftservo/FTServo_Python
- Commit: `9e6e342f6011dc727ba7144fc835bca5357b3ca9`
- Source directory: https://github.com/ftservo/FTServo_Python/tree/9e6e342f6011dc727ba7144fc835bca5357b3ca9/scservo_sdk
- License: upstream MIT, reproduced in `LICENSE`.
- `UPSTREAM_README.md` is the unmodified upstream README at that commit.

All ten Python source files are copied without changes. Their relative imports
allow use as `robot_io.vendor.scservo_sdk`, without importing a possibly different
system `scservo_sdk`. The external dependency is `pyserial` (tested project
version: 3.5); the application also uses NumPy. Upstream documents testing with
Python 3.5.3, and all included sources pass Python 3.6 grammar checks.

The `sms_sts` backend uses `PortHandler(port)`, `sms_sts(port_handler)`,
`GroupSyncRead(handler, address, length)`, and
`GroupSyncWrite(handler, address, length)`. This differs from the unofficial
`feetech-servo-sdk` PyPI package used by older LeRobot releases; do not mix their
constructors or group-read return conventions.

Calibration and normalization semantics follow the official LeRobot sources:

- https://github.com/huggingface/lerobot/blob/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3/src/lerobot/motors/motors_bus.py
- https://github.com/huggingface/lerobot/blob/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3/src/lerobot/robots/so_follower/so_follower.py

No LeRobot module is imported. In particular, the application's norm modes are
explicit configuration, because the model's training units must be verified and
cannot be inferred from a newer LeRobot default.
