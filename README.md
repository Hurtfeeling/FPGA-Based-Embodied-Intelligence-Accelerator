# FPGA-Based Embodied Intelligence Accelerator

[English](README.md) · [简体中文](README_CN.md)

A ZU15EG/PYNQ project for accelerating matrix multiplication and Softmax in robot-side vision-language-action inference. This repository brings together custom RTL, Vivado integration files, a three-GEMM-plus-Softmax deployment package, a local LeRobot/SmolVLA source snapshot, and PYNQ inference and robot-integration code.

The hardware uses a **PS-PL heterogeneous architecture**: the processing system (PS) prepares data and schedules work, while programmable logic (PL) executes configurable compute kernels. SmolVLA and the SO-101 arm provide the application context.

> **Snapshot scope:** the repository contains source code, packaged hardware, implementation reports, and historical board-side results. The packaged bitstream has not been rebuilt after the shared DMA source consolidation. The included robot entry point has local mock-test coverage; a complete physical robot control loop has not been validated as part of this repository snapshot.

## Architecture

```text
Dual RGB cameras, language instruction, robot state
                         |
                         v
           PS: PYNQ / ONNX Runtime / Python
           preprocessing, scheduling, action output
                  |                 |
         AXI-Lite control       CPU DDR access
                  |                 |
                  v                 v
       PL: 3 x GEMM + 1 x Softmax <--> DDR
                  128-bit AXI through SmartConnect
                  and the shared PS HP0 port

           PS USB / serial interface --> SO-101
```

Each accelerator has its own control registers and a 128-bit AXI data interface. All four data interfaces share the same HP0/DDR path, so adding compute cores does not multiply the available external-memory bandwidth.

### GEMM: INT8 systolic-array computation

- Three independently controlled GEMM cores, each with a **32 x 32 systolic array** containing 1,024 processing elements.
- INT8 inputs and INT32 accumulation. A is stored as an `M x K` row-major matrix; B is stored pre-transposed as `BT`, an `N x K` row-major matrix.
- Separate **512 KiB A and BT caches per core**, each split equally between BRAM and URAM, plus a 256 KiB URAM result cache.
- Macro-tile traversal follows a serpentine schedule to reuse resident input strips. Strip-ready computation and early result writeback overlap loading, computation, and output transfers.
- Multi-core row partitioning lets the cores share read-only BT while writing separate output-row ranges.

```text
C[i, j] = sum(A[i, k] * BT[j, k], k = 0 ... K - 1)
```

M, N, and K must be positive multiples of 32; the current K limit is 16,352. At 250 MHz, the arithmetic peak is **0.512 TOPS per core**, or **1.536 TOPS across three cores**, counting one multiplication and one addition as two operations. These are theoretical peaks, excluding fill/drain cycles, memory stalls, and software overhead.

See [GEMM documentation](GEMM_V2/README.md), [validation instructions](GEMM_V2/VALIDATION.md), and the [technical report](docs/report.md).

### Softmax: four-lane FP32 pipeline

- Row-wise stable Softmax with up to **4,096 elements per row**.
- Four FP32 lanes, exponential lookup-table interpolation, fixed-point aligned summation, and reciprocal-based normalization.
- Two row buffers support data reuse and overlap between adjacent rows without buffering the entire matrix.
- AXI-Lite control, 128-bit AXI transfers, tail-lane masking, and invalid-input/error handling.

```text
m    = max(x)
e[j] = exp(x[j] - m)
p[j] = e[j] / sum(e)
```

The hardware IP and CPU/PL Softmax experiments are included. Availability of the IP does not mean every model configuration routes Softmax to PL; use the matching runtime configuration and numerical checks.

See the [Softmax overview](Softmax/README.md) and [RTL implementation details](Softmax/RTLImplementation.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [`GEMM_V2/`](GEMM_V2/) | GEMM RTL, register ABI, DMA, caches, systolic array, Vivado scripts, and testbenches. |
| [`Softmax/`](Softmax/) | FP32 Softmax RTL, lookup-table data, math tests, and end-to-end simulation scripts. |
| [`three_gemm_softmax/`](three_gemm_softmax/) | Standalone Python driver, deployment instructions, dependencies, and matching bitstream/HWH files. |
| [`fpga/`](fpga/) | Vivado Block Design, Tcl reconstruction scripts, and project creation helpers for `xczu15eg-ffvb1156-2-i`. |
| [`lerobot-clean/`](lerobot-clean/) | LeRobot/SmolVLA source, training and distillation code, quantization, ONNX export, and numerical reference tools. |
| [`tests/`](tests/) | Host driver tests and the PYNQ delivery snapshot under `tests/pynq_robot/pynq/`. |
| [`docs/`](docs/) | Technical explanations, architecture and datapath figures, and editable Draw.io sources. |
| [`results/implementation/system/`](results/implementation/system/) | System placement, routing, timing, and utilization reports. |
| [`SOURCE_ORIGINS.json`](SOURCE_ORIGINS.json) | Source and packaging provenance. |
| [`PACKAGE_MANIFEST.json`](PACKAGE_MANIFEST.json) | File sizes and SHA-256 hashes recorded during snapshot packaging; not a live index of later Git commits. |
| [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md) | Project licensing status and third-party notices. |

## Getting started

### 1. Clone the source snapshot

The LeRobot test fixtures include **50 LFS pointer files without their binary payloads**. To obtain the source without attempting to download those missing objects:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Hurtfeeling/FPGA-Based-Embodied-Intelligence-Accelerator.git
cd FPGA-Based-Embodied-Intelligence-Accelerator
```

Tests that depend on those fixtures require the matching payloads from the original source repository. Model checkpoints, tokenizer/VLM assets, and robot datasets also need to be prepared separately.

### 2. Run the host driver tests

These tests use mock registers and do not require a PYNQ board or an RTL simulator:

```bash
python -m unittest discover -s tests -p test_accelerator_driver.py -v
```

The packaging record reports **21 passing host tests** on November 4, 2026. Root-level `pytest` is configured to discover this host suite only. LeRobot and board-side tests use separate environments. See [test documentation](tests/README.md).

### 3. Prepare the SmolVLA software environment

The local LeRobot snapshot declares Python **3.12+** and package version `0.6.2`. Its exact upstream commit is not recorded; the version field alone does not identify an upstream release.

```bash
cd lerobot-clean
uv sync --locked --extra smolvla --extra dataset
uv run python -m scripts.quantize_smolvla_w8a8 --help
```

Use the [project-specific software guide](lerobot-clean/README_PROJECT.md) for checkpoint paths, calibration data, distillation, and additional ONNX/FP8 dependencies. The upstream [LeRobot README](lerobot-clean/README.md) and [license](lerobot-clean/LICENSE) are preserved.

Custom INT8/FP8 export domains require their corresponding reference runtimes; they are not automatically executable by standard ONNX Runtime or the FPGA. The host model-preparation environment and the PYNQ ARM64 environment must be set up separately.

### 4. Load the packaged hardware on PYNQ

Copy `three_gemm_softmax/` to a compatible board and run the following from that directory:

```python
from pathlib import Path
from pynq import Overlay
from accelerator_driver import AcceleratorDriver

hardware = Path("hardware").resolve()
overlay = Overlay(str(hardware / "design_1.bit"))
acc = AcceleratorDriver.from_hwh(hardware / "design_1.hwh")
print(acc.addresses)
```

The driver requires PYNQ and Python 3.6+. It configures accelerators through MMIO; callers provide valid DDR physical addresses and manage buffer allocation, lifetime, and CPU-cache synchronization.

| Module | Packaged AXI-Lite base address |
| --- | --- |
| GEMM 0 | `0x80000000` |
| GEMM 1 | `0x80001000` |
| GEMM 2 | `0x80002000` |
| Softmax | `0x90000000` |

These addresses belong to the packaged integrated design; a standalone module's default address may differ. The packaged hardware date is November 2, 2026. Keep each `.bit` file paired with its matching `.hwh` and verify the accompanying manifest before deployment.

See the [deployment overview](three_gemm_softmax/README.md) and [driver API and usage guide](three_gemm_softmax/USAGE.md).

## Building and simulation

From the repository root, with Icarus Verilog and Vivado available as appropriate:

```bash
# GEMM functional regression
python GEMM_V2/tb/run_regression.py --iverilog iverilog --vvp vvp --jobs 2

# Softmax math and end-to-end regression
python Softmax/tb/test_math.py
python Softmax/scripts/run_tests.py

# GEMM interface and synthesis checks
vivado -mode batch -source GEMM_V2/scripts/check_interfaces.tcl
vivado -mode batch -source GEMM_V2/scripts/synth.tcl
```

For full-system reconstruction, follow the [FPGA integration guide](fpga/README.md) and `fpga/create_project.tcl`. The Block Design Tcl was generated with Vivado 2020.1 and may need upgrading for other versions. Add the Softmax ROM initialization file and use the shared DMA sources from `GEMM_V2/rtl`.

Changing RTL or DMA requires a new full-system build and matching hardware deployment. The checked-in bitstream is not automatically updated by simulation or standalone synthesis commands.

## PYNQ inference and robot integration

The board-side snapshot is under [`tests/pynq_robot/pynq/`](tests/pynq_robot/pynq/README.md):

| Task | Entry point |
| --- | --- |
| GEMM driver and board benchmarks | [GEMM guide](tests/pynq_robot/pynq/GEMM/README.md) |
| Softmax benchmarks and CPU/PL experiments | [Softmax guide](tests/pynq_robot/pynq/Softmax/README.md) |
| SmolVLA ONNX inference and runtime setup | [ONNX guide](tests/pynq_robot/pynq/ONNX/README.md) |
| Dual cameras, joint feedback, and action execution | [Robot integration guide](tests/pynq_robot/pynq/ONNX/README_ROBOT.md) |

`run_robot.py` reuses the ONNX Runtime bridge and GEMM driver, adding two-camera acquisition, joint-state feedback, preprocessing, and optional action execution. Without `--execute`, it reads state and records predictions without sending motor targets. Real execution requires the correct devices, training-time joint calibration, joint order, units, and configured motion limits.

## Validation status and limitations

- **Host protocol tests:** the packaging record reports 21 passing tests for task submission, status handling, address validation, and recovery. This does not validate physical hardware execution.
- **RTL tests:** Softmax regression passed after consolidating shared DMA dependencies under `GEMM_V2/rtl`. The packaged bitstream/HWH has not been rebuilt from that consolidated source tree.
- **Implementation reports:** system reports include a 250 MHz constraint and WNS of +0.016 ns for the recorded implementation. Tool reports do not establish board power, sustained memory bandwidth, complete model latency, or robot task success.
- **Board-side materials:** historical results, runtime configurations, code, and file manifests are included. Repository packaging did not rerun board benchmarks or validate a complete physical robot loop. The robot entry point was tested locally with mocks without commanding real motors.
- **LeRobot fixtures:** 50 test-artifact files are LFS pointers only. Their binary data is absent, and the full LeRobot test suite has not been validated as part of this snapshot.
- **Documentation versions:** some technical documents describe earlier project stages or retain original machine paths. Use the component guides and matching manifests to identify the intended source, runtime, and hardware combination.

## Licensing

No project-wide open-source license has been selected for the FPGA submission package. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for its current status.

The included LeRobot source retains its Apache-2.0 license and copyright headers. ONNX Runtime, vendor code, model assets, and datasets retain their respective terms. LeRobot's license does not apply automatically to the entire FPGA project.
