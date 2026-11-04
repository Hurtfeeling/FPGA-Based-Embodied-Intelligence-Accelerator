# 基于 FPGA 的具身智能加速器

本目录是面向 ZU15EG/PYNQ 平台的工程快照，包含 GEMM_V2、Softmax RTL 与 Vivado 工程、LeRobot/SmolVLA 模型软件，以及板端 CPU/PL 混合推理交付材料。新增的 LeRobot 项目已整理到单层 `lerobot-clean/`，上游 README、许可证和源码结构均保留。

## 目录结构

| 路径                               | 内容与用途                                                                                                                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GEMM_V2/`                       | 当前唯一保留的 GEMM 硬件实现：RTL、AXI-Lite 控制寄存器、AXI DMA、片上缓存、脉动阵列、Vivado 综合脚本、仿真 testbench 和验证说明。                                                     |
| `Softmax/`                       | FP32 Softmax 硬件实现：四路计算流水、指数查表 ROM、行缓存、AXI 接口、数学单元测试和端到端仿真脚本。                                                                                   |
| `three_gemm_softmax/`            | 三个 GEMM 核和一个 Softmax 的部署包：Python 驱动、调用说明、依赖列表，以及对应的 `design_1.bit` 和 `design_1.hwh`。                                                               |
| `fpga/`                          | Vivado Block Design 的 `.bd`、Tcl 重建脚本和工程创建辅助脚本。目标器件为 `xczu15eg-ffvb1156-2-i`。                                                                                |
| `lerobot-clean/`                 | LeRobot 本地代码快照：`src/lerobot/` 策略与训练框架、SmolVLA 蒸馏、`scripts/` 量化与 ONNX 导出/数值参考工具。项目入口见 [`README_PROJECT.md`](lerobot-clean/README_PROJECT.md)。 |
| `tests/`                         | 主机侧驱动协议测试；`tests/pynq_robot/pynq/` 保存相机、电机与 SmolVLA 联动快照。入口见 [`tests/README.md`](tests/README.md)。                                                     |
| `FPGA创新创业竞赛.pdf`           | 当前项目竞赛文档。`docs/` 保留补充技术说明、架构/数据通路图和可编辑的 Draw.io 图源。                                                                                                 |
| `docs/`                          | GEMM/Softmax 的补充技术推导、架构/数据通路图和可编辑的 Draw.io 图源。                                                                                                                  |
| `results/implementation/system/` | 系统级 Vivado 放置、布线和资源报告；不包含历史仿真输出。                                                                                                                              |
| `LICENSE_NOTICE.md`              | 当前许可状态和第三方资源的再分发提示。                                                                                                                                                |
| `SOURCE_ORIGINS.json`            | 文件来源记录；保留原整理包来源，并登记新增软件和板端快照。                                                                                                                            |
| `PACKAGE_MANIFEST.json`          | 本次整理后的文件大小与 SHA256 清单，不包含清单自身。                                                                                                                                  |

## 从哪里开始

| 工作内容                                         | 入口                                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| 阅读 FPGA 创新创业竞赛文档                     | [`FPGA创新创业竞赛.pdf`](FPGA创新创业竞赛.pdf)                                                                          |
| 修改算子 RTL、仿真或重建 Vivado 工程             | [`GEMM_V2/README.md`](GEMM_V2/README.md)、[`Softmax/README.md`](Softmax/README.md)、[`fpga/README.md`](fpga/README.md) |
| 训练、蒸馏、量化或导出 SmolVLA                   | [`lerobot-clean/README_PROJECT.md`](lerobot-clean/README_PROJECT.md)              |
| 在 PYNQ 上查看 GEMM/Softmax/ONNX 板端快照 | [`tests/pynq_robot/pynq/README.md`](tests/pynq_robot/pynq/README.md)、[`tests/pynq_robot/pynq/ONNX/README.md`](tests/pynq_robot/pynq/ONNX/README.md) |
| 检查或运行 PYNQ 相机、电机与 SmolVLA 联动 | [`tests/pynq_robot/pynq/README.md`](tests/pynq_robot/pynq/README.md)、[`README_ROBOT.md`](tests/pynq_robot/pynq/ONNX/README_ROBOT.md) |
| 在主机检查寄存器协议与错误处理                   | [`tests/README.md`](tests/README.md)                                                                                     |

LeRobot 项目根目录现在是 `lerobot-clean/`，不再需要进入第二层同名目录。本地 `pyproject.toml` 声明版本 `0.6.2`、Python `>=3.12`；具体上游 commit 未记录，不能仅凭版本号推定快照等同于某个发布版本。从本仓库根目录准备 SmolVLA 软件环境：

```bash
cd lerobot-clean
uv sync --locked --extra smolvla --extra dataset
uv run python scripts/quantize_smolvla_w8a8.py --help
```

ONNX/FP8 脚本的额外依赖、各导出格式和路径配置见项目说明。模型、数据集和本地输出放在各项目忽略的目录或仓库外；快照中的原训练机路径需要按实际环境替换。LeRobot 的 Python 环境与原 PYNQ ARM64 环境分别准备。

## 两个核心计算单元

### GEMM_V2

GEMM 计算 `C = A × B`。输入 `A` 为 `M×K` 的 INT8 行优先矩阵；权重以 `BT = Bᵀ`（`N×K`）存放，使每个输出列对应的 K 个权重连续，便于突发读取。输出元素为：

```text
C[i,j] = Σ(k=0..K-1) A[i,k] × BT[j,k]
```

每个 INT8 乘积先扩展为有符号 32 位，再累加为 INT32。32×32 脉动阵列含 1024 个 PE，每个有效推进周期最多执行 1024 次乘加；若乘法与加法各算一次操作，250 MHz 下单核理论峰值为 `1024 × 250 MHz × 2 = 0.512 TOPS`。该峰值不包含阵列填充/排空、搬运等待或软件开销。

大矩阵由 32×32 输出微块组成。A 与 BT 的输入条带暂存在混合 BRAM/URAM 缓存中，调度器根据缓存容量选择宏块跨度，并按蛇形顺序遍历以复用驻留条带；满足条件的结果会尽早写回。硬件周期、输入读取量和理论吞吐的推导详见 [`docs/report.md`](docs/report.md) 与 [`GEMM_V2/VALIDATION.md`](GEMM_V2/VALIDATION.md)。

### Softmax

Softmax 对每一行独立计算。对有效输入 `x[j]`，先求 `m = max(x[j])`，再计算 `e[j] = exp(x[j] - m)`，最后得到 `p[j] = e[j] / Σe[j]`。减去行最大值用于稳定指数计算；无效尾 lane 不参与求和，`-inf` 可作为 mask。

RTL 每拍处理四个 FP32 lane，指数单元使用查表插值，行和经定点对齐和进位保存后转换为 FP32，再求倒数并并行归一化。两块行缓存承接输入与指数结果，数据经 AXI 从 DDR 读取并写回；支持的单行长度上限为 4096。端到端公式、定点格式、流水周期及误差定义见 [`Softmax/RTLImplementation.md`](Softmax/RTLImplementation.md) 和 [`docs/report.md`](docs/report.md)。

## SmolVLA 与系统集成范围

`lerobot-clean/` 提供 SmolVLA 策略、训练/蒸馏实现，以及 FP32、W8A8、strict INT8 和 FP8 等导出或数值参考脚本。自定义 INT8/FP8 域有各自的参考运行时，不能直接作为默认 ONNX Runtime 或 FPGA 的可执行图。W8A8 量化导出统一使用 `lerobot-clean/scripts/quantize_smolvla_w8a8.py`，调用方式见该项目说明。

当前板端代码快照位于 `tests/pynq_robot/pynq/`，包含 GEMM、Softmax、ONNX 推理以及相机、电机联动入口。`ONNX/run_robot.py` 复用 ONNX Runtime 与 GEMM 驱动，并增加双相机采集、关节反馈、预处理和可选动作下发。运行真实状态推理前必须核对设备及训练对应的关节校准、顺序和单位；不传 `--execute` 时不会写电机目标。机器人联动仅做过本地模拟验证，未执行电机动作，完整板端闭环仍需标定和联调。详细命令以快照 README 为准。

`three_gemm_softmax/` 部署包对应三个 GEMM 核和一个 Softmax，bitstream 日期为 2026-11-02。控制基地址分别为 `0x80000000`、`0x80001000`、`0x80002000` 和 `0x90000000`。四个数据主接口经互连共享 PS 的 HP0 DDR 通路，单接口的理论线速不能直接视作系统持续带宽。驱动用法和寄存器配置示例见 [`three_gemm_softmax/USAGE.md`](three_gemm_softmax/USAGE.md)。`tests/pynq_robot/pynq/GEMM/hardware/` 保留板端交付自身的配套硬件，请按随附清单核对版本。

## 验证状态

- 2026-11-04 本次整理重新运行主机侧驱动测试，21 项通过；根目录默认 `pytest` 同样仅收集并通过这 21 项。运行方式见 [`tests/README.md`](tests/README.md)。
- 原整理包记录 Softmax 默认端到端回归 21 个用例通过，覆盖异常输入、AXI 错误恢复和随机背压；本次目录整理未重新运行 RTL 仿真。
- Softmax 的 DMA 依赖现已统一到 `GEMM_V2/rtl`，Icarus 仿真回归通过；现有 bitstream/HWH 尚未用该统一后的源码重新构建，需要重新构建并上板验证后才能作为当前 RTL 的部署结果。
- `tests/pynq_robot/pynq/` 保存相机、电机与 SmolVLA 联动板端快照，随附 `FILE_MANIFEST.json` 与 `SHA256SUMS.txt`。机器人入口仅做本地模拟测试，未执行电机动作；本次仓库整理未重新进行板端性能测量或完整机器人闭环验证。
- LeRobot 测试需独立依赖环境；随包 `tests/artifacts/` 中 50 个文件是 Git LFS 指针，没有对应的二进制载荷。本目录无 `.git`，不能直接依靠此快照执行 `git lfs pull` 恢复上游测试数据。
- 整机布线报告属于实现工具结果，不等同于板端功耗、持续带宽、完整 SmolVLA 推理延迟或机器人任务成功率。

复现条件和接口说明见各算子 README、[`three_gemm_softmax/USAGE.md`](three_gemm_softmax/USAGE.md) 以及 [`fpga/README.md`](fpga/README.md)。综合/实现流程需要对应版本的 Vivado；板端运行需要匹配的 ZU15EG/PYNQ 环境。
