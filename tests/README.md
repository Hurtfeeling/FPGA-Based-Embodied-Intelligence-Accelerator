# 测试与验证入口

本仓库包含主机驱动测试、LeRobot 软件测试，以及 PYNQ 机器人联动板端交付快照。它们使用不同环境，请按下面的入口分别运行。

## 主机驱动协议测试

[`test_accelerator_driver.py`](test_accelerator_driver.py) 使用模拟寄存器检查任务提交、完成状态、地址范围及错误恢复，不需要 PYNQ 板卡或 RTL 仿真环境。它加载本目录的 [`accelerator_driver.py`](accelerator_driver.py)。

在仓库根目录执行：

```bash
python -m unittest discover -s tests -p test_accelerator_driver.py -v
```

2026-11-04 本次整理重新运行该命令，21 项通过；根目录默认 `pytest` 也验证为仅收集并通过这 21 项。根目录的 [`pytest.ini`](../pytest.ini) 将默认发现范围限定为该文件，避免递归收集其他项目的测试。

## LeRobot 软件测试

LeRobot 的测试保存在 [`lerobot-clean/tests/`](../lerobot-clean/tests/)，依赖和运行约定见其 [`AGENTS.md`](../lerobot-clean/AGENTS.md)、[`pyproject.toml`](../lerobot-clean/pyproject.toml) 与 [`CLEAN_EXPORT.md`](../lerobot-clean/CLEAN_EXPORT.md)。请在对应的 Python 3.12+ 环境中准备所选测试所需的依赖后，从仓库根目录进入软件项目：

```bash
cd lerobot-clean
uv run pytest tests
```

部分测试需要可选依赖、GPU、仿真环境或 Git LFS 二进制测试数据。`tests/artifacts/` 包含 50 个 LFS 指针和 3 个生成工件的 Python 源码，没有指针对应的二进制内容；需要从原仓库取得匹配数据。LeRobot 完整测试套件未在本次整理中验证。

## PYNQ 机器人联动快照

[`pynq_robot/pynq/README.md`](pynq_robot/pynq/README.md) 是包含相机、电机和 SmolVLA ONNX 联动代码的独立板端快照。主要入口如下：

- [`GEMM/`](pynq_robot/pynq/GEMM/README.md)：共享 GEMM 驱动、PL 硬件和板端基准。
- [`Softmax/`](pynq_robot/pynq/Softmax/README.md)：Softmax 板端测试与 CPU/PL 流水实验。
- [`ONNX/`](pynq_robot/pynq/ONNX/README.md)：SmolVLA ONNX 推理、运行时和模型资源准备。
- [`ONNX/README_ROBOT.md`](pynq_robot/pynq/ONNX/README_ROBOT.md)：双相机、舵机状态和动作执行说明；运行入口为 `run_robot.py`。

机器人快照同样面向匹配的 PYNQ ARM64 Linux 环境。运行真实状态推理前需核对相机/串口设备和训练对应的关节校准、顺序及单位；未显式传入 `--execute` 时只读状态并保存预测。随包记录的机器人联动测试是本地模拟测试，未执行电机动作，也不能替代板端标定和联调。来源及哈希记录见 [`FILE_MANIFEST.json`](pynq_robot/pynq/FILE_MANIFEST.json) 和 [`SHA256SUMS.txt`](pynq_robot/pynq/SHA256SUMS.txt)。
