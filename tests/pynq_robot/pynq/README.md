# PYNQ 代码交付（2026-11-04）

新增摄像头、电机及 ONNX 联动入口在 `ONNX/run_robot.py`；运行方式见 [联动说明](ONNX/README_ROBOT.md)。
已配置用户确认的 `/dev/video0` 腕部和 `/dev/video2` 侧边相机。硬件断开后仅做本地模拟验证，未执行电机动作。

按要求停止后续优化和板端测试。本包为板端已用代码的整理快照，仅做路径整理和静态检查。

```text
pynq/
  GEMM/     # CPU 四线程 NEON、PL 单/双/三核、快速 AXI-Lite、DMA、bit/HWH
  Softmax/  # 1024×1024 测试、PL 流水、CPU/PL 并行源码
  ONNX/     # SmolVLA CPU/PL 推理、图转换、ORT、配置及最近的 NEON 实验快照
```

先看各目录 README.md。共享硬件及正式驱动保存在 GEMM 中，请保留三个目录的相对位置。
目标环境为原 PYNQ ARM64 Linux、PYNQ/XLNK、NumPy、GCC/G++、OpenMP。
Notebook 另需 Jupyter、pandas。预编译 .so 供同板使用，源码均附带；DMA .ko 要求内核匹配。

## 使用顺序

1. 将整个 pynq 目录放在板上可写且空间充足的位置。
2. 从本包加载一次配套 overlay。GEMM/setup_fast_gemm_board.py 是原板端加载流程的相对路径版，设置 PL 250 MHz；它在已有 overlay 时拒绝重载。首次使用建议重启后运行：

   ```bash
   cd /path/to/pynq
   sudo python3 GEMM/setup_fast_gemm_board.py
   sudo python3 GEMM/set_cpu_highest.py
   ```

3. 按各目录说明选择测试。每次只运行一个持有硬件的程序，任务结束前保留 CMA 缓冲区。

CPU 请求最高可用 CPUFreq 档位 1199999 kHz（约 1.2 GHz），历史硬件读回约 1049.9895 MHz；
PL 读回约 249.9975 MHz。结果记录实际频率，请不要把请求档位当成实测频率。

## 模型及版本说明

本包包含代码、硬件 bit/HWH、ARM64 ORT、验证输入和历史结果；大型 SmolVLA ONNX 权重未重复打包。
原板端模型在 `/home/xilinx/jupyter_notebooks/smolvla_final/GEMM_V2/eval_wrist_beside_20261103/model`。
本机模型在 `G:/smolvla/smolvla_20261103_wrist_beside_base_200k_3gpu/200000_w8a8_fp32`。
可直接引用原模型路径，或将所需模型及同名 manifest 一起复制到 ONNX/models。

默认 ONNX 路径为大矩阵 GEMM 使用 PL、原生 CPU Softmax、80 MiB 常驻 CMA、四线程 CPU。
PL Softmax 的整图试验和最新 NEON A 打包试验保留为独立入口，详见 ONNX/README.md。
DMA 在独立 GEMM 测试中使用；正式 ONNX 整图尚未接入 PS DMA。

`FILE_MANIFEST.json` 记录每个交付文件的 SHA256、来源和路径调整；
`SHA256SUMS.txt` 可在 pynq 目录执行 `sha256sum -c SHA256SUMS.txt` 核验。
本次交付未重新运行任何板端计算。
