# GEMM 测试及共享驱动

测试 INT8 A×INT8 B→INT32 C；物理任务的 M/N/K 均为 32 的倍数。
2048×2048 测试的三核分行是 704/672/672。INT8 吞吐按 `2*M*N*K` 计 OP。

## 入口（当前目录为 GEMM）

```bash
# 原始完整计时及硬件计数：CPU 四线程 NEON、PL 单/双/三核
sudo python3 benchmark_gemm_2048.py --repeats 6

# 改进快速驱动：一个核配置完立即启动，再配置下一个核
sudo python3 fast_gemm/benchmark_fast_gemm.py --repeats 8

# AXI-Lite 配置/提交/收尾耗时
sudo python3 fast_gemm/benchmark_control_path.py
```

`gemm_cpu_pl123_2048.ipynb` 展示 CPU、单双三核的完整耗时、硬件 GOPS 和分项数据。
Notebook 从 GEMM 目录打开，已清除历史输出；历史实测另保存在 benchmark_results/。
`benchmark_gemm_2048.py` 使用通用驱动并重新加载 bit；快速测试使用 fast_gemm 改进驱动且不重载。
运行快速测试前，应已从本包 hardware/design_1.bit 路径加载 overlay（历史入口会比较路径）。

## 代码对应关系

- `fast_gemm/fast_gemm.py`、`fast_axilite.c`：快速 MMIO。每核参数和启动 6 次写；正常收尾不读性能计数，可单独读计数。
- `accelerator_driver.py`：通用 GEMM/Softmax 控制、HWH 解析与检查。
- `runtime/neon_runtime.py`、`gemm_benchmark_neon.cpp`：四线程 CPU GEMM、B 转置/打包，含 ARM NEON 源码和预编译库。
- `runtime/operators.py`：正式 ONNX 张量接口、32 补齐、分块、批量任务、常驻权重和共享 CMA 暂存。
- `hardware/`：从原板端取回的配套 bitstream 和 HWH。
- `fast_gemm/dispatch_aligned32*.json`、`DISPATCH*.md`：32 对齐的方阵、长矩阵、宽矩阵分流证据。

完整计时包含普通内存 A→CMA、B 转置、缓存同步、软件提交/等待/收尾、C→独立普通输出；
分配、初始化、正确性校验另列。硬件 GOPS 根据硬件周期换算，不等于端到端 GOPS。

## DMA（独立测试，可选）

批量 DMA 走 Linux DMAEngine 的 PS DMA，将普通内存 A 搬入 CMA、C 搬回普通内存。
注册固定地址可复用；B 仍使用四线程 NEON 转置。关闭 DMA fd 后才能释放 CMA。

```bash
sudo insmod dma_batch/ps_dma_batch.ko
sudo python3 fast_gemm/benchmark_dma_batched_ac.py --repeats 8
# 上述程序完成并退出后
sudo rmmod ps_dma_batch
```

预编译模块来自原板 Linux 5.4.0-xilinx-v2020.1，依赖 XLNK、ZynqMP DMA 驱动、
`dma1chan0`/`dma2chan0`。`dma_batch/` 内含 C 源码、Makefile、Python DmaCopy 和 .ko。
重建需要匹配内核源码/头文件；保留的构建器先运行 `python3 dma_trial/build_trial.py`，
再运行 `python3 dma_batch/build_batch.py`。该准备步骤会复制 `/usr/src/kernel` 到 dma_trial/kernel_build，
代码包未附带内核构建树，请先确保板端磁盘空间充足。

历史 `fast_gemm/benchmark_dma_input.py` 使用另一模块 `/dev/ps_dma_trial`；
对应旧源码及 .ko 在 dma_trial，不能用 ps_dma_batch 替代它。默认采用上面的批量 A/C 入口。
历史 README 里的旧板端绝对路径仅作为实验记录，当前运行路径以本文件为准。

NEON/MMIO 库缺失或源码较新时会在 ARM64 板端自动编译，无需原 GEMM_V1 目录。
首次编译和驱动初始化不属于稳态计时。本次仅打包，未重新做性能测量。
