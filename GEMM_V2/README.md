# GEMM_V2：单 AXI、蛇形复用与混合输入缓存

本目录包含当前 GEMM_V2 硬件实现及其软件接口。顶层为 `gemm_accelerator_top`，具有 32 位 AXI-Lite 控制接口、128 位 AXI4 读写主接口和 ABI 10 寄存器定义。

## 数据与缓存

- 计算：INT8 A × INT8 B，INT32 C；32×32 脉动阵列。
- DDR 输入格式：A 为 M×K 行优先，B 预先转置为 N×K 的 BT；C 为 M×N 行优先。
- M、N、K 为 32 的倍数，K 上限为 16352；地址和启动规则见寄存器说明。
- A、BT 各有 512 KiB 缓存，每个缓存各自一半 BRAM、一半 URAM；C 默认仍为 256 KiB URAM。

| 每个输入缓存的 256 位字地址 | 字节范围 | 推断资源 |
| --- | --- | --- |
| 0～8191 | 0～256 KiB | 4 个 2048×256 BRAM bank |
| 8192～16383 | 256～512 KiB | 2 个 4096×256 URAM bank |

两种存储统一采用 3 拍读取接口。切换 bank 或 BRAM/URAM 半区时不需要插入额外空拍。复位只清除控制流水线，不逐字清空 RAM。

## 调度与性能指标

宏块尺寸由 K 和 512 KiB 缓存容量决定；每个宏块由多个 32×32 输出块组成。蛇形遍历在相邻宏块之间复用 A 或 BT 条带，减少重复读取。任一 A/BT 条带对完整写入缓存后即可计算；结果水位允许后交由 writer 提前写回。整个任务完成前，所有结果、写数据和 AXI B 响应必须排空。

32×32 阵列共有 1024 个 PE，每个有效推进周期最多执行 1024 次乘加。若乘法与加法各计一次操作，频率为 f 时单核理论峰值为 1024×f×2 次操作/秒。实测吞吐应以完整运算量 2MNK 除以相同计时范围内的任务时延计算；理论峰值不是板端结果。

硬件周期、数据流量和资源的统计口径见 [docs/report.md](../docs/report.md)。验证命令见 [VALIDATION.md](VALIDATION.md)，当前无需提交历史仿真输出。

## DMA 地址生成

下一行首地址提前计算并保存在寄存器中，当前行结束后直接选用；连续地址拆为 4 KiB 页内低位更新及页号预增，burst 限长使用饱和后的 9 位数比较。该实现保留两笔 outstanding 窗口、4 KiB 边界拆分及跨行写入语义。

## 文件与接入

- `rtl/gemm_core.v`：宏块调度、缓存命中和并行回写控制。
- `rtl/gemm_axi_dma128.v`：读写 DMA、burst 限长和地址预计算。
- `rtl/gemm_sram256.v`：混合输入缓存和 C 结果缓存。
- `rtl/gemm_accelerator_top.v`、`rtl/gemm_axil_regs.v`、`gemm_regs.h`：顶层接口与寄存器 ABI。
- `rtl/add_to_bd.tcl`：在 Vivado Block Design 中添加模块引用。
- `scripts/check_interfaces.tcl`：检查 S_AXI 与单路 M_AXI 接口。
- `scripts/synth.tcl`、`scripts/synth_memories.tcl`：独立综合及缓存推断检查。
- `tb/run_regression.py`：Icarus Verilog 功能回归。
- `tb/run_performance_compare.py`：按需运行完整 RTL 性能仿真。

软件仍使用 ABI 10。板端使用前，必须用当前 GEMM_V2 与 Softmax RTL 生成匹配的 bitstream/HWH，并确认 PYNQ 驱动的地址映射一致。

```powershell
python GEMM_V2/tb/run_regression.py --iverilog iverilog --vvp vvp --jobs 2
vivado -mode batch -source GEMM_V2/scripts/check_interfaces.tcl
vivado -mode batch -source GEMM_V2/scripts/synth.tcl
```

