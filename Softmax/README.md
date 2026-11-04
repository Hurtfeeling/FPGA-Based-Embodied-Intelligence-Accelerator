# Softmax FP32 accelerator

本目录实现 AXI-Lite32 控制、AXI-full128 数据的四路 FP32 Softmax。RTL 使用 Verilog `.v`，公共函数放在 `.vh`，可用 Icarus `-g2005` 编译。

## 功能与存储

- 一次 START 自动执行 M×K 矩阵的逐行 Softmax；1≤M≤65536，1≤K≤4096。
- 固定 input/output 两个 1024×128-bit 同步 1R1W 行 BRAM，总计 32 KiB；每块由 4 个 1024×32-bit bank 组成，不缓存完整矩阵。
- 加载时求最大值；四路减法和指数写入 output，同时进行定点对齐求和；归一化结果原位写回 output；最后经 AXI 写 DDR。
- EXP 阶段逐字回收 input 并加载下一行。默认允许 STORE 上一行与 EXP 下一行重叠。
- RX 队列为 32 拍；STORE 返回/TX 队列为 32 拍；EXP/NORM 读返回队列各为 4 拍。
- 指数单元采用 1024 段整数 ROM 线性插值；FP32 运算采用 RNE，输入 subnormal 使用 DAZ，输出 subnormal 使用 FTZ。
- 行和采用 UQ16.48 和进位保存累加；每行一次求倒数，再进行四路归一化。

计划、数据流与逐模块计算说明见 [Plan.md](Plan.md)、[ComputeFlow.md](ComputeFlow.md) 和 [RTLImplementation.md](RTLImplementation.md)。

## 流水结构

最大值计算由 key 捕获、四路独立 lane 最大值和两级合并组成；末尾需等待最大值流水及 DMA 排空。指数流水包含 10 级 FP32 减法、8 级 log2(e) 乘法和 9 级 LUT 后端，共 27 级；归一化采用 8 级 FP32 乘法。两条计算流水均为 II=1，每拍处理 4 个 FP32。

行和采用 4 级定点转换、前向和反馈 CSA；完整行和在行尾转换为 FP32。倒数采用迭代实现，2 的整数次幂走旁路。具体寄存边界、握手和行尾调度见 [RTLImplementation.md](RTLImplementation.md)。250 MHz 是设计时钟目标；是否满足时序以对应版本的 Vivado 实现报告为准。

## 软件接口

默认 AXI-Lite 基址为 0x80010000，可由 C_S_AXI_BASEADDR 修改。顶层为 `softmax_accelerator_top`，共用 aclk/aresetn；32 位 AXI 地址，128 位 AXI 数据，AXI ID 宽度可配置。

| 偏移 | 寄存器 |
|---:|---|
| 0x00 | ENABLE bit0；BUSY bit1、DONE bit2、ERROR bit3、ERROR_CODE bits7:4 只读；写 START bit8、CLEAR bit9 |
| 0x04 | SRC_ADDR |
| 0x08 | DST_ADDR |
| 0x0c | 低16位 M-1，高16位 K-1 |

SRC/DST 地址按 16-byte 对齐且不得重叠；每行跨度为 16×ceil(K/4) 字节。尾 lane 由 mask 处理，输出尾部补 +0。错误码、地址范围、复位和缓存一致性要求见本文件其余章节的接口表和 [RTLImplementation.md](RTLImplementation.md)。

## 构建与验证

所有命令从仓库根目录运行。ROM 文件为 `Softmax/rtl/exp_lut.mem`，必须作为 Memory Initialization Files 加入 Vivado 工程。独立综合脚本会加入 ROM 并传入对应路径。

```powershell
python Softmax/tb/test_math.py
python Softmax/scripts/run_tests.py
python Softmax/scripts/run_tests.py --large --case matrix_1024x1024
```

默认回归覆盖小型和中型矩阵、mask、零、subnormal/下溢、NaN/+inf、AXI 错误恢复、非法配置和随机背压。完整 1024×1024 矩阵使用 `--large` 运行。结果保存在本机 `Softmax/tb/build/`，不随源码提交。

综合与布局布线命令：

```powershell
vivado -mode batch -source Softmax/scripts/synth.tcl -tclargs Softmax/tb/build/synth 4.0
vivado -mode batch -source Softmax/scripts/implement.tcl -tclargs Softmax/tb/build/synth/softmax_accelerator_top_synth.dcp Softmax/tb/build/impl 4.0 0
```

这些命令生成本地综合与实现报告，不自动更新随包的 bitstream/HWH。若修改 RTL 或 DMA，须重新运行整机 Vivado 构建后才能将对应硬件部署到 PYNQ。