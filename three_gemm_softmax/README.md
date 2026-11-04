# 三核 GEMM + Softmax 部署包

一个 `accelerator_driver.py` 就能控制三个 GEMM 核和一个 Softmax。把整个 `three_gemm_softmax/` 文件夹复制到 PYNQ 板端即可使用。

```text
three_gemm_softmax/
├── accelerator_driver.py   # 唯一的运行时 Python 文件
├── README.md               # 部署与快速调用
├── USAGE.md                # 完整接口、布局、并行和错误恢复说明
├── requirements.txt        # 板端依赖：PYNQ
├── MANIFEST.json           # 文件来源、SHA256、硬件地址
└── hardware/
    ├── design_1.bit        # 当前工程三核 GEMM + Softmax bitstream
    └── design_1.hwh        # 同工程导出的硬件描述，与 bit 保持同名
```

运行环境为 Python 3.6+ 和板端已安装的 PYNQ。驱动只调用 `pynq.MMIO`，其余 Python 导入均属于标准库；此文件夹包含全部项目内运行依赖。PYNQ 及其底层设备访问库由板卡镜像提供，通常无需重新安装 `requirements.txt` 中的软件包。

## 加载硬件并连接 driver

先进入本文件夹，再在板端 Python 或 Notebook 中运行：

```python
from pathlib import Path
from pynq import Overlay
from accelerator_driver import AcceleratorDriver

hardware = Path("hardware").resolve()
overlay = Overlay(str(hardware / "design_1.bit"))
acc = AcceleratorDriver.from_hwh(hardware / "design_1.hwh")
print(acc.addresses)
```

如果已经加载了这套硬件，直接执行 `acc = AcceleratorDriver()` 即可连接，不必重复下载。driver 自身不下载 bitstream、不分配或映射矩阵内存；调用方提供已准备好的 DDR 物理地址，并负责内存生命周期和 CPU cache 同步。

| 模块 | AXI-Lite 基址 |
|---|---|
| GEMM 0 | `0x80000000` |
| GEMM 1 | `0x80001000` |
| GEMM 2 | `0x80002000` |
| Softmax | `0x90000000` |

随附 HWH 中四个模块的 DDR 窗口均为 `0x00000000–0x7FFFFFFF`；`from_hwh()`会按此检查地址范围。这个映射不表示整段 DDR 已为应用保留，实际传入地址必须属于调用方持有的内存。

## 调用

下面的 `A_ADDR` 等变量由调用方提供，都是实际 DDR 物理地址：

```python
# 三个 GEMM 共同完成一个矩阵，按32行块切分M，共享BT。
results = acc.run_gemm_rows(
    a_addr=A_ADDR, bt_addr=BT_ADDR, c_addr=C_ADDR,
    m=2048, n=512, k=2048,
)

# 独立FP32矩阵的Softmax。
result = acc.run_softmax(
    src_addr=SCORE_FP32_ADDR, dst_addr=PROB_FP32_ADDR,
    m=1024, k=1024,
)
acc.close()
```

GEMM使用INT8 A和预转置INT8 BT，输出INT32；Softmax输入/输出为FP32。GEMM输出接Softmax时，先完成缩放及INT32转FP32。三核独立任务、四个模块并行提交、矩阵步长与超时恢复见 [USAGE.md](USAGE.md)。

## 打包记录

硬件复制自当前15eg工程已完成的bitstream生成结果，bit文件头时间为2026-11-02 19:40:53，器件为`xczu15eg-ffvb1156-2-i`。已核对HWH中的三个GEMM和一个Softmax、控制地址及DDR映射，并校验复制文件的SHA256；具体来源见 [MANIFEST.json](MANIFEST.json)。

驱动源码与21项软件单测通过的版本一致。此次仅整理部署文件并检查独立导入，未运行RTL仿真、综合、重新生成bitstream或板端计算。
