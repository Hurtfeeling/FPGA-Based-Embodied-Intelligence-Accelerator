# 三个 GEMM + 一个 Softmax：物理地址驱动

新驱动为 [accelerator_driver.py](accelerator_driver.py)，Python 3.6+，运行时仅需 PYNQ 的 MMIO。它连接已经加载好的 FPGA，接收 DDR 物理地址和矩阵尺寸；不申请 CMA、不映射矩阵内存、不复制/转置数据，也不加载 bitstream 或修改时钟。本部署包中唯一的运行时 Python 文件就是该驱动，环境与硬件加载步骤见 [README.md](README.md)。

当前 BD 和导出 HWH 的控制窗口如下，大小均为 `0x200`：

| 模块 | 控制基址 |
|---|---|
| GEMM 0 | `0x80000000` |
| GEMM 1 | `0x80001000` |
| GEMM 2 | `0x80002000` |
| Softmax | `0x90000000` |

本包 `hardware/` 提供当前三核 GEMM 加 Softmax 的 bit/HWH。板端先加载它，再连接驱动；启动时检查三个 GEMM 的 ABI 签名 `0x0A080020`。Softmax 没有版本签名寄存器，须使用配套 HWH。核序号按控制基址升序排列。

```python
from accelerator_driver import AcceleratorDriver, GemmJob

# 已加载当前工程，按上表地址连接；不执行内存分配。
acc = AcceleratorDriver()

# 或：解析实际加载硬件的 HWH，确认三个 GEMM + 一个 Softmax，
# 检查控制地址和 RTL 参数一致，并按各模块的 DDR 映射验证数据范围。
# acc = AcceleratorDriver.from_hwh("/home/xilinx/three_gemm_softmax/hardware/design_1.hwh")
print(acc.addresses)
```

只有地址和32位范围检查无法确认 Linux 内存所有权。传入的 DDR 区域必须已经由调用方保留、可被 PL 访问，并在全部任务结束前保持有效；不要把普通 NumPy 的虚拟地址传给驱动。直接构造时不限制旧版三个16 MiB固定窗口，`from_hwh()` 额外检查实际 DDR 映射。

## 三个核共同计算一个矩阵

输入为连续 INT8 `A[M,K]` 和预先转置好的连续 INT8 `BT[N,K]`，输出为连续小端 INT32 `C[M,N]`。全部基址32字节对齐；M/N/K为32的正倍数，M/N≤2097120、K≤16352。完整矩阵范围须位于32位地址空间内，C不能覆盖输入。

以下地址只作为调用示例，需替换为调用方实际持有的 DDR 地址。

```python
A_ADDR, BT_ADDR, C_ADDR = 0x10000000, 0x20000000, 0x30000000

group = acc.submit_gemm_rows(
    a_addr=A_ADDR, bt_addr=BT_ADDR, c_addr=C_ADDR,
    m=2048, n=512, k=2048,
    cores=(0, 1, 2),
)
results = group.wait(timeout_s=10.0)
for result in results:
    print(result["device"], result["row_start"], result["shape"], result["counters"])
```

2048行按32行块分成704、672、672行。三核共享只读BT，A基址分别偏移`row_start*K`，C基址偏移`row_start*N*4`；结果直接组成完整C，不需要软件拼接。少于三组32行时自动减少使用核数，`cores=(2,)`表示只使用第2核。

驱动先配置所有参与核，再连续写START，然后通过各核ACCEPTED_JOB、WRITE_DONE_JOB和BUSY/WRITE_BUSY确认任务完成，包括最终AXI写响应。软件连续START存在MMIO调用间隔，不是硬件同拍触发。硬件就绪和DDR争用决定实际并行收益。

阻塞形式为`acc.run_gemm_rows(...)`，参数相同，另可传`timeout_s`。所有返回值都是任务元数据/统计，矩阵结果仍位于调用方提供的C地址，不返回或创建NumPy数组。

## 三个核运行不同任务

```python
batch = acc.submit_gemm_batch([
    GemmJob(core=0, a_addr=0x10000000, bt_addr=0x20000000,
            c_addr=0x30000000, m=256, n=256, k=1024),
    GemmJob(core=1, a_addr=0x11000000, bt_addr=0x21000000,
            c_addr=0x31000000, m=512, n=128, k=1024),
    GemmJob(core=2, a_addr=0x12000000, bt_addr=0x22000000,
            c_addr=0x32000000, m=128, n=512, k=1024),
])
results = batch.wait()

# 单核异步任务也可逐个提交，每个核最多一个待回收任务。
task = acc.submit_gemm(core=1, a_addr=0x10000000, bt_addr=0x20000000,
                       c_addr=0x30000000, m=256, n=256, k=1024)
result = task.wait()
# 阻塞调用：acc.run_gemm(...)
```

## Softmax 和 GEMM 并行

Softmax输入输出均为小端FP32；1≤M≤65536、1≤K≤4096。源/目的地址16字节对齐且不能重叠。每行跨度为`16*ceil(K/4)`字节，例如K=5时占32字节；输入尾部lane忽略，输出尾部补零。参数传实际M/K，驱动编码为`(M-1)|((K-1)<<16)`。

```python
group = acc.submit_gemm_rows(A_ADDR, BT_ADDR, C_ADDR, m=2048, n=512, k=2048)

# 这里处理一张已经准备好的、独立的FP32矩阵，可与三个GEMM同时运行。
soft = acc.submit_softmax(src_addr=0x40000000, dst_addr=0x50000000,
                          m=1024, k=1024)
results = acc.wait_all(group.tasks + (soft,), timeout_s=10.0)

# 单独阻塞执行：
# result = acc.run_softmax(src_addr=..., dst_addr=..., m=..., k=...)
```

GEMM输出INT32不是Softmax所需的FP32。QK之后必须等待GEMM完成，由调用方或另一个模块执行`logit = float32(acc_int32) * score_scale`及所需mask处理，再提交FP32矩阵给Softmax；不能仅把同一段INT32位模式当成FP32。Softmax后的INT8 PV输入量化也在本驱动之外。

驱动允许多个任务共享只读输入，拒绝并行任务之间的输出重叠或输入/输出依赖。已完成任务也应先`poll()`或`wait()`回收，再提交依赖它的任务。调用方负责在START前使CPU修改对PL可见，PL写回完成后再使结果对CPU可见；裸物理地址无法自动执行cache同步。纯PL生产/消费数据时按实际系统一致性要求处理。

## 查询、超时和错误

`task.poll()`未完成返回None，完成返回结果字典；`group.poll()`返回对应结果元组。`acc.pending`列出待回收句柄。GEMM返回每核总周期、计算/加载/writer活跃周期及读写字节；驱动每次任务前清计数，不需要手动做跨任务差值。compute/load/store互有重叠，不能相加。Softmax没有周期寄存器，只返回软件观察到的耗时，不能作为纯硬件周期。

```python
try:
    results = group.wait(timeout_s=1.0)
except TimeoutError:
    # 超时不会取消硬件，也不会释放内存；保留同一组句柄继续等待。
    results = group.wait(timeout_s=30.0)

acc.close()   # 有未回收任务时拒绝关闭；不释放调用方内存。
```

硬件ERROR出现后继续等待BUSY和writer排空，再抛`HardwareError`；Softmax异常带`code`。批次等待会轮询所有所选任务，全部结束后以`BatchError.errors/results`报告失败及成功结果。下一次提交会在空闲时清除旧错误。部分START的MMIO写失败会抛`SubmissionError`，其`tasks`包含已经尝试START的任务，应保留它们和DDR并调用`acc.wait_all(error.tasks)`。若START是否送达始终无法确认，驱动保持pending，不会自行取消或假报完成；需按整机AXI复位流程恢复后重新建立控制对象。

同一FPGA控制窗口应由一个进程中的一个驱动实例独占，运行期间不要从其他软件写寄存器或重载硬件。`with`块发生超时/提交异常时保留pending状态和原异常，可继续使用原对象等待。

## 软件验证

打包的driver与21项软件单测通过的源文件逐字节一致。原测试以模拟MMIO检查三核地址分工、批量配置/启动顺序、任务编号回绕、Softmax维度/stride、四核并行、依赖检查、超时续等、错误排空及部分提交失败。

部署包已做独立导入、HWH解析、文档链接和文件SHA256校验；本次没有运行RTL仿真或板端计算。硬件bit/HWH来自当前工程现有输出，文件来源见MANIFEST.json。
