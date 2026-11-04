# GEMM_V2 最少寄存器访问的快速提交

A DMA复制与四线程NEON转置的后续实测见 [README_DMA_INPUT.md](README_DMA_INPUT.md)：三核完整调用中位数约32.701 ms、525.36 GOPS。该结果复用已注册缓冲区，注册开销另计。

批量排队DMA及C CMA到普通内存输出DMA实测见 [README_DMA_BATCHED_AC.md](README_DMA_BATCHED_AC.md)：三核A/C双向DMA、C不重复sync的端到端中位数28.879 ms、约595 GOPS；DMA注册12.39 ms另计。

这是独立的 `FastGemm` 接口，已部署在板端 `GEMM_V2/fast_gemm`，保留现有 `accelerator_driver.py`。2026-11-03 已完成原生编译、小矩阵单双三核测试以及 2048 方阵完整对比，状态记录在 `validation.json`。

## 路径和访问次数

- 初始化：确认 ABI 和所有核心空闲，CLEAR 并确认，读取实际任务编号，标记 READY。
- `prepare(a, bt, c, cores)`：校验 PYNQ 缓冲区、DDR 范围及形状，计算 32 行对齐的分工，构建可复用参数。没有 MMIO。
- `plan.submit()`：软件检查 READY；**每核5次参数写 + 1次START，0次读取**。严格先写 core0 五个参数并 START，再配置/START core1，再配置/START core2。三核共 **18 次 MMIO 写入**。
- `task.wait()`：轮询硬件状态，等待各核完整回写，验证本轮 WRITE_DONE_JOB 和 ACCEPTED_JOB；错误时等待硬件排空。**不读取性能寄存器**。所有核完成后 CLEAR、确认状态并读取 CLEAR 后的实际任务编号，恢复 READY。
- 收尾读取任务编号放在上一轮末尾，兼容 CLEAR 保留或重置编号两种实现；它没有被伪称为“消失”。完成轮询次数取决于运行时间，完整一轮不限于18次访问。

核心参数每轮仍全部重写，允许切换地址、M/N/K；三核 2048 行继续分为704/672/672行。地址和形状改变时准备新 plan，或在初始化时为各组工作缓冲区预先准备多个 plan。普通内存到 CMA 的复制及缓存同步仍由调用者显式执行，不混入寄存器提交。

## 调用方法

把整个 `fast_gemm/` 目录放在板端 `GEMM_V2/` 下，从 `GEMM_V2` 目录使用以下代码。第一次 `backend="native"` 会在板上用 GCC 自动编译 `fast_axilite.c`。它只访问当前进程持有的真实 PYNQ MMIO 映射；配置后用 ARM `dsb sy` 排序再发 START，避免 Python 逐个调用 MMIO。`backend="python"` 使用同样的状态协议，便于比较和测试。

```python
from fast_gemm import FastGemm

driver = FastGemm("hardware/design_1.hwh", backend="native")
plan = driver.prepare(a_cma, bt_cma, c_cma, cores=(0, 1, 2))

# 输入内容可以逐轮更新；BT 必须已经转置成 [N,K]。
a_cma.sync_to_device()
bt_cma.sync_to_device()
c_cma.sync_to_device()

task = plan.submit()  # 18 writes, 0 reads
task.wait()          # 完成 + 清状态，不读取性能计数
c_cma.sync_from_device()

# 单独测性能时，必须在 CLEAR 前显式选择计数收尾接口。
profile_task = plan.submit()
result = profile_task.wait_with_counters()
counters = driver.read_counters()  # 返回刚采集的快照，不再访问寄存器
driver.close()
```

`wait_with_counters()` 是显式性能接口，会在清状态前读取计数；普通 `wait()` 已清除计数后不能再补读该轮计数。再次提交会使上一轮性能快照失效。

调用者在任务结束前必须保持 CMA 分配有效，不得显式 freebuffer、覆盖硬件仍在读取的输入或并发重载 bitstream/访问同一组寄存器。plan 和在途任务保留缓冲区与映射引用；超时仍保留任务，允许再次 wait；部分提交失败时用 `driver.active` 保留的任务和 `driver.recover()` 排空恢复，结果判为无效。不得在任务未排空时释放 CMA。

## 验证与计时

`python test_fast_gemm.py`：本地寄存器协议模型验证18次写入和顺序、快速收尾不读取计数、显式计数在清除前采集、任务编号保留/重置兼容、重复提交拒绝、超时保留、部分提交失败恢复、新地址、硬件错误排空及已释放缓冲区拒绝。

`python benchmark_fast_gemm.py --repeats 8`：在已加载 GEMM_V2 的 PYNQ 上比较现有驱动、快速 Python、快速 native 的单/双/三核。使用2048方阵、四线程NEON参考、8轮正式测量；交替使用两组实际分配的A/C地址，每轮逐元素校验。计时分别记录初始化、预配置、输入准备、submit、wait与清状态、输出同步复制和完整调用。性能计数单独运行一次，不混入快速延迟样本。脚本不改PL时钟；实际CPU/PL频率保存在结果中。

## 板卡实测结果

最终数据：`fast_submit_results_20261103_052046.json`。每组预热 1 次、正式 8 次，以下为中位数，单位 ms。72 轮正式结果全部与四线程 NEON CPU 参考逐元素一致。CPU 选择最高支持档位 1199999 kHz，实际读数 **1049.9895 MHz**；PL **249.9975 MHz**。

| 核数 | 原驱动提交 | native 提交 | 原驱动提交至清状态完成 | native 提交至清状态完成 | 原驱动完整调用 | native 完整调用 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.842 | 0.224 | 37.234 | 36.500 | 58.065 | 56.828 |
| 2 | 1.279 | 0.235 | 19.946 | 19.022 | 40.272 | 41.474 |
| 3 | 1.737 | 0.240 | 14.916 | 13.378 | 36.743 | 33.677 |

完整调用包含 A 从普通内存复制到 CMA、四线程 NEON B 转置到 BT CMA、缓存同步、提交、等待并清状态、C 同步及复制为独立输出。初始化和 prepare 分别记录，不计入重复调用；动态地址场景测试中交替使用两组 A/C 实际地址。完整调用的复制耗时会波动，双核 native 此轮完整时间反而更长，因此不将提交加速等同于所有场景端到端加速。

三核快速 Python：提交 **0.483 ms**，提交至清状态 **13.345 ms**，完整调用 **33.600 ms**。native 的启动最快，但这轮三核完整调用与 Python 快速版很接近，不能声称 native 的总时间稳定优于 Python。

`benchmark_control_path.py` 用同一组缓冲区交叉组合 Python/native 提交与轮询，数据在 `control_path_results.json`。三核 native 的 C 提交函数含 ctypes 调用约 **0.024 ms**，整个 plan.submit() 约 **0.132 ms**；完整复制场景的 plan.submit() 为上表 **0.240 ms**，两者计时上下文不同。native CLEAR+确认状态+读取实际任务号函数约 **0.053 ms**。MMIO 映射地址及清状态数组已缓存，收尾不再重复准备这些对象。

同一实验 native/native 和 native/Python 总时间约 13.301/13.326 ms，Python/native 和 Python/Python 约 13.080/13.097 ms。差异主要随启动方式变化，切换轮询方式影响较小；更紧密启动也改变 DDR 请求的重叠，硬件周期可能变长。实验没有直接测 DDR 带宽，不能单凭这些数据确定带宽瓶颈。

正常 `wait()` 从不读取性能计数；需要性能时另发一轮 `wait_with_counters()`。性能运行已排除在正常延迟统计之外。启动仅有18次写；完成轮询、任务编号校验和收尾的访问仍需另外计算。
