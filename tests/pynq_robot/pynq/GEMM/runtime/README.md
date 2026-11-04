# GEMM_V2 驱动统一入口

从板端 `GEMM_V2` 目录调用：

```python
from runtime import PynqOperators, FastGemm, AcceleratorDriver, DmaCopy
```

| 接口 | 用途 | 内存与同步 |
|---|---|---|
| `PynqOperators` | 普通张量输入输出，适配ONNX的UINT8/INT8 MatMulInteger及FP32 Softmax | 内部复用36 MiB CMA，补齐32维度、转置、缓存同步、分块调度和输出复制 |
| `FastGemm` | 已准备好的CMA A/BT/C提交；配置一个核后立即启动，再配置下一核 | 调用者管理CMA和缓存同步；每核启动6写0读，正常收尾不读性能计数 |
| `AcceleratorDriver` | 通用物理地址GEMM/Softmax控制与HWH验证，保留原有兼容接口 | 调用者持有DDR缓冲区；有状态/编号/回写完成校验 |
| `DmaCopy` | 实验性Linux DMAEngine普通内存与XLNK CMA复制 | 需要显式加载模块、注册并保持固定源目标地址；关闭fd后才能释放缓冲区 |

`PynqOperators` 适配已验证的 `UINT8(q_signed+128), zero point=128` 与INT8权重零点0。MatMul输入至少2维，支持批量广播；其他zero point或float GEMM不能直接交给此接口。Softmax沿最后一维，K≤4096，分块处理。超出staging容量的矩阵会明确报错。

```python
ops = PynqOperators()
try:
    ops.matmul_integer(a_uint8, b_int8, c_int32)
    ops.softmax(logits_fp32, probabilities_fp32)
    print(ops.metrics)
finally:
    ops.close()
```

默认按每个矩阵补齐后的运算量分流：`2 * up32(M) * up32(N) * up32(K) <= 12058624` 时使用四线程 NEON，其余使用 PL。批量输入逐矩阵判断，最多三个 PL 矩阵合并提交；单矩阵仍可按32行对齐分给三个核。传入 `PynqOperators(cpu_threshold_ops=0)` 可关闭 CPU 分流。阈值来自37组板端实测，CPU实际约1.05 GHz、PL约250 MHz，包含搬运和调度，详见 [测试口径及原始报告](../fast_gemm/DISPATCH_THRESHOLD.md)。该阈值是保守经验值，尚未代表新版整图推理收益。

K/N均为32倍数时，B的首次转置和动态B转置统一使用四线程NEON，静态B后续可命中BT缓存。N较大时，首次转置成本与缓存命中成绩差别明显，详见 [宽矩阵专项测试与首次转置修复](../fast_gemm/DISPATCH_WIDE32.md)。

张量接口的输入输出复制目前使用CPU/NEON；ONNX全图未启用PS DMA。动态图中每个新地址注册DMA的成本可能大于复制收益。`DmaCopy`保留为独立可选接口。

NEON模块及C++源码已收进本目录，首次调用自动编译，无需`GEMM_V1`目录。GEMM原生MMIO helper在`../fast_gemm`中自动编译。只允许一个驱动持有一组GEMM寄存器；本接口内部由FastGemm控制三核、通用驱动仅提交Softmax。

历史测试脚本和原始结果保留在`../fast_gemm`，DMA模块源码在`../dma_batch`。配套`hardware/design_1.bit/.hwh`需已加载；驱动不会重载bitstream或调PL时钟。

## 常驻权重与共享 CMA

`resident_cache_slots` 接受图转换器生成的常驻权重清单；对应静态 B 已在图内转为 `[N,K]`，第一次调用时才补齐、复制到 CMA 并 flush。后续命中直接复用，不再转置、复制或同步权重。权重必须在驱动生命周期内保持不变。`resident_cache_bytes` 设置预算；没有常驻槽时不分配权重池。

`shared_scratch=True` 可将 GEMM 和 Softmax 的暂存区合并。Softmax 每个输入/输出缓冲为 2 Mi 个 FP32 元素时，独立暂存占 44 MiB，共享后占 28 MiB，释放的 16 MiB 可用于权重池。共享区使 GEMM 与 Softmax 互斥；它不提供两类 IP 之间的并发。发生超时应先收尾硬件任务，不能覆盖仍在使用的内存。调用 `close()` 会清理计划、视图和底层 CMA 分配。

## 可选 Softmax 并行路径

`softmax_parallel.CpuSoftmax` 调用同版本 ORT 的原生 FP32 Softmax；`SoftmaxRouter` 可按行分配 CPU/PL 工作，`softmax_pipeline` 在 PL 分块期间准备下一块输入和搬出上一块输出。CPU 与 PL 使用不同的行，回调返回前等待两者完成。小于等于 2 Mi 元素的混合请求走 CPU，避免增加小任务开销。

这些路径需要显式配置，原有默认调用保持不变。PL Softmax 与 CPU 的微小浮点差异可能影响后续量化，必须比较完整动作输出；不能仅凭单算子误差选择配置。`overlap_seconds` 统计 CPU 原生调用与完整 PL 路径的软件时间区间交集，包含拷贝、同步和等待，不是 IP 周期计数。

两相机模型的复测入口为 `../run_runtime_sweep_remote.py`，支持 `deploy/start/status/fetch`，同一 session 内比较不同 CMA 与 Softmax 策略。原始数据、完整计时范围和精度结果见模型目录下 `pynq_eval/CMA_SOFTMAX_PARALLEL_RESULTS_20261104.md`。板端实测 CPU 约 1.05 GHz、PL 约 250 MHz；cpufreq 请求 1.2 GHz 不代表 PLL 已达到 1.2 GHz。
