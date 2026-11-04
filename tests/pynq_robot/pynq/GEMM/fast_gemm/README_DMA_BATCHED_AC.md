# PS DMA 批量队列、A输入与C输出

## “一次搬完”的含义

普通 NumPy 数组虚拟地址连续，但物理页通常分散。DMA 不能把整段虚拟地址当作一个物理源地址，因此 Linux DMA API 把它映射为 scatter-gather 段。逻辑上可以一次调用 DMA copy；实现上可能包含多个硬件描述符。

此前的试验驱动每完成一个段就提交并等待，A 的 41 段需要41轮等待。新隔离驱动先排队多个段、启动一批并只在批次末等待。实测分配下，A 为71段、C 为201段；PS DMA 驱动每批最多接收32个描述符，故分别是3批和7批。若整块 CMA 或 DMA coherent 连续区作为普通内存源/目标，则可以是一段、一个描述符；本实验使用的普通数组不保证物理连续。不要把“一个 ioctl”说成“一个硬件描述符”。

## A 输入和 C 输出实测

文件 `dma_batched_ac_20261103_060918.json`。2048² INT8 GEMM，四线程 NEON 转置B；结果复制到独立普通 INT32 输出。预热后8轮中位数，单位ms。A输入、缓存同步、GEMM完整等待/清状态、C同步及独立输出复制均计入；DMA映射注册和分配单独，不计入稳态。

| 三核路径 | 完整时间 | GOPS |
|---|---:|---:|
| CPU复制A + CPU同步/复制C | 31.247 | 550 |
| 批量DMA复制A + CPU同步/复制C | 30.609 | 561 |
| 批量DMA复制A + DMA复制C（仍额外PYNQ sync） | 30.286 | 567 |
| 批量DMA复制A + DMA复制C（DMA API接管C同步） | **28.879** | **595** |

最后一行相对同轮 CPU A/C 基线减少 **2.369 ms，约7.6%**。C CPU路径约1.38 ms PYNQ sync + 7.26 ms复制；C DMA路径约6.94 ms，DMA API自己执行传输前后的缓存维护，因此跳过冗余的 `c.sync_from_device()`，避免重复同步。`host_c` 每轮逐元素验证与四线程NEON参考完全一致。若用户要求返回仍在CMA中的借用视图，则不需要把C搬出，这段工作可直接省掉，比DMA复制更好。

A DMA稳态约1.78 ms，而 CPU copy+PYNQ sync约2.33 ms；三核约省0.55 ms。A与四线程NEON转置并行没有明显收益，转置会从约4.84 ms升至约6.4 ms，说明两项工作争用CPU/缓存/内存通路。

测试合计144轮正式输出和18次性能采样均通过；PL为249.9975 MHz，CPU实际1049.9895 MHz。分配对物理页映射/合并有影响，所以段数会在不同轮次或重启后改变。本次 A/C 同时注册耗时约12.39 ms，其中A约1.79 ms，C约10.42 ms。若每轮都要换普通内存地址并重新注册，稳态收益会被抵消。复用 host A 与 host C 地址可以保持注册；无法复用时需按真实调用频率计入注册成本。

## 数据方向和所有权

- A 输入：普通 NumPy源是 `DMA_TO_DEVICE`，PYNQ CMA目标是 `DMA_FROM_DEVICE`。copy ioctl 同步A源/目标缓存；完成后GEMM读取CMA里的A。copy完成前不要访问host A或CMA A。
- C 输出：GEMM写完C CMA后，PS DMA把C搬到普通 host C。DMA API在copy前后同步映射；完成后host C可读。采用这个接口时不要再对同一C CMA额外执行 `sync_from_device()`，否则重复维护缓存。
- fd打开期间，内核持有普通页和XLNK页引用。普通host数组和CMA对象都必须保持有效；DMA完成并关闭fd后才能释放CMA。脚本不接受裸物理地址。
- A/B/C CMA及普通缓冲区分配、DMA注册和Neon结果参考均在计时外；CMA空间不够时应复用缓冲区，或让上游直接写入CMA。

## 可复现文件

内核试验模块源码/构建器/用户态映射接口在 `GEMM_V2/dma_batch/`。模块root-only，版本适配板端Linux 5.4.0-xilinx-v2020.1；只使用`dma1chan0`和`dma2chan0`，不修改PL bitstream或时钟。批次上限默认32，是底层描述符准备失败时的安全flush界限。测试脚本 `benchmark_dma_batched_ac.py --repeats 8`，A/C分别走DMA时使用不同channel；正式输出每轮完整校验。原始逐轮记录和内核profile在上述JSON。

该内核模块只用于隔离实验，不是可直接交付的任意buffer生产驱动。主 `accelerator_driver.py` 未改动；测试结束后 `ps_dma_batch` 已卸载，三核状态均为空闲。
