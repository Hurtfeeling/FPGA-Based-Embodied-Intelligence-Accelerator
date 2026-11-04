# PYNQ PS DMA 内存复制试验

板端目录：`/home/xilinx/jupyter_notebooks/smolvla_final/dma_trial_20261102`。

使用 Linux DMAEngine 独占申请 `dma1chan0`（ZynqMP PS DMA），不需要修改 bitstream。试验仅操作本进程分配的普通 NumPy 数组及 `pynq.allocate(cacheable=1)` CMA，未改动 GEMM/GEMM_V1 主驱动或固定窗口。现有 GEMM 的 AXI master 本身已经负责 PL 侧访存；这里测试的是 PL 启动前、完成后原本由 CPU 执行的额外复制。

## 已验证结果

每次 4 MiB，每方向 1 次预热、8 轮正式测量，CPU/DMA 顺序交替，每轮改变输入并逐字节验证。数值为中位数，单位 ms；CPU 和 DMA 都计入当前路径必需的缓存同步，DMA 还包括 ioctl、描述符、启动、完成等待。

| 路径 | CPU 完整路径 | DMA 完整路径 | DMA 减少耗时 |
|---|---:|---:|---:|
| 普通 NumPy → 可缓存 CMA | 2.416 | 1.884 | 22.0% |
| 可缓存 CMA → 普通 NumPy | 2.248 | 2.032 | 9.6% |

- DMA 传输、分段提交及等待：约 0.815 / 0.952 ms；缓存同步：约 1.011 / 1.029 ms。
- 进程 CPU 时间：CPU 路径约 2.418 / 2.251 ms，DMA 路径约 1.136 / 1.314 ms；不包含硬中断等全系统 CPU 成本。
- 两端首次注册合计约 1.648 / 1.633 ms，作为初始化单独统计，未计入稳态复制。每次换数组、重新注册时需要重新支付，独立输出分配成本也未计入。
- 普通内存经 DMA 映射后为 6 / 24 段，当前保守实现逐段提交并等待；分段数量依赖分配结果，会影响耗时。
- 全部 32 条正式 CPU/DMA 结果及预热均验证通过。
- 这不是完整 GEMM 的新端到端结果，不能直接宣称 GEMM 已加速。固定窗口整合和输出所有权策略尚未修改。

结果：`dma_directional_results.json`。另保留 `dma_results.json`（非缓存一致性测试映射，不能用于 GEMM CPU 性能比较）和 `dma_cma_results.json`（双方都使用保守双向同步的初版）。初版约 2.15 ms，缓存同步过重；最终版本按源/目标方向分别映射，才获得上述收益。

## 使用方式

板上已编译好 `ps_dma_trial.ko`。重新构建执行 `python3 build_trial.py`；脚本在本目录复制内核头文件并生成缺失的构建工具，不修改系统内核树。板上 GCC 7 不支持当前内核默认的每任务栈保护选项，因此仅本试验模块构建禁用相关选项，不改变运行内核配置。

```sh
cd /home/xilinx/jupyter_notebooks/smolvla_final/dma_trial_20261102
sudo sh -c 'set -eu; insmod ./ps_dma_trial.ko; trap "rmmod ps_dma_trial" EXIT; python3 -u benchmark_directional.py'
```

复制步骤：打开 `/dev/ps_dma_trial`，先选方向、注册 CMA 和普通内存，然后复用映射提交复制，最后关闭设备解除映射，再释放 CMA。

```python
fd = os.open('/dev/ps_dma_trial', os.O_RDWR)
try:
    fcntl.ioctl(fd, 0xD105, 1)  # 1: host -> CMA; 0: CMA -> host
    fcntl.ioctl(fd, 0xD104, struct.pack('=2Q', cma.ctypes.data, 4194304))
    fcntl.ioctl(fd, 0xD101, struct.pack('=2Q', host.ctypes.data, 4194304))
    profile = dma_copy(fd, 0xD102)  # reverse direction uses 0xD103
finally:
    os.close(fd)
# Only after closing fd may cma.freebuffer() be called.
```

本原型限定本机 Linux 5.4、ARM64、4 MiB、XLNK 后端，设备仅 root 可用。普通内存通过 GUP 固定页面；CMA 只接受当前进程实际映射的 `/dev/xlnk`，由 VMA 推导页面，不接受物理地址。调用者必须保持数组存活、不得并发读写或释放，关闭 DMA 设备后再释放原 CMA 分配。驱动使用 DMA API 建立总线地址并同步缓存，超时先同步终止 DMA，再释放映射。它不是支持任意用户、任意缓冲区生命周期的通用生产驱动。

试验结束已卸载临时模块并停止 dmatest，不设开机加载。
