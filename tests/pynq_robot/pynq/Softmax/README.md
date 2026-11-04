# Softmax 测试

当前目录为 Softmax，共用 ../GEMM 中的硬件和控制驱动。先按根 README 加载 overlay。

```bash
# FP32 1024×1024，CPU NumPy 参考与 PL 完整路径
sudo python3 softmax_1024_benchmark.py

# PL 流水：与上一块输出搬出/下一块输入准备重叠
sudo python3 benchmark_softmax_pipeline.py --output softmax_pipeline_new.json --repeats 5
```

1024² 的 PL 计时含普通内存→CMA、缓存同步、提交/等待、CMA→独立输出。
结果写入 benchmark_results；CPU 参考为 NumPy，不是“四线程 NEON 优化 Softmax”基线。
Softmax 驱动没有硬件周期计数器，submit/wait 是软件观察时间。
Softmax 不采用 GEMM 的 M/N/K 全部 32 对齐约束，内部按行和 FP32 数据布局处理。

`softmax_pipeline.py` 为流水源码；`softmax_parallel.py` 中 CpuSoftmax/SoftmaxRouter
实现按不重叠行拆分 CPU/PL。CPU 分支调用与 ONNX 相同版本的 ORT 原生 Softmax，
桥接 C++、构建脚本、softmax_rows.onnx、.so 和 ORT 在 ../ONNX。
本目录源码方便独立查看；正式 ONNX 使用同源的 ../GEMM/runtime/softmax_*.py。

CPU/PL 并行的整图入口是 ../ONNX/benchmark_runtime_sweep.py，需完整模型、manifest、
输入及 CPU 基线。没有独立的 parallel 性能 main；不要把并行模块的单元测试视为板端性能测试。
本地逻辑测试源码保存在 test_softmax_pipeline.py、test_softmax_parallel.py。

PL Softmax 单算子误差较小，但当前 SmolVLA 整图经过后续量化后存在动作输出偏差，
故默认 ONNX 保留 CPU Softmax。相关配置、历史结果均保留在 ONNX。
