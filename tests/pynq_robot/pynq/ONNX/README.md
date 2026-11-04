# SmolVLA ONNX CPU / PL 推理

实时双相机、关节反馈与电机动作联动见 [README_ROBOT.md](README_ROBOT.md)，入口为 `run_robot.py`。

正式默认：四线程 ORT + 大矩阵 PL GEMM + 80 MiB 常驻权重 + 共享 CMA 暂存，原生 CPU Softmax。
图转换使用已测的 32 对齐方阵/长矩阵/宽矩阵数据筛选任务，小矩阵继续 CPU。

## 模型准备

本代码包不含大型 SmolVLA 权重，包含匹配 manifest 和 validation_0.npz、cpu.npy/cpu.json。
原板模型目录：

```text
/home/xilinx/jupyter_notebooks/smolvla_final/GEMM_V2/eval_wrist_beside_20261103/model
```

需要的模型文件：

- CPU：smolvla_w8a8_fp32.onnx
- 默认 PL：smolvla_selective_pynq_resident88.onnx
- PL Softmax 实验：smolvla_selective_pynq_softmax_resident88.onnx

直接传原路径即可。若复制到本目录 models/，必须把同名 .manifest.json 也放在模型旁边。
manifest 与模型对应，不能混用；缓存槽在图内预先转置后按 manifest 加载。
小的 softmax_rows.onnx 已在包内。

## 运行（当前目录为 ONNX）

先按根 README 加载 GEMM/hardware/design_1.bit。推理入口不会自行重载 bitstream。

```bash
MODEL=/home/xilinx/jupyter_notebooks/smolvla_final/GEMM_V2/eval_wrist_beside_20261103/model

# 同板 CPU 四线程；结果保存为 results/cpu_new.json 与 .npy
sudo bash run.sh cpu "$MODEL/smolvla_w8a8_fp32.onnx" --output-prefix "$PWD/results/cpu_new"

# 大矩阵使用 PL，其他算子使用 CPU；和上一步输出比较
sudo bash run.sh hybrid "$MODEL/smolvla_selective_pynq_resident88.onnx" \
  "$PWD/results/cpu_new.npy" --output-prefix "$PWD/results/hybrid_new"
```

也可用随包 cpu.npy 作原验证输入的同板参考。更换输入时必须重新生成相应 CPU 基线。
结果前缀必须是新的，脚本拒绝覆盖已有报告。加载 session 与稳态推理耗时分别记录；
稳态计时含回调、输入打包、CMA 同步、调度、CPU/PL 计算及输出搬运。

run.sh 设置四核绑定、ORT intra=4/inter=1、sequential、graph_opt=disable、arena=0、
memory_pattern=0、allow_spinning=1；调用 set_cpu_highest.py 请求最高 CPUFreq 档，
实际时钟写入报告。它设置 LD_LIBRARY_PATH，使随包 ARM64 ORT 可在新路径加载。
脚本默认使用 libsmolvla_ort_sweep.so，与这组环境变量配套。

## 源码及重编译

- `infer_board.py`：CPU/混合推理、回调及计时/误差报告。
- `ort_cpu_bridge.cpp`、`pynq_custom_ops.h`：ORT 与自定义 GEMM/Softmax 的 C++ 桥接。
- `prepare_selective_graph.py`：本机图转换和分流证据筛选，需 onnx/numpy/onnxruntime，建议本机执行。
- `reorder_camera_weight_pairs.py`：保留的图调度实验源码，非默认推理步骤；必须显式传 --input/--output/--report。
- `benchmark_runtime_sweep.py`、`summarize_runtime_sweep.py`：CMA/Softmax 配置对比及汇总。
- `../GEMM/runtime`：板端正式驱动；`../GEMM/dma_batch` 提供可选独立 DMA 接口。

```bash
bash build_runtime.sh ./onnxruntime-linux-aarch64-1.16.3 libsmolvla_ort_sweep.so
bash build_softmax_cpu.sh ./onnxruntime-linux-aarch64-1.16.3
```

随包 .so 取自同板，旧 rpath 指向原路径；run.sh 的 LD_LIBRARY_PATH 或在目标位置重编译可解决重定位。
ORT 1.16.3 的头文件、实际非空 ARM64 动态库、LICENSE 和第三方声明均已附带。

## Softmax 与最近的 NEON 试验

`runtime_cma88_cpu_softmax.json`、`runtime_cma88_parallel_softmax.json` 用于带自定义 Softmax
节点的图，和默认原生 CPU Softmax 图区别见文件名。CPU/PL 行级并行、PL 流水均保留源码；
目前 PL Softmax 整图存在动作偏差，因此默认 run.sh 采用 runtime_cma80_native_softmax.json。

`experimental_neon/` 保留停止前最后一次完整实验：NEON A 转换/32 补齐直接写 CMA、
自适应选择和原始配置/结果。它有独立 runtime，不覆盖正式版本。
历史同一 session 的 5 轮中位数由 34.2975 s 降至 33.9248 s，输出与同板 CPU 逐元素一致；
这属于既有实测快照，打包后未重新测量。该整图仍未接 PS DMA。
默认版本与该实验的区别、依赖详见 experimental_neon/README.md。

正式历史参考：同输入 CPU 58.018452 s、native80 PL GEMM 路径 34.167594 s；
这些是不同已存记录，不应和最新实验混合计算提升率。
