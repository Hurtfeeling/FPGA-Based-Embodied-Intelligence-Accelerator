# 最后一次 NEON A 打包实验快照（非默认）

取自原板 /dev/shm/graphopt_20261104；未在打包时继续运行或调优。
runtime/pack_a_neon.cpp 与 pack_a_neon.py 将 UINT8→signed INT8 转换及补零融合，
大任务使用四线程 NEON 直接写入 CMA；runtime/operators.py 包含该实验选择逻辑。

benchmark_graphopt.py 通过显式 --runtime-root 选择本目录，configs_v4.json 为最后一轮配置，
a_pack_sweep_v3/v4.json 与 a_pack_micro.json 为历史原始结果。
驱动、HWH 和 infer_board 依赖已随快照整理；bitstream 统一使用 ../../GEMM/hardware 的已加载版本。
运行时还需 ../ 中 ORT 桥接库、CPU Softmax 库、softmax_rows.onnx、验证输入、模型和同板 CPU 基线。
入口的 --help 列出这些参数；此快照用于保存最后的工作，不作为默认 run.sh。

本轮结果中位数 baseline_full_clear_v4=34.297492077 s，adaptive_neon_cma_v4=33.924750127 s，
各 1 次 warmup、5 次正式测量。该整图未使用 PS DMA，不能标为 DMA+NEON 整图结果。
