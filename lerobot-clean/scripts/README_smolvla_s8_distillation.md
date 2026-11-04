# S8 双相机学生模型与蒸馏训练

已实现可训练、可保存重载的 S8，教师为本机最新的 wrist / beside / base 200k steps 原始检查点。这里的 base 是训练版本名称；实际输入是腕部和侧边两路相机。

## 已实现的结构

| 组件 | 教师 | S8 学生 |
| --- | --- | --- |
| 视觉编码器 | 12 层、768 维 | 保留教师权重并冻结 |
| 语言骨干 | 16 层、960 维、FFN 2560 | 8 层、576 维、FFN 1536 |
| 注意力 Q / KV 头数 | 15 / 5 | 9 / 3，head_dim 64 |
| 动作专家 | 16 层、720 维、FFN 2048 | 8 层、432 维、FFN 1280 |
| 动作输出 | 50 × 6 | 保持 50 × 6，内部补齐到 32 维 |
| 去噪步数 / tokenizer | 原模型配置 | 保持不变 |

学生参数总量为 197,428,832，包含部署时不使用的生成头。默认可训练参数为 80,950,496，包含三个仅用于训练的特征投影层。语言骨干、Embedding、连接器、动作专家和输入输出投影共同参与蒸馏；视觉编码器、教师以及不参与动作推理的分支冻结。

深度初始化选择教师层 `0,1,4,5,8,9,14,15`，保留自注意力 / 交叉注意力的交替关系。宽度采用一致的注意力头、残差通道、FFN 通道裁剪；这是确定性的结构化初始化，未进行重要性搜索。学生模型的所有权重均来自教师切片；训练专用的特征投影采用确定性初始化。

## 双卡启动训练

当前已支持单机 DDP，双卡入口为：

```bash
cd /home/ial-mohd/lerobot
SMOLVLA_GPUS=0,1 bash scripts/train_smolvla_s8_distill_2gpu.sh
```

默认使用 GPU 0、1；每卡 batch size 2、梯度累积 2，全局有效 batch 为 `2 × 2 × 2 = 8`，保持与原单卡配置相同。学习率、20,000 总步数、损失和验证频率均保持原值。输出目录为 `outputs/train/smolvla_s8_wrist_beside_distill_2gpu`。每张卡放一份冻结教师和学生，独立处理分配到的样本，DDP 同步学生及训练专用特征投影的梯度；梯度累积期间通过 `no_sync()` 减少同步次数。

双卡采样对同一个全局随机批次按 rank 分片，避免各卡重复使用同一条随机流；有放回采样本身仍可能抽到相同帧。不同 rank 的有效动作 / 文本 token 数可能不同，训练损失和验证指标按各自有效数量归一化。验证子集按 rank 分片且不补重复样本；默认 64 帧是全局总量，不是每张卡 64 帧。只有 rank 0 写日志、模型、`best` / `last` 链接；检查点包含各 rank 独立的 flow / PyTorch 随机状态。

选择其他 GPU、更长训练或扩大验证子集可追加参数，例如：

```bash
SMOLVLA_GPUS=0,1 bash scripts/train_smolvla_s8_distill_2gpu.sh \
  --steps 50000 \
  --eval-batches 256 \
  --output outputs/train/smolvla_s8_wrist_beside_distill_2gpu_50k
```

20k / 50k 是实验预算，不能保证恢复教师效果。20k、有效 batch 8 对应 160,000 次样本抽取，约为当前 46,248 个训练帧的 3.46 倍；应根据留出验证和实机表现判断是否足够。当前没有自动早停。

双卡续训需继续使用双卡入口，并保持原训练的参数：

```bash
SMOLVLA_GPUS=0,1 bash scripts/train_smolvla_s8_distill_2gpu.sh \
  --resume outputs/train/smolvla_s8_wrist_beside_distill_2gpu/checkpoints/last
```

检查点要求 GPU 数量不变；可更换物理 GPU 编号。若开始时自定义了总步数、batch、累积步数、损失权重等，续训也需传入相同值。当前不能把 20k 计划直接改成 50k 续训；学习率调度依赖原总步数。

双卡短测：

```bash
SMOLVLA_GPUS=0,1 bash scripts/train_smolvla_s8_distill_2gpu.sh \
  --smoke-test \
  --output outputs/diagnostics/smolvla_s8_my_ddp_smoke_test
```

双卡短测使用两步、每卡 batch 1、梯度累积 2，覆盖跨卡梯度同步、跨卡权重一致性、验证汇总、保存重载和完整推理。单元测试还用有 padding / 不同有效数量的双进程计算，对比单进程合并样本的参考梯度。已通过双卡真实数据短测；正式训练的默认有效 batch 8 另有两步训练与续训验证。

## 单卡启动训练

```bash
cd /home/ial-mohd/lerobot
bash scripts/train_smolvla_s8_distill.sh
```

脚本已配置本机教师、离线 tokenizer / VLM 配置，以及数据集当前位置 `/data/ial-dataset/ial-mohd/test_20261102_142034_cleaned_wrist_beside`。默认使用已验证的 `/home/ial-mohd/miniforge3/envs/lerobot/bin/python`；可用环境变量 `SMOLVLA_PYTHON` 替换解释器。

此单卡脚本默认 `cuda:0`，batch size 2，梯度累积 4，有效 batch size 8；训练 20,000 个优化器更新，学习率 `3e-5`，预热 300 步。训练和教师计算均使用 FP32，AMP 和 TF32 关闭，不使用 FP16 / BF16 / FP8。多卡请使用上面的双卡启动脚本。

默认损失为：

```text
loss = demo + kd + 0.1 * feature
```

- `demo`：学生对真实示范 flow velocity 的误差。
- `kd`：相同动作噪声和时间下，学生与教师 flow velocity 的误差。
- `feature`：三个语言中间层经投影后的余弦距离。
- `teacher_demo`：教师对同一真实目标的误差，仅作为监控指标。

动作误差仅统计有效的 6 个动作维度和非 padding 时间步。当前联合训练上述损失，尚未实现 QAT。

数据按每项任务的最后 20% episode 留出，当前为 94 个训练 episode、24 个验证 episode。默认每 500 步在固定的 64 帧验证子集上评估，并非每次遍历全部验证帧；以验证 `demo` 最小值选取 `best`。每 2,000 步、验证改善时以及训练结束时保存检查点。20k 是可调整的训练起点，实际停止点应结合验证曲线和实机表现判断。

修改参数可直接追加，例如：

```bash
bash scripts/train_smolvla_s8_distill.sh \
  --device cuda:1 \
  --output outputs/train/smolvla_s8_wrist_beside_distill_run2
```

训练输出：

```text
outputs/train/smolvla_s8_wrist_beside_distill/
  run_config.json              # 本次蒸馏的实际参数与教师权重哈希
  episode_split.json           # 训练 / 验证划分
  evaluation.jsonl             # 验证指标
  student_assets/              # S8 结构和 tokenizer 配置
  checkpoints/
    000500/pretrained_model/   # 示例；实际取决于保存时机
    000500/training_state.pt   # 优化器、特征投影、噪声 RNG 等
    last/                     # 最近一次完整检查点的符号链接
    best/                     # 验证 demo 最低检查点的符号链接
```

`train_config.json` 保留原始训练来源信息；本入口的实际参数以 `run_config.json` 为准，续训请继续使用此蒸馏入口。

## 断点续训

```bash
cd /home/ial-mohd/lerobot
bash scripts/train_smolvla_s8_distill.sh \
  --resume outputs/train/smolvla_s8_wrist_beside_distill/checkpoints/last
```

保持总步数、batch、累积步数、学习率、预热、随机种子、损失权重和数据划分与原训练一致；更换 GPU 或数据读取 workers 不影响上述约束。自定义训练参数也需要在续训命令中传入。`--steps` 表示整个训练的总步数，不是本次追加步数。程序检查教师哈希，恢复优化器、特征投影、采样位置和噪声生成器。检查点先完整写入临时目录，再发布最终目录。

## 两步验证与当前完成情况

```bash
bash scripts/train_smolvla_s8_distill.sh \
  --smoke-test \
  --output outputs/diagnostics/smolvla_s8_my_smoke_test
```

该命令强制使用两步、batch 1、累积 1，执行真实数据的前向 / 反向、验证、保存重载和完整 10 步去噪推理。输出目录需要为空。

本机已通过 19 项单元测试和上述真实数据验证；保存重载后权重逐值一致，动作输出为 FP32 `[1,50,6]`。单卡、双卡均验证了从第 1 步恢复到第 2 步，与连续训练的权重一致。证明材料位于：

- `outputs/diagnostics/smolvla_s8_distillation_validation/smoke_test_report.json`
- `outputs/diagnostics/smolvla_s8_resume_check/resume_test_report.json`
- `outputs/diagnostics/smolvla_s8_ddp_validation/smoke_test_report.json`
- `outputs/diagnostics/smolvla_s8_ddp_resume_check/resume_test_report.json`

尚未执行正式 20k 蒸馏，也未验证实机成功率。诊断目录的两步权重仅用于程序验证。

## 训练后导出 W8A8 + Embedding INT8

训练先保存 FP32 权重，单份模型权重约 790 MB。之前的 **176～186 MB** 是量化 ONNX 的预算，需要对语言 Embedding 也做 INT8；这是估算，实际大小和误差以导出报告为准。

已为原有 W8A8 / FP32 导出脚本增加 `--quantize-language-embedding`。完成训练后运行：

```bash
cd /home/ial-mohd/lerobot
PYTHONPATH=outputs/tools/onnx_fp32_deps:src:. \
  /home/ial-mohd/miniforge3/envs/lerobot/bin/python \
  scripts/export_smolvla_w8a8_fp32_onnx.py \
  --checkpoint outputs/train/smolvla_s8_wrist_beside_distill/checkpoints/best/pretrained_model \
  --vlm-assets outputs/train/smolvla_s8_wrist_beside_distill/checkpoints/best/pretrained_model/vlm_assets \
  --output outputs/onnx/smolvla_s8_wrist_beside_distill/best_w8a8_fp32 \
  --device cuda:0 \
  --quantize-language-embedding
```

该导出对线性层采用 W8A8、Embedding 使用 INT8 存储并在索引后反量化，其余浮点算子保持 FP32。它属于训练后量化，不等同于 QAT。当前已验证小图的 INT8 Embedding 导出和 ONNX Runtime 运行；尚未导出完整训练后的 S8 ONNX。

检查点中的本地 assets 路径为绝对路径，复制到其他服务器时需同步资源并调整配置路径。上面的导出命令使用单卡训练默认目录；双卡训练的结果请将检查点路径改为 `outputs/train/smolvla_s8_wrist_beside_distill_2gpu/checkpoints/best/pretrained_model`，assets 路径同步修改。
