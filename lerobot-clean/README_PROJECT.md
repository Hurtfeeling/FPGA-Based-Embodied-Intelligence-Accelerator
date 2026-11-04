# LeRobot 与 FPGA 工程的项目入口

本目录是随主工程保存的 LeRobot 本地源码快照，包含项目使用的 SmolVLA 量化、导出和 S8 蒸馏代码。`pyproject.toml` 的版本字段为 `0.6.2`，要求 **Python >=3.12**；该字段不能确定快照对应的上游 tag，当前也未记录可核对的上游 commit。

上游介绍见 [README.md](README.md)，开发约定见 [AGENTS.md](AGENTS.md)，快照边界见 [CLEAN_EXPORT.md](CLEAN_EXPORT.md)。上游 README、LICENSE 和其他原有文档保留原样；本页补充本工程的入口与使用范围。

## 在主工程中的位置

- `src/lerobot/`：LeRobot 源码，包括 SmolVLA 策略、数据集、训练、评估、采集和机器人控制等通用功能。
- `scripts/`：本工程的 SmolVLA 模型准备与数值验证入口；S8 实现位于 `src/lerobot/policies/smolvla/distillation*.py`。
- [板端 ONNX 入口](../tests/pynq/ONNX/README.md)：PYNQ 上 CPU / PL 混合推理、图转换、运行库和板端验证材料。

这里提供模型准备和参考计算能力。自定义 ONNX 域、INT8 / FP8 张量格式不会自动成为板端可执行模型；需要匹配的图转换、运行库、硬件与配置。板端工作应以上述板端说明为入口。当前整理包不构成从相机采集、策略推理到机器人动作执行的完整机器人闭环。

## 项目专用脚本

以下只登记本工程使用的自定义入口，不穷举上游 CLI。路径均相对于本目录。

| 分组 | 入口 | 作用与边界 |
|---|---|---|
| 硬件交换包 | [quantize_smolvla_w8a8.py](scripts/quantize_smolvla_w8a8.py) | 标定、量化、导出自定义 HLS/PL W8A8 交换包；包含浮点模拟的数值参考，不是 FPGA 推理后端。 |
| 标准 ONNX | [export_smolvla_fp32_onnx.py](scripts/export_smolvla_fp32_onnx.py) | 导出原始权重的全 FP32 ONNX。 |
| 标准 ONNX | [export_smolvla_w8a8_onnx.py](scripts/export_smolvla_w8a8_onnx.py) | 从自定义 W8A8 交换包导出标准整数算子 ONNX。 |
| 标准 ONNX | [export_smolvla_w8a8_fp32_onnx.py](scripts/export_smolvla_w8a8_fp32_onnx.py) | 从检查点导出 W8A8 矩阵乘、其余浮点运算为 FP32 的图；可选语言 Embedding INT8 存储。 |
| ONNX 运行 | [run_smolvla_fp32_onnx.py](scripts/run_smolvla_fp32_onnx.py)、[run_smolvla_onnx.py](scripts/run_smolvla_onnx.py) | 使用 ONNX Runtime 运行对应标准图和已记录输入。 |
| 自定义 INT8 | [export_smolvla_strict_int8.py](scripts/export_smolvla_strict_int8.py)、[run_smolvla_strict_int8.py](scripts/run_smolvla_strict_int8.py) | 自定义整数域模型，配套 [strict_int8_runtime.py](scripts/strict_int8_runtime.py)；默认 ONNX Runtime 不能直接加载该域。 |
| 自定义 FP8 | [export_smolvla_w8a8_fp8.py](scripts/export_smolvla_w8a8_fp8.py)、[run_smolvla_w8a8_fp8.py](scripts/run_smolvla_w8a8_fp8.py) | W8A8 与 FP8 边界参考图，配套 [w8a8_fp8_runtime.py](scripts/w8a8_fp8_runtime.py)；内部包含 FP32 参考算术，不代表原生全 FP8 硬件执行。 |
| S8 蒸馏 | [train_smolvla_s8_distill.py](scripts/train_smolvla_s8_distill.py) | 教师初始化、FP32 蒸馏、保存重载、续训及可选 DDP。 |
| S8 启动配置 | [train_smolvla_s8_distill.sh](scripts/train_smolvla_s8_distill.sh)、[train_smolvla_s8_distill_2gpu.sh](scripts/train_smolvla_s8_distill_2gpu.sh) | 原训练环境的单卡 / 双卡启动配置；使用前替换固定解释器、权重、assets 和数据路径。 |

S8 结构、损失和原训练环境验证记录见 [蒸馏说明](scripts/README_smolvla_s8_distillation.md)。其中指向 `outputs/diagnostics/` 的报告不包含在此快照中；原环境的测试记录不等于本次整理已重新运行训练或验证。

## 本地环境与命令示例

以下为 Bash 示例，在主工程根目录执行第一组命令。先安装 uv 并准备 Python 3.12 或更高版本：

```bash
cd lerobot-clean
uv sync --locked --extra smolvla --extra dataset
```

模型、tokenizer / VLM assets 和训练数据需在本地另行准备。将示例中的 `/path/to/...` 替换为实际路径，选择新的输出目录；检查点及其配置中引用的资源路径也应存在。脚本和原有文档保留了原机器的固定路径，跨机器使用前需检查并替换。

导出自定义 W8A8 交换包：

```bash
uv run --extra smolvla --extra dataset python -m scripts.quantize_smolvla_w8a8 \
  --checkpoint /path/to/checkpoint \
  --output outputs/exports/smolvla_w8a8 \
  --calibration-frames 64 \
  --validation-frames 16 \
  --device cuda:0 \
  --linear-activation-scheme per_token
```

导出 W8A8 + FP32 标准 ONNX，并量化语言 Embedding：

```bash
uv run --extra smolvla --extra dataset --with onnx --with onnxruntime \
  python -m scripts.export_smolvla_w8a8_fp32_onnx \
  --checkpoint /path/to/checkpoint \
  --vlm-assets /path/to/vlm_assets \
  --output outputs/onnx/smolvla_w8a8_fp32 \
  --device cuda:0 \
  --quantize-language-embedding
```

`smolvla` extra 未包含这些项目自定义导出脚本所需的全部 ONNX 依赖。`--with onnx --with onnxruntime` 在本次运行环境中临时补充依赖，不修改 `pyproject.toml` 或 `uv.lock`。FP8 参考运行还需要 `ml_dtypes`，可同样临时添加；例如已有 W8A8 + FP32 图及同目录的 `validation_*.npz` 时：

```bash
uv run --with onnx --with ml_dtypes python -m scripts.export_smolvla_w8a8_fp8 \
  --source /path/to/smolvla_w8a8_fp32.onnx \
  --output outputs/onnx/smolvla_w8a8_fp8
```

S8 真实数据两步短测入口如下。它会执行训练、验证及保存重载，需要可用的教师、assets、双相机 wrist / beside 数据和对应计算环境：

```bash
uv run --extra smolvla --extra dataset python -m scripts.train_smolvla_s8_distill \
  --teacher /path/to/teacher_checkpoint \
  --vlm-assets /path/to/vlm_assets \
  --dataset-root /path/to/dataset \
  --output outputs/diagnostics/s8_smoke \
  --device cuda:0 \
  --smoke-test
```

以上命令参数已与脚本的 CLI 定义核对；本次整理未安装依赖、下载资源或执行这些脚本。

## 测试工件、来源与许可

- `tests/artifacts/` 保留 50 个 Git LFS 指针及 3 个生成工件的 Python 源码，没有指针对应的二进制对象。依赖这些工件的测试需先从对应原仓库取得数据。
- 此快照不带 `.git`，不能在这里通过 `git lfs pull` 恢复原仓库的 LFS 对象；应在有正确 remote 和历史的原仓库中获取后再按对应路径准备。
- 上游来源登记在 `pyproject.toml` 的 `[project.urls]`；当前快照的精确上游修订未知。
- 本目录保留的上游许可见 [LICENSE](LICENSE)。主工程许可说明见 [../LICENSE_NOTICE.md](../LICENSE_NOTICE.md)，模型、数据和其他第三方组件各自遵循其原有条款。
