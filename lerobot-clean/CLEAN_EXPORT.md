# 干净代码目录说明

此目录是原仓库当前工作区的代码快照，保留已跟踪文件和新增的源码、测试、文档，以及原有相对符号链接。

- 未包含 `.git`、本地模型、训练数据、`outputs/`、根目录 `artifacts/`、日志、环境和缓存。
- 根目录 `.gitignore` 补充了 `/artifacts/`、`/datasets/`、`/models/`、`/checkpoints/`。
- `tests/artifacts/` 保留上游测试源码和 Git LFS 指针；没有打包指针对应的二进制测试数据。
- 原目录没有改动，也没有执行 Git 暂存、提交或发布。
- 除本说明与忽略规则外，源码保持原样；脚本内已有的本机路径、可选依赖声明问题尚未改写。
- 仅验证文件复制完整性和本次改动 Python 文件的语法；未运行训练或测试套件。

- 本工程已将同名包装目录压平，源码入口现位于 `lerobot-clean/`；项目专用说明见 [README_PROJECT.md](README_PROJECT.md)。
