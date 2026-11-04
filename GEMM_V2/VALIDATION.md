# GEMM_V2 验证与复现

本文件说明如何从当前 RTL 源码重新运行验证。测试结果输出到本机的 `tb/build/`，生成文件不作为历史结果提交。

## 功能回归

从仓库根目录运行：

```powershell
python GEMM_V2/tb/run_regression.py --iverilog iverilog --vvp vvp --jobs 2
```

可用 `--test <name>` 选择单个用例，并用 `--build-dir <path>` 指定本地输出目录。测试覆盖配置/寄存器协议、DMA 边界与背压、输入缓存、脉动阵列、蛇形规划、C 结果队列、错误恢复和端到端输出比较。具体用例由当前 `run_regression.py` 的清单决定。

## 性能测量

```powershell
python GEMM_V2/tb/run_performance_compare.py --backend verilator --shape 288,288,2048 --shape 2048,512,2048 --jobs 2 --make-jobs 4 --build-dir GEMM_V2/tb/build/performance_verilator
```

性能仿真使用指定 DDR 服务模型，不代表实体 DDR 的时序或 PYNQ 端到端耗时。发布结果时应记录被测 RTL 的 Git commit、输入矩阵形状、AXI 模型、时钟、工具版本、结果校验与计时边界。

## 综合与接口检查

```powershell
vivado -mode batch -source GEMM_V2/scripts/check_interfaces.tcl
vivado -mode batch -source GEMM_V2/scripts/synth.tcl
vivado -mode batch -source GEMM_V2/scripts/synth_memories.tcl
```

这些脚本生成本地接口、资源和时序报告。OOC 综合不等于完整系统布局布线或板端频率验证。当前整机实现报告保存在 `results/implementation/system/`。
