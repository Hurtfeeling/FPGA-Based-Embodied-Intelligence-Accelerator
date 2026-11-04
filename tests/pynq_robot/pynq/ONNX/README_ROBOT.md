# 摄像头、电机与 SmolVLA 联动

入口为 `run_robot.py`，复用已经测试过的 C++ ONNX Runtime 桥接及 GEMM 驱动。
同一进程只加载一次模型/session；两路相机后台采集最新帧，主线程读关节状态、推理、记录或下发动作。
不安装完整 LeRobot/PyTorch；适配板端 Python 3.6，依赖现有 NumPy、OpenCV、pyserial。
Feetech 官方 SDK 源码固定版本附在 `robot_io/vendor`，避免与同名第三方 SDK 混用。

## 已确认的设备

| 模型输入 | 物理位置 | 当前节点 |
|---|---|---|
| camera2 | 腕部，用户已确认 | /dev/video0，icSpring |
| camera3 | 侧边，用户已确认 | /dev/video2，USB Camera |
| 电机 USB 串口 | 尚需核对舵机身份/训练校准 | /dev/ttyACM0 |

`robot_config.json` 使用 `/dev/v4l/by-id/` 和 `/dev/serial/by-id/` 的稳定路径，避免插拔后节点编号变化。
相机帧为 640×480；BGR→RGB、除以255、HWC→NCHW 后提供 `[1,3,480,640]`。
图内已包含512等比缩放/补边及[-1,1]转换，外部不再做 ImageNet mean/std。

## 当前还需填入的配置

运行真实状态推理前，需要训练时对应 follower 的校准 JSON，设置 `calibration` 路径，
确认六维关节顺序、ID、单位，填写 `motors.norm_modes`。
本模型导出文件未保存训练数据集的关节 feature names 和 `use_degrees`；
不能仅凭当前 LeRobot 默认值推断是角度还是[-100,100]校准空间。
可用模式字符串是 `range_m100_100`、`range_0_100`、`degrees`，必须逐关节明确填写。
配置中的标准 SO-101 顺序/ID 是待核对的模板，不代表已经验证训练对应关系。

校准 JSON 为 LeRobot follower 格式：每个关节包含 `id`、`drive_mode`、`homing_offset`、
`range_min`、`range_max`。使用原 follower 校准，不套用 leader 的文件。
脚本只读取并核对舵机已有校准，不自动校准、不写 ID/波特率/零偏/EEPROM。
舵机已应用 homing_offset，读取位置后不能在软件里再次减去零偏。

动作执行另外需填写 `control` 与 `motors` 中的 min/max/max_step，
`control.max_velocity`（每秒的训练动作单位）、`control.max_state_drift`。
确认后将 `training_mapping_confirmed` 设为 true，再显式传 `--execute`。
不传该选项时，只读状态并保存预测，不写目标或使能电机。

## 先检查设备

```bash
cd /path/to/pynq/ONNX

# 只拍照与读取舵机身份、位置及配置，不写电机寄存器
python3 inspect_robot.py --cameras --motors --output-dir /dev/shm/robot_inventory_new
```

检查结果和两张图片在输出目录。读取原始舵机身份不需要校准文件。

## 模型预处理资源

`robot_assets/stats.json` 从原 safetensors 精确导出，不需要板上安装 safetensors/torch。
状态公式为 `(raw_state - mean) / (std + 1e-8)`；输出还原为 `action * std + mean`。
`robot_assets/language.npz` 为随包验证任务的预分词结果：

```text
only grab the blue cube and put into the box
```

保持末尾换行。板上无需 tokenizer。任务改变时，在有 tokenizers 的本机执行：

```bash
python3 prepare_robot_assets.py --help
```

该工具从原模型目录读取 safetensors 与 tokenizer.json，生成新的 stats/language 资源；
也支持从 validation_0.npz 提取既有语言 tokens。不要把旧任务 tokens 当作新的文字指令。
例如在本机执行：

```bash
python3 prepare_robot_assets.py --model-dir /path/to/200000_w8a8_fp32 \
  --task "only grab the blue cube and put into the box" --output-dir robot_assets
```
noise 每轮使用 NumPy 标准正态 `[1,50,32]`；同种子不保证与 PyTorch 噪声逐值一致。

## 推理与联动

先按根 README 加载配套 overlay，模型仍引用原目录，不复制大型权重。
如果缺校准文件，可先用完整已有输入验证新入口，不接触相机或串口：

```bash
sudo bash run_robot.sh --mode hybrid --replay-inputs validation_0.npz --cycles 1
```

校准及单位填写完成后，接入实时双相机和电机反馈：

```bash
# 只读取真实状态，推理并保存动作；无电机写操作
sudo bash run_robot.sh --cycles 1

# 核对配置和预测结果后，下发受限动作
sudo bash run_robot.sh --cycles 1 --execute
```

CPU 对照使用 `--mode cpu`；PL 使用 `--mode hybrid`，保留大矩阵 PL GEMM + 原生 CPU Softmax。
输入、归一化动作、还原动作、推理时间和每条下发记录在 `robot_runs/时间_模式/`。

当前板端稳态整图约34秒，50步动作块不是50次新视觉观测。程序采用“拍摄/读状态→等待推理→
有限步动作→重新观察”的流程，没有不断增长的过期任务队列。默认每轮只执行1步；
`command_hz` 是一个动作块内的发送节拍，不是视觉闭环频率。
`max_observation_age_s=60` 只是适配当前推理延迟的上限，需按实际任务调整；
同时检查推理期间关节漂移，过期预测或反馈异常直接终止发送。

可用 Ctrl+C 或在 ONNX 目录创建 `STOP_ROBOT` 停止发送；删除该文件后才能再次运行。
ONNX 在原生调用内运行时，信号处理可能等到当前推理返回；推理期间不下发新动作。
默认 `release_on_close=false` 保持舵机原有保持状态，避免机械臂直接掉落；
设为 true 时仅关闭本程序自己使能的扭矩。软件停止不替代机械臂的硬件急停。

## 官方来源

- [LeRobot](https://github.com/huggingface/lerobot)
- [LeRobot SO follower](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
- [Feetech 官方 Python SDK](https://github.com/ftservo/FTServo_Python)

实际 SDK 固定 commit、许可及文件清单见 `robot_io/vendor/`。本包没有运行任何电机动作。

本地模拟测试：`python3 -m unittest discover -s tests -v`，不访问真实摄像头、串口或 FPGA。
测试验证协议调用顺序、只读零写入、归一化、最终发送限位/限速、超时、停止和资源生命周期；
不能替代机械臂上的标定和联调。
