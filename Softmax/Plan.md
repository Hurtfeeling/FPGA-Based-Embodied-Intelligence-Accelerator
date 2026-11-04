# Softmax FP32 输入/输出行 BRAM 四阶段架构计划

更新日期：2026-11-02。状态：已实现 Verilog .v 源码，目标时钟为 250 MHz（4 ns）；公共函数使用 .vh，不依赖 SystemVerilog package/import。行缓存采用 4-bank BRAM 推断结构。当前源码的验证状态以 [README.md](README.md) 为准。RTL 实施细节见 [RTLImplementation.md](RTLImplementation.md)。
## 1. 数据与任务约定

沿每一行 K 维计算稳定 Softmax：

~~~
m[row]   = max_j x[row,j]
z[row,j] = x[row,j] - m[row]
e[row,j] = exp(z[row,j])
S[row]   = sum_j e[row,j]
p[row,j] = e[row,j] / S[row]
~~~

- M：行数，1..65,536。
- K：每行元素数，1..4,096；同一任务各行的 K 相同。
- 输入：已完成 QK 反量化、缩放、temperature 和 mask 处理的 FP32 logits。
- 输出：FP32 概率；后续 PV 所需的 INT8 量化由后续模块处理。
- AXI-Lite32 控制，AXI-full128 数据，每个 beat 含 4 个 FP32。
- 32-bit SRC_ADDR、DST_ADDR 均须 16-byte 对齐，两块矩阵地址范围不重叠。

维度寄存器：

~~~
DIM_MK[15:0]  = M - 1
DIM_MK[31:16] = K - 1
~~~

解码时先扩展至至少 17 bit 再加 1。M 字段 0xffff 表示 65,536；K 字段最大合法值为 0x0fff，表示 4096，超过范围在 START 时拒绝。一条 START 自动处理全部 M 行。

## 2. 两块 BRAM 的职责和容量

这里的“两块行 BRAM”指一块输入、一块输出，总共两块；不是两块输入 BRAM 再另加输出 BRAM。

~~~
W             = ceil(K/4)
W_MAX         = 1024
input_buffer  = 1024 × 128 bit = 16 KiB
output_buffer = 1024 × 128 bit = 16 KiB
行 BRAM 总量 = 32 KiB
~~~

| 存储 | 内容/用途 | 数据容量 |
|---|---|---:|
| input_buffer | 当前行 x，已读位置可装下一行 x | 16 KiB |
| output_buffer | EXP 阶段存 e，NORM 阶段原位改为 p | 16 KiB |
| RX FIFO | 32×128 bit，AXI R 与加载解耦 | 512 B |
| TX FIFO | 32×128 bit，output 读出与 AXI W 解耦 | 512 B |
| 合计 | 行数据和收发 FIFO payload | 33 KiB |

每个 128-bit word 保存连续的 4 列，lane0 对应列 4j，lane3 对应列 4j+3。K=1024 时每块 BRAM 使用前 256 个 word、4 KiB，物理容量仍按最大 K=4096 分配。

两块 BRAM 各自采用同步简单双口 1R1W：一个 128-bit 读口、一个 128-bit 写口。它们必须能独立访问。33 KiB 不包括 exp ROM、浮点流水、tag、DMA 描述符和求和寄存器，物理 BRAM 分配还受器件粒度影响。

`softmax_row_bram`内显式生成4个1024×32-bit bank，每个bank对应一个FP32 lane，并带`ram_style="block"`属性。bank共用地址和使能，每拍合并读/写4个FP32；同步读延迟1拍，读使能无效时保持输出，不复位阵列或读数据。按目标器件每个行缓存预计4个RAMB36，两块共8个；exp ROM保留原有block推断。浅层RX和读返回队列保留LUTRAM/寄存器实现。

## 3. 四阶段数据流

~~~
① LOAD_MAX
   AXI R → RX FIFO → input_buffer
                         └→ 每拍4路比较 → 行 max

② EXP_SUM
   input_buffer → 4路 FP32 (x-max) → 4路 exp
                                         ├→ output_buffer 写 e
                                         └→ 四路指数/尾数对齐 → UQ16.48 → CSA累加 → S
   同时：AXI R → RX FIFO → input_buffer 已回收位置，加载下一行并求其 max

③ NORMALIZE
   output_buffer 读 e → p=e/S → output_buffer 原位写 p

④ STORE
   output_buffer 读 p → TX FIFO → AXI W → DDR
~~~

### 3.1 LOAD_MAX：加载即比较

每次有效握手写入4个FP32，同时生成并寄存4个可排序的IEEE编码key。下一拍分别更新4个lane最大值，随后用两级寄存比较合并为行最大值；每条反馈路径只有一个无符号比较器。输入subnormal先按DAZ处理，负数key取按位反、非负数key把最高位置1，最终再还原FP32。K非4倍数时尾word无效lane按-inf处理，不参与有效值判断。

这条max路径分为key捕获、4路独立累积、两两合并、最终合并4级。最后一个加载beat之后必须排空4级valid，再发布行max；输入加载保持每拍4个元素。

只有本行 W 个 word 全部实际写入 BRAM，且本行 max 比较/归约流水完成后，才发布 INPUT_READY。最后一个 AXI beat 到达和最终 max 有效不一定是同一拍。

必须锁存当前行的 max。预取下一行时使用另一份 max 状态，不能把 EXP 正在使用的 max 覆盖。

### 3.2 EXP_SUM：四路减法、指数和求和

每次从 input_buffer 读 4 个 x，同时计算：

~~~
z0=x0-m  z1=x1-m  z2=x2-m  z3=x3-m
e0=exp(z0) ... e3=exp(z3)
~~~

对于有效有限元素 z<=0。减法输出通过 valid/ready 直接进入 exp 流水，不需要缓存整行差值。

exp 输出分为两个消费者：

- 将 e0..e3 打包为一个 128-bit word，写 output_buffer 对应地址；
- 四路解包指数及24位有效尾数，对齐到64位无符号定点（48位小数），经进位保存压缩树计算S。

两路必须准确消费同一批 e，各一次。若某路停顿，要通过联合握手或独立弹性队列保证不漏写、不重复累加；归一化使用的 S 对应实际保存的 e。

求和使用q=roundEven(e×2^48)，正常FP32对应q=roundEven(M×2^(E-102))。四输入前向压成U/V，与反馈A/B再做4:2压缩，每拍更新两个64位寄存器。行末才做完整加法T=A+B并转回S=FP32(T×2^-48)。入口定点转换分4级，前向4:2压缩占1级，反馈4:2压缩占1级；每拍接收4个exp。行末另做一次64位进位加法：从最后一个输入握手沿到m_sum有效沿相隔6拍，随后通过独立5级sum_to_fp32转换模块。反馈两层3:2压缩在单拍内完成。K≤4096时T≤2^60，至少61位，统一采用64位。入口定点化的总和绝对误差≤K×2^-49，不包含exp近似及最终FP32舍入。output BRAM继续保存FP32 exp。

只有全部 e 写入 output 且求和归约完成，S 才有效。最后一个 input 读请求发出，不等于 EXP_SUM 完成。

### 3.3 NORMALIZE：读取 e 并原位写 p

数学运算为：

~~~
p[i] = e[i] / S
~~~

硬件每行只求一次r=1/S，再用8级四路FP32乘法流水计算p=e*r，每拍接收4个元素。一般倒数执行48次恢复除法迭代，再分别比较余数、RNE舍入和打包，共3拍收尾；2的整数次幂走1级精确旁路。若要求与逐元素 FP32 除法逐位一致，需要直接除法或相应数值规范；一次倒数再乘法与直接除法可能有舍入差异。

本阶段占 output_buffer 的读口和写口。每个旧 e 先被安全读出，运算结果随后携带原地址写回 p。读写地址、row_id、valid、有效 lane 随流水对齐，不重复读取已覆盖的位置。

只有全部 W 个 p 写回后，才将整行标记为 OUTPUT_READY，进入本行 STORE。当前计划不在同一行 NORM 尚未完成时抢占 output 读口提前 STORE。

### 3.4 STORE：读概率并写 DDR

从 output_buffer 按地址顺序读取 p，通过 TX FIFO 和 AXI W 写入 DST 对应行。尾 word 的无效 lane 写 FP32 +0。

数据被 TX FIFO/写通路可靠接管后，可释放对应 BRAM 地址；任务完成仍需等待相关 W 发送和 B 响应。DDR 停顿时须为已经发出的 BRAM 读请求保留返回空间。

## 4. EXP 同时预取下一行：input 逐字回收

input_buffer 在 EXP_SUM 期间，读口读取当前行 x，写口可以加载下一行。新数据只能覆盖已安全读出并被减法/exp 前级可靠接管的位置：

~~~
input_read_safe_count = 当前行已安全读出的连续 word 数
next_load_count       = 下一行已实际写入的连续 word 数

允许下一行写：next_load_count < input_read_safe_count
0 <= next_load_count <= input_read_safe_count <= W
~~~

发出 BRAM 读请求不代表数据已经安全；要等同步读返回、下游有保证地保存数据后才增加 safe_count。使用上拍已确认的计数准许新写，避免同周期同地址读写。W=1024 时计数至少 11 bit。

若 BRAM 写口前增加多级请求流水，还需单独记录已预留写地址与已提交计数，保证不重复分配地址。

当前行 EXP 的 max/row_id 与下一行 LOAD 的 max/row_id 分离。input 只能预取一行；下一行装满后等待它开始 EXP，不能继续把再下一行覆盖上去。若预取未及时完成，可以在当前行 NORM/STORE 期间继续加载。

## 5. output 端口约束与调度

| 当前 output 用途 | 读口 | 写口 |
|---|---|---|
| EXP_SUM | 空闲 | 写 e |
| NORMALIZE | 读 e | 原位写 p |
| STORE | 读 p | 基础调度下空闲 |
| STORE 旧行 + EXP 新行（可选重叠） | 旧行 p | 新行 e，仅写已回收位置 |

NORM 已占用 output 写口，不能同时再让下一行 EXP 满速写同一块 BRAM；NORM 与 STORE 也需要同一个读口，当前计划分阶段运行。仅把 BRAM 改称“双端口”并不能消除这些冲突。

### 5.1 基础调度：输入预取重叠

L=加载求max，E=指数求和，N=归一化写回 BRAM，W=输出到DDR：

| 时段 | input_buffer | output_buffer |
|---|---|---|
| 0 | L 行0 | 空闲 |
| 1 | E 行0读取，同时逐字 L 行1 | E 行0写 e |
| 2 | 行1等待，或继续完成 L 行1 | N 行0：e→p |
| 3 | 行1等待，或继续完成 L 行1 | W 行0 |
| 4 | E 行1读取，同时逐字 L 行2 | E 行1写 e |
| 后续 | 重复 | N 行1，再 W 行1 |

基础调度先等旧行 STORE 完成，再开始下一行 EXP；实现时可先以相关 B 响应也返回作为保守切换条件。它已经满足“当前行 EXP 时预取下一行”，每行逻辑仍为四阶段。

### 5.2 默认重叠：STORE 旧行，同时 EXP 下一行

默认OVERLAP_STORE_EXP=1，在第四步用output的空闲写口接收下一行e。前提是：旧行 NORM 已全部完成、下一行 input/max 已就绪，并对 output 也采用逐字回收：

~~~
output_store_safe_count = 旧行 p 已被 TX/写通路可靠接管的连续 word 数
next_exp_write_count    = 新行 e 已写入 output 的连续 word 数

允许新行 e 写入：next_exp_write_count < output_store_safe_count
~~~

这样能同时工作：

~~~
input 读口：EXP 行1
input 写口：逐字 LOAD 行2
output 读口：STORE 行0
output 写口：逐字写入行1的 e
~~~

如果 DDR 写入停顿，output 可回收空间耗尽，须暂停新行 e 的提交，并让 exp/减法/input 读/下一行 LOAD 按背压依次停止。对不能停顿的浮点 IP，必须在发射时为全部在途结果预留空间或安全地址，不能只在结果到达后才发现 output 无处可写。

旧行 p 全部安全离开 output、新行 e 全部写入且 sum 完成后，才能将 output 交给新行 NORM。旧行可能仍有 TX/W/B 待完成，保留独立输出描述符，直到事务结束。

默认重叠时序：

~~~
L0
E0 + L1
N0
W0 + E1 + L2
N1
W1 + E2 + L3
N2
...
最后一行 W
~~~

这是在每行四阶段不变的基础上做行间重叠。NORM 阶段仍单独占用 output，不能沿用三阶段架构每 W 拍完成一行的估算。

## 6. exp 路径与附加资源

exp 接收 z=x-max，内部可令 d=-z，再做范围归约与 LUT 插值：

~~~
t = d × log2(e)
n = floor(t)
f = t - n
j = floor(1024 × f)
a = 1024 × f - j
base[j] = roundEven(2^(-j/1024) × 2^30)
drop[j] = base[j] - base[j+1]
b = base[j] - roundEven(a × drop[j])
e = FP32(b × 2^(-30-n))
~~~

当前RTL采用1024段整数ROM，源文件中base/drop各以32位存储，段内插值保留13位。FP32减法和乘log2(e)后，使用定点插值再转FP32。每份配对表1024×64 bit，即8 KiB；两份双读表供4个lane并行读取。历史综合ROM占4个RAMB36，工具会去掉常量高位；行BRAM另占8个RAMB36。

采用两份双读 ROM 时，行 BRAM+RX/TX FIFO+exp ROM 的已列 payload 共约49 KiB；浮点流水、求和、tag、控制寄存器及 BRAM 物理粒度另计。

乘法已经映射DSP：4路log2(e)乘法共8个DSP，4路插值共4个DSP，4路归一化共8个DSP，启动时矩阵尺寸乘法占1个DSP，总21个。FP32的24×24尾数乘法拆为24×17和24×7部分积，分别寄存后再合并；插值drop最大726572，仅需20位，20×13乘法每路占1个DSP。2^(-n) 可调整 FP32 指数，但必须处理零、下溢、边界舍入及极端差值，不能直接截断/环绕指数位。极端有限数相减也可能得到无穷，要在转换整数索引前处理范围。

模块约定-inf为mask，e/p=0；NaN/+inf、全mask行报错。尾部无效lane不参与max/sum。可综合FP32逻辑采用RNE，输入subnormal按DAZ、舍入后的subnormal结果按FTZ处理，max比较使用一致的零语义。倒数每行一次采用迭代除法，归一化四路FP32乘法；误差与吞吐由参考模型和仿真验证。

### 6.1 250 MHz流水划分

| 路径 | 寄存级数/处理方式 | 吞吐 |
|---|---|---|
| 加载求max | key捕获、4路独立最大值、两级合并，共4级 | 每拍4元素 |
| FP32减法 | 捕获、解包、幅值选择、对阶、加减、前导1、规格化、下溢、RNE、打包，共10级 | II=1 |
| 乘log2(e) | 捕获、解包、DSP部分积、合并、规格化、下溢、RNE、打包，共8级 | II=1 |
| exp LUT后端 | 移位解码、定点转换、ROM、ROM输出寄存、插值乘法、drop舍入、减法、RNE、打包，共9级 | II=1 |
| 完整softmax_exp4 | 10+8+9=27级，不含上游BRAM读返回与隔离寄存器 | 每拍4元素 |
| 求和 | 定点转换4级、前向CSA、反馈CSA、行尾进位加法 | 每拍4元素 |
| 行和转FP32 | 5级 | 每行一次 |
| 倒数 | 一般48次迭代+3拍收尾；2的整数次幂1级旁路 | 每行一次 |
| 归一化 | 8级FP32乘法 | 每拍4元素 |

这里的级数描述无背压时的寄存流水深度；握手停顿、BRAM读返回、状态切换和DMA响应另计。流水加深增加首结果和行尾延迟，不降低连续输入时每拍4个元素的吞吐。valid、keep、tag与数据同步，输出背压时保持稳定。

## 7. 周期估算

令 W=ceil(K/4)。以下仅假设四路数据通路均能每拍处理一组，忽略 FP 流水尾部、max/sum归约、倒数、逐字回收启动滞后和 AXI 停顿：

| 指标 | 基础调度 | 可选 STORE/EXP 重叠 |
|---|---:|---:|
| 首行完整写出 | 约4W拍 | 约4W拍 |
| 稳态完成一行的间隔 | 约3W拍 | 约2W拍 |
| M行矩阵主体周期 | 约(3M+1)W | 约(2M+2)W |
| K=4096，W=1024 | 稳态约3072拍/行 | 稳态约2048拍/行 |
| K=1024，W=256 | 稳态约768拍/行 | 稳态约512拍/行 |

这些是理想主体估算，不是实测性能。每行外部读写仍各一次，但 output 的额外原位归一化与独立 STORE 带来三次扫描：EXP写、NORM读写、STORE读。每行最大外部流量为16 KiB读+16 KiB写。

250 MHz 下每拍处理 4 个 FP32，对应单方向接口数据率 4 GB/s、流水入口吞吐 10 亿元素/秒；这是流水线入口能力，不代表完整 Softmax 吞吐。M 行矩阵主体周期按表中的基础调度约 (3M+1)W，启用 STORE/EXP 重叠时约 (2M+2)W。实际总时延还包括两次计算扫描、行尾流水排空与归约、倒数、共享 DDR 带宽及 DMA 停顿。## 8. DDR 布局与 AXI 控制

~~~
W            = ceil(K/4)
ROW_BYTES    = 16×W
row_addr     = BASE + row_index×ROW_BYTES
MATRIX_BYTES = M×ROW_BYTES
~~~

K 非4倍数时每行补齐到16字节，输入无效 lane 忽略，输出填+0。软件需按该行跨度准备数据。M最大且K=4096时单矩阵1 GiB，地址计算使用足够宽的中间结果检查32-bit地址溢出和输入输出范围重叠。

START后立即置BUSY，先用4拍完成矩阵word数乘法、字节数形成、33位范围端点加法以及合法性判定，通过后才调度DMA。读写下一行地址分别按ROW_BYTES递增，避免把row_index乘法放在每行调度路径。配置非法时结束BUSY并置ERROR_CODE=1。

内部按行生成DMA描述符；burst须遵守最大长度和4 KiB边界。R/W通道独立调度，软件只启动一次。若DMA不能支持预期的背压，发请求前必须为相应burst预留容量。

保持四个AXI-Lite32寄存器：

| 偏移 | 内容 |
|---:|---|
| 0x00 | ENABLE bit0；读 BUSY bit1、DONE bit2、ERROR bit3、ERROR_CODE bits7:4；写 START bit8、CLEAR bit9 |
| 0x04 | SRC_ADDR |
| 0x08 | DST_ADDR |
| 0x0C | DIM_MK，低16位 M-1，高16位 K-1 |

软件配置地址、维度，最后写 CTRL_STATUS=0x101。BUSY期间锁定配置，不允许再次START/CLEAR；忙时修改配置、再次START/CLEAR及非法寄存器访问返回SLVERR。

## 9. 阶段完成与异常收尾

每行状态依次为：

~~~
INPUT_READY → EXP_SUM → SUM_READY → NORMALIZE → OUTPUT_READY → STORE
~~~

LOAD引擎可与计算独立工作；如果启用STORE/EXP重叠，保留旧行STORE与新行EXP两套上下文，不能用一个row_id代表两者。

阶段切换检查实际写提交、读返回、max流水排空和归约valid；流水尾部延迟不能替代有效握手。同步BRAM访问通过地址和水位约束避免同周期同地址冲突，不依赖未定义read-during-write行为。

DONE须等全部M行输出、所有读事务、W beat、B响应及内部流水完成。数据离开BRAM允许提前复用，但不代表任务完成。发生错误停止新行调度并收尾已接受的AXI事务；输出可能部分更新，ERROR时软件不得使用结果。

## 10. 后续验证重点

RTL提供OVERLAP_STORE_EXP参数，默认开启重叠，参数0用于基础调度对照。250 MHz改造后的官方算术单元、21个端到端用例和完整1024×1024矩阵回归均已通过。验证重点包括：

- K=1、非4倍数、1024、4096，以及M=1/2/3的启动和收尾。
- 当前行max已冻结，预取下一行不会覆盖它。
- exp存储与求和分支对同一数据各消费一次。
- 等待max/sum/倒数及最后p写回，不提前切换阶段。
- input覆盖仅发生在旧x安全读出后；output覆盖仅发生在旧p安全离开后。
- DDR停顿、FP流水停顿、两个安全水位和在途credit正确传播。
- output在NORM和STORE之间不混淆e/p，旧行地址和B响应不因复用丢失。
- 参考模型核对LUT误差、FTZ、归约舍入，以及倒数乘法与直接除法的差异。

当前 BRAM 分 bank 版本需在目标 Vivado 环境重新运行模块级仿真、综合和布局布线；模块级时序不能代替外部 AXI、完整系统和板端验证。