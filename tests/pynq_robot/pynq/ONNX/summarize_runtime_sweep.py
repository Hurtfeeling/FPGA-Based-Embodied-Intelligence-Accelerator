#!/usr/bin/env python3
"""Render the fetched PYNQ sweep JSON as a Chinese Markdown report locally.

This script uses only the Python standard library and never connects to PYNQ.
An incomplete sweep is rendered explicitly as a partial report.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE / 'results' / 'cma_softmax_sweep_v2_20261104.json'
DEFAULT_OUTPUT = BASE / 'CMA_SOFTMAX_PARALLEL_RESULTS_20261104.md'


def number(value, digits=3):
    if value is None:
        return '未记录'
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return ('{:.' + str(digits) + 'f}').format(value)


def scientific(value):
    return '未记录' if value is None else '{:.7g}'.format(float(value))


def median(values):
    values = [float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def maximum(values):
    values = [float(v) for v in values if v is not None]
    return max(values) if values else None


def escaped(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def shape_text(shape):
    return '[' + ','.join(str(v) for v in shape) + ']'


def table(headers, rows):
    lines = ['| ' + ' | '.join(map(escaped, headers)) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines.extend('| ' + ' | '.join(map(escaped, row)) + ' |' for row in rows)
    return '\n'.join(lines)


def router_sum(record, field):
    return sum(float(router.get(field, 0.0)) for router in record.get('routers', {}).values())


def read_optional(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def overlap_audit(report):
    """Recompute interval intersections independently of saved aggregates."""
    checked = 0
    interval_error = aggregate_error = 0.0
    for record in report.get('records', []):
        for router in record.get('routers', {}).values():
            intervals = router.get('interval_records', [])
            if not intervals:
                continue
            total = 0.0
            for item in intervals:
                cpu = item.get('cpu_interval')
                pl = item.get('pl_path_interval')
                overlap = (max(0.0, min(cpu[1], pl[1]) - max(cpu[0], pl[0]))
                           if cpu and pl else 0.0)
                interval_error = max(interval_error,
                    abs(overlap - float(item.get('overlap_seconds', 0.0))))
                total += overlap
                checked += 1
            aggregate_error = max(aggregate_error,
                abs(total - float(router.get('overlap_seconds', 0.0))))
    return checked, interval_error, aggregate_error


def config_results(report):
    configurations = {row['name']: dict(row) for row in report.get('configurations', [])}
    for row in report.get('summaries', []):
        config = row.get('config', {})
        if isinstance(config, dict) and config.get('name'):
            configurations.setdefault(config['name'], dict(config))
    for record in report.get('records', []):
        configurations.setdefault(record['config'], dict(name=record['config']))
    results = []
    for name, config in configurations.items():
        rounds = [r for r in report.get('records', []) if r.get('config') == name]
        measured = [r for r in rounds if r.get('phase') == 'measure']
        if not measured:
            results.append(dict(config=config, measured=[], warmups=sum(
                r.get('phase') == 'warmup' for r in rounds)))
            continue
        results.append(dict(config=config, measured=measured,
            warmups=sum(r.get('phase') == 'warmup' for r in rounds),
            seconds=median(r.get('seconds') for r in measured),
            softmax=median(r.get('softmax_seconds') for r in measured),
            gemm=median(r.get('operators', {}).get('gemm_seconds') for r in measured),
            mae=maximum(r.get('vs_board_cpu', {}).get('mae') for r in measured),
            max_abs=maximum(r.get('vs_board_cpu', {}).get('max_abs') for r in measured),
            overlap=median(router_sum(r, 'overlap_seconds') for r in measured),
            overlap_per_call_ms=median((router_sum(r, 'overlap_seconds') * 1000 /
                max(1, router_sum(r, 'calls'))) for r in measured),
            overlap_per_mixed_ms=median((router_sum(r, 'overlap_seconds') * 1000 /
                max(1, router_sum(r, 'mixed_calls'))) for r in measured),
            mixed_calls=median(router_sum(r, 'mixed_calls') for r in measured),
            cache_hits=median(r.get('operators', {}).get('resident_weight_cache_hits') for r in measured),
            cache_fills=median(r.get('operators', {}).get('resident_weight_cache_fills') for r in measured),
            cache_copy_mib=median((r.get('operators', {}).get('resident_weight_cache_bytes_copied', 0)
                                   / 1024**2) for r in measured),
            pl_tasks=median(r.get('operators', {}).get('softmax_tasks', 0) for r in measured),
            cma_free_mib=median((r.get('cma', {}).get('CmaFree_kib', 0) / 1024)
                                for r in measured)))
    return results


def historical_baselines(source, input_hash):
    rows = []
    for filename, label in [('cpu.json', 'CPU-only 历史基线'),
                            ('hybrid_resident64.json', 'GEMM-only + 64 MiB 常驻缓存历史基线')]:
        path = source.parent / filename
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if input_hash and value.get('inputs_sha256') != input_hash:
            continue
        rows.append((label, value.get('median_inference_seconds'), filename))
    return rows


def native_sections(native, source, current_results, input_hash):
    sections = ['## 原图 CPU Softmax 与常驻缓存的独立扫描']
    if native is None:
        sections.append('尚未取得 `results/cma_native_sweep_20261104.json`；'
                        '原图 CPU Softmax 扫描不参与本报告的最终最快结论。')
        return sections
    complete = native.get('status') == 'complete'
    same_input = bool(input_hash and native.get('inputs_sha256') == input_hash)
    sections.append('此扫描状态为 `{}`，{}。模型保持普通 ONNX CPU Softmax 节点，'
        '只将选定 GEMM 节点交给 PL，64/80/88 MiB 配置共用一次 GEMM-only 图加载。'
        'session 加载耗时 **{} s**，输入哈希与 Softmax 路由扫描{}。'.format(
            native.get('status', 'unknown'),
            '以下是完整结果' if complete else '**结果尚未完成，下表仅为已记录样本，不能作为最终比较**',
            number(native.get('session_load_seconds')), '一致' if same_input else '**不一致或尚未记录**'))
    sections.append('原图扫描使用 `allow_spinning=1`，前面的 Softmax 路由扫描使用 '
        '`allow_spinning=0`。同时还改变了 Softmax 是否经过 Python 回调与独立 ORT 小 session。'
        '**两组之间不是只改变缓存容量的受控实验**；跨组速度差不能全部归因于扩大 CMA 缓存。'
        '同一次原图扫描内部的 64/80/88 MiB 比较更适合判断缓存增量的效果。'
        'native 中记录的 `softmax_seconds=0` 表示没有 Softmax 回调计时，'
        '不是 Softmax 计算耗时为零。')
    configs = config_results(native)
    rows = []
    for item in configs:
        config = item['config']
        cache = config.get('cache_mib')
        scratch = 28 if config.get('shared') else 44
        rows.append([config['name'], '{}/{}'.format(item['warmups'], len(item['measured'])),
            ', '.join(number(r['seconds']) for r in item['measured']) or '尚未测量',
            number(item.get('seconds')), number(item.get('gemm')),
            str(cache) if cache is not None else '未记录',
            str(scratch + cache) if cache is not None else '未记录',
            number(item.get('cma_free_mib')), number(item.get('cache_hits'), 0),
            scientific(item.get('mae')), scientific(item.get('max_abs'))])
    sections.append(table(['配置', 'warmup/正式轮', '逐轮延时 s', '总中位 s', 'GEMM 路径中位 s',
                          '权重 MiB', 'CMA 申请 MiB', '空闲 MiB', '命中/轮', '动作 MAE', '动作 max abs'], rows))
    eligible = [r for r in configs if r['measured'] and r.get('mae') == 0 and r.get('max_abs') == 0]
    if eligible:
        winner = min(eligible, key=lambda r: r['seconds'])
        sections.append('{}原图扫描内逐元素一致的最快配置是 `{}`，中位数 **{} s**。'.format(
            '' if complete else '截至当前已记录样本，', winner['config']['name'], number(winner['seconds'])))
        if complete:
            sections.append('每组只有三次正式测量，且逐轮时间有波动；这里的“最快”指已测中位数最低，'
                '尚未证明小于一秒的组间差异在所有输入和长期运行中都稳定成立。'
                '缓存增大后 GEMM 子路径缩短，也不保证完整 CPU 图的墙钟时间同步缩短。')
            config_file = 'runtime_cma{}_native_softmax.json'.format(winner['config']['cache_mib'])
            if (source.parent.parent / config_file).is_file():
                sections.append('复用本次最低中位延时配置：`{}`。使用 **GEMM-only 的 '
                    '`smolvla_selective_pynq_resident88.onnx`**（按配置截取常驻槽），'
                    '不使用带 `_softmax` 的实验图。板端模型目录下可执行：'.format(config_file))
                sections.append('```bash\n'
                    'sudo env SMOL_ORT_ALLOW_SPINNING=1 SMOL_ORT_EXECUTION_MODE=sequential \\\n'
                    '  SMOL_ORT_INTRA_OP_THREADS=4 SMOL_ORT_INTER_OP_THREADS=1 \\\n'
                    '  SMOL_ORT_GRAPH_OPT=disable SMOL_ORT_CPU_ARENA=0 SMOL_ORT_MEMORY_PATTERN=0 \\\n'
                    '  OMP_WAIT_POLICY=PASSIVE taskset -c 0,1,2,3 python3 infer_board.py \\\n'
                    '  --mode hybrid --model smolvla_selective_pynq_resident88.onnx \\\n'
                    '  --inputs validation_0.npz --cpu-baseline cpu.npy \\\n'
                    '  --runtime-root .. --library ./libsmolvla_ort_sweep.so \\\n'
                    '  --runtime-config {} --warmups 1 --repeats 3 \\\n'
                    '  --output-prefix native_cache_verify\n```'.format(config_file))
                sections.append('本轮完整推理未启用 PS DMA，复制、UINT8→INT8 转换、缓存维护均已计入延时。'
                    '配置与报告已同步到板端；原始模型和原有默认配置保持独立，未覆盖。')
    # Include the earlier exact result so a slower new experiment is never
    # called the global best merely because it is the most recent one.
    candidates = [(r['seconds'], 'Softmax 路由扫描/' + r['config']['name']) for r in current_results
                  if r['measured'] and r.get('mae') == 0 and r.get('max_abs') == 0]
    if complete and same_input:
        candidates.extend((r['seconds'], '原图扫描/' + r['config']['name']) for r in eligible)
    history_path = source.parent / 'hybrid_resident64.json'
    historical = read_optional(history_path)
    if historical and historical.get('inputs_sha256') == input_hash:
        measured = [r for r in historical.get('records', []) if r.get('phase') == 'measure']
        if measured and all(r.get('vs_board_cpu', {}).get('mae') == 0 and
                            r.get('vs_board_cpu', {}).get('max_abs') == 0 for r in measured):
            seconds = historical.get('median_inference_seconds')
            if seconds is not None:
                candidates.append((seconds, '此前 GEMM-only + 64 MiB 常驻缓存'))
    if candidates:
        seconds, label = min(candidates)
        sections.append('在同一输入且已完成的这些实验与历史基线中，'
                        '**全局已测最低且逐元素一致**的是「{}」，中位数 **{} s**。'
                        '这里比较已测配置的结果，不表示跨组单变量因果结论。'.format(label, number(seconds)))
    if native.get('error'):
        sections.append('原图扫描错误：\n\n```text\n{}\n```'.format(native['error']))
    return sections


def render_report(report, source):
    results = config_results(report)
    measured = [r for r in results if r['measured']]
    complete = report.get('status') == 'complete'
    fastest = min(measured, key=lambda r: r['seconds']) if measured else None
    exact = [r for r in measured if r.get('mae') == 0 and r.get('max_abs') == 0]
    best_exact = min(exact, key=lambda r: r['seconds']) if exact else None
    sections = ['# CMA、Softmax 并行与整图延时测试（2026-11-04）']
    native = read_optional(source.parent / 'cma_native_sweep_20261104.json')
    input_hash = report.get('inputs_sha256')
    exact_candidates = [(r['seconds'], r['config']['name']) for r in exact] if complete else []
    if (native and native.get('status') == 'complete' and input_hash and
            native.get('inputs_sha256') == input_hash):
        exact_candidates.extend((r['seconds'], r['config']['name'])
            for r in config_results(native) if r['measured'] and r.get('mae') == 0 and r.get('max_abs') == 0)
    historical = read_optional(source.parent / 'hybrid_resident64.json')
    if historical and historical.get('status') == 'complete' and historical.get('inputs_sha256') == input_hash:
        rounds = [r for r in historical.get('records', []) if r.get('phase') == 'measure']
        if rounds and all(r.get('vs_board_cpu', {}).get('mae') == 0 and
                          r.get('vs_board_cpu', {}).get('max_abs') == 0 for r in rounds):
            exact_candidates.append((historical['median_inference_seconds'], '历史 GEMM-only + 64 MiB'))
    if exact_candidates:
        seconds, label = min(exact_candidates)
        sections.append('同一验证输入下，已完成测试中与 CPU 动作输出逐元素一致的最低中位延时为 '
                        '**{} s**，配置为 `{}`。以下分别列出原图 CPU Softmax、实验性 '
                        'CPU/PL Softmax、卷积与对齐的证据。'.format(number(seconds), label))
    if not complete:
        sections.append('**本报告为未完成的中间结果。** 原始状态为 `{}`；'
                        '尚未运行完的配置不参与最终结论，当前最快值仅供进度参考。'.format(
                            escaped(report.get('status', 'unknown'))))
    if fastest:
        sections.append('当前 Softmax 路由扫描{}整图最快配置为 `{}`，中位数 **{} s**，'
            '相对 CPU 动作输出的最大绝对误差为 `{}`。{}'.format(
                '已完成测试中的' if complete else '已记录样本中的',
                fastest['config']['name'], number(fastest['seconds']),
                scientific(fastest['max_abs']),
                ('与 CPU 输出逐元素一致的最快配置为 `{}`，中位数 **{} s**。'.format(
                    best_exact['config']['name'], number(best_exact['seconds']))
                 if best_exact else '当前已记录配置尚无与 CPU 输出逐元素一致的结果。')))
    else:
        sections.append('尚未取得完整模型的正式推理样本；下列内容仅描述测试方法与已记录的微测试。')

    sections.append('## 条件与计时范围')
    clock_rows = [report.get('clock', {})] + [r.get('clock', {}) for r in report.get('records', [])]
    cpu_clocks = [r.get('cpu_mhz') for r in clock_rows if r.get('cpu_mhz') is not None]
    pl_clocks = [r.get('pl_mhz') for r in clock_rows if r.get('pl_mhz') is not None]
    def clock_range(values):
        return ('未记录' if not values else '{}–{} MHz'.format(
            number(min(values), 4), number(max(values), 4)))
    sections.append('CPU 固定使用 4 个核；实际时钟读数为 **{}**，PL 为 **{}**。'
                    'Linux 的最高请求档位和实际 PLL 读数分开看，不能将设置 1.2 GHz 当作实测已达到 1.2 GHz。'
                    '完整 ONNX session 加载耗时 **{} s**，所有配置共用这次加载。'.format(
                        clock_range(cpu_clocks), clock_range(pl_clocks),
                        number(report.get('session_load_seconds'))))
    ort = report.get('ort', {})
    sections.append('报告中的 ORT 配置：`{}`。每个配置分别 warmup，只有 `phase=measure` '
        '的样本计入中位数。完整延时为一次 `smol_run` 的墙钟时间，包含 CPU 图执行、回调、'
        '输入预处理、CMA 复制、缓存同步、AXI 配置、等待硬件与输出复制；'
        'session 加载、驱动初始化、报告和 NPY 写盘均排除。GEMM 与 Softmax 的子路径时间包含软件开销，'
        '不能据此反推纯硬件周期。'.format(json.dumps(ort, ensure_ascii=False, sort_keys=True)))
    sections.append('模型：`{}`  \n模型 SHA-256：`{}`  \n'
                    '输入 SHA-256：`{}`  \nCPU 基线：`{}`'.format(
                        report.get('model', '未记录'), report.get('model_sha256', '未记录'),
                        report.get('inputs_sha256', '未记录'), report.get('cpu_baseline', '未记录')))
    cpu_sidecar = source.parent / 'cpu.json'
    if cpu_sidecar.exists():
        previous = json.loads(cpu_sidecar.read_text(encoding='utf-8'))
        same = previous.get('inputs_sha256') == report.get('inputs_sha256')
        sections.append('本地 `results/cpu.json` 的输入哈希与本次{}。输出比较使用板端 CPU 的 '
            '`[1,50,6]` FP32 动作张量；下表误差取各正式轮中的最大值，'
            '逐元素一致要求 MAE 和最大绝对误差同时为 0。'.format('一致' if same else '**不一致，需核实基线**'))
    else:
        sections.append('本地未找到 CPU 基线侧车 JSON，无法在报告生成阶段复核其输入哈希。'
                        '输出误差沿用板端程序记录的 CPU 基线比较值。')

    sections.append('## 完整模型的正式结果')
    rows = []
    for item in results:
        records = item['measured']
        rows.append([item['config']['name'], '{}/{}'.format(item['warmups'], len(records)),
            ', '.join(number(r['seconds']) for r in records) if records else '尚未测量',
            number(item.get('seconds')), scientific(item.get('mae')),
            scientific(item.get('max_abs'))])
    sections.append(table(['配置', 'warmup/正式轮数', '逐轮延时 s', '中位数 s',
                           '动作 MAE（最坏轮）', '动作 max abs（最坏轮）'], rows))
    if measured:
        sections.append(table(['配置', 'GEMM 路径中位 s', 'Softmax 回调墙钟中位 s',
                              'CPU/PL 重叠中位 s', '混合调用数/轮', 'Softmax PL 任务/轮',
                              '权重命中/填充每轮', '常驻权重新填充 MiB/轮'], [
            [r['config']['name'], number(r['gemm']), number(r['softmax']), number(r['overlap']),
             number(r['mixed_calls'], 0), number(r['pl_tasks'], 0),
             '{}/{}'.format(number(r['cache_hits'], 0), number(r['cache_fills'], 0)),
             number(r['cache_copy_mib'])] for r in measured]))
    history = historical_baselines(source, report.get('inputs_sha256'))
    if history:
        sections.append('同一输入哈希的历史参考如下。它们来自之前独立运行，'
            '与本次改动后的回调和线程池设置存在区别，不能把跨次差值全部归因于单项优化。')
        sections.append(table(['历史配置', '推理中位数 s', '原始文件'],
            [[label, number(seconds), filename] for label, seconds, filename in history]))

    sections.extend(native_sections(native, source, results, report.get('inputs_sha256')))

    sections.append('## CMA 分配和常驻缓存')
    idle = report.get('idle_cma', {})
    sections.append('本板 CMA 总量记录为 **{} MiB**，测试开始前空闲约 **{} MiB**'
        '（此前读数约 121 MiB）。Softmax 输入、输出各有 2,097,152 个 FP32 元素，即各 8 MiB。'
        '独立暂存区为 A 4 + BT 16 + C 8 + Softmax 16 = 44 MiB；'
        '共享暂存区为 28 MiB，因此释放了 16 MiB 给常驻权重。'.format(
            number(idle.get('CmaTotal_kib', 131072) / 1024, 3),
            number(idle.get('CmaFree_kib') / 1024 if idle.get('CmaFree_kib') is not None else None)))
    memory_rows = []
    for r in results:
        config = r['config']
        shared = config.get('shared')
        cache = config.get('cache_mib')
        scratch = 28 if shared else 44
        memory_rows.append([config['name'], '共享' if shared else '独立', scratch,
            cache if cache is not None else '未记录',
            scratch + cache if cache is not None else '未记录', number(r.get('cma_free_mib'))])
    sections.append(table(['配置', '暂存区模式', '暂存 MiB', '权重池 MiB',
                           '申请总计 MiB', '正式轮 CMA 空闲中位 MiB'], memory_rows))
    sections.append('共享布局的 A/BT/C 区域分别为 0–4 / 4–20 / 20–28 MiB，'
        'Softmax 输入复用 0–8 MiB、输出复用 20–28 MiB。两类算子用同一把锁串行使用，'
        'Softmax 覆盖的瞬态 BT 缓存会失效，独立常驻权重池不受影响。'
        '所有 CPU 写入的输入区域在启动前显式 flush，输出在完成后 invalidate；'
        '共享区首次使用额外整体清理一次。超时的未完成任务阻止复用。')
    sections.append('缓存扫描使用同一份 88 MiB 缓存图和稳定的 cache ID；'
        '64/80 MiB 预算保留能容纳的槽位，其余 ID 回退到普通 BT 暂存。'
        '不存在的槽位允许回退，已存在槽位的形状错误仍报错。'
        'Legacy XLNK 分配还要求释放所有 NumPy 视图；关闭驱动时清除计划和缓冲引用后执行 GC，'
        '避免微测试留下的 CMA 占用影响后续配置。')
    if report.get('after_micro_cma'):
        sections.append('微测试关闭后 CMA 空闲读数为 **{} MiB**。'.format(
            number(report['after_micro_cma'].get('CmaFree_kib', 0) / 1024)))

    sections.append('## Softmax 微测试与并行的实际含义')
    sections.append('微测试使用固定种子的正态随机 FP32 输入，每个候选先 warmup 一次；'
        '下表展示所有记录到的正式轮。CPU 部分调用同一版本 ORT 的独立 Softmax 小模型，'
        '直接读写外部张量。CPU/PL 混合策略按不相交的行分区；'
        'PL 分块流水线在硬件运行当前块时，CPU 准备下一块输入或搬出上一块输出。'
        '随机分布下的获胜策略只是筛选结果，完整模型结果仍以上一节为准。')
    micro_rows = []
    for item in report.get('microbenchmark', []):
        policy = item.get('policy', {})
        metrics = item.get('router_metrics', {})
        calls = max(1, int(metrics.get('calls', len(item.get('times', []))) or 1))
        micro_rows.append([shape_text(item.get('shape', [])), policy.get('name', '未记录'),
            '{}% / {}线程'.format(number(float(policy.get('fraction', 0)) * 100, 0),
                                 policy.get('threads', '未记录')),
            '是' if policy.get('pipeline') else '否',
            ', '.join(number(float(t) * 1000) for t in item.get('times', [])),
            number(float(item.get('median_seconds', 0)) * 1000),
            number(float(metrics.get('overlap_seconds', 0)) / calls * 1000),
            scientific(item.get('vs_ort_cpu', {}).get('max_abs'))])
    if micro_rows:
        sections.append(table(['形状', '策略', 'CPU 行比例/线程', 'PL 流水线开关',
                              '逐轮 ms', '中位 ms', '平均重叠 ms/调用', '相对 CPU max abs'], micro_rows))
    else:
        sections.append('尚未记录微测试结果。')
    selected = report.get('micro_selected_policies', {})
    if selected:
        sections.append(table(['形状', '微测试选出的策略'],
            [[shape, policy.get('name', '未记录')] for shape, policy in selected.items()]))
    sections.append('`overlap_seconds` 是 CPU Softmax 调用区间与**完整 PL 路径调用区间**的交集时长；'
        '完整 PL 路径包含 CPU 拷贝、缓存维护、提交、等待与输出搬运。'
        '它证明两个软件路径存在同时执行的时段，不等于 Softmax IP 忙周期，'
        '也不能证明 GEMM 与 Softmax 两个 IP 同时计算。'
        '共享 CMA 在当前设计下让 GEMM/Softmax 互斥；并行来自单个 Softmax 内部 CPU/PL 行分工、'
        '以及 PL 分块与 CPU 搬运的流水。对两个时间分量直接相加会重复计算重叠部分。')
    if measured:
        sections.append(table(['配置', '每轮累计重叠中位 s', '平均重叠 ms/全部 Softmax 调用',
                              '平均重叠 ms/混合调用', '混合调用数/轮'], [
            [r['config']['name'], number(r['overlap'], 6), number(r['overlap_per_call_ms'], 6),
             number(r['overlap_per_mixed_ms'], 6) if r['mixed_calls'] else '不适用',
             number(r['mixed_calls'], 0)] for r in measured]))
    checked, interval_error, aggregate_error = overlap_audit(report)
    sections.append('报告生成时独立重算了 {} 条回调区间交集；'
        '与逐调用记录的最大差为 `{}` s，与每个 router 累计值的最大差为 `{}` s。'
        '上表先逐轮计算「累计重叠÷调用数」，再对正式轮取中位数；'
        '全部调用分母包含 CPU-only 或 PL-only，混合调用分母只计两边均有行的调用。'.format(
            checked, scientific(interval_error), scientific(aggregate_error)))
    overlaps = [router_sum(r, 'overlap_seconds') for r in report.get('records', [])
                if r.get('phase') == 'measure']
    micro_overlaps = [float(r.get('router_metrics', {}).get('overlap_seconds', 0))
                      for r in report.get('microbenchmark', [])]
    sections.append('记录中的微测试最大累计区间重叠为 **{} s**，'
        '完整模型正式轮最大累计区间重叠为 **{} s**。'
        '是否降低整图延时，应比较完整模型中位数，不能只看重叠是否大于零。'.format(
            number(max(micro_overlaps), 6) if micro_overlaps else '未记录',
            number(max(overlaps), 6) if overlaps else '未记录'))

    sections.append('## 数值结果与使用边界')
    sections.append('CPU 和 PL 的浮点 Softmax 实现可能有很小的差异。'
        '本模型在 Softmax 后执行 ReduceMax、缩放、Round、Clip 和整数量化；'
        '微小差异若跨过量化舍入边界，可能改变后续整数结果。'
        '若本次出现“小算子误差、较大动作输出误差”，这一现象**与舍入误差传播放大的机制一致**；'
        '本次没有逐层定位首个量化分歧，不能认定它是已经证实的唯一原因。'
        '共享 CMA 或更大的缓存只改变存储和调度，并不会自动消除 PL Softmax 的数值差异。')
    sections.append('测试只覆盖固定 `validation_0.npz` 的一组输入；'
        '三次重复衡量的是该输入下的执行稳定性，尚未验证数据集精度或实际控制效果。'
        '“逐元素一致最快”仅指该 CPU 基线输入，“原始最快”可能有动作偏差，二者分别报告。')

    sections.append('## 前端卷积改成 im2col 的可行性')
    sections.append('原始 ONNX 只有两次 Conv，分别处理两路相机，并共享同一组 FP32 权重。'
        '输入 `[1,3,512,512]`，权重 `[768,3,16,16]`，bias `[768]`；'
        'kernel 为 16×16，stride 为 16×16，dilation=1，group=1，VALID，'
        '输出 `[1,768,32,32]`。')
    sections.append('im2col 后可表示为 A `[1024,768]` × BT `[768,768]`，'
        'M/N/K 全部已经对齐 32。每路乘加按 2 ops 计为 **1.207959552 GOP**，'
        '两路共 **2.415919104 GOP**。由于 patch 不重叠，FP32 im2col 是 **3 MiB/路**，'
        '与原输入元素数量相同，没有滑窗重叠造成的多倍展开；但仍需重新排列布局。')
    sections.append('若转换成 INT8，A 为 0.75 MiB、BT 为 0.5625 MiB、INT32 C 为 3 MiB，'
        '现有暂存区能容纳。当前 PL 核执行 INT8×INT8→INT32，原 Conv 是 FP32，'
        '因此不能直接替换并宣称数值等价，需要量化输入/权重并恢复 scale 和 bias。'
        'CPU FP32 im2col+GEMM 可保持原运算语义，浮点求和顺序变化仍可能带来小差异。'
        '缓存中的 x86 主机 profile 两次 Conv 为 5.192/3.280 ms，这不是 PYNQ 数据，未用于板端收益估计。')
    conv = read_optional(source.parent / 'patch_conv_cpu4_20261104.json')
    if conv:
        conv_s = conv.get('conv_s', {}).get('median')
        im2col_s = conv.get('im2col_total_s', {}).get('median')
        sections.append('另做了 PYNQ 单个 patch-embedding Conv 的 CPU 对照：{} 线程、'
            '{} 次 warmup、{} 次正式测量，CPU/PL 读数为 {}/{} MHz。'
            '显式 im2col 的总计时包含打包线程启动、ORT MatMul、bias 与输出 NCHW 布局恢复；'
            '它没有使用 INT8 PL。'.format(conv.get('threads', '未记录'), conv.get('warmups', '未记录'),
                conv.get('repeats', '未记录'), number(conv.get('cpu_mhz'), 4), number(conv.get('pl_mhz'), 4)))
        conv_rows = []
        for key, label in [('conv_s', '原始 FP32 ORT Conv'), ('im2col_pack_s', '显式 im2col 打包'),
                           ('ort_matmul_bias_layout_s', 'ORT MatMul + bias + 布局恢复'),
                           ('im2col_total_s', 'im2col 全路径总计')]:
            stats = conv.get(key, {})
            conv_rows.append([label, number(stats.get('median', 0) * 1000),
                number(stats.get('min', 0) * 1000), number(stats.get('max', 0) * 1000)])
        sections.append(table(['路径', '中位 ms', '最小 ms', '最大 ms'], conv_rows))
        sections.append('im2col 路径相对原 Conv 的 MAE 为 `{}`，最大绝对误差 `{}`，'
            '记录的 allclose 验证结果为 `{}`。{}'.format(
                scientific(conv.get('mae_vs_conv')), scientific(conv.get('max_abs_vs_conv')),
                conv.get('outputs_verified', '未记录'),
                ('完整 im2col 路径比 Conv 慢 **{}%**（{} ms），因此本次实测支持保留原 Conv。'.format(
                    number((im2col_s / conv_s - 1) * 100, 2), number((im2col_s - conv_s) * 1000))
                 if conv_s and im2col_s else '')))
        sections.append('表内分段中位数不必相加等于总路径中位数。'
            '这是单层、固定测试输入上的结果，不能把两路耗时简单相加后宣称整图必然节省或增加相同时间。'
            '它也没有测量量化后 INT8 im2col + PL GEMM 的速度或模型精度。')
    else:
        sections.append('尚未取得 `results/patch_conv_cpu4_20261104.json`，'
                        '本报告不据此宣称显式 im2col 在板端更快。')

    sections.append('## 不规则矩阵预先对齐的作用')
    padding = []
    for logical in [(50, 720, 720), (50, 2048, 720), (177, 960, 960), (1024, 768, 768)]:
        padded = tuple((v + 31) // 32 * 32 for v in logical)
        ratio = math.prod(padded) / math.prod(logical)
        padding.append([shape_text(logical), shape_text(padded), number((ratio - 1) * 100, 2) + '%'])
    sections.append(table(['逻辑 M,N,K', '补齐 M,N,K', '额外理论算术量'], padding))
    sections.append('当前静态 B 已离线转置；**padding 在常驻槽首次装入 CMA 时完成，'
        '尚未离线预生成 padded 权重文件**。常驻槽命中后无需再次转置、补齐或同步。'
        '动态 A 可以预分配固定地址和布局，但有效值仍必须随输入更新。'
        '预先对齐能省部分软件清零、转置和打包，不能让硬件自动跳过 padding 零值的计算。'
        '共享暂存区跨形状复用可能污染零填充区域，不能直接删除每轮清零；'
        '需要明确的边缘清零或按形状独立缓冲。若把 padded 形状沿整个图传递，'
        'LayerNorm、Softmax 和归约必须保留逻辑尺寸或 mask，避免填充改变结果。')
    sections.append('对已选择的 PL 节点和当前分块规则做静态统计：每轮 A 整块清零写入 '
        '**571,441,152 B**，真正需要维持零值的补齐边界为 **14,281,280 B**。'
        '若改为只清边界，理论上可少写 **557,159,872 B/轮**（约 531.35 MiB）。'
        '**这项边界清零优化尚未实现，也未测量延时收益**；统计仅说明可减少的软件写入量。'
        '同一批 PL 节点的逻辑工作量约 **575.091 GOP/轮**，按补齐后的尺寸计约 '
        '**608.682 GOP/轮**，增加约 **5.84%**。'
        '这些是由形状推导的算术量，不是计数器实测硬件周期。')

    sections.append('## 原始材料与复现')
    sections.append('- 本次原始结果：`{}`\n'
        '- 可选原图 CPU Softmax 扫描：`results/cma_native_sweep_20261104.json`\n'
        '- 可选板端 patch Conv 对照：`results/patch_conv_cpu4_20261104.json`\n'
        '- 测试程序：`benchmark_runtime_sweep.py`\n'
        '- 本报告生成器：`summarize_runtime_sweep.py`\n'
        '- 运行时：`../../../pynq/GEMM_V2_test/runtime/operators.py`、'
        '`softmax_parallel.py`、`softmax_pipeline.py`\n'
        '- 卷积形状：原始 `../smolvla_w8a8_fp32.onnx`；主机 profile：'
        '`host_shapes_2026-11-03_21-30-45_948.json.gz`'.format(source.name))
    if report.get('error'):
        sections.append('原始运行错误：\n\n```text\n{}\n```'.format(report['error']))
    return '\n\n'.join(sections) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('Result JSON does not exist: {}'.format(args.input))
    report = json.loads(args.input.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report, args.input.resolve()), encoding='utf-8')
    print('REPORT', str(args.output.resolve()))
    print('SOURCE_STATUS', report.get('status', 'unknown'))


if __name__ == '__main__':
    main()
