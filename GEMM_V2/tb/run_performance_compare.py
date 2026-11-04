#!/usr/bin/env python3
"""原 GEMM / 当前 V2 的完整 RTL 性能对照；不修改被测 RTL。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TB = HERE / 'tb_gemm_performance.sv'


def shape(text):
    values = tuple(map(int, re.split('[xX,]', text)))
    if len(values) != 3 or any(v <= 0 or v % 32 for v in values):
        raise argparse.ArgumentTypeError('Use M,N,K, each a positive multiple of 32')
    return values


def read_oracle(m, n, k, version):
    cap = min(256, (524288 // (32 * k)) * 32)
    tm, tn = min(m, cap), min(n, cap)
    cm, cn = (m + tm - 1) // tm, (n + tn - 1) // tn
    paths = []
    if version == 'GEMM':
        paths.append([(i * tm, j * tn) for i in range(cm) for j in range(cn)])
    else:
        for columns in (False, True):
            for reverse in (False, True):
                path = []
                for outer in range(cn if columns else cm):
                    inner = list(range(cm if columns else cn))
                    if reverse ^ bool(outer % 2):
                        inner.reverse()
                    path.extend((i * tm, outer * tn) if columns else (outer * tm, i * tn)
                                for i in inner)
                paths.append(path)
    def traffic(path):
        a = b = 0
        previous = (-1, -1)
        for row, col in path:
            if row != previous[0]:
                a += min(tm, m - row) * k
            if col != previous[1]:
                b += min(tn, n - col) * k
            previous = row, col
        return a, b
    return min(map(traffic, paths), key=lambda counts: sum(counts))


def reference(path, m, n, k):
    # 原测试数据分别以 31 行、29 列为周期，只算不同的行/列再展开。
    # 与直接 INT32 A@BT.T 等价，避免 Python 参考计算成为大矩阵瓶颈。
    kk = np.arange(k, dtype=np.int32)[None, :]
    a = ((np.arange(min(m, 31), dtype=np.int32)[:, None] * 7 +
          kk * 5 + (kk // 3) * 3 + 61) % 31) - 15
    bt = ((np.arange(min(n, 29), dtype=np.int32)[:, None] * 11 +
           kk * 3 + (kk // 5) * 7 + 183) % 29) - 14
    unique = a @ bt.T
    result = unique[np.arange(m)[:, None] % 31, np.arange(n)[None, :] % 29]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join('%08x\n' % (int(v) & 0xffffffff) for v in result.flat), encoding='ascii')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shape', action='append', type=shape)
    ap.add_argument('--backend', choices=['xsim', 'iverilog', 'verilator'], default='xsim')
    ap.add_argument('--vivado-bin', type=Path, default=Path('vivado'))
    ap.add_argument('--iverilog', default='iverilog')
    ap.add_argument('--vvp', default='vvp')
    ap.add_argument('--verilator-root', type=Path,
                    default=HERE / 'build' / 'tools' / 'verilator_wheel' / 'verilator')
    ap.add_argument('--make-jobs', type=int, default=4)
    ap.add_argument('--build-dir', type=Path, default=HERE / 'build' / 'performance_compare')
    ap.add_argument('--version', action='append', choices=['GEMM', 'GEMM_V2'])
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--dense-ddr', type=int, choices=[0, 1], default=1)
    ap.add_argument('--burst', type=int, default=256)
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()
    if args.jobs < 1 or not 1 <= args.burst <= 256:
        ap.error('jobs must be positive and burst must be 1..256')
    shapes = args.shape or [(288, 288, 2048), (2048, 512, 2048)]
    versions = args.version or ['GEMM', 'GEMM_V2']
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    references = {}
    for dims in shapes:
        path = build / ('reference_' + '_'.join(map(str, dims)) + '.hex')
        references[dims] = (path, reference(path, *dims))
    flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}

    def run(item):
        version, dims = item
        directory = build / (version + '_' + '_'.join(map(str, dims)))
        directory.mkdir(parents=True, exist_ok=True)
        rtl = ROOT / version / 'rtl'
        sources = [rtl / name.strip() for name in (rtl / 'gemm_sources.f').read_text().splitlines()
                   if name.strip()]
        ref, ref_hash = references[dims]
        params = dict(zip(('PM', 'PN', 'PK'), dims))
        params.update(DMA_BURST=args.burst, DENSE_DDR=args.dense_ddr)
        result = dict(version=version, shape=dims, backend=args.backend, parameters=params,
                      reference_sha256=ref_hash, tb_sha256=hashlib.sha256(TB.read_bytes()).hexdigest(),
                      rtl_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      passed=False)
        start = time.perf_counter()
        def command(argv, name, env=None):
            print('START %s %s %s' % (version, dims, name), flush=True)
            argv = list(map(str, argv))
            if os.name == 'nt' and argv[0].lower().endswith('.bat'):
                # Vivado 包装脚本会把未加引号的 NAME=VALUE 拆开；保留整段参数。
                # 参数均来自本脚本的受控路径/数字，拒绝 cmd 的展开或控制字符。
                if any(any(c in arg for c in '\"%\r\n') for arg in argv):
                    raise ValueError('Unsupported character in batch command argument')
                batch = directory / (name + '.cmd')
                batch.write_text('@echo off\n' + ' '.join('"' + a + '"' for a in argv) + '\n',
                                 encoding='utf-8')
                argv = [os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', str(batch)]
            with (directory / (name + '.stdout.log')).open('w', encoding='utf-8') as output:
                process = subprocess.run(argv, cwd=directory, stdout=output,
                                         stderr=subprocess.STDOUT, timeout=args.timeout, env=env, **flags)
            if process.returncode:
                raise RuntimeError('%s failed (%s)' % (name, process.returncode))
            return (directory / (name + '.stdout.log')).read_text(errors='replace')
        try:
            if args.backend == 'xsim':
                command([args.vivado_bin / 'xvlog.bat', '--sv', '--work', 'xil_defaultlib'] +
                        sources + [TB], 'compile')
                elab = [args.vivado_bin / 'xelab.bat', '--O3', '--mt', '4',
                        'xil_defaultlib.tb_gemm_performance', '--snapshot', 'perf']
                for key, value in params.items():
                    elab += ['--generic_top', '%s=%d' % (key, value)]
                command(elab, 'elaborate')
                output = command([args.vivado_bin / 'xsim.bat', 'perf', '--runall', '--onerror',
                                  'quit', '--onfinish', 'quit', '--testplusarg',
                                  'REFERENCE=' + ref.as_posix()], 'simulate')
            elif args.backend == 'verilator':
                vroot = args.verilator_root.resolve()
                venv = os.environ.copy()
                venv['VERILATOR_ROOT'] = vroot.as_posix()
                obj = directory / 'obj'
                # Verilator 的可变长度循环不支持对字节数组排队 NBA。
                # C 内存仅在 B 响应提交时写入，AXI 从不读 C；参考在完成后才读。
                # 只在编译副本中改成阻塞赋值，硬件 RTL 与原始 TB 保持不变。
                compiled_tb = directory / 'tb_gemm_performance_verilator.sv'
                original_tb = TB.read_text(encoding='utf-8')
                old = 'c_memory[32+wr_addr-Y_BASE+bi*16+bj]<=write_burst[bi][bj*8+:8];'
                new = 'c_memory[32+wr_addr-Y_BASE+bi*16+bj]=write_burst[bi][bj*8+:8];'
                if original_tb.count(old) != 1:
                    raise RuntimeError('Expected one DDR C commit assignment')
                compiled_tb.write_text(original_tb.replace(old, new), encoding='utf-8')
                result['compiled_tb_sha256'] = hashlib.sha256(compiled_tb.read_bytes()).hexdigest()
                result['tb_adaptation'] = 'Only DDR C commit byte array NBA changed to blocking in compiled copy'
                command([vroot / 'bin' / 'verilator_bin.exe', '--cc', '--exe', '--main',
                         '--timing', '--assert', '-Wno-fatal', '-O3', '--output-split', '20000',
                         '--output-split-cfuncs', '1000', '--top-module', 'tb_gemm_performance',
                         '--Mdir', obj] + ['-G%s=%d' % pair for pair in params.items()] +
                        sources + [compiled_tb], 'verilate', venv)
                command(['make', '-C', obj, '-f', 'Vtb_gemm_performance.mk', '-j', str(args.make_jobs),
                         'CXX=g++', 'LINK=g++', 'AR=ar', 'PYTHON3=python', 'CXXFLAGS=-std=c++20',
                         'CFG_CXXFLAGS_PCH=-x c++-header', 'CFG_CXXFLAGS_PCH_I=-include', 'OPT_FAST=-O3',
                         'UNAME_S=Windows_NT'], 'build', venv)
                executable = obj / ('Vtb_gemm_performance.exe' if os.name == 'nt' else 'Vtb_gemm_performance')
                output = command([executable, '+REFERENCE=' + ref.as_posix()], 'simulate', venv)
            else:
                image = directory / 'perf.vvp'
                command([args.iverilog, '-g2012', '-s', 'tb_gemm_performance', '-o', image] +
                        ['-Ptb_gemm_performance.%s=%d' % pair for pair in params.items()] +
                        sources + [TB], 'compile')
                output = command([args.vvp, image, '+REFERENCE=' + ref.as_posix()], 'simulate')
            match = re.search(r'^PERF (.+)$', output, re.MULTILINE)
            if not match or 'PASS tb_gemm_performance' not in output:
                raise RuntimeError('Simulation did not finish with PERF and PASS')
            perf = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', match[1])}
            expected_a, expected_b = read_oracle(*dims, version)
            if (perf['a_read_bytes'], perf['bt_read_bytes']) != (expected_a, expected_b):
                raise RuntimeError('Actual input traffic differs from independent cache-path oracle')
            result.update(perf=perf, passed=True)
        except Exception as error:
            result['error'] = str(error)
        result['seconds'] = round(time.perf_counter() - start, 3)
        (directory / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(('PASS ' if result['passed'] else 'FAIL ') + version + ' ' + str(dims) +
              ' ' + str(result.get('perf', result.get('error'))), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, [(v, s) for s in shapes for v in versions]))
    comparison = []
    for dims in shapes:
        both = {r['version']: r for r in results if tuple(r['shape']) == dims and r['passed']}
        if len(both) == 2:
            a, b = (both[v]['perf'] for v in ('GEMM', 'GEMM_V2'))
            comparison.append(dict(shape=dims, original_job_cycles=a['job_cycles'],
                                   v2_job_cycles=b['job_cycles'],
                                   cycle_reduction_pct=100 * (1 - b['job_cycles'] / a['job_cycles']),
                                   speedup=a['job_cycles'] / b['job_cycles'],
                                   original_read_bytes=a['read_bytes'], v2_read_bytes=b['read_bytes'],
                                   read_reduction_pct=100 * (1 - b['read_bytes'] / a['read_bytes'])))
    report = dict(passed=all(r['passed'] for r in results),
                  model='Full 1024-PE RTL; complete INT32 CPU reference; identical independent R/W DDR model',
                  results=results, comparison=comparison)
    (build / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(comparison, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
