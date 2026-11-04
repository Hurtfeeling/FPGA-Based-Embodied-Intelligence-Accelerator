"""GEMM_V2 only: INT8 2048-square GEMM, 32-row aligned 1/2/3-core runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'runtime'))
from accelerator_driver import AcceleratorDriver, discover_hardware
from neon_runtime import FastCpu, FastPack, highest_cpu_environment, frequency_report
from pynq import Clocks, MMIO, Overlay, PL, allocate


def stamp():
    return time.strftime('%Y%m%d_%H%M%S')


def main(repeats=6):
    size = 2048
    bitfile = BASE / 'hardware/design_1.bit'
    hwh = bitfile.with_suffix('.hwh')
    hardware = discover_hardware(hwh)
    # Do not replace an overlay while any known GEMM is still using DDR.
    for name, ip in PL.ip_dict.items():
        if 'gemm' in name.lower():
            flags = int(MMIO(ip['phys_addr'], 0x200).read(0x18))
            if flags & (1 | 64):
                raise RuntimeError('Existing GEMM still busy: '+name)
    prefix = stamp()
    result_dir = BASE / 'benchmark_results'
    result_dir.mkdir(exist_ok=True)
    output_path = result_dir / ('gemm_2048_'+prefix+'.json')
    owners, driver = [], None
    results = dict(shape=[size, size, size], dtype='INT8 x INT8 -> INT32',
                   seed=1, repeats=repeats, hardware=hardware, records=[],
                   measurement='ordinary host A/B -> independent ordinary host C; A copy, NEON B transpose, cache sync, submit, full writeback wait, C copy included; allocations/setup/validation excluded',
                   softmax_tested=False, dma_copy_used=False,
                   driver_sha256=hashlib.sha256((BASE/'accelerator_driver.py').read_bytes()).hexdigest(),
                   bitfile_sha256=hashlib.sha256(bitfile.read_bytes()).hexdigest(),
                   hwh_sha256=hashlib.sha256(hwh.read_bytes()).hexdigest())
    try:
        with highest_cpu_environment(keep_highest=True) as cpu_env:
            setup_start = time.perf_counter()
            overlay = Overlay(str(bitfile), download=True)
            # HWH supplies clock0 divisors; use the supported PYNQ setter only if needed.
            if abs(Clocks.fclk0_mhz - 250) > .1:
                Clocks.fclk0_mhz = 250
            clock = float(Clocks.fclk0_mhz)
            if abs(clock-250) > .1:
                raise RuntimeError('PL clock must be 250 MHz, got '+str(clock))
            driver = AcceleratorDriver.from_hwh(hwh, poll_interval=0.0)
            rng = np.random.RandomState(1)
            a = rng.randint(-8, 8, (size, size)).astype(np.int8)
            b = rng.randint(-8, 8, (size, size)).astype(np.int8)
            reference = np.empty((size, size), dtype=np.int32)
            cpu = FastCpu(a, b, reference, (0, 1, 2, 3))
            cpu_plan = dict(variant=2, mb=32, nb=128, pad=0, mode=1, block=32, pack_threads=4)
            for shape, dtype in [((size,size),np.int8),((size,size),np.int8),((size,size),np.int32)]:
                owners.append(allocate(shape=shape, dtype=dtype, cacheable=1))
            aa, bt, cc = owners
            for value in owners:
                if int(value.device_address) % 32 or int(value.device_address)+value.nbytes > 0x80000000:
                    raise RuntimeError('CMA address must be aligned and inside HWH DDR window')
            if any(np.shares_memory(x,y) for i,x in enumerate(owners) for y in owners[i+1:]):
                raise RuntimeError('Allocated buffers overlap')
            packer = FastPack(b, np.asarray(bt), (0,1,2,3))
            pack_plan = dict(mode=1, block=64, threads=4)
            results.update(clock_mhz=clock, cpu_mhz=float(Clocks.cpu_mhz),
                           cpu_frequency=frequency_report(cpu_env), cpu_plan=cpu_plan,
                           pack_plan=pack_plan,
                           buffers={n:dict(address=int(v.device_address),bytes=int(v.nbytes),
                                           coherent=bool(v.coherent)) for n,v in zip(['A','BT','C'],owners)},
                           input_sha256={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in [('A',a),('B',b)]})
            results['setup_s'] = time.perf_counter()-setup_start
            # Native 4-thread CPU warmup/reference, independently verify sample dots.
            cpu.run(None, cpu_plan)
            probe_rng = np.random.RandomState(27)
            probes=[(0,0),(size-1,size-1),(704,1376),(1376,704)]
            probes += list(zip(probe_rng.randint(0,size,64),probe_rng.randint(0,size,64)))
            for row,col in probes:
                exact = int(np.dot(a[row].astype(np.int64),b[:,col].astype(np.int64)))
                assert int(reference[row,col])==exact, ('reference',row,col)
            print('SETUP',json.dumps({k:results[k] for k in ['shape','clock_mhz','cpu_mhz','buffers']}),flush=True)

            def run_pl(cores, repeat, warmup=False):
                started=time.perf_counter()
                np.copyto(aa,a,casting='no');copied_a=time.perf_counter()
                packer.run(None,pack_plan);packed=time.perf_counter()
                aa.sync_to_device();bt.sync_to_device();cc.sync_to_device()
                synced=time.perf_counter()
                group=driver.submit_gemm_rows(int(aa.device_address),int(bt.device_address),
                    int(cc.device_address),size,size,size,cores=cores)
                submitted=time.perf_counter()
                tasks=group.wait(timeout_s=30.0);finished=time.perf_counter()
                cc.sync_from_device();output_synced=time.perf_counter()
                output=np.array(cc,dtype=np.int32,copy=True,subok=False)
                ended=time.perf_counter()
                rows=[dict(core=int(t['device'][4:]),start=int(t['row_start']),
                           rows=int(t['shape'][0]),counters=t['counters']) for t in tasks]
                for part in rows:
                    assert part['start']%32==0 and part['rows']%32==0
                    assert not part['counters']['saturated']
                assert sum(p['rows'] for p in rows)==size
                assert not np.shares_memory(output,cc)
                np.testing.assert_array_equal(output,reference)
                record=dict(mode='PL'+str(len(cores)),repeat=repeat,cores=list(cores),
                    total_s=ended-started,copy_a_s=copied_a-started,
                    transpose_call_s=packed-copied_a,transpose_profile=dict(packer.last_profile),
                    input_sync_s=synced-packed,submit_s=submitted-synced,
                    wait_s=finished-submitted,output_sync_s=output_synced-finished,
                    output_copy_s=ended-output_synced,validation_s=time.perf_counter()-ended,
                    rows=rows,hardware_max_s=max(p['counters']['cycles'] for p in rows)/(clock*1e6),
                    cpu_mhz=float(Clocks.cpu_mhz),pl_mhz=float(Clocks.fclk0_mhz),verified=True)
                assert abs(record['pl_mhz']-clock)<.01
                if not warmup:results['records'].append(record)
                print('WARMUP' if warmup else 'MEASURE',record['mode'],list(cores),
                      'total_ms',round(record['total_s']*1000,3),
                      'hw_ms',round(record['hardware_max_s']*1000,3),
                      'cycles',[p['counters']['cycles'] for p in rows],
                      'rows',[p['rows'] for p in rows],'VERIFIED',flush=True)

            # Check all three cores individually with the full square problem.
            for core in range(3):run_pl((core,),0,warmup=True)
            run_pl((0,1),0,warmup=True)
            run_pl((0,1,2),0,warmup=True)
            for repeat in range(1,repeats+1):
                profile=cpu.run(None,cpu_plan)
                copy_start=time.perf_counter();cpu_output=reference.copy();copy_end=time.perf_counter()
                results['records'].append(dict(mode='CPU',repeat=repeat,
                    total_s=profile['seconds']+copy_end-copy_start,
                    output_copy_s=copy_end-copy_start,profile=profile,verified=True))
                order=[(0,),(0,1),(0,1,2)]
                if repeat%2==0:order.reverse()
                for cores in order:run_pl(cores,repeat)
                output_path.write_text(json.dumps(results,indent=2))
            summary={}
            for mode in ['CPU','PL1','PL2','PL3']:
                records=[r for r in results['records'] if r['mode']==mode]
                stats={}
                fields=['total_s'] if mode=='CPU' else ['total_s','hardware_max_s','copy_a_s',
                    'transpose_call_s','input_sync_s','submit_s','wait_s','output_sync_s','output_copy_s']
                for field in fields:
                    values=[r[field] for r in records]
                    stats[field]=dict(median=statistics.median(values),mean=statistics.mean(values),
                                      minimum=min(values),maximum=max(values))
                if mode!='CPU':
                    stats['core_counters']={str(core):{
                        key:statistics.median(p['counters'][key] for r in records for p in r['rows'] if p['core']==core)
                        for key in ['cycles','compute_cycles','load_cycles','store_cycles','read_bytes','write_bytes']
                    } for core in records[0]['cores']}
                summary[mode]=stats
            results.update(summary=summary,all_verified=True,
                           environment_final=dict(cpu_mhz=float(Clocks.cpu_mhz),
                               pl_mhz=float(Clocks.fclk0_mhz),cpu=frequency_report(cpu_env)))
            output_path.write_text(json.dumps(results,indent=2))
            print('RESULT',str(output_path),flush=True)
            print('SUMMARY',json.dumps(summary),flush=True)
            return results
    finally:
        # Never free DMA-owned memory on a timeout. Drain before releasing owners.
        safe=True
        if driver is not None:
            if driver.pending:
                try:driver.wait_all(timeout_s=30.0)
                except Exception:
                    safe=not bool(driver.pending)
                    if not safe:
                        print('PENDING DMA: retaining owners, process must stay alive',flush=True)
                        while driver.pending:
                            try:driver.wait_all(timeout_s=30.0)
                            except Exception:time.sleep(.1)
                        safe=True
            driver.close()
        if safe:
            for owner in reversed(owners):owner.freebuffer()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=6)
    args=parser.parse_args()
    main(args.repeats)
