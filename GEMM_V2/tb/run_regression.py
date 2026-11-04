#!/usr/bin/env python3
"""GEMM V2 ABI 10 regression: snake cache reuse, overlapped early writeback, mixed operand RAM."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,shutil,subprocess,time
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
SRC=PROJECT/'rtl'
BUILD=HERE/'build'
ALL=[SRC/x.strip() for x in (SRC/'gemm_sources.f').read_text().splitlines() if x.strip()]
CASES=[["axil32","tb_gemm_axil_regs",["gemm_axil_regs.v"],{}],["dma17","tb_gemm_axi_dma128",["gemm_axi_dma128.v"],{"BURST":17}],["dma256","tb_gemm_axi_dma128",["gemm_axi_dma128.v"],{"BURST":256}],["duplex17","tb_gemm_axi_duplex128",["gemm_axi_dma128.v","gemm_axi_duplex128.v"],{"BURST":17}],["duplex256","tb_gemm_axi_duplex128",["gemm_axi_dma128.v","gemm_axi_duplex128.v"],{"BURST":256}],["array","tb_gemm_systolic_array",["gemm_systolic_array.v"],{}],["streams","tb_gemm_streams",["gemm_sram256.v"],{}],["writer512","tb_gemm_output_writer512",["gemm_output_writer512.v","gemm_sram256.v"],{}],["status","tb_gemm_job_status",None,{}],["e2e32","tb_gemm_accelerator",None,{"CASE":0}],["reuse32","tb_gemm_accelerator",None,{"CASE":1}],["k_limit","tb_gemm_accelerator",None,{"CASE":2}],["serial32","tb_gemm_accelerator",None,{"CASE":3}],["errors32","tb_gemm_accelerator",None,{"CASE":4}],["mn32","tb_gemm_accelerator",None,{"CASE":5}],["row4096","tb_gemm_accelerator",None,{"CASE":6,"DMA_BURST":256}],["serial_error32","tb_gemm_accelerator",None,{"CASE":7}],["shapes32","tb_gemm_accelerator",None,{"CASE":8}],["tiles2d","tb_gemm_accelerator",None,{"CASE":9}]]
CASES += [["result_ram512","tb_gemm_result_sram512",["gemm_sram256.v"],{}],["counter64","tb_gemm_counter64",None,{}],["e2e_bram","tb_gemm_accelerator",None,{"CASE":0,"C_USE_URAM":0}],["serial_bram","tb_gemm_accelerator",None,{"CASE":3,"C_USE_URAM":0}]]
LONG={'k_limit','row4096'}
CASES += [["pipeline32","tb_gemm_accelerator",None,{"CASE":10}]]
CASES += [["axi_dense1","tb_gemm_axi_dma128",["gemm_axi_dma128.v"],{"BURST":1,"NO_BACKPRESSURE":1,"BW_ONLY":1}],
          ["axi_dense17","tb_gemm_axi_dma128",["gemm_axi_dma128.v"],{"BURST":17,"NO_BACKPRESSURE":1,"BW_ONLY":1}],
          ["axi_dense256","tb_gemm_axi_dma128",["gemm_axi_dma128.v"],{"BURST":256,"NO_BACKPRESSURE":1,"BW_ONLY":1}]]
CASES += [["dense_e2e","tb_gemm_accelerator",None,{"CASE":0,"DMA_BURST":256,"DENSE_DDR":1}]]
for case in CASES:
    if case[0]=="row4096":case[3]["DENSE_DDR"]=1
CASES += [["bt_early","tb_gemm_accelerator",None,{"CASE":11,"DMA_BURST":256,"DENSE_DDR":1}],
          ["bt_early_bram","tb_gemm_accelerator",None,{"CASE":11,"DMA_BURST":256,"DENSE_DDR":1,"C_USE_URAM":0}],
          ["bt_background_error","tb_gemm_accelerator",None,{"CASE":12,"DMA_BURST":17,"DENSE_DDR":1}]]
CASES += [["bt_k8192","tb_gemm_accelerator",None,{"CASE":13,"DMA_BURST":256,"DENSE_DDR":1}],
          ["bt_k1024","tb_gemm_accelerator",None,{"CASE":14,"DMA_BURST":256,"DENSE_DDR":1}]]
LONG.add('bt_k8192')
CASES += [["bt_overlap_reset","tb_gemm_accelerator",None,{"CASE":15,"DMA_BURST":256,"DENSE_DDR":1}]]
CASES += [["c_capacity64","tb_gemm_accelerator",None,{"CASE":16,"DMA_BURST":256,"DENSE_DDR":1}],
          ["c_capacity_error","tb_gemm_accelerator",None,{"CASE":17,"DMA_BURST":17,"DENSE_DDR":1}],
          ["c_queue_reset","tb_gemm_accelerator",None,{"CASE":18,"DMA_BURST":256,"DENSE_DDR":1}],
          ["c_older_write_error","tb_gemm_accelerator",None,{"CASE":19,"DMA_BURST":17,"DENSE_DDR":1}]]
CASES += [["input_stream256","tb_gemm_input_stream256",["gemm_sram256.v"],{}]]
CASES += [["serial_full_a","tb_gemm_accelerator",None,{"CASE":20,"DMA_BURST":256,"DENSE_DDR":1}]]
LONG.add('serial_full_a')
CASES += [["axi_window1","tb_gemm_axi_window",["gemm_axi_dma128.v"],{"BURST":1}],
          ["axi_window17","tb_gemm_axi_window",["gemm_axi_dma128.v"],{"BURST":17}],
          ["axi_window256","tb_gemm_axi_window",["gemm_axi_dma128.v"],{"BURST":256}]]
CASES += [["writer_half_major","tb_gemm_output_writer512",["gemm_output_writer512.v","gemm_sram256.v"],{"HALF_MAJOR":1}]]
CASES += [["a_early_exact","tb_gemm_accelerator",None,{"CASE":21,"DMA_BURST":256,"DENSE_DDR":1}],
          ["a_background_error","tb_gemm_accelerator",None,{"CASE":22,"DMA_BURST":17,"DENSE_DDR":1}]]
CASES += [["operand_sram_pipe","tb_gemm_operand_sram256",["gemm_sram256.v"],{}],
          ["input_stream_pipe3","tb_gemm_input_stream256",["gemm_sram256.v"],{"READ_LATENCY":3}]]

CASES += [["snake_row_forward","tb_gemm_accelerator",None,{"CASE":23,"DMA_BURST":256,"DENSE_DDR":1}],
          ["snake_row_reverse","tb_gemm_accelerator",None,{"CASE":24,"DMA_BURST":17,"DENSE_DDR":1}],
          ["snake_col_forward","tb_gemm_accelerator",None,{"CASE":25,"DMA_BURST":256,"DENSE_DDR":1}],
          ["snake_col_reverse","tb_gemm_accelerator",None,{"CASE":26,"DMA_BURST":17,"DENSE_DDR":1}]]
CASES += [["mixed_sram256","tb_gemm_mixed_sram256",["gemm_sram256.v"],{}]]
CASES += [["snake_plan","tb_gemm_snake_plan",None,{}]]
CASES += [["early_cross_macro","tb_gemm_accelerator",None,{"CASE":27,"DMA_BURST":256,"DENSE_DDR":1}]]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iverilog',default=shutil.which('iverilog'))
    ap.add_argument('--vvp',default=shutil.which('vvp'))
    ap.add_argument('--test',action='append',choices=[c[0] for c in CASES])
    ap.add_argument('--long',action='store_true')
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--timeout',type=int,default=900)
    ap.add_argument('--build-dir',type=Path,default=BUILD)
    ap.add_argument('--summary',type=Path)
    args=ap.parse_args()
    if not args.iverilog or not args.vvp:ap.error('Install Icarus or supply --iverilog and --vvp paths')
    if args.jobs<1 or args.timeout<1:ap.error('jobs and timeout must be positive')
    build=args.build_dir.resolve()
    build.mkdir(parents=True,exist_ok=True)
    if args.summary is None:args.summary=build/'abi10_regression_summary.json'
    def run(case):
        name,top,deps,params=case
        image=build/(name+'.vvp');sources=ALL if deps is None else [SRC/s for s in deps]
        cmd=[args.iverilog,'-g2012','-s',top,'-o',str(image)]
        cmd+=['-P'+top+'.'+k+'='+str(v) for k,v in params.items()]
        cmd+=[str(s) for s in sources]+[str(HERE/(top+'.sv'))]
        start=time.perf_counter();text='';ok=False;code=None
        flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        try:
            compiled=subprocess.run(cmd,cwd=SRC,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout,**flags)
            text+=compiled.stdout;code=compiled.returncode
            if code==0:
                executed=subprocess.run([args.vvp,str(image)],cwd=SRC,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=args.timeout,**flags)
                text+=executed.stdout;code=executed.returncode;ok=code==0 and 'PASS' in text
        except subprocess.TimeoutExpired as exc:
            out=exc.stdout or '';text+=(out.decode(errors='replace') if isinstance(out,bytes) else out)+'\nTIMEOUT\n'
        log=build/(name+'.log');log.write_text(text,encoding='utf-8')
        result=dict(test=name,passed=ok,returncode=code,seconds=round(time.perf_counter()-start,3),parameters=params,log=str(log))
        print(('PASS ' if ok else 'FAIL ')+name,flush=True)
        return result
    selected=[c for c in CASES if (c[0] in args.test if args.test else (args.long or c[0] not in LONG))]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:results=list(pool.map(run,selected))
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(dict(abi=10,passed=all(r['passed'] for r in results),results=results),indent=2),encoding='utf-8')
    return 0 if all(r['passed'] for r in results) else 1
if __name__=='__main__':raise SystemExit(main())
