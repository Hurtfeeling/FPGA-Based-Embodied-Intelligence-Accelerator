# 快速检查带跨行写入的 DMA 地址生成路径；不打开现有工程。
# vivado -mode batch -source GEMM_V2/scripts/synth_dma.tcl -tclargs ?out_dir? ?rtl_file? ?period_ns?
set dma_check_root [file normalize [file join [file dirname [info script]] ..]]
set dma_check_out [file join $dma_check_root tb build dma_synth]
set dma_check_rtl [file join $dma_check_root rtl gemm_axi_dma128.v]
set dma_check_period 4.000
if {[llength $argv] > 0} {set dma_check_out [file normalize [lindex $argv 0]]}
if {[llength $argv] > 1} {set dma_check_rtl [file normalize [lindex $argv 1]]}
if {[llength $argv] > 2} {set dma_check_period [lindex $argv 2]}
if {![string is double -strict $dma_check_period] || $dma_check_period <= 0} {
    error "Invalid clock period"
}
file mkdir $dma_check_out
cd $dma_check_out
set_param general.maxThreads 4
create_project -in_memory -part xczu15eg-ffvb1156-2-i
read_verilog $dma_check_rtl
set dma_check_fh [open [file join $dma_check_out clock.xdc] w]
puts $dma_check_fh [format {create_clock -name clk -period %.3f [get_ports clk]} $dma_check_period]
close $dma_check_fh
read_xdc [file join $dma_check_out clock.xdc]
synth_design -top gemm_axi_dma128 -generic {STRIDED_WRITE=1 MAX_BURST=256} -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -file [file join $dma_check_out utilization.rpt]
report_timing_summary -max_paths 10 -file [file join $dma_check_out timing_summary.rpt]
write_checkpoint -force [file join $dma_check_out dma_synth.dcp]
puts "PASS GEMM_V2 DMA standalone synthesis: $dma_check_rtl"
