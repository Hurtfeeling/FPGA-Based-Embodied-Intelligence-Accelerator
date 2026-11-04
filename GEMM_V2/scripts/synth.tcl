# 独立综合，不打开或修改现有 15eg 工程，也不会生成/下载 bitstream。
# vivado -mode batch -source GEMM_V2/scripts/synth.tcl -tclargs ?out_dir? ?period_ns?
set gemm_v2_root [file normalize [file join [file dirname [info script]] ..]]
set gemm_v2_out [file join $gemm_v2_root tb build synth]
set gemm_v2_period 4.000
if {[llength $argv] > 0} {set gemm_v2_out [file normalize [lindex $argv 0]]}
if {[llength $argv] > 1} {set gemm_v2_period [lindex $argv 1]}
if {![string is double -strict $gemm_v2_period] || $gemm_v2_period <= 0} {
    error "Invalid clock period"
}
file mkdir $gemm_v2_out
cd $gemm_v2_out
set_param general.maxThreads 4
create_project -in_memory -part xczu15eg-ffvb1156-2-i
set gemm_v2_fh [open [file join $gemm_v2_root rtl gemm_sources.f] r]
set gemm_v2_sources [split [read $gemm_v2_fh] "\n"]
close $gemm_v2_fh
foreach gemm_v2_file $gemm_v2_sources {
    set gemm_v2_file [string trim $gemm_v2_file]
    if {$gemm_v2_file ne ""} {
        read_verilog [file join $gemm_v2_root rtl $gemm_v2_file]
    }
}
set gemm_v2_fh [open [file join $gemm_v2_out clock.xdc] w]
puts $gemm_v2_fh [format {create_clock -name aclk -period %.3f [get_ports aclk]} $gemm_v2_period]
close $gemm_v2_fh
read_xdc [file join $gemm_v2_out clock.xdc]
synth_design -top gemm_accelerator_top -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -hierarchical -file [file join $gemm_v2_out utilization.rpt]
report_timing_summary -max_paths 10 -file [file join $gemm_v2_out timing_summary.rpt]
report_drc -file [file join $gemm_v2_out drc.rpt]
if {[llength [info commands report_ram_utilization]]} {
    report_ram_utilization -file [file join $gemm_v2_out ram_utilization.rpt]
}
write_checkpoint -force [file join $gemm_v2_out gemm_v2_synth.dcp]
puts "PASS GEMM_V2 standalone OOC synthesis: $gemm_v2_out"
