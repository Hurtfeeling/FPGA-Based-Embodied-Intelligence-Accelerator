# Standalone Vivado synthesis; does not open or modify an existing project.
# Run from any directory:
# vivado -mode batch -source Softmax/scripts/synth.tcl \
#        -tclargs ?output_directory? ?clock_period_ns? ?top_module?
# Default clock is an evaluation target, not a claim of timing closure.
set softmax_script_dir [file dirname [file normalize [info script]]]
set softmax_repo_root [file normalize [file join $softmax_script_dir ../..]]
set softmax_out_dir [file join $softmax_repo_root Softmax tb build synth]
set softmax_clock_period 4.000
set softmax_top softmax_accelerator_top
set softmax_part xczu15eg-ffvb1156-2-e
if {[llength $argv] > 0} {set softmax_out_dir [file normalize [lindex $argv 0]]}
if {[llength $argv] > 1} {set softmax_clock_period [lindex $argv 1]}
if {[llength $argv] > 2} {set softmax_top [lindex $argv 2]}
if {![string is double -strict $softmax_clock_period] || $softmax_clock_period <= 0} {
    error "Clock period must be a positive number of nanoseconds"
}
file mkdir $softmax_out_dir
cd $softmax_repo_root
set_param general.maxThreads 4
create_project -in_memory -part $softmax_part

# Helpers are module-local Verilog functions included from the RTL directory.
# Reuse the current GEMM_V2 DMA implementation; no historical build is read.
set_property include_dirs [file join $softmax_repo_root Softmax rtl] [current_fileset]
foreach softmax_source [lsort [glob [file join $softmax_repo_root Softmax rtl *.v]]] {
    read_verilog $softmax_source
}
read_verilog [file join $softmax_repo_root GEMM_V2 rtl gemm_axi_dma128.v]
read_verilog [file join $softmax_repo_root GEMM_V2 rtl gemm_axi_duplex128.v]

# Track the ROM as a project dependency. Non-project synthesis also gets an
# absolute parameter value, independent of Vivado's synthesis working directory.
set softmax_lut [file join $softmax_repo_root Softmax rtl exp_lut.mem]
if {![file isfile $softmax_lut] || ![file readable $softmax_lut]} {
    error "Missing or unreadable Softmax ROM: $softmax_lut"
}
add_files -norecurse $softmax_lut
set_property file_type {Memory Initialization Files} [get_files $softmax_lut]
set softmax_lut_generic {}
if {$softmax_top eq "softmax_exp4"} {
    set softmax_lut_generic [list -generic [format {LUT_FILE="%s"} $softmax_lut]]
} elseif {$softmax_top eq "softmax_accelerator_top" || $softmax_top eq "softmax_core"} {
    set softmax_lut_generic [list -generic [format {EXP_LUT_FILE="%s"} $softmax_lut]]
}

set softmax_clock_port clk
if {$softmax_top eq "softmax_accelerator_top"} {set softmax_clock_port aclk}
set softmax_xdc [file join $softmax_out_dir clock.xdc]
set softmax_fh [open $softmax_xdc w]
puts $softmax_fh [format {create_clock -name core_clk -period %.3f [get_ports %s]} \
    $softmax_clock_period $softmax_clock_port]
close $softmax_fh
read_xdc $softmax_xdc
synth_design -top $softmax_top -part $softmax_part -mode out_of_context \
    -flatten_hierarchy rebuilt {*}$softmax_lut_generic

report_utilization -hierarchical -file [file join $softmax_out_dir utilization.rpt]
report_timing_summary -max_paths 10 -file [file join $softmax_out_dir timing_summary.rpt]
report_timing -max_paths 20 -file [file join $softmax_out_dir timing_paths.rpt]
report_drc -file [file join $softmax_out_dir drc.rpt]
if {[llength [info commands report_ram_utilization]] != 0} {
    report_ram_utilization -file [file join $softmax_out_dir ram_utilization.rpt]
}
write_checkpoint -force [file join $softmax_out_dir [format {%s_synth.dcp} $softmax_top]]
set softmax_fh [open [file join $softmax_out_dir scope.txt] w]
puts $softmax_fh "Top: $softmax_top"
puts $softmax_fh "Part: $softmax_part"
puts $softmax_fh "Clock evaluation target: $softmax_clock_period ns"
puts $softmax_fh "Out-of-context synthesis only; no placement, routing or bitstream."
puts $softmax_fh "Only the clock is constrained. External I/O timing is not signed off."
puts $softmax_fh "Existing FPGA project and GEMM_V2 source files were not modified."
close $softmax_fh
puts "SOFTMAX_SYNTH_DONE $softmax_out_dir"
