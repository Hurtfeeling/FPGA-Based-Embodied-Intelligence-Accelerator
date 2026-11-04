# Vivado 2020.1 helper. Creates a project and registers current module sources.
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ..]]
set out_dir [file normalize [file join $repo_root fpga vivado_project]]
set part xczu15eg-ffvb1156-2-i
file mkdir $out_dir
create_project embodied_accelerator $out_dir -part $part -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
foreach f [glob -nocomplain [file join $repo_root GEMM_V2 rtl *.v]] { add_files -norecurse $f }
foreach f [glob -nocomplain [file join $repo_root Softmax rtl *.v]] { add_files -norecurse $f }
add_files -norecurse [file join $repo_root Softmax rtl softmax_fp32_funcs.vh]
add_files -norecurse [file join $repo_root Softmax rtl exp_lut.mem]
set_property file_type {Memory Initialization Files} [get_files [file join $repo_root Softmax rtl exp_lut.mem]]
source [file join $repo_root fpga bd design_1_bd.tcl]
update_compile_order -fileset sources_1
validate_bd_design
save_bd_design
puts PROJECT_CREATED
