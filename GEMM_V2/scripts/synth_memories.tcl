# 独立存储综合，验证 A 和 BT 各 256 KiB BRAM + 256 KiB URAM，C 为 256 KiB URAM。
# vivado -mode batch -source GEMM_V2/scripts/synth_memories.tcl -tclargs ?output_dir? ?part?
set script_dir [file normalize [file dirname [info script]]]
set project_dir [file dirname $script_dir]
set out_dir [file join $project_dir tb build memory_synth]
set part xczu15eg-ffvb1156-2-i
if {$argc > 0} {set out_dir [file normalize [lindex $argv 0]]}
if {$argc > 1} {set part [lindex $argv 1]}
file mkdir $out_dir
set_param general.maxThreads 4
create_project -in_memory -part $part
read_verilog [file join $project_dir rtl gemm_sram256.v]
read_verilog [file join $script_dir gemm_v2_memories.v]
synth_design -top gemm_v2_memories -part $part -mode out_of_context -flatten_hierarchy none
create_clock -name aclk -period 4.000 [get_ports clk]
report_utilization -hierarchical -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing_synth.rpt]
set counts [dict create]
foreach group {u_a u_bt u_c} {
    foreach family {RAMB36 RAMB18 URAM URAM_PIPELINED} {dict set counts $group $family 0}
    foreach cell [get_cells -quiet -hierarchical -filter "NAME =~ $group/* && IS_PRIMITIVE == 1"] {
        set ref [get_property REF_NAME $cell]
        if {[string match URAM* $ref]} {
            dict set counts $group URAM [expr {[dict get $counts $group URAM]+1}]
            if {[get_property OREG_A $cell] || [get_property OREG_B $cell]} {
                dict set counts $group URAM_PIPELINED [expr {[dict get $counts $group URAM_PIPELINED]+1}]
            }
        }
        if {[string match RAMB36* $ref]} {dict set counts $group RAMB36 [expr {[dict get $counts $group RAMB36]+1}]}
        if {[string match RAMB18* $ref]} {dict set counts $group RAMB18 [expr {[dict get $counts $group RAMB18]+1}]}
    }
    puts "MEMORY_MAPPING $group [dict get $counts $group]"
}
set passed 1
foreach group {u_a u_bt} {
    if {[dict get $counts $group RAMB36]!=56 || [dict get $counts $group RAMB18]!=4 ||
            [dict get $counts $group URAM]!=8 || [dict get $counts $group URAM_PIPELINED]!=8} {set passed 0}
}
if {[dict get $counts u_c RAMB36]!=0 || [dict get $counts u_c RAMB18]!=0 || [dict get $counts u_c URAM]!=8} {set passed 0}
set fh [open [file join $out_dir summary.json] w]
puts $fh "\{"
puts $fh [format {"part":"%s","passed":%s,} $part [expr {$passed?"true":"false"}]]
foreach group {u_a u_bt u_c} {
    set fields {}
    foreach family {RAMB36 RAMB18 URAM URAM_PIPELINED} {
        lappend fields [format {"%s":%d} $family [dict get $counts $group $family]]
    }
    puts $fh [format {"%s":{%s}%s} $group [join $fields ,] [expr {$group eq "u_c"?"":","}]]
}
puts $fh "\}"
close $fh
write_checkpoint -force [file join $out_dir memories_synth.dcp]
if {!$passed} {error "Mixed memory inference mismatch; inspect summary.json and utilization.rpt."}
puts "PASS GEMM_V2 memory mapping: each A/BT 56 RAMB36 +4 RAMB18 +8 pipelined URAM; C 8 URAM."
