# Place and route an existing out-of-context Softmax synthesis checkpoint.
# This does not open or modify an FPGA project and does not create a bitstream.
# Run from the repository root (or use absolute paths):
# vivado -mode batch -source Softmax/scripts/implement.tcl -tclargs \
#   Softmax/tb/build/synth_250/softmax_accelerator_top_synth.dcp \
#   Softmax/tb/build/impl_250 4.0 0
# Arguments: checkpoint, optional output directory, expected clock period ns,
# optional post-route phys_opt (0 or 1). The checkpoint supplies the part and
# the clock constraint. No false paths or multicycle paths are introduced.

if {[llength $argv] < 1 || [llength $argv] > 4} {
    error "Usage: implement.tcl checkpoint ?output_directory? ?expected_period_ns=4.0? ?post_route_phys_opt=0?"
}
set softmax_impl_dcp [file normalize [lindex $argv 0]]
if {![file isfile $softmax_impl_dcp]} {
    error "Synthesis checkpoint does not exist: $softmax_impl_dcp"
}
set softmax_impl_dir [file join [file dirname $softmax_impl_dcp] implementation]
set softmax_expected_period 4.000
set softmax_post_route_opt 0
if {[llength $argv] > 1} {set softmax_impl_dir [file normalize [lindex $argv 1]]}
if {[llength $argv] > 2} {set softmax_expected_period [lindex $argv 2]}
if {[llength $argv] > 3} {set softmax_post_route_opt [lindex $argv 3]}
if {![string is double -strict $softmax_expected_period] || $softmax_expected_period <= 0} {
    error "Expected clock period must be a positive number of nanoseconds"
}
if {$softmax_post_route_opt ni {0 1}} {
    error "post_route_phys_opt must be 0 or 1"
}
file mkdir $softmax_impl_dir
set_param general.maxThreads 4
open_checkpoint $softmax_impl_dcp

# Preserve the synthesis constraint and fail early if a stale checkpoint has
# a different target. In particular a 5 ns checkpoint must not be reported as
# a successful 250 MHz implementation by changing only the report title.
set softmax_impl_clocks [get_clocks -quiet]
if {[llength $softmax_impl_clocks] != 1} {
    error "Expected one constrained clock in the OOC checkpoint; found [llength $softmax_impl_clocks]"
}
set softmax_impl_clock [lindex $softmax_impl_clocks 0]
set softmax_actual_period [get_property PERIOD $softmax_impl_clock]
if {abs($softmax_actual_period - $softmax_expected_period) > 0.0005} {
    error "Checkpoint clock is $softmax_actual_period ns; requested $softmax_expected_period ns. Re-run synthesis with the intended constraint."
}
set softmax_impl_top [get_property NAME [current_design]]
set softmax_impl_part [get_property PART [current_design]]

# Fixed directives make this script repeatable for a given checkpoint and
# Vivado version. The device is inherited from the checkpoint (the standard
# synthesis script currently uses xczu15eg-ffvb1156-2-e).
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
write_checkpoint -force [file join $softmax_impl_dir ${softmax_impl_top}_placed.dcp]
report_timing_summary -delay_type min_max -max_paths 10 \
    -file [file join $softmax_impl_dir placed_timing_summary.rpt]

route_design -directive Explore
if {$softmax_post_route_opt} {
    # Vivado selects post-route physical optimization from the design state.
    # Re-run routing afterwards to reconnect any physically modified nets.
    phys_opt_design -directive Explore
    route_design -directive Explore
}
write_checkpoint -force [file join $softmax_impl_dir ${softmax_impl_top}_routed.dcp]
report_timing_summary -delay_type min_max -check_timing_verbose \
    -report_unconstrained -max_paths 20 \
    -file [file join $softmax_impl_dir routed_timing_summary.rpt]
report_timing -delay_type max -max_paths 50 -sort_by group \
    -file [file join $softmax_impl_dir routed_setup_paths.rpt]
report_timing -delay_type min -max_paths 20 -sort_by group \
    -file [file join $softmax_impl_dir routed_hold_paths.rpt]
report_utilization -hierarchical -file [file join $softmax_impl_dir routed_utilization.rpt]
report_route_status -file [file join $softmax_impl_dir route_status.rpt]
report_drc -file [file join $softmax_impl_dir routed_drc.rpt]
report_clocks -file [file join $softmax_impl_dir clocks.rpt]

# Scope is deliberately explicit: passing internal setup/hold in isolation
# does not establish AXI boundary timing in the integrated board design.
set softmax_impl_fh [open [file join $softmax_impl_dir scope.txt] w]
puts $softmax_impl_fh "Vivado: [version -short]"
puts $softmax_impl_fh "Input checkpoint: $softmax_impl_dcp"
puts $softmax_impl_fh "Input checkpoint bytes: [file size $softmax_impl_dcp]"
puts $softmax_impl_fh "Input checkpoint modified UTC: [clock format [file mtime $softmax_impl_dcp] -gmt 1 -format {%Y-%m-%dT%H:%M:%SZ}]"
puts $softmax_impl_fh "Top: $softmax_impl_top"
puts $softmax_impl_fh "Part: $softmax_impl_part"
puts $softmax_impl_fh "Clock: $softmax_impl_clock; preserved period: $softmax_actual_period ns"
puts $softmax_impl_fh "Flow: opt_design; place Explore; phys_opt Explore; route Explore"
puts $softmax_impl_fh "Post-route phys_opt and reroute: $softmax_post_route_opt"
puts $softmax_impl_fh "Out-of-context place and route; no bitstream generated."
puts $softmax_impl_fh "The inherited OOC clock constraint is preserved. No timing exceptions are added."
puts $softmax_impl_fh "External I/O delays are not constrained by this script. AXI and board I/O timing are not signed off."
puts $softmax_impl_fh "Inspect routed_timing_summary.rpt, route_status.rpt and routed_drc.rpt before assessing internal timing."
puts $softmax_impl_fh "Existing FPGA project and RTL source files were not modified."
close $softmax_impl_fh
puts "SOFTMAX_IMPLEMENT_DONE $softmax_impl_dir"
