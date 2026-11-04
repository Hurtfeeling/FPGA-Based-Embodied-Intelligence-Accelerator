# 独立检查 Vivado 模块引用是否仍为一个 AXI-Lite + 一个 AXI4 主接口。
# vivado -mode batch -source GEMM_V2/scripts/check_interfaces.tcl
set gemm_v2_root [file normalize [file join [file dirname [info script]] ..]]
set gemm_v2_check_dir [file join $gemm_v2_root tb build interfaces]
file mkdir $gemm_v2_check_dir
cd $gemm_v2_check_dir
create_project -in_memory -part xczu15eg-ffvb1156-2-i
set_property source_mgmt_mode All [current_project]
create_bd_design gemm_v2_interface_check
source [file join $gemm_v2_root rtl add_to_bd.tcl]
set gemm_v2_cell [get_bd_cells gemm_0]
if {[llength $gemm_v2_cell] != 1} {error "Expected one GEMM_V2 module reference"}
set gemm_v2_fh [open [file join $gemm_v2_check_dir interfaces.txt] w]
foreach {gemm_v2_name gemm_v2_mode} {S_AXI Slave M_AXI Master} {
    set gemm_v2_pin [get_bd_intf_pins $gemm_v2_cell/$gemm_v2_name]
    if {[llength $gemm_v2_pin] != 1} {error "Missing interface $gemm_v2_name"}
    if {[get_property MODE $gemm_v2_pin] ne $gemm_v2_mode} {
        error "Incorrect mode for $gemm_v2_name"
    }
    set gemm_v2_expected_width [expr {$gemm_v2_name eq "S_AXI" ? 32 : 128}]
    if {[get_property CONFIG.DATA_WIDTH $gemm_v2_pin] != $gemm_v2_expected_width ||
        [get_property CONFIG.ADDR_WIDTH $gemm_v2_pin] != 32} {
        error "Incorrect width for $gemm_v2_name"
    }
    puts $gemm_v2_fh "$gemm_v2_name [get_property MODE $gemm_v2_pin] [get_property CONFIG.PROTOCOL $gemm_v2_pin] [get_property CONFIG.DATA_WIDTH $gemm_v2_pin] [get_property CONFIG.READ_WRITE_MODE $gemm_v2_pin]"
}
if {[llength [get_bd_intf_pins $gemm_v2_cell/*]] != 2} {
    error "Expected exactly S_AXI and M_AXI"
}
if {[get_property CONFIG.READ_WRITE_MODE [get_bd_intf_pins $gemm_v2_cell/M_AXI]] ne "READ_WRITE"} {
    error "Expected one read/write AXI master"
}
if {[get_property CONFIG.PROTOCOL [get_bd_intf_pins $gemm_v2_cell/S_AXI]] ne "AXI4LITE"} {
    error "Expected unchanged AXI-Lite"
}
if {[get_property CONFIG.PROTOCOL [get_bd_intf_pins $gemm_v2_cell/M_AXI]] ne "AXI4"} {
    error "Expected AXI4 master"
}
close $gemm_v2_fh
puts "PASS GEMM_V2 interfaces: S_AXI, M_AXI"
