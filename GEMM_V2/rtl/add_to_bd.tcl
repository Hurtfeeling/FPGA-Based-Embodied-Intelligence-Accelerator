# Run in the Vivado Tcl Console with a project and Block Design open.
# Adds sources and one module-reference cell; does not compile/synthesize/package IP.
namespace eval ::gemm_bd {
    variable rtl_dir [file normalize [file dirname [info script]]]
    proc add {{cell_name ""} {base_addr 0x80000000}} {
        variable rtl_dir
        if {[current_project -quiet] eq ""} {
            error "Open a Vivado project before adding GEMM."
        }
        if {[current_bd_design -quiet] eq ""} {
            error "Create/open a Block Design before adding GEMM."
        }
        # Remove only obsolete project references; never delete source files.
        set legacy_names {gemm_accelerator_top.sv gemm_axil_regs.sv gemm_axi_dma128.sv gemm_axi_duplex128.sv gemm_core.sv gemm_quarter_writer.sv gemm_quarter_writer.v gemm_output_writer256.sv gemm_output_writer256.v gemm_sram256.sv gemm_systolic_array.sv}
        foreach source [get_files -quiet -all] {
            if {[lsearch -exact $legacy_names [file tail $source]] >= 0} {
                remove_files $source
            }
        }
        set handle [open [file join $rtl_dir gemm_sources.f] r]
        set source_list [split [read $handle] "\n"]
        close $handle
        foreach line $source_list {
            set name [string trim $line]
            if {$name eq ""} {continue}
            set source [file normalize [file join $rtl_dir $name]]
            if {![file isfile $source]} {error "Missing GEMM source: $source"}
            if {[llength [get_files -quiet $source]] == 0} {
                add_files -norecurse $source
            }
            if {[file extension $source] ne ".v"} {error "Expected Verilog source: $source"}
            set_property file_type Verilog [get_files $source]
        }
        update_compile_order -fileset sources_1
        if {$cell_name eq ""} {
            set matches {}
            foreach cell [get_bd_cells -quiet -hierarchical] {
                if {[get_property TYPE $cell] eq "module" &&
                    [get_property REF_NAME $cell] eq "gemm_accelerator_top"} {
                    lappend matches $cell
                }
            }
            if {[llength $matches] > 1} {
                error "Multiple GEMM cells exist; call ::gemm_bd::add with an explicit instance name."
            }
            set cell_name [expr {[llength $matches] == 1 ? [lindex $matches 0] : "gemm_0"}]
        }
        set existing [get_bd_cells -quiet $cell_name]
        if {[llength $existing] > 0} {
            if {[get_property TYPE $existing] ne "module" ||
                [get_property REF_NAME $existing] ne "gemm_accelerator_top"} {
                error "Cell $cell_name already exists and is not the GEMM module."
            }
            update_module_reference $cell_name
            # Refresh can preserve an old explicit value: set it deliberately.
            set_property CONFIG.C_S_AXI_BASEADDR $base_addr $existing
            puts "GEMM cell $cell_name already exists; sources and module reference refreshed."
            puts "C_S_AXI_BASEADDR=$base_addr; keep the S_AXI address assignment at this base."
            return $existing
        }
        set cell [create_bd_cell -type module -reference gemm_accelerator_top $cell_name]
        set_property CONFIG.C_S_AXI_BASEADDR $base_addr $cell
        puts "Added GEMM cell $cell_name. Connect S_AXI/M_AXI/aclk/aresetn."
        puts "C_S_AXI_BASEADDR=$base_addr; assign S_AXI to this same base."
        puts "No synthesis, implementation or IP Catalog package was run."
        return $cell
    }
}
::gemm_bd::add
