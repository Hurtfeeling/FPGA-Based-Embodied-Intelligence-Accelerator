# Run in the Vivado Tcl Console with the target project open:
# From repository root: source [file normalize ./Softmax/scripts/setup_lut.tcl]
# Adds the ROM and refreshes existing Softmax BD cells. No simulation,
# synthesis, implementation, address assignment or wiring changes are run.
namespace eval ::softmax_lut {
    variable rtl_dir [file normalize [file join [file dirname [info script]] ../rtl]]

    proc configure {} {
        variable rtl_dir
        if {[current_project -quiet] eq ""} {
            error "Open the target Vivado project first."
        }
        set lut [file join $rtl_dir exp_lut.mem]
        if {![file isfile $lut] || ![file readable $lut]} {
            error "Missing or unreadable Softmax ROM: $lut"
        }
        if {[llength [get_files -quiet $lut]] == 0} {
            add_files -norecurse $lut
        }
        set_property file_type {Memory Initialization Files} [get_files $lut]
        set_property used_in_synthesis true [get_files $lut]
        set_property used_in_simulation true [get_files $lut]
        update_compile_order -fileset sources_1

        # Open the sole BD automatically; with several designs the user must
        # select one, so this helper cannot accidentally refresh another BD.
        if {[current_bd_design -quiet] eq ""} {
            set designs [get_files -quiet *.bd]
            if {[llength $designs] == 1} {
                open_bd_design [lindex $designs 0]
            } elseif {[llength $designs] > 1} {
                error "ROM added. Open the target Block Design and source this script again."
            }
        }
        set refreshed 0
        if {[current_bd_design -quiet] ne ""} {
            foreach cell [get_bd_cells -quiet -hierarchical] {
                if {[get_property TYPE $cell] ne "module"} {continue}
                if {[get_property REF_NAME $cell] ne "softmax_accelerator_top"} {continue}
                update_module_reference $cell
                # Existing wrappers may retain the old explicit relative path
                # even after the RTL default changes. Replace it deliberately.
                set_property CONFIG.EXP_LUT_FILE $lut [get_bd_cells $cell]
                incr refreshed
            }
            if {$refreshed > 0} {
                save_bd_design
                set bd_name [current_bd_design]
                generate_target all [get_files ${bd_name}.bd]
            }
        }
        puts "SOFTMAX_LUT_READY: $lut; refreshed $refreshed Softmax BD cell(s)."
        puts "Rerun the affected synthesis run to apply ROM initialization."
        puts "After relocating this repository, source this script again to update the absolute BD path."
    }
}
::softmax_lut::configure
