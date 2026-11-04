# Read-only physical review for an ALREADY OPEN synthesized/implemented design.
# No launches, placement changes, LOCs, pblocks, run resets or checkpoint opens.
namespace eval ::gemm_physical_review {
    proc prop {key obj} {
        if {$obj eq ""} {return ""}
        if {[catch {get_property $key $obj} value]} {return ""}
        return $value
    }
    proc csv {values} {
        set row {}
        foreach value $values {lappend row [format {"%s"} [string map [list "\"" "\"\""] $value]]}
        return [join $row ,]
    }
    proc location {site} {
        if {$site eq ""} {return [list "" "" "" "" ""]}
        set tile [lindex [get_tiles -quiet -of_objects $site] 0]
        set name [prop NAME $tile]
        set x "";set y ""
        regexp {_X([0-9]+)Y([0-9]+)$} $name all x y
        set cr [prop CLOCK_REGION $site]
        set slr [prop NAME [lindex [get_slrs -quiet -of_objects $site] 0]]
        return [list $name $x $y $cr $slr]
    }
    proc run_report {command errors} {
        upvar 1 $errors failures
        if {[catch {uplevel #0 $command} message]} {lappend failures [list $command $message]}
    }
    proc report {out_dir {cell_pattern *gemm_accelerator_top*}} {
        if {[current_design -quiet] eq ""} {error "Open an existing synthesized/implemented design first; this script does not open or build one."}
        set out_dir [file normalize $out_dir]
        file mkdir $out_dir
        set fh [open [file join $out_dir resource_sites.csv] w]
        puts $fh [csv {site type tile tile_x tile_y clock_region slr}]
        set census [dict create]
        foreach site [get_sites -quiet {URAM288* RAMB36* DSP48*}] {
            set name [prop NAME $site]
            set type [prop SITE_TYPE $site]
            lassign [location $site] tile x y cr slr
            puts $fh [csv [list $name $type $tile $x $y $cr $slr]]
            if {[string match URAM* $name]} {set family URAM} elseif {[string match RAMB36* $name]} {set family BRAM36} else {set family DSP}
            dict incr census "$cr/$family"
        }
        close $fh
        set fh [open [file join $out_dir placed_gemm_cells.csv] w]
        puts $fh [csv {cell primitive group site tile tile_x tile_y clock_region slr}]
        set placed 0
        foreach cell [get_cells -hierarchical -quiet -filter {REF_NAME =~ RAMB* || REF_NAME =~ URAM* || REF_NAME == DSP48E2 || NAME =~ *ab_wr_data* || NAME =~ *group_q*}] {
            set name [prop NAME $cell]
            if {![string match $cell_pattern $name]} {continue}
            set type [prop REF_NAME $cell]
            set group other
            if {[regexp {AB\[([0-9]+)\]\.(a|b)/} $name all lane operand]} {
                set group [format {%s_bank_group_%d} $operand [expr {$lane/8}]]
            } elseif {[string match */u_a/* $name]} {
                set group A_input256
            } elseif {[string match */u_bt/* $name]} {
                set group BT_input256
            } elseif {[string match */u_c/* $name]} {
                set group C_result512
            } elseif {[regexp {g_row\[([0-9]+)\]\.g_pe\[([0-9]+)\]} $name all row col]} {
                set group "PE_[expr {$row/8}]_[expr {$col/8}]"
            } elseif {[string match *ab_wr_data* $name]} {set group AB_write_pipeline} elseif {[string match *group_q* $name]} {set group result_pipeline}
            set loc [prop LOC $cell]
            set site ""
            if {$loc ne ""} {set site [lindex [get_sites -quiet $loc] 0]}
            lassign [location $site] tile x y cr slr
            if {$site ne ""} {incr placed}
            puts $fh [csv [list $name $type $group $loc $tile $x $y $cr $slr]]
        }
        close $fh
        set failures {}
        run_report [list report_utilization -hierarchical -file [file join $out_dir utilization.rpt]] failures
        run_report [list report_timing -max_paths 50 -nworst 1 -file [file join $out_dir timing.rpt]] failures
        run_report [list report_high_fanout_nets -max_nets 30 -file [file join $out_dir high_fanout.rpt]] failures
        run_report [list report_design_analysis -logic_level_distribution -file [file join $out_dir logic_levels.rpt]] failures
        if {$placed>0} {run_report [list report_design_analysis -congestion -file [file join $out_dir congestion.rpt]] failures}
        set fh [open [file join $out_dir summary.txt] w]
        puts $fh "Part: [prop PART [current_design]]"
        puts $fh "Design: [current_design]"
        puts $fh "Placed GEMM resource/pipeline cells: $placed"
        foreach key [lsort [dict keys $census]] {puts $fh "$key [dict get $census $key]"}
        puts $fh "Coordinates tile_x/tile_y use the shared device tile grid. Resource-specific SITE X/Y indices must not be compared across BRAM/URAM/DSP types."
        puts $fh "This script changes no placement/constraints and starts no run. Resource counts and timing must match the C_USE_URAM mode being compared."
        foreach failure $failures {puts $fh "REPORT FAILED: $failure"}
        close $fh
        puts "GEMM physical review exported to $out_dir; unavailable report count: [llength $failures]"
        return $out_dir
    }
}
puts {Use: ::gemm_physical_review::report {path/to/output} ?*GEMM_instance_name*?}
