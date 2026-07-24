# AUTO-GENERATED -- SAIF-annotated power for MXFP4_g16 (MXFP4_g16_SystolicArray).
# Re-powers the already-synthesized .ddc; only switching activity differs.
set top "MXFP4_g16_SystolicArray"
set search_path [list "." "<PATH-TO-DIRECTORY-CONTAINING-THE-.db-TIMING-LIBRARY>"]
set target_library [list "<STANDARD-CELL-TIMING-LIBRARY>.db"]
set_app_var target_library [concat $target_library]
set_app_var link_library [concat "*" $target_library]
set LIB_NAME [file rootname "<STANDARD-CELL-TIMING-LIBRARY>.db"]

# Requires syn/MXFP4_g16 to have been run first (it writes the .ddc reused here).
read_file -format ddc "../../syn/MXFP4_g16/output/${top}.ddc"
current_design $top
link

set_operating_conditions "<OPERATING-CONDITION-NAME>" -library $LIB_NAME
create_clock -name clk -period 2 [get_ports clk]

reset_switching_activity
read_saif -input "${top}.saif" -instance_name "tb/dut"

report_saif -hier > saif_coverage.rpt
report_power -hier -analysis_effort medium -nosplit > power_saif.rpt
quit
