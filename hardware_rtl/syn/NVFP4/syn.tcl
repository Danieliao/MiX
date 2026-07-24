#==============================================================================
# Synthesis script -- NVFP4   (top-level module: NVFP4_SystolicArray)
#
# Reproduces the "NVFP4" row of ../../results_28nm_iso512.csv.
#
# HOW TO RUN
#   1. Fill in the SITE CONFIGURATION block below for your standard-cell library.
#   2. From this directory:   make syn        (equivalently: dc_shell -f syn.tcl)
#   3. Read the results:
#        Area  -> log/NVFP4_SystolicArray.2.28nm.syn_area.28nm.rpt    ("Total cell area", um^2)
#        Power -> log/NVFP4_SystolicArray.2.28nm.syn_power.28nm.rpt   ("Total Power", mW)
#        Timing-> log/NVFP4_SystolicArray.2.28nm.syn_timing.28nm.rpt  (slack must be >= 0)
#
# The published numbers were produced with TSMC 28nm tcbn28hpcplusbwp30p140,
# worst-case corner ssg0p81v125c (SS, 0.81 V, 125 C), 2 ns clock (500 MHz),
# compile_ultra. A different 28nm library will run fine but will shift the
# absolute area/power; the relative comparison between designs is preserved.
#==============================================================================

#-------------------- SITE CONFIGURATION -- EDIT BEFORE RUNNING ---------------
# Directory that contains the .db timing library (appended to search_path).
set PDK_DB_PATH "<PATH-TO-DIRECTORY-CONTAINING-THE-.db-TIMING-LIBRARY>"
# The .db timing library file name.   Paper used: tcbn28hpcplusbwp30p140ssg0p81v125c.db
set TARGET_DB   "<STANDARD-CELL-TIMING-LIBRARY>.db"
# Operating condition defined inside that .db.  Paper used: ssg0p81v125c
set OPCOND      "<OPERATING-CONDITION-NAME>"
# Inverter cell used to model input drive strength.  Paper used: INVD18BWP30P140
set DRIVE_CELL  "<INPUT-DRIVING-CELL>"
#------------------------------------------------------------------------------

# Library name as seen by DC = the .db file name without its extension.
set LIB_NAME [file rootname $TARGET_DB]

# Set Multicore Functionality
set_host_options -max_cores 16

set tech_node 28

set search_path [list "." $PDK_DB_PATH ]

set TARGET_LIBS [list $TARGET_DB]

set_app_var target_library [concat $TARGET_LIBS]
set_app_var link_library [concat "*" $TARGET_LIBS]

# Set top level name
set top_level "NVFP4_SystolicArray"

# RTL source -- relative to this directory, no editing needed
set VERILOG_DIR "../../src/systolic_arrays/NVFP4"

set RTL_SRC_FILES [list \
"$VERILOG_DIR/NVFP4_MUL.sv" \
"$VERILOG_DIR/NVFP4_PE.sv" \
"$VERILOG_DIR/NVFP4_SystolicArray.sv" \
]

analyze -format sverilog $RTL_SRC_FILES

elaborate $top_level
list_designs
current_design $top_level

# Clock period
set clk_period 2
#in ns

set clk_uncertainty 0.01
set clk_transition 0.1

#Create real clock if clock port is found
if {[sizeof_collection [get_ports clk]] > 0} {
  set clk_name "clk"
  set clk_port "clk"
  #If no waveform is specified, 50% duty cycle is assumed
  create_clock -name $clk_name -period $clk_period [get_ports $clk_port]
  set_drive 0 [get_clocks $clk_name]
}

set_clock_uncertainty $clk_uncertainty [get_clocks $clk_name]
set_clock_transition $clk_transition [get_clocks $clk_name]

set_operating_conditions $OPCOND -library $LIB_NAME
set_wire_load_mode "segmented"

set typical_input_delay 0.01
set typical_output_delay 0.01
set typical_wire_load 0.010

# Set maximum fanout of gates
set_max_fanout 16 $top_level

# Configure the clock network
set_fix_hold [all_clocks]
set_dont_touch_network $clk_port

set_driving_cell -lib_cell $DRIVE_CELL [all_inputs]
set_input_delay $typical_input_delay [all_inputs] -clock $clk_name
remove_input_delay -clock $clk_name [get_ports $clk_port]
set_output_delay $typical_output_delay [all_outputs] -clock $clk_name

# Set loading of outputs
set_load $typical_wire_load [all_outputs]

# Verify the design
check_design

compile_ultra
balance_registers

exec mkdir -p ./output
exec mkdir -p ./log
# Generate structural verilog netlist
write_file -hierarchy -format verilog -output "./output/${top_level}.${clk_period}.${tech_node}nm.syn.v"
# Save current design (used by the optional SAIF power flow in ../../power_saif)
write_file -hierarchy -format ddc -output "./output/${top_level}.ddc"

# Generate Standard Delay Format (SDF) file
write_sdf -context verilog "./output/${top_level}.${clk_period}.syn.${tech_node}nm.sdf"

# Generate timing constraints file
write_sdc "./output/${top_level}.${clk_period}.syn.${tech_node}nm.sdc"

# Generate report file
set maxpaths 100
set rpt_file "./log/${top_level}.${clk_period}.${tech_node}nm.syn"

check_design > ${rpt_file}_chk_design.${tech_node}nm.rpt
report_area -hierarchy -nosplit > ${rpt_file}_area.${tech_node}nm.rpt
report_power -hier -analysis_effort medium -nosplit > ${rpt_file}_power.${tech_node}nm.rpt
report_design -nosplit > ${rpt_file}
report_cell -nosplit > ${rpt_file}
report_port -nosplit -verbose > ${rpt_file}
report_compile_options -nosplit > ${rpt_file}
report_constraint -all_violators -verbose -nosplit > ${rpt_file}
report_timing -path full -delay max -max_paths $maxpaths -nworst 500 -nosplit > ${rpt_file}_timing.${tech_node}nm.rpt

# Exit dc_shell
quit
