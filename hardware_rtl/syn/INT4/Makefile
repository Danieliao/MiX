# Synthesis driver for INT4 (top: INT4_SystolicArray).
#   make syn    -- run Design Compiler (edit the SITE CONFIGURATION block in
#                  syn.tcl first)
#   make clean  -- remove all generated output
TOPLEVEL = INT4_SystolicArray

syn:
	dc_shell -f syn.tcl -output_log_file syn.log

clean:
	rm -rf output log alib-* command.log default.svf *.pvl *.syn *.mr syn.log
