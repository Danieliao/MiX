# Synthesis driver for MiX45b_MXINT5 (top: MiX45b_MXINT5_SystolicArray).
#   make syn    -- run Design Compiler (edit the SITE CONFIGURATION block in
#                  syn.tcl first)
#   make clean  -- remove all generated output
TOPLEVEL = MiX45b_MXINT5_SystolicArray

syn:
	dc_shell -f syn.tcl -output_log_file syn.log

clean:
	rm -rf output log alib-* command.log default.svf *.pvl *.syn *.mr syn.log
