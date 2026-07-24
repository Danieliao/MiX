# Synthesis driver for k_smoother (top: K_SMOOTH_ARRAY).
#   make syn    -- run Design Compiler (edit the SITE CONFIGURATION block in
#                  syn.tcl first)
#   make clean  -- remove all generated output
TOPLEVEL = K_SMOOTH_ARRAY

syn:
	dc_shell -f syn.tcl -output_log_file syn.log

clean:
	rm -rf output log alib-* command.log default.svf *.pvl *.syn *.mr syn.log
