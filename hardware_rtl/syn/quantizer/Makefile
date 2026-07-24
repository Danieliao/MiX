# Synthesis driver for quantizer (top: MiX_MX_Quantizer).
#   make syn    -- run Design Compiler (edit the SITE CONFIGURATION block in
#                  syn.tcl first)
#   make clean  -- remove all generated output
TOPLEVEL = MiX_MX_Quantizer

syn:
	dc_shell -f syn.tcl -output_log_file syn.log

clean:
	rm -rf output log alib-* command.log default.svf *.pvl *.syn *.mr syn.log
