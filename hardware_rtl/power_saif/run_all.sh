#!/bin/bash
# SAIF-annotated power flow for all designs.
#  1. VCS gate-level zero-delay sim of the synthesized netlist -> backward SAIF
#  2. dc_shell (via synop module loads, in tcsh): read .ddc + read_saif + report_power
# Run from the repo root or anywhere; paths are resolved from this script's dir.
# Usage: ./run_all.sh [DesignName ...]   (no args = all designs)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
#-------------------- SITE CONFIGURATION -- EDIT BEFORE RUNNING ---------------
# 1) Gate-level Verilog simulation model of your standard-cell library (the .v
#    with the cell behavioural models, NOT the .db). Paper used TSMC 28nm
#    tcbn28hpcplusbwp30p140.v (found under .../Front_End/verilog/... in the PDK).
LIB=<PATH-TO-STANDARD-CELL-VERILOG-SIMULATION-MODEL>.v
# 2) Command that puts dc_shell on PATH inside a tcsh subshell. At the paper's
#    site this is what the local `synop` alias does; replace with whatever your
#    site needs (e.g. "source /path/to/synopsys/setup.csh").
SYNOP='module load <YOUR-DESIGN-COMPILER-MODULE>'
#------------------------------------------------------------------------------

# Design  Top  SynDir
read -r -d '' TABLE <<'EOF'
INT4            INT4_SystolicArray                  INT4
INT8            INT8_SystolicArray              INT8
MXINT4          MXINT4_SystolicArray                MXINT4
MXINT4_g16      MXINT4_g16_SystolicArray        MXINT4_g16
MXFP4           MXFP4_SystolicArray                 MXFP4
MXFP4_g16       MXFP4_g16_SystolicArray         MXFP4_g16
NVFP4           NVFP4_SystolicArray             NVFP4
AMXFP4          AMXFP4_SystolicArray                AMXFP4
MXFP4_PLUS      MXFP4_PLUS_SystolicArray            MXFP4_PLUS
MiX425b_MXINT4  MiX425b_MXINT4_SystolicArray        MiX425b_MXINT4
MiX425b_MXFP4   MiX425b_MXFP4_SystolicArray         MiX425b_MXFP4
MiX45b_MXINT4   MiX45b_MXINT4_SystolicArray   MiX45b_MXINT4
MiX45b_MXFP4    MiX45b_MXFP4_SystolicArray          MiX45b_MXFP4
MiX45b_MXINT5   MiX45b_MXINT5_SystolicArray   MiX45b_MXINT5
FP16            FP16_SystolicArray                  FP16
EOF

WANT="$*"
SUMMARY="$HERE/power_summary.txt"
: > "$SUMMARY"

while read -r DESIGN TOP SYN; do
    [ -z "$DESIGN" ] && continue
    if [ -n "$WANT" ] && ! grep -qw "$DESIGN" <<<"$WANT"; then continue; fi
    DIR="$HERE/$DESIGN"
    TB="$DIR/${DESIGN,,}_power_tb.sv"
    NET=$(ls "$ROOT/syn/$SYN/output/"*SystolicArray*.syn.v 2>/dev/null | head -1)
    echo "================ $DESIGN ($TOP) ================"
    if [ -z "$NET" ] || [ ! -f "$TB" ]; then echo "  MISSING netlist or TB, skip"; continue; fi
    cd "$DIR" || continue

    # 1) VCS compile + run -> SAIF
    vcs -sverilog -full64 +v2k -timescale=1ns/10ps +nospecify +notimingcheck +lint=none \
        -l vcs_compile.log -o simv "$NET" "$LIB" "$TB" >/dev/null 2>&1
    if [ ! -x ./simv ]; then echo "  VCS COMPILE FAILED (see vcs_compile.log)"; continue; fi
    ./simv -l sim_run.log >/dev/null 2>&1
    if [ ! -f "$TOP.saif" ]; then echo "  SIM produced no SAIF (see sim_run.log)"; continue; fi

    # 2) DC read_saif + report_power
    tcsh -c "$SYNOP; dc_shell -f power_saif.tcl -output_log_file dc_power.log" >/dev/null 2>&1

    COV=$(grep -m1 'Nets' saif_coverage.rpt 2>/dev/null | grep -oE '[0-9.]+%' | head -1)
    PWR=$(grep -E "^$TOP " power_saif.rpt 2>/dev/null | awk -v t="$TOP" '$1==t && NF>=5 && $NF ~ /^[0-9]/ {print $(NF-1)}' | head -1)
    echo "  SAIF net coverage: ${COV:-?}   Total power: ${PWR:-FAIL} mW"
    printf "%-16s %s\n" "$DESIGN" "${PWR:-FAIL}" >> "$SUMMARY"
    cd "$HERE"
done <<< "$TABLE"

echo ""
echo "=================== SUMMARY (mW) ==================="
cat "$SUMMARY"
