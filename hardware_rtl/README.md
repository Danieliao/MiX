# INVMX — Hardware RTL Artifact

SystemVerilog RTL, testbenches and synthesis scripts for the systolic-array
accelerators evaluated in the paper. This artifact reproduces
**`results_28nm_iso512.csv`** — the area / power / efficiency table for 15
quantization formats, plus the MiX/MX quantizer and the K-smoother.

Everything here is self-contained: no external IP, no generated netlists, and no
site-specific paths (you fill in your own standard-cell library in one clearly
marked block per script).

---

## 1. Repository layout

```
hardware_rtl/
├── README.md                     this file
├── results_28nm_iso512.csv       the reference result to reproduce
├── design_mapping.txt            Design name <-> directory <-> top module
├── src/
│   ├── systolic_arrays/<Design>/ 15 designs: <Design>_{MUL,PE,SystolicArray}.sv
│   │                             + a self-checking testbench <design>_tb.sv
│   ├── quantizer/                MiX/MX quantizer + testbench
│   └── k_smoother/               K-smoothing unit + testbench
├── syn/<Design>/                 Design Compiler script (syn.tcl) + Makefile
└── power_saif/                   SAIF-annotated power measurement (Power_mW; see §6)
```

Each design follows the same three-module hierarchy:

```
<Design>_SystolicArray    ROWS x COLS grid of PEs with systolic data routing
  └── <Design>_PE         one PE: MUL + FP32/INT32 accumulator + pass-through regs
        └── <Design>_MUL  the format-specific dot-product datapath (registered)
```

## 2. Prerequisites

| Step | Tool | Notes |
|------|------|-------|
| Synthesis  | Synopsys Design Compiler (`dc_shell`) | area, timing, and the `.ddc` used for power |
| Simulation | Synopsys VCS | testbenches **and** the gate-level SAIF capture for power (§6) |
| Power      | Design Compiler `read_saif` + `report_power` | the **`Power_mW`** column (§6) |
| Library    | A 28 nm standard-cell `.db` timing library + its Verilog sim model | see below |

The published numbers used **TSMC 28 nm `tcbn28hpcplusbwp30p140`**, worst-case
corner **`ssg0p81v125c`** (SS, 0.81 V, 125 °C), a **2 ns clock (500 MHz)**, and
`compile_ultra`. **`Power_mW` is SAIF-annotated** — measured under real switching
activity, not Design Compiler's default estimate (see §6). The library is *not*
redistributable, so you must point the scripts at your own. Any 28 nm library
will run; absolute area/power will shift, but the *relative* comparison between
designs — which is what the paper claims — is preserved.

## 3. Reproducing `results_28nm_iso512.csv`

**Step 1 — configure your library.** Every `syn/<Design>/syn.tcl` starts
with a single block to edit:

```tcl
#-------------------- SITE CONFIGURATION -- EDIT BEFORE RUNNING ---------------
set PDK_DB_PATH "<PATH-TO-DIRECTORY-CONTAINING-THE-.db-TIMING-LIBRARY>"
set TARGET_DB   "<STANDARD-CELL-TIMING-LIBRARY>.db"   ;# paper: tcbn28hpcplusbwp30p140ssg0p81v125c.db
set OPCOND      "<OPERATING-CONDITION-NAME>"          ;# paper: ssg0p81v125c
set DRIVE_CELL  "<INPUT-DRIVING-CELL>"                ;# paper: INVD18BWP30P140
#------------------------------------------------------------------------------
```

Nothing else needs changing — the RTL path is relative.

**Step 2 — synthesize** (gives `Area_um2`, timing, and the `.ddc` the power step
reuses).

```bash
cd syn/MiX45b_MXINT4      # or any other design
make syn                  # == dc_shell -f syn.tcl
```

**Step 3 — measure SAIF-annotated power** (gives `Power_mW`). This is a required
second stage, not an add-on: the paper's power is measured under real switching
activity, so `report_power` alone (Step 2's synthesis log) is **not** the
reported number. The full procedure is in **§6**; in brief, from `power_saif/`:

```bash
cd power_saif
./run_all.sh MiX45b_MXINT4    # gate-level sim -> SAIF -> read_saif + report_power
```

**Step 4 — read the numbers and compare to `results_28nm_iso512.csv`.**

| CSV column | Where it comes from |
|---|---|
| `Area_um2`  | `syn/<Design>/log/<top>.2.28nm.syn_area.28nm.rpt` → **Total cell area** |
| `Power_mW`  | `power_saif/<Design>/power_saif.rpt` → **Total Power** (SAIF-annotated, §6) |
| timing      | `syn/<Design>/log/<top>.2.28nm.syn_timing.28nm.rpt` → slack must be ≥ 0 |

The remaining columns are derived:

```
MACs_per_cycle  = ROWS * COLS * group_size          (= 512 for every design)
FLOPs_per_cycle = 2 * MACs_per_cycle                (= 1024)
FLOPs_per_mm2   = FLOPs_per_cycle * 1e6 / Area_um2
FLOPs_per_mW    = FLOPs_per_cycle / Power_mW         (Power_mW = SAIF power)
```

`power_saif/update_csv.py` automates Step 4 for power: it reads every
`power_saif.rpt` and writes `results_28nm_iso512.reproduced.csv`, which you can
`diff` against the shipped reference.

**FP16 caveat.** FP16 is a dense 1-MAC-per-PE baseline. Both its synthesis script
and its SAIF flow build the full 32×32 array (1024 MAC/cycle); **both** its
`Area_um2` and `Power_mW` CSV entries are that result **divided by 2** to
normalize to 512 MAC/cycle (a 16×32 mesh). `update_csv.py` applies the ÷2
automatically.

## 4. Why every design is 512 MAC/cycle ("iso-block-size")

Comparing formats fairly requires more than equal array dimensions. A PE that
covers 32 elements amortizes its FP32 normalize+accumulate backend and its
systolic pass-through registers over 32 MACs; a 16-element PE amortizes the same
backend over only 16, and would carry ~2× the backend area per MAC regardless of
its element format.

Every design is therefore built with **32 elements per PE in a 4×4 grid**
(512 MAC/cycle). Formats whose native scale block is 16 (INT8, MXINT4_g16,
MXFP4_g16, NVFP4) keep their per-16 scale — so their bit-width is unchanged —
and simply process **two 16-element sub-blocks per PE**, folding both into the
accumulator with one fused 3-input floating-point add.

**Fairness of the accumulator backend.** The FP32 normalize+accumulate is a
*recurrence*: it must stay fused in a single pipeline stage. Splitting normalize
and accumulate into separate, individually relaxed stages lets the synthesizer
meet the same clock with much smaller cells, which understates area and power.
All MX-family designs here use the same fused backend, so no design gets an
unearned advantage.

## 5. Testbenches

Every design ships one self-checking testbench that drives its **PE** (which
instantiates the MUL, i.e. the entire compute datapath) and compares against a
software reference model written in the testbench. Each prints `PASS` or
`RESULT: n / N FAILED`.

```bash
# example: MiX-4.5b x MXINT4
cd src/systolic_arrays/MiX45b_MXINT4
vcs -sverilog -full64 +v2k -timescale=1ns/10ps \
    MiX45b_MXINT4_MUL.sv MiX45b_MXINT4_PE.sv mix45b_mxint4_tb.sv -R
```

Integer formats (INT4, INT8), the INVMX/MiX formats and the quantizer are
checked **exactly** (the reference replays the identical integer operations).
The formats that fold two independently-scaled sub-blocks into one fused
floating-point add (MXINT4_g16, MXFP4_g16, NVFP4, MiX45b_*) use a small relative
tolerance, because that add aligns to the larger exponent and truncates.

Two conventions worth knowing when reading the testbenches:

* **Accumulator format.** All FP-accumulator designs use a custom
  `{sign, 8b exp, 23b mant}` layout whose exponent carries an *implicit* bias
  inherited from the input scales rather than the IEEE-754 bias. Zero is
  `exp=0, mant=0`; there are no subnormals and the exponent saturates rather
  than wrapping. The decode offset per family is stated at the top of each
  testbench (254 for E8M0×E8M0 formats, 14 for NVFP4's E4M3 scales, 127 for the
  MiX formats where only the weight side is biased, 50 for FP16).
* **Pipeline depth.** Most PEs are 2-stage (MUL → fused normalize+accumulate),
  so the block result appears 2 clocks after reset release. AMXFP4 and
  MiX45b_MXFP4 keep normalize in its own register and are 3-stage.

## 6. SAIF-annotated power (the `Power_mW` column)

 Design Compiler's built-in `report_power` (run inside `syn.tcl`) uses a *default* switching-activity
estimate that under-counts data-dependent toggling, so it is **not** the reported
number. `power_saif/` instead measures power under *real* switching activity
captured from a gate-level simulation of the synthesized netlist.

**Prerequisite.** Run `syn/<Design>` first (§3, Step 2) — it writes the
`output/<top>.ddc` and `.syn.v` netlist that this stage reuses. No re-synthesis
happens here.

**Configure the site block** at the top of `run_all.sh` (the gate-level Verilog
simulation model `.v` of your standard-cell library, and the command that puts
`dc_shell` on `PATH`) and of `gen_tb.py` / each `power_saif.tcl` (the same `.db`
library / operating-condition values used in `syn.tcl`).

**Run it.** From `power_saif/`:

```bash
cd power_saif
./run_all.sh                 # all designs   (or:  ./run_all.sh MiX45b_MXINT4 INT8 ...)
python3 update_csv.py        # -> ../results_28nm_iso512.reproduced.csv
diff ../results_28nm_iso512.csv ../results_28nm_iso512.reproduced.csv
```

For each design `run_all.sh`:

1. compiles the synthesized netlist + `<design>_power_tb.sv` in VCS and runs a
   gate-level simulation that drives every input bus with uniform-random data and
   dumps a SAIF (`gen_tb.py` regenerates these testbenches from each top-level
   port list if you need to);
2. re-runs `report_power` on the already-built `.ddc` with `read_saif`
   (`power_saif.tcl`), giving the **SAIF-annotated Total Power** in
   `power_saif/<Design>/power_saif.rpt`;
3. reports the SAIF-annotation coverage — expect ~100 % of nets/pins annotated.

**Methodology note.** The stimulus is uniform-random per cycle, i.e. a
high-activity, uncorrelated-data operating point; every design uses identical
stimulus, so the comparison across formats is apples-to-apples.
