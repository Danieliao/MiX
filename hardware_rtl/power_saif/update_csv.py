#!/usr/bin/env python3
"""Collect the SAIF-annotated power for every design (from each
power_saif/<Design>/power_saif.rpt, produced by run_all.sh) and write a
reproduced copy of the results table.

Reads the shipped reference ../results_28nm_iso512.csv (for Area_um2 and the
other columns) and writes ../results_28nm_iso512.reproduced.csv with Power_mW and
FLOPs_per_mW replaced by the freshly measured SAIF numbers. The shipped reference
is left untouched so you can `diff` the two to confirm reproduction.

FP16's netlist is the full 32x32 array (1024 MAC/cycle), so its power is divided
by 2 to normalize to the 512-MAC/cycle row (same convention as Area)."""
import csv, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# CSV design name -> (subdir, top, divide_by)
MAP = {
 "INT4":           ("INT4", "INT4_SystolicArray", 1),
 "INT8":           ("INT8", "INT8_SystolicArray", 1),
 "MXINT4":         ("MXINT4", "MXINT4_SystolicArray", 1),
 "MXINT4_g16":     ("MXINT4_g16", "MXINT4_g16_SystolicArray", 1),
 "MXFP4":          ("MXFP4", "MXFP4_SystolicArray", 1),
 "MXFP4_g16":      ("MXFP4_g16", "MXFP4_g16_SystolicArray", 1),
 "NVFP4":          ("NVFP4", "NVFP4_SystolicArray", 1),
 "AMXFP4":         ("AMXFP4", "AMXFP4_SystolicArray", 1),
 "MXFP4_PLUS":     ("MXFP4_PLUS", "MXFP4_PLUS_SystolicArray", 1),
 "MiX425b_MXINT4": ("MiX425b_MXINT4", "MiX425b_MXINT4_SystolicArray", 1),
 "MiX425b_MXFP4":  ("MiX425b_MXFP4", "MiX425b_MXFP4_SystolicArray", 1),
 "MiX45b_MXINT4":  ("MiX45b_MXINT4", "MiX45b_MXINT4_SystolicArray", 1),
 "MiX45b_MXFP4":   ("MiX45b_MXFP4", "MiX45b_MXFP4_SystolicArray", 1),
 "MiX45b_MXINT5":  ("MiX45b_MXINT5", "MiX45b_MXINT5_SystolicArray", 1),
 "FP16":           ("FP16", "FP16_SystolicArray", 2),
}

def total_power(subdir, top):
    rpt = os.path.join(HERE, subdir, "power_saif.rpt")
    if not os.path.exists(rpt):
        return None
    for line in open(rpt):
        f = line.split()
        # total-power row: starts with top name, >=5 fields, ends with the % column
        if len(f) >= 5 and f[0] == top and re.match(r"^[0-9]", f[-1]):
            try:
                return float(f[-2])
            except ValueError:
                continue
    return None

def main():
    src = os.path.join(REPO, "results_28nm_iso512.csv")             # shipped reference
    dst = os.path.join(REPO, "results_28nm_iso512.reproduced.csv")  # measured copy
    rows = list(csv.DictReader(open(src)))
    fields = list(rows[0].keys())  # keep the reference column layout
    miss = []
    for r in rows:
        d = r["Design"]
        sub, top, div = MAP.get(d, (None, None, 1))
        p = total_power(sub, top) if sub else None
        if p is None:
            miss.append(d); continue
        p = p / div
        r["Power_mW"] = f"{p:.3f}"          # measured SAIF power
        r["FLOPs_per_mW"] = f"{float(r['FLOPs_per_cycle'])/p:.2f}"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("Wrote", dst)
    print("Compare against the reference with:  diff", os.path.basename(src), os.path.basename(dst))
    if miss:
        print("MISSING power for (run run_all.sh first):", ", ".join(miss))

if __name__ == "__main__":
    main()
