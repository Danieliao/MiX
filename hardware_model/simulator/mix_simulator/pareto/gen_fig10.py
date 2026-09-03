#!/usr/bin/env python3
"""Figure 10 — PE-efficiency Pareto frontier (composite efficiency x accuracy).

Standalone, self-contained generator (supersedes pareto_efficiency_plot.py):
  X = composite PE efficiency = TOPs/mm^2 * TOPs/W   (joint area+power; higher better)
  Y = average task accuracy over Qwen2-VL-7B and LLaVA-OneVision-7B (higher better)

Inputs:
  - simulator/mix_simulator/results_28nm_iso512_saif.csv  (iso-512 SAIF area/power)
  - accuracy_result/table2.json                           (per-format accuracy)

Special points:
  - INT8 : assumed lossless (accuracy = FP16); hardware from the CSV INT8 row.
  (Focus is excluded: running on the dense FP16 array, its PE efficiency is an
   outlier far to the left that compresses the 4-bit cluster.)

Usage:  python -m simulator.mix_simulator.pareto.gen_fig10
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(HERE, "..")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV_PATH = os.path.join(SIM_DIR, "results_28nm_iso512_saif.csv")
T2_PATH = os.path.join(ROOT, "accuracy_result", "table2.json")
OUT_DIR = os.path.join(HERE, "output")

# (csv_name, table2_key, label, is_mix, color)
#   csv_name   -> "Design" column of results_28nm_iso512_saif.csv
#   table2_key -> format directory key in accuracy_result/table2.json, i.e. the
#                 first column of FORMATS in accuracy/accuracy_result/
#                 generate_table2_json.py. Keep the two in sync: a key that is
#                 not in table2.json used to be dropped silently, which removed
#                 every MiX point from the figure.
#   table2_key None -> INT8 uses FP16 (lossless) accuracy.
FORMATS = [
    ("INT8",           None,                "INT8",             False, "#606060"),
    ("AMXFP4",         "amxfp4",            "AMXFP4",           False, "#8c564b"),
    ("NVFP4",          "nvfp4",             "NVFP4",            False, "#aa72fe"),
    ("MXFP4_PLUS",     "mxfp4plus",         "MXFP4+",           False, "#6aa84f"),
    ("MXFP4_g16",      "mxfp4g16",          r"MXFP4$_{g16}$",   False, "#A61A57"),
    ("MiX45b_MXFP4",   "mix-4.5b-fp4",      r"MiX-FP4$_{g16}$", True,  "#EA5193"),
    ("MXINT4_g16",     "mxint4g16",         r"MXINT4$_{g16}$",  False, "#B5A4D4"),
    ("MiX45b_MXINT4",  "mix-4.5b-int4",     r"MiX-INT4$_{g16}$",True,  "#9467bd"),
    ("MiX45b_MXINT5",  "mix-4.5b-int5",     r"MiX-INT5$_{g16}$",True,  "#6A3D9A"),
    ("MXFP4",          "mxfp4",             "MXFP4",            False, "#4D8941"),
    ("MiX425b_MXFP4",  "mix-4.25b-fp4",     "MiX-FP4",          True,  "#6DEC53"),
    ("MXINT4",         "mxint4",            "MXINT4",           False, "#C96B19"),
    ("MiX425b_MXINT4", "mix-4.25b-int4",    "MiX-INT4",         True,  "#FFA85B"),
]

# Models averaged on the Y axis (paper Section "The PPA-Accuracy Pareto
# Frontier": "the average accuracy across six VLM benchmarks and two models
# (Qwen2-VL-7B and LLaVA-OneVision-7B)").
ACC_MODELS = ("qwen2vl", "llava-onevision")

# Per-point label nudges (dx in composite-eff units, dy in % accuracy), ha, va.
LABEL_OFFSETS = {
    "Focus":            (4,  0.0,  "left",  "center"),
    "INT8":             (4,  0.3,  "left",  "bottom"),
    "AMXFP4":           (4, -0.2,  "left",  "top"),
    "NVFP4":            (-4, 0.2,  "right", "bottom"),
    "MXFP4_PLUS":       (4,  0.3,  "left",  "bottom"),
    "MXFP4_g16":        (-4,-0.3,  "right", "top"),
    "MiX45b_MXFP4":     (4, -0.4,  "left",  "top"),
    "MXINT4_g16":       (4, -0.3,  "left",  "top"),
    "MiX45b_MXINT4":    (4,  0.3,  "left",  "bottom"),
    "MiX45b_MXINT5":    (-4, 0.3,  "right", "bottom"),
    "MXFP4":            (-4,-0.3,  "right", "top"),
    "MiX425b_MXFP4":    (4,  0.3,  "left",  "bottom"),
    "MXINT4":           (-4,-0.3,  "right", "top"),
    "MiX425b_MXINT4":   (4,  0.3,  "left",  "bottom"),
}


def composite_eff(row):
    fl = float(row["FLOPs_per_cycle"]); f = float(row["Frequency_MHz"])
    a = float(row["Area_um2"]); p = float(row["Power_mW"])
    am = fl * f / a               # TOPs/mm^2
    pw = fl * f / (1000.0 * p)    # TOPs/W
    return am * pw


def pareto_frontier(points):
    """Max-x, max-y frontier. points: list of (x, y, name). Returns sorted-by-x."""
    pts = sorted(points, key=lambda t: t[0])
    out, ymax = [], -1e9
    for x, y, n in reversed(pts):
        if y >= ymax:
            out.append((x, y, n)); ymax = y
    return list(reversed(out))


def main():
    hw = {r["Design"]: r for r in csv.DictReader(open(CSV_PATH))}
    t2 = json.load(open(T2_PATH))

    missing = [k for _, k, _, _, _ in FORMATS
               if k is not None and not any(k in t2.get(m, {}) for m in ACC_MODELS)]
    if missing:
        raise KeyError(
            "table2.json has no entry for format key(s) " + ", ".join(missing) +
            ".\nKeys present: " +
            ", ".join(sorted(set().union(*(t2.get(m, {}).keys() for m in ACC_MODELS)))) +
            "\nFORMATS in this script must use the same directory keys as "
            "accuracy/accuracy_result/generate_table2_json.py.")

    def acc(key):
        """Mean `avg` over ACC_MODELS. `avg` is None unless a model has all six
        benchmarks, so a partially-run table2.json is reported, not averaged."""
        vs = [t2[m][key]["avg"] for m in ACC_MODELS
              if key in t2.get(m, {}) and t2[m][key].get("avg") is not None]
        if len(vs) < len(ACC_MODELS):
            print(f"  warn {key}: only {len(vs)}/{len(ACC_MODELS)} model(s) have "
                  f"a complete 6-benchmark average")
        return sum(vs) / len(vs) if vs else None

    fp16_acc = acc("fp16")

    data = []
    for csv_name, t2_key, label, is_mix, color in FORMATS:
        ce = composite_eff(hw[csv_name])
        if label == "INT8":
            a = fp16_acc            # INT8 assumed lossless
        else:
            a = acc(t2_key)
        if a is None:
            print(f"  skip {label}: no accuracy"); continue
        data.append(dict(name=csv_name, label=label, x=ce, y=a,
                         is_mix=is_mix, color=color))
        print(f"  {label:16s} comp={ce:6.1f}  acc={a:5.1f}")

    # ── Plot ──────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.grid(True, linestyle="--", alpha=0.3, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    for d in data:
        ax.scatter(d["x"], d["y"], s=(380 if d["is_mix"] else 170),
                   c=d["color"], marker=("*" if d["is_mix"] else "o"),
                   edgecolors="black", linewidth=1.0,
                   zorder=(4 if d["is_mix"] else 3))

    # Pareto frontier (true max-xy). AMXFP4 is plotted but excluded from the
    # frontier line: it is only non-dominated by virtue of Qwen's unusually high
    # AMXFP4 accuracy, at an impractical <50 composite efficiency.
    FRONTIER_EXCLUDE = {"AMXFP4"}
    front = pareto_frontier([(d["x"], d["y"], d["label"]) for d in data
                             if d["name"] not in FRONTIER_EXCLUDE])
    if len(front) > 1:
        fx, fy, _ = zip(*front)
        ax.plot(fx, fy, color="#444444", linestyle="--", linewidth=1.4,
                alpha=0.85, zorder=2)
    print("\nPareto frontier:", " -> ".join(n for _, _, n in front))

    for d in data:
        dx, dy, ha, va = LABEL_OFFSETS.get(d["name"], (4, 0.3, "left", "bottom"))
        ax.annotate(d["label"], xy=(d["x"], d["y"]),
                    xytext=(d["x"] + dx, d["y"] + dy),
                    fontsize=10.5, fontweight="bold", ha=ha, va=va,
                    color="#222222", zorder=5)

    ax.set_xlabel(r"$\rightarrow$ Better" "\n"
                  r"Composite PE Efficiency  (TOPs/mm$^2$ $\times$ TOPs/W)",
                  fontweight="bold")
    ax.set_ylabel("Avg. Accuracy (%)\n" r"$\rightarrow$ Better",
                  fontweight="bold")
    ax.set_title("PE-Efficiency Pareto Frontier", fontweight="bold")

    xs = np.array([d["x"] for d in data]); ys = np.array([d["y"] for d in data])
    xpad = (xs.max() - xs.min()) * 0.10
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - 1.5, ys.max() + 1.5)

    legend = [
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#888",
                   markeredgecolor="black", markersize=15, label="MiX (Ours)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
                   markeredgecolor="black", markersize=10, label="Baseline"),
        plt.Line2D([0], [0], color="#444", linestyle="--", linewidth=1.4,
                   label="Pareto frontier"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True,
              framealpha=0.95, edgecolor="#cccccc", fontsize=10)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "pe_efficiency_pareto.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
