#!/usr/bin/env python3
"""Publication-quality plots for MiX energy & speedup simulator.

Generates two 1x2 side-by-side subplot figures (Qwen2-VL | LLaVA-OneVision),
matching the aesthetic of BitMoD Fig. 7/8 (HPCA '25):
  1. Normalized Energy Breakdown (stacked Core/SRAM/DRAM, hierarchical x-axis)
  2. Iso-Area Speedup (grouped bars with hatching, hierarchical x-axis)

Usage:
    python -m simulator.mix_simulator.plot_fig9
    python -m simulator.mix_simulator.plot_fig9 --qwen2vl path/to/qwen2vl/results.json \
                                               --llava path/to/llava/results.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.5
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Format definitions (6 formats: 4 baselines + 2 MiX) ──────────────
# Bar order: FP16, INT8, Focus, NVFP4, MiX-INT4_g16 (4.5b), MiX-INT4 (4.25b)
FORMATS = [
    {"key": "FP16",         "label": "FP16"},
    {"key": "INT8",         "label": "INT8"},
    {"key": "Focus",        "label": "Focus"},
    {"key": "NVFP4",        "label": "NVFP4"},
    {"key": "MXFP4+",       "label": "MXFP4+"},
    {"key": "AMXFP4",       "label": "AMXFP4"},
    {"key": "MiX-INT4 g16", "label": "MiX-INT4$_{g16}$"},
    {"key": "MiX-INT4",     "label": "MiX-INT4"},
]

# MMMU is evaluated at its 128-token generation cap (chain-of-thought, memory-bound);
# all tasks are included in the geomean "Mean".
TASKS_KEYS = ["ChartQA", "TextVQA", "VizWiz", "OCRBench", "MMMU", "SEED-2+"]
MEAN_EXCLUDE = set()
TASKS_LABELS = ["ChartQA", "TextVQA", "VizWiz", "OCRBench", "MMMU", "SEED2+"]

MODELS = [
    {"name": "LLaVA-OneVision",  "tag": "llava"},
    {"name": "MiniCPM-V-2.6",    "tag": "minicpm_v"},
]

# ── Colors ──────────────────────────────────────────────────────────────
COLOR_CORE = "#1f77b4"
COLOR_SRAM = "#aec7e8"
COLOR_DRAM = "#ffbb78"

SPEEDUP_COLORS = {
    "FP16":         "#000000",
    "INT8":         "#7F7F7F",
    "Focus":        "#50C1BD",  # forest green — distinct from MiX purples
    "NVFP4":        "#aa72fe",
    "MXFP4+":       "#6aa84f",  # green — FP4 outlier-aware baseline
    "AMXFP4":       "#8c564b",  # brown — FP4 outlier-aware baseline (large area)
    "MiX-INT4 g16": "#f3cb47",  # dark purple — 4.5b tier
    "MiX-INT4":     "#e78641",  # light purple — 4.25b tier
}



def _geomean(vals):
    product = 1.0
    for v in vals:
        product *= v
    return product ** (1.0 / len(vals))


def _setup_rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
    })


def _style_ax(ax):
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════════

def _safe(v):
    """Treat NaN / None as 0 so plot bars degrade to empty instead of crashing."""
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN check
        return 0.0
    return f


def load_model_data(results_path):
    with open(results_path) as f:
        raw = json.load(f)

    baseline_key = "FP16"
    energy_core, energy_sram, energy_dram, speedups = [], [], [], []

    for fmt in FORMATS:
        fkey = fmt["key"]
        ec, es, ed, sp = [], [], [], []
        for task in TASKS_KEYS:
            bt = (_safe(raw[baseline_key][task].get("core_energy_uJ"))
                  + _safe(raw[baseline_key][task].get("sram_energy_uJ"))
                  + _safe(raw[baseline_key][task].get("dram_energy_uJ")))
            entry = raw.get(fkey, {}).get(task, {})
            if bt > 0:
                ec.append(_safe(entry.get("core_energy_uJ")) / bt)
                es.append(_safe(entry.get("sram_energy_uJ")) / bt)
                ed.append(_safe(entry.get("dram_energy_uJ")) / bt)
            else:
                ec.append(0.0); es.append(0.0); ed.append(0.0)
            sp.append(_safe(entry.get("iso_area_speedup")))

        mean_idx = [i for i, t in enumerate(TASKS_KEYS) if t not in MEAN_EXCLUDE]
        totals = [c + s + d for c, s, d in zip(ec, es, ed)]
        nonzero = [totals[i] for i in mean_idx if totals[i] > 0]
        gm = _geomean(nonzero) if nonzero else 0.0
        ac = sum(ec[i] for i in mean_idx) / len(mean_idx)
        asv = sum(es[i] for i in mean_idx) / len(mean_idx)
        ad = sum(ed[i] for i in mean_idx) / len(mean_idx)
        at = ac + asv + ad
        ec.append(gm * ac / at if at > 0 else 0)
        es.append(gm * asv / at if at > 0 else 0)
        ed.append(gm * ad / at if at > 0 else 0)
        nonzero_sp = [sp[i] for i in mean_idx if sp[i] > 0]
        sp.append(_geomean(nonzero_sp) if nonzero_sp else 0.0)

        energy_core.append(ec)
        energy_sram.append(es)
        energy_dram.append(ed)
        speedups.append(sp)

    return energy_core, energy_sram, energy_dram, speedups


# ═══════════════════════════════════════════════════════════════════════
#  Hierarchical X-axis helper
# ═══════════════════════════════════════════════════════════════════════

def _draw_break_marks(ax, x_center, y_clip, bar_width, value_label):
    """Draw a pair of zig-zag break marks just below y_clip and write the
    real value as a vertical text label rising from above the slashes.
    Used for any bar whose height exceeds y_clip (FP16 = 1.0, Focus on
    MiniCPM-V = 0.69, etc.)."""
    brk_y = y_clip * 0.7
    brk_dy = y_clip * 0.04
    brk_dx = bar_width * 0.22
    ax.plot([x_center - brk_dx, x_center + brk_dx],
            [brk_y - brk_dy, brk_y + brk_dy],
            color="black", linewidth=1.0, clip_on=True, zorder=5)
    ax.plot([x_center - brk_dx, x_center + brk_dx],
            [brk_y - brk_dy * 2.2, brk_y - brk_dy * 0.2],
            color="black", linewidth=1.0, clip_on=True, zorder=5)
    # Vertical label sitting INSIDE the truncated bar (just above the
    # slash marks, baseline near the middle so the text body stays below
    # the bar's top edge).
    ax.text(x_center, y_clip * 0.76, value_label,
            ha="center", va="bottom", rotation=90,
            fontsize=8, fontweight="bold", zorder=5)


def _build_hierarchical_xaxis(ax, group_centers, bar_width, n_fmts, dataset_names):
    """Format labels (bold, rotated 90 deg) under bars; dataset names below."""
    ax.set_xticks([])

    fmt_labels = [f["label"] for f in FORMATS]

    y_fmt = -0.04
    y_ds  = -0.50

    for g, center in enumerate(group_centers):
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = center + offset
            ax.text(x, y_fmt, fmt_labels[i],
                    ha="center", va="top", fontsize=7, fontweight="bold",
                    rotation=90, transform=ax.get_xaxis_transform())

        ax.text(center, y_ds, dataset_names[g],
                ha="center", va="top", fontsize=9.5, fontweight="bold",
                transform=ax.get_xaxis_transform())


# ═══════════════════════════════════════════════════════════════════════
#  Plot 1: Normalized Energy Breakdown (1x2 side-by-side)
# ═══════════════════════════════════════════════════════════════════════

def plot_energy(model_data, output_dir):
    _setup_rcparams()
    n_fmts = len(FORMATS)
    dataset_names = TASKS_LABELS + ["Mean"]
    n_groups = len(dataset_names)

    bar_width = 0.12
    group_width = n_fmts * bar_width
    group_gap = 0.10
    group_centers = np.arange(n_groups) * (group_width + group_gap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 2.5), sharey=True)

    for col, (model, ax) in enumerate(zip(MODELS, axes)):
        ec, es, ed, _ = model_data[model["tag"]]
        _style_ax(ax)

        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers + offset
            c, s, d = ec[i], es[i], ed[i]
            bot_s = c
            bot_d = [cv + sv for cv, sv in zip(c, s)]

            ax.bar(x, c, bar_width * 0.88, color=COLOR_CORE,
                   edgecolor="black", linewidth=0.4, zorder=3)
            ax.bar(x, s, bar_width * 0.88, bottom=bot_s, color=COLOR_SRAM,
                   edgecolor="black", linewidth=0.4, zorder=3)
            ax.bar(x, d, bar_width * 0.88, bottom=bot_d, color=COLOR_DRAM,
                   edgecolor="black", linewidth=0.4, zorder=3)

        # Vertical separator before Geo Mean
        sep_x = group_centers[-2] + (group_width + group_gap) / 2
        ax.axvline(x=sep_x, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        # Clip y; bars exceeding y_clip get break marks + a real-value label.
        y_clip = 0.45
        ax.set_ylim(0, y_clip)
        ax.set_xlim(group_centers[0] - group_width * 0.7,
                    group_centers[-1] + group_width * 0.7)

        if col == 0:
            ax.set_ylabel("Normalized Energy", fontweight="bold")

        # Panel title
        panel = "(a)" if col == 0 else "(b)"
        ax.set_title(f"{panel} {model['name']}", fontsize=12, fontweight="bold")

        # Break marks for any bar that exceeds y_clip (FP16 always, Focus
        # may exceed on some MiniCPM-V tasks where the speedup is small).
        # Skip the rightmost (Mean) group — its label is drawn explicitly
        # below to avoid a duplicate annotation.
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            for g in range(n_groups - 1):
                total = ec[i][g] + es[i][g] + ed[i][g]
                if total > y_clip:
                    xf = group_centers[g] + offset
                    _draw_break_marks(ax, xf, y_clip, bar_width, f"{total:.2f}$\\times$")

        # Annotate Geo Mean totals. For bars that exceed y_clip (FP16 always,
        # Focus on MiniCPM-V), draw break marks at the top of the truncated
        # bar with the real value as the label. Otherwise draw the inline
        # value label above the bar.
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers[-1] + offset
            total = ec[i][-1] + es[i][-1] + ed[i][-1]
            if total > y_clip:
                _draw_break_marks(ax, x, y_clip, bar_width, f"{total:.2f}$\\times$")
            else:
                ax.text(x, total + 0.005, f"{total:.3f}$\\times$", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", rotation=90)

        _build_hierarchical_xaxis(ax, group_centers, bar_width, n_fmts, dataset_names)

    # Component legend at top center
    comp_handles = [
        mpatches.Patch(facecolor=COLOR_CORE, edgecolor="black", linewidth=0.5, label="Core"),
        mpatches.Patch(facecolor=COLOR_SRAM, edgecolor="black", linewidth=0.5, label="SRAM"),
        mpatches.Patch(facecolor=COLOR_DRAM, edgecolor="black", linewidth=0.5, label="DRAM"),
    ]
    fig.legend(handles=comp_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=True,
               framealpha=0.95, edgecolor="#cccccc", handlelength=1.8,
               handletextpad=0.5, columnspacing=1.5,
               prop={"weight": "bold", "size": 10})

    fig.subplots_adjust(wspace=0.06)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "energy_stacked_hierarchical.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Plot 2: Iso-Area Speedup (1x2 side-by-side, hatched bars)
# ═══════════════════════════════════════════════════════════════════════

def plot_speedup(model_data, output_dir):
    _setup_rcparams()
    n_fmts = len(FORMATS)
    dataset_names = TASKS_LABELS + ["Mean"]
    n_groups = len(dataset_names)

    bar_width = 0.12
    group_width = n_fmts * bar_width
    group_gap = 0.10
    group_centers = np.arange(n_groups) * (group_width + group_gap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 2.5), sharey=True)

    for col, (model, ax) in enumerate(zip(MODELS, axes)):
        _, _, _, speedups = model_data[model["tag"]]
        _style_ax(ax)

        for i, fmt in enumerate(FORMATS):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers + offset
            ax.bar(x, speedups[i], bar_width * 0.88,
                   color=SPEEDUP_COLORS[fmt["key"]],
                   edgecolor="black", linewidth=0.6, zorder=3)

        # Baseline line
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

        # Vertical separator before Geo Mean
        sep_x = group_centers[-2] + (group_width + group_gap) / 2
        ax.axvline(x=sep_x, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        if col == 0:
            ax.set_ylabel("Speedup", fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.set_xlim(group_centers[0] - group_width * 0.7,
                    group_centers[-1] + group_width * 0.7)

        # Panel title
        panel = "(a)" if col == 0 else "(b)"
        ax.set_title(f"{panel} {model['name']}", fontsize=12, fontweight="bold")

        # Annotate Geo Mean values (stagger 3 levels)
        stagger = [(0.10, 0.45, 0.80)[i % 3] for i in range(n_fmts)]
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers[-1] + offset
            val = speedups[i][-1]
            y_pad = stagger[i]
            ax.text(x, val + y_pad, f"{val:.2f}$\\times$", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", rotation=90)

        _build_hierarchical_xaxis(ax, group_centers, bar_width, n_fmts, dataset_names)

    # Format legend at top center
    fmt_handles = []
    for fmt in FORMATS:
        p = mpatches.Patch(facecolor=SPEEDUP_COLORS[fmt["key"]],
                           edgecolor="black", linewidth=0.6,
                           label=fmt["label"])
        fmt_handles.append(p)
    fig.legend(handles=fmt_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=len(FORMATS), frameon=True,
               framealpha=0.95, edgecolor="#cccccc", handlelength=1.8,
               handletextpad=0.4, columnspacing=1.0,
               prop={"weight": "bold", "size": 10})

    fig.subplots_adjust(wspace=0.06)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "speedup_grouped_hierarchical.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Combined 2x2: Speedup (top) + Energy (bottom)
# ═══════════════════════════════════════════════════════════════════════

def plot_combined(model_data, output_dir):
    """Single figure: 2x2 grid. Top row = speedup, bottom row = energy.
    Model titles at top of each column (pushed to edges).
    Hierarchical x-axis only on bottom row.
    """
    _setup_rcparams()
    n_fmts = len(FORMATS)
    dataset_names = TASKS_LABELS + ["Mean"]
    n_groups = len(dataset_names)

    bar_width = 0.12
    group_width = n_fmts * bar_width
    group_gap = 0.10
    group_centers = np.arange(n_groups) * (group_width + group_gap)

    fig, axes = plt.subplots(2, 2, figsize=(14, 4.5),
                             gridspec_kw={"height_ratios": [1, 1]})

    for col, model in enumerate(MODELS):
        ec, es, ed, speedups = model_data[model["tag"]]
        ax_sp = axes[0, col]  # top: speedup
        ax_en = axes[1, col]  # bottom: energy

        # ── Top row: Speedup ──────────────────────────────────────
        _style_ax(ax_sp)
        for i, fmt in enumerate(FORMATS):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers + offset
            ax_sp.bar(x, speedups[i], bar_width * 0.88,
                      color=SPEEDUP_COLORS[fmt["key"]],
                      edgecolor="black", linewidth=0.6, zorder=3)

        ax_sp.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
        sep_x = group_centers[-2] + (group_width + group_gap) / 2
        ax_sp.axvline(x=sep_x, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_sp.set_ylim(bottom=0)
        ax_sp.set_xlim(group_centers[0] - group_width * 0.7,
                       group_centers[-1] + group_width * 0.7)
        ax_sp.set_xticks([])  # no x-labels on top row

        if col == 0:
            ax_sp.set_ylabel("Speedup", fontweight="bold")

        # (model title is drawn at the BOTTOM of the energy row instead of
        # the top of the speedup row — see ax_en annotation below)

        # Annotate Mean values (stagger)
        stagger = [(0.10, 0.45, 0.80)[i % 3] for i in range(n_fmts)]
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers[-1] + offset
            val = speedups[i][-1]
            ax_sp.text(x, val + stagger[i], f"{val:.2f}$\\times$",
                       ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=90)

        # ── Bottom row: Energy ────────────────────────────────────
        _style_ax(ax_en)
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers + offset
            c, s, d = ec[i], es[i], ed[i]
            bot_s = c
            bot_d = [cv + sv for cv, sv in zip(c, s)]

            ax_en.bar(x, c, bar_width * 0.88, color=COLOR_CORE,
                      edgecolor="black", linewidth=0.4, zorder=3)
            ax_en.bar(x, s, bar_width * 0.88, bottom=bot_s, color=COLOR_SRAM,
                      edgecolor="black", linewidth=0.4, zorder=3)
            ax_en.bar(x, d, bar_width * 0.88, bottom=bot_d, color=COLOR_DRAM,
                      edgecolor="black", linewidth=0.4, zorder=3)

        ax_en.axvline(x=sep_x, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        y_clip = 0.48
        ax_en.set_ylim(0, y_clip)
        ax_en.set_xlim(group_centers[0] - group_width * 0.7,
                       group_centers[-1] + group_width * 0.7)

        if col == 0:
            ax_en.set_ylabel("Norm. Energy", fontweight="bold")

        # Break marks for any bar that exceeds y_clip (FP16 always, Focus
        # may exceed on some MiniCPM-V tasks where the speedup is small).
        # Skip the rightmost (Mean) group — its label is drawn explicitly
        # below to avoid a duplicate annotation.
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            for g in range(n_groups - 1):
                total = ec[i][g] + es[i][g] + ed[i][g]
                if total > y_clip:
                    xf = group_centers[g] + offset
                    _draw_break_marks(ax_en, xf, y_clip, bar_width, f"{total:.3f}$\\times$")

        # Annotate Mean totals. For bars that exceed y_clip (FP16 always,
        # Focus on MiniCPM-V), draw break marks at the top of the truncated
        # bar with the real value as the label. Otherwise draw the inline
        # value label above the bar.
        for i in range(n_fmts):
            offset = (i - (n_fmts - 1) / 2) * bar_width
            x = group_centers[-1] + offset
            total = ec[i][-1] + es[i][-1] + ed[i][-1]
            if total > y_clip:
                _draw_break_marks(ax_en, x, y_clip, bar_width, f"{total:.3f}$\\times$")
            else:
                ax_en.text(x, total + 0.005, f"{total:.3f}$\\times$", ha="center", va="bottom",
                           fontsize=8, fontweight="bold", rotation=90)

        # Hierarchical x-axis only on bottom row
        _build_hierarchical_xaxis(ax_en, group_centers, bar_width, n_fmts, dataset_names)

        # Model title BELOW the hierarchical x-axis (panel label + model name)
        panel = "(a)" if col == 0 else "(b)"
        ax_en.text(0.5, -0.62, f"{panel} {model['name']}",
                   transform=ax_en.transAxes,
                   ha="center", va="top",
                   fontsize=12, fontweight="bold")

    # Share y-axes within each row
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)

    # ── Legends ──
    # Speedup format legend (top-left area)
    fmt_handles = [mpatches.Patch(facecolor=SPEEDUP_COLORS[f["key"]],
                                   edgecolor="black", linewidth=0.6,
                                   label=f["label"]) for f in FORMATS]
    leg_sp = fig.legend(handles=fmt_handles, loc="upper center",
                        bbox_to_anchor=(0.32, 0.99), ncol=len(FORMATS), frameon=True,
                        framealpha=0.95, edgecolor="#cccccc", handlelength=1.5,
                        handletextpad=0.3, columnspacing=0.8,
                        prop={"weight": "bold", "size": 9})
    fig.add_artist(leg_sp)

    # Energy component legend (top-right area)
    comp_handles = [
        mpatches.Patch(facecolor=COLOR_CORE, edgecolor="black", linewidth=0.5, label="Core"),
        mpatches.Patch(facecolor=COLOR_SRAM, edgecolor="black", linewidth=0.5, label="SRAM"),
        mpatches.Patch(facecolor=COLOR_DRAM, edgecolor="black", linewidth=0.5, label="DRAM"),
    ]
    fig.legend(handles=comp_handles, loc="upper center",
               bbox_to_anchor=(0.75, 0.99), ncol=3, frameon=True,
               framealpha=0.95, edgecolor="#cccccc", handlelength=1.5,
               handletextpad=0.3, columnspacing=0.8,
               prop={"weight": "bold", "size": 9})

    fig.subplots_adjust(wspace=0.03, hspace=0.12)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "combined_speedup_energy.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Publication-quality MiX plots (hierarchical)")
    parser.add_argument("--llava", type=str,
                        default="simulator/mix_simulator/output/llava/results.json")
    parser.add_argument("--minicpm_v", type=str,
                        default="simulator/mix_simulator/output/minicpm_v/results.json")
    parser.add_argument("--output_dir", type=str,
                        default="simulator/mix_simulator/output")
    args = parser.parse_args()

    model_data = {
        "llava":     load_model_data(args.llava),
        "minicpm_v": load_model_data(args.minicpm_v),
    }

    plot_energy(model_data, args.output_dir)
    plot_speedup(model_data, args.output_dir)
    plot_combined(model_data, args.output_dir)
    print(f"\nDone! Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
