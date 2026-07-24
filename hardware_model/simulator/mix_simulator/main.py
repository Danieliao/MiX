#!/usr/bin/env python3
"""MiX Energy & Speedup Simulator — Entry Point.

Simulates speedup and energy (Core/SRAM/DRAM) for quantization formats
across multiple VLM benchmark tasks, producing grouped bar charts
similar to BitMoD Fig 8 and Focus Fig 9.

Usage:
    python -m simulator.mix_simulator.main [--model qwen2vl]
"""

import argparse
import json
import os
from typing import Dict, List

from .config import (
    AcceleratorConfig, TaskConfig,
    FP16, INT8, MXINT4_G16, MXFP4_G16,
    NVFP4, MIX_INT4_G16, MIX_FP4_G16, FOCUS,
    DEFAULT_ACCELS, DEFAULT_TASKS, QWEN3_TASKS, MINICPM_TASKS,
    CHARTQA, TEXTVQA, VIZWIZ, OCRBENCH, MMMU, SEEDBENCH,
)
from .model_profile import get_model_profile
from .energy import simulate_task, SimResult


def _get_speedup(accel, baseline_accel, baseline_r, accel_r, task, model_tag):
    """Compute iso-area speedup for `accel` on `task`, honoring overrides."""
    if accel.external_only or accel.speedup_overrides:
        ov = accel.speedup_overrides.get(model_tag, {})
        if task.name in ov:
            return ov[task.name]
        return float("nan")
    area_scale = baseline_accel.area_um2 / accel.area_um2
    baseline_total = max(baseline_r.compute_cycles, baseline_r.dram_cycles)
    scaled_compute = accel_r.compute_cycles / area_scale
    scaled_total = max(scaled_compute, accel_r.dram_cycles)
    return baseline_total / scaled_total


def _get_energy(accel, accel_r, task, model_tag, baseline_r=None, baseline_accel=None,
                speedup=None):
    """Return (core_uJ, sram_uJ, dram_uJ, total_uJ) for `accel` on `task`.

    Override priority:
    1. Explicit energy_overrides[model_tag][task_name]  -> use as-is.
    2. external_only accelerator with no explicit energy override but
       a speedup override -> derive Core/SRAM/DRAM by scaling the FP16
       baseline by 1/speedup (Focus runs on the same FP16 dense hardware,
       so its per-cycle energy matches FP16; total energy is FP16/speedup).
    3. Otherwise -> use the simulated SimResult.
    """
    if accel.energy_overrides:
        ov = accel.energy_overrides.get(model_tag, {}).get(task.name)
        if ov is not None:
            c, s, d = ov["core_uJ"], ov["sram_uJ"], ov["dram_uJ"]
            return c, s, d, c + s + d
    if accel.external_only:
        # Derive from FP16 baseline / speedup
        if (baseline_r is not None and speedup is not None and
                speedup == speedup and speedup > 0):
            c = baseline_r.core_energy_pJ / 1e6 / speedup
            s = baseline_r.sram_energy_pJ / 1e6 / speedup
            d = baseline_r.dram_energy_pJ / 1e6 / speedup
            return c, s, d, c + s + d
        return float("nan"), float("nan"), float("nan"), float("nan")
    if accel_r is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (accel_r.core_energy_pJ / 1e6,
            accel_r.sram_energy_pJ / 1e6,
            accel_r.dram_energy_pJ / 1e6,
            accel_r.total_energy_pJ / 1e6)


def print_results(
    results: Dict[str, Dict[str, SimResult]],
    accels: List[AcceleratorConfig],
    tasks: List[TaskConfig],
    model_tag: str,
    baseline_name: str = "FP16",
):
    """Print a summary table: tasks as columns, accelerators as rows."""
    # Header
    task_names = [t.name for t in tasks]
    col_w = 14
    header = f"{'Accelerator':<20}" + "".join(f"{t:>{col_w}}" for t in task_names)
    baseline_accel = [a for a in accels if a.name == baseline_name][0]

    print(f"\n{'='*len(header)}")
    print(f"  ISO-AREA SPEEDUP (normalized to {baseline_name})")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for accel in accels:
        row = f"{accel.name.replace(chr(10), ' '):<20}"
        for task in tasks:
            baseline_r = results[baseline_name].get(task.name)
            accel_r = results[accel.name].get(task.name)
            speedup = _get_speedup(accel, baseline_accel, baseline_r, accel_r, task, model_tag)
            row += f"{speedup:>{col_w}.2f}x"
        print(row)

    # Energy table (normalized to FP16 total)
    print(f"\n{'='*len(header)}")
    print(f"  NORMALIZED ENERGY (total, relative to {baseline_name})")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for accel in accels:
        row = f"{accel.name.replace(chr(10), ' '):<20}"
        for task in tasks:
            baseline_r = results[baseline_name].get(task.name)
            accel_r = results[accel.name].get(task.name)
            speedup = _get_speedup(accel, baseline_accel, baseline_r, accel_r, task, model_tag)
            _, _, _, baseline_total = _get_energy(
                baseline_accel, baseline_r, task, model_tag,
                baseline_r=baseline_r, baseline_accel=baseline_accel, speedup=1.0)
            _, _, _, accel_total = _get_energy(
                accel, accel_r, task, model_tag,
                baseline_r=baseline_r, baseline_accel=baseline_accel, speedup=speedup)
            if baseline_total and baseline_total == baseline_total:  # not nan
                norm_e = accel_total / baseline_total
            else:
                norm_e = float("nan")
            row += f"{norm_e:>{col_w}.4f}"
        print(row)

    # Per-task detail for one accelerator
    mix_name = MIX_INT4_G16.name
    print(f"\n{'='*len(header)}")
    print(f"  ENERGY BREAKDOWN: {mix_name.replace(chr(10), ' ')} (uJ)")
    print(f"{'='*len(header)}")
    detail_header = f"{'Task':<14} {'Core(uJ)':>12} {'SRAM(uJ)':>12} {'DRAM(uJ)':>12} {'Total(uJ)':>12}"
    print(detail_header)
    print("-" * len(detail_header))
    for task in tasks:
        r = results[mix_name][task.name]
        print(f"{task.name:<14} {r.core_energy_pJ/1e6:>12.2f} {r.sram_energy_pJ/1e6:>12.2f} "
              f"{r.dram_energy_pJ/1e6:>12.2f} {r.total_energy_pJ/1e6:>12.2f}")


def save_results_json(
    results: Dict[str, Dict[str, SimResult]],
    accels: List[AcceleratorConfig],
    tasks: List[TaskConfig],
    output_dir: str,
    model_tag: str,
    baseline_name: str = "FP16",
):
    """Save all results as JSON."""
    baseline_accel = [a for a in accels if a.name == baseline_name][0]
    data = {}
    for accel in accels:
        accel_key = accel.name.replace("\n", " ")
        data[accel_key] = {}
        for task in tasks:
            r = results[accel.name].get(task.name)
            baseline_r = results[baseline_name].get(task.name)

            speedup = _get_speedup(accel, baseline_accel, baseline_r, r, task, model_tag)
            c, s, d, total = _get_energy(
                accel, r, task, model_tag,
                baseline_r=baseline_r, baseline_accel=baseline_accel, speedup=speedup)
            _, _, _, baseline_total = _get_energy(
                baseline_accel, baseline_r, task, model_tag,
                baseline_r=baseline_r, baseline_accel=baseline_accel, speedup=1.0)

            if total != total or baseline_total != baseline_total or baseline_total == 0:
                normalized_energy = float("nan")
            else:
                normalized_energy = total / baseline_total

            data[accel_key][task.name] = {
                "total_cycles": (r.total_cycles if r is not None else None),
                "iso_area_speedup": speedup,
                "core_energy_uJ": c,
                "sram_energy_uJ": s,
                "dram_energy_uJ": d,
                "total_energy_uJ": total,
                "normalized_energy": normalized_energy,
            }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved results JSON: {path}")


def main():
    parser = argparse.ArgumentParser(description="MiX Energy & Speedup Simulator")
    parser.add_argument("--model", type=str, default="qwen2vl",
                        choices=["qwen2vl", "qwen3vl", "llava", "minicpm_v"],
                        help="Model architecture to simulate")
    parser.add_argument("--output_dir", type=str,
                        default="simulator/mix_simulator/output",
                        help="Output directory for charts and results")
    args = parser.parse_args()

    model = get_model_profile(args.model)
    accels = DEFAULT_ACCELS
    if args.model == "qwen3vl":
        tasks = QWEN3_TASKS
    elif args.model == "minicpm_v":
        tasks = MINICPM_TASKS
    else:
        tasks = DEFAULT_TASKS

    print(f"Model: {model.name}")
    print(f"Accelerators: {[a.name.replace(chr(10), ' ') for a in accels]}")
    print(f"Tasks: {[t.name for t in tasks]}")
    print(f"  Task details:")
    for t in tasks:
        print(f"    {t.name}: {t.num_images} img, {t.vision_tokens} vis_tok, "
              f"{t.input_text_tokens} txt_tok, {t.max_output_tokens} out_tok "
              f"(prefill={t.prefill_seq_len})")

    # Simulate: results[accel_name][task_name] = SimResult
    # Skip external_only accelerators (e.g., Focus) — their numbers come
    # from speedup_overrides / energy_overrides at JSON-write time.
    results: Dict[str, Dict[str, SimResult]] = {}
    for accel in accels:
        results[accel.name] = {}
        if accel.external_only:
            continue
        for task in tasks:
            results[accel.name][task.name] = simulate_task(model, accel, task)

    # Print summary
    print_results(results, accels, tasks, model_tag=args.model)

    # Save JSON
    save_results_json(results, accels, tasks, args.output_dir, model_tag=args.model)

    print(f"\nDone! Results saved to {args.output_dir}/"
          f"\nRun 'python -m simulator.mix_simulator.plot_fig9' to generate plots.")


if __name__ == "__main__":
    main()
