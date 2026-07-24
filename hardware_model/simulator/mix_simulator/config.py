"""Accelerator, task, and model configurations for the MiX energy simulator.

Each quantization format is modeled as a distinct accelerator with its own
grouped-PE systolic array specs from RTL synthesis (hardware.csv).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# AcceleratorConfig: unified format + hardware specification
# ---------------------------------------------------------------------------
@dataclass
class AcceleratorConfig:
    """Combined quantization format and hardware accelerator specification.

    Each format defines its own grouped-PE systolic array design.
    PE does dot(block_a, block_w) → 1 partial sum = block_size MACs/cycle.
    """
    name: str
    # --- Format properties ---
    w_eff_bits: float              # Weight effective bits (incl. metadata)
    a_eff_bits: float              # Activation effective bits
    bmm_q_eff_bits: float          # Q tensor effective bits for attention BMM
    bmm_kv_eff_bits: float         # K/V tensor effective bits for attention BMM
    # --- PE properties (from RTL synthesis / hardware.csv) ---
    block_size: int                # MACs per PE per cycle (= Group_Size: 32 or 16)
    pe_rows: int                   # Systolic array rows
    pe_cols: int                   # Systolic array columns
    power_mW: float                # Total systolic array power (milliwatts)
    area_um2: float                # Total systolic array area (µm², from RTL synthesis)
    # --- Memory subsystem ---
    freq_hz: float = 500e6
    sram_w_size_bytes: int = 288 * 1024
    sram_i_size_bytes: int = 288 * 1024
    sram_o_size_bytes: int = 128 * 1024   # Output accumulator SRAM
    acc_bits: int = 32                     # Accumulator precision (FP32)
    sram_rd_energy_pJ_per_byte: float = 0.641   # ARM SRAM Compiler 288KB
    sram_wr_energy_pJ_per_byte: float = 0.643
    dram_bw_bytes_per_sec: float = 68.26e9    # LPDDR5X-8533, 64-bit bus (JEDEC JESD209-5C): 8533 MT/s x 64 bit
    dram_energy_pJ_per_byte: float = 99.98
    # --- Plotting ---
    color: str = "#333333"
    # --- External-result overrides ---
    # When set, the simulator BYPASSES cycle/energy computation and uses
    # the provided per-(model_tag, task_name) numbers verbatim. Used for
    # reference accelerators (e.g., Focus) whose speedup is sparsity-driven
    # and not derivable from hardware specs alone.
    speedup_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    energy_overrides:  Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # If true, this accelerator is treated as an external reference and the
    # main loop should skip its `simulate_task()` call entirely.
    external_only: bool = False


# ---------------------------------------------------------------------------
# TaskConfig: per-benchmark workload specification
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig:
    """VLM benchmark task specification.

    Defines the token counts that determine FLOPs for each benchmark.
    LLM prefill seq_len = vision_tokens + input_text_tokens.
    """
    name: str
    num_images: int          # images per sample
    vision_tokens: int       # tokens from vision encoder (after projector)
    input_text_tokens: int   # text prompt tokens
    max_output_tokens: int   # decode steps (max_new_tokens)

    @property
    def prefill_seq_len(self) -> int:
        return self.vision_tokens + self.input_text_tokens


# ---------------------------------------------------------------------------
# Pre-defined accelerator configs (from hardware.csv RTL synthesis)
# ---------------------------------------------------------------------------
# All designs at 500 MHz. Effective bits:
#   g32 formats: 4 + 8/32 = 4.25 bits
#   g16 formats: 4 + 8/16 = 4.5 bits
#   INT4: 4.0 bits (pure), INT8: 8.0 bits

INT8 = AcceleratorConfig(
    name="INT8",
    w_eff_bits=8.0, a_eff_bits=8.0,
    bmm_q_eff_bits=8.0, bmm_kv_eff_bits=8.0,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=61.415, area_um2=95842.78,
    color="#A0A0A0",
)

MXINT4_G16 = AcceleratorConfig(
    name="MXINT4\ng16",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=34.242, area_um2=56668.50,
    color="#2E75B6",
)

MXFP4_G16 = AcceleratorConfig(
    name="MXFP4\ng16",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=33.858, area_um2=57422.48,
    color="#548235",
)

NVFP4 = AcceleratorConfig(
    name="NVFP4",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=36.472, area_um2=70390.66,
    color="#ED7D31",
)

FP16 = AcceleratorConfig(
    name="FP16",
    w_eff_bits=16.0, a_eff_bits=16.0,
    bmm_q_eff_bits=16.0, bmm_kv_eff_bits=16.0,
    block_size=1, pe_rows=16, pe_cols=32, power_mW=275.762, area_um2=734235.82,
    color="#666666",
)

MIX_INT4_G16 = AcceleratorConfig(
    name="MiX-INT4\ng16",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=31.822, area_um2=56124.81,
    color="#FF0000",
)

MIX_FP4_G16 = AcceleratorConfig(
    name="MiX-FP4\ng16",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=37.119, area_um2=65458.26,
    color="#D63384",
)

# 4.25-bit tier MiX-INT4 (group_size=32). From hardware.csv:MiX425b_MXINT4.
# 4x4 PE array, 512 MACs/cycle. Best area + power efficiency in the
# 4.25b tier (11.0 TOPs/mm^2, 29.1 TOPs/W at 500 MHz).
MIX_INT4 = AcceleratorConfig(
    name="MiX-INT4",
    w_eff_bits=4.25, a_eff_bits=4.25,
    bmm_q_eff_bits=4.25, bmm_kv_eff_bits=4.25,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=26.820, area_um2=46407.19,
    color="#7B1FA2",
)

# 4.5-bit FP4 outlier-aware baselines (reviewer-requested), iso-512 SAIF.
# MX+ stores the block-max at E2M3; AMXFP4 uses two asymmetric FP8 scales
# (its dual-scale datapath makes it the largest 4.5b array, ~2x others).
MXFP4_PLUS = AcceleratorConfig(
    name="MXFP4+",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=32.481, area_um2=58394.32,
    color="#9C6500",
)

AMXFP4 = AcceleratorConfig(
    name="AMXFP4",
    w_eff_bits=4.5, a_eff_bits=4.5,
    bmm_q_eff_bits=4.5, bmm_kv_eff_bits=4.5,
    block_size=32, pe_rows=4, pe_cols=4, power_mW=47.668, area_um2=115379.96,
    color="#C00000",
)

# ---------------------------------------------------------------------------
# Focus (HPCA 2026): sparsity-driven accelerator on FP16 dense systolic
# ---------------------------------------------------------------------------
# Focus runs on the SAME dense FP16 32x32 systolic array as our FP16 baseline,
# so its area/power match FP16 exactly. The actual speedup comes from the
# token-redundancy elimination algorithm and is loaded from external Focus
# simulator output via `load_focus_overrides()` below.
def _make_focus_accelerator() -> "AcceleratorConfig":
    speedup_overrides, energy_overrides = load_focus_overrides()
    return AcceleratorConfig(
        name="Focus",
        # Format: dense FP16 (no quantization)
        w_eff_bits=16.0, a_eff_bits=16.0,
        bmm_q_eff_bits=16.0, bmm_kv_eff_bits=16.0,
        # Hardware: identical to FP16 baseline (16x32 dense systolic, iso-512 @ 500MHz)
        block_size=1, pe_rows=16, pe_cols=32,
        power_mW=275.762, area_um2=734235.82,
        color="#9467bd",
        speedup_overrides=speedup_overrides,
        energy_overrides=energy_overrides,
        external_only=True,
    )


def load_focus_overrides():
    """Load per-(model_tag, task_name) Focus speedup + energy from JSON.

    Returns (speedup_overrides, energy_overrides) where:
      speedup_overrides[model_tag][task_name] = float
      energy_overrides[model_tag][task_name]  = {core_uJ, sram_uJ, dram_uJ}

    The JSON lives at simulator/focus_eval/focus_results.json. If the file
    is missing or empty, returns empty dicts (the simulator will then leave
    Focus's task entries as None and the plot script will skip them).
    """
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.normpath(
        os.path.join(here, "..", "focus_eval", "focus_results.json"))
    if not os.path.exists(json_path):
        return {}, {}
    with open(json_path) as f:
        raw = json.load(f)
    speedup, energy = {}, {}
    for model_tag, tasks in raw.items():
        if model_tag.startswith("_") or not isinstance(tasks, dict):
            continue  # skip comment fields
        speedup[model_tag] = {}
        energy[model_tag] = {}
        for task_name, entry in tasks.items():
            if not isinstance(entry, dict):
                continue
            if "speedup" in entry:
                speedup[model_tag][task_name] = float(entry["speedup"])
            if "core_uJ" in entry and "sram_uJ" in entry and "dram_uJ" in entry:
                energy[model_tag][task_name] = {
                    "core_uJ": float(entry["core_uJ"]),
                    "sram_uJ": float(entry["sram_uJ"]),
                    "dram_uJ": float(entry["dram_uJ"]),
                }
    return speedup, energy


FOCUS = _make_focus_accelerator()


DEFAULT_ACCELS = [
    FP16, INT8, MXINT4_G16, MXFP4_G16,
    NVFP4, MXFP4_PLUS, AMXFP4, MIX_INT4_G16, MIX_FP4_G16, MIX_INT4, FOCUS,
]


# ---------------------------------------------------------------------------
# Pre-defined task configs (typical per-benchmark token counts)
# ---------------------------------------------------------------------------
# input_text_tokens: measured mean prompt tokens + ~15 chat template overhead.
#   Measured with Qwen2-VL tokenizer over full benchmark splits.
#   Chat template overhead: Qwen2-VL ~22 tokens, Qwen3-VL ~11 tokens.
# max_output_tokens: measured actual average output lengths (not max_new_tokens caps).
#   Qwen3-VL generates much longer outputs on MC benchmarks (chain-of-thought).
#   Use QWEN2_TASKS for Qwen2-VL/LLaVA, QWEN3_TASKS for Qwen3-VL.
# vision_tokens: model-dependent (Qwen2-VL: 256/img, Qwen3-VL: 196/img, LLaVA: 729).
#   Set to 256 here (Qwen2-VL default); override per model as needed.
# num_images: MMMU has 95.2% single-image samples (mean=1.09), use 1.

CHARTQA = TaskConfig(
    name="ChartQA", num_images=1,
    vision_tokens=256, input_text_tokens=40, max_output_tokens=4,
)

TEXTVQA = TaskConfig(
    name="TextVQA", num_images=1,
    vision_tokens=256, input_text_tokens=35, max_output_tokens=4,
)

VIZWIZ = TaskConfig(
    name="VizWiz", num_images=1,
    vision_tokens=256, input_text_tokens=45, max_output_tokens=4,
)

OCRBENCH = TaskConfig(
    name="OCRBench", num_images=1,
    vision_tokens=256, input_text_tokens=30, max_output_tokens=16,
)

# MMMU is evaluated at its 128-token generation cap (chain-of-thought): unlike the
# short-answer benchmarks (~1-2 tokens, compute-bound), the long decode re-reads all
# weights each step and shifts MMMU into the DRAM-bound regime.
MMMU = TaskConfig(
    name="MMMU", num_images=1,
    vision_tokens=256, input_text_tokens=110, max_output_tokens=128,
)

SEEDBENCH = TaskConfig(
    name="SEED-2+", num_images=1,
    vision_tokens=256, input_text_tokens=70, max_output_tokens=2,
)

DEFAULT_TASKS = [CHARTQA, TEXTVQA, VIZWIZ, OCRBENCH, MMMU, SEEDBENCH]

# --- Qwen3-VL task overrides (chain-of-thought generates much longer outputs) ---
# Measured on 30 samples: MMMU mean=680 tokens, SEED-2+ mean=131 tokens.
# Short-answer benchmarks are similar to Qwen2-VL (~3 tokens).
MMMU_Q3 = TaskConfig(
    name="MMMU", num_images=1,
    vision_tokens=196, input_text_tokens=110, max_output_tokens=680,
)

SEEDBENCH_Q3 = TaskConfig(
    name="SEED-2+", num_images=1,
    vision_tokens=196, input_text_tokens=70, max_output_tokens=131,
)

CHARTQA_Q3 = TaskConfig(
    name="ChartQA", num_images=1,
    vision_tokens=196, input_text_tokens=40, max_output_tokens=3,
)

TEXTVQA_Q3 = TaskConfig(
    name="TextVQA", num_images=1,
    vision_tokens=196, input_text_tokens=35, max_output_tokens=3,
)

VIZWIZ_Q3 = TaskConfig(
    name="VizWiz", num_images=1,
    vision_tokens=196, input_text_tokens=45, max_output_tokens=3,
)

OCRBENCH_Q3 = TaskConfig(
    name="OCRBench", num_images=1,
    vision_tokens=196, input_text_tokens=30, max_output_tokens=2,
)

QWEN3_TASKS = [CHARTQA_Q3, TEXTVQA_Q3, VIZWIZ_Q3, OCRBENCH_Q3, MMMU_Q3, SEEDBENCH_Q3]


# --- MiniCPM-V-2.6 task overrides (resampler reduces vision to 64 tokens) ---
# MiniCPM-V's resampler outputs only 64 visual tokens regardless of input
# image resolution, so the LLM-side prefill is much shorter than Qwen2-VL.
CHARTQA_MINICPM = TaskConfig(
    name="ChartQA", num_images=1,
    vision_tokens=64, input_text_tokens=40, max_output_tokens=4,
)
TEXTVQA_MINICPM = TaskConfig(
    name="TextVQA", num_images=1,
    vision_tokens=64, input_text_tokens=35, max_output_tokens=4,
)
VIZWIZ_MINICPM = TaskConfig(
    name="VizWiz", num_images=1,
    vision_tokens=64, input_text_tokens=45, max_output_tokens=4,
)
OCRBENCH_MINICPM = TaskConfig(
    name="OCRBench", num_images=1,
    vision_tokens=64, input_text_tokens=30, max_output_tokens=16,
)
MMMU_MINICPM = TaskConfig(
    name="MMMU", num_images=1,
    vision_tokens=64, input_text_tokens=110, max_output_tokens=128,
)
SEEDBENCH_MINICPM = TaskConfig(
    name="SEED-2+", num_images=1,
    vision_tokens=64, input_text_tokens=70, max_output_tokens=2,
)
MINICPM_TASKS = [CHARTQA_MINICPM, TEXTVQA_MINICPM, VIZWIZ_MINICPM,
                 OCRBENCH_MINICPM, MMMU_MINICPM, SEEDBENCH_MINICPM]
