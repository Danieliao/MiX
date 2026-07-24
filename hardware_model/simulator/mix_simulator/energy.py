"""Core simulation engine: compute cycles and energy (Core, SRAM, DRAM) per layer.

Grouped-PE output-stationary systolic array model:
- Each PE computes dot(block_a, block_w) → 1 partial sum = block_size MACs/cycle
- Cycles: max(compute_cycles, dram_cycles) per layer
- Core energy: compute_cycles / freq × power_mW (from RTL synthesis)
- Two-level tiling: matrix-level (check first) then K-tile-level
- SRAM energy: separate read/write accounting; writes = DRAM reads; stationary re-reads
- Output buffer: FP32 accumulators, spill to DRAM if output SRAM too small
"""

import math
from dataclasses import dataclass
from typing import Dict

from .config import AcceleratorConfig, TaskConfig
from .model_profile import ModelProfile, LinearOp, AttentionBMM


@dataclass
class LayerResult:
    """Energy and cycle results for a single layer."""
    name: str
    compute_cycles: float
    dram_cycles: float
    total_cycles: float
    core_energy_pJ: float
    sram_energy_pJ: float
    dram_energy_pJ: float
    total_energy_pJ: float


@dataclass
class SimResult:
    """Aggregated simulation results."""
    accel_name: str
    task_name: str

    total_cycles: float = 0.0
    compute_cycles: float = 0.0
    dram_cycles: float = 0.0
    core_energy_pJ: float = 0.0
    sram_energy_pJ: float = 0.0
    dram_energy_pJ: float = 0.0

    @property
    def total_energy_pJ(self) -> float:
        return self.core_energy_pJ + self.sram_energy_pJ + self.dram_energy_pJ

    def add(self, lr: LayerResult):
        self.total_cycles += lr.total_cycles
        self.compute_cycles += lr.compute_cycles
        self.dram_cycles += lr.dram_cycles
        self.core_energy_pJ += lr.core_energy_pJ
        self.sram_energy_pJ += lr.sram_energy_pJ
        self.dram_energy_pJ += lr.dram_energy_pJ


def _sim_linear(
    M: int, K: int, N: int, count: int,
    w_bits: float, a_bits: float,
    accel: AcceleratorConfig,
) -> LayerResult:
    """Simulate one type of linear layer (repeated `count` times).

    GEMM: C(M, N) = A(M, K) × W^T(K, N)
    Output-stationary dataflow with grouped-PE systolic array.
    """
    # -- Compute cycles --
    total_macs = M * K * N * count
    macs_per_cycle = accel.block_size * accel.pe_rows * accel.pe_cols
    compute_cycles = math.ceil(total_macs / macs_per_cycle)

    # -- Data sizes (total across all `count` instances) --
    w_bytes = K * N * w_bits / 8 * count
    a_bytes = M * K * a_bits / 8 * count

    # -- Two-level tiling --
    # Effective tile dimensions: pe_dim × block_size (e.g., 8×16=128 or 4×32=128)
    eff_tile_m = accel.pe_rows * accel.block_size
    eff_tile_n = accel.pe_cols * accel.block_size

    # Per-layer sizes (for SRAM fit checks)
    w_matrix_bytes = K * N * w_bits / 8
    a_matrix_bytes = M * K * a_bits / 8
    w_tile_bytes = eff_tile_n * K * w_bits / 8   # pe_cols×block_size weight columns, K rows
    a_tile_bytes = eff_tile_m * K * a_bits / 8   # pe_rows×block_size activation rows, K cols
    num_m_tiles = math.ceil(M / eff_tile_m)
    num_n_tiles = math.ceil(N / eff_tile_n)

    # Level 1: Matrix-level (check first)
    w_matrix_fits = (w_matrix_bytes <= accel.sram_w_size_bytes)
    a_matrix_fits = (a_matrix_bytes <= accel.sram_i_size_bytes)

    # Track stationarity for SRAM read accounting
    k_mult_w = 1
    k_mult_a = 1
    matrix_level_fits = False
    n_outer_chosen = True  # default

    if w_matrix_fits or a_matrix_fits:
        # One side fits entirely → permanently stationary, no reloading
        matrix_level_fits = True
        w_bytes_dram = w_bytes
        a_bytes_dram = a_bytes
    else:
        # Neither matrix fits → check K-tile level, then M/N reloading

        # Level 2: K-tile-level
        num_k_tiles_w = max(1, math.ceil(w_tile_bytes / accel.sram_w_size_bytes))
        num_k_tiles_a = max(1, math.ceil(a_tile_bytes / accel.sram_i_size_bytes))

        if num_k_tiles_w > 1 and num_k_tiles_a > 1:
            # Both overflow K → must reload one side per K-partition of the other
            cost_w_k = w_tile_bytes + a_tile_bytes * num_k_tiles_w
            cost_a_k = a_tile_bytes + w_tile_bytes * num_k_tiles_a
            if cost_w_k <= cost_a_k:
                k_mult_w = 1
                k_mult_a = num_k_tiles_w
            else:
                k_mult_w = num_k_tiles_a
                k_mult_a = 1
        # else: at most one overflows K → fitting side stays, no K-refetching

        # M/N-level reloading
        cost_n_outer = w_bytes * k_mult_w + a_bytes * k_mult_a * num_n_tiles
        cost_m_outer = a_bytes * k_mult_a + w_bytes * k_mult_w * num_m_tiles
        if cost_n_outer <= cost_m_outer:
            n_outer_chosen = True
            w_bytes_dram = w_bytes * k_mult_w
            a_bytes_dram = a_bytes * k_mult_a * num_n_tiles
        else:
            n_outer_chosen = False
            w_bytes_dram = w_bytes * k_mult_w * num_m_tiles
            a_bytes_dram = a_bytes * k_mult_a

    # -- Output DRAM --
    output_buffer_needed = eff_tile_m * N * accel.acc_bits / 8
    if output_buffer_needed <= accel.sram_o_size_bytes:
        # Accumulate in SRAM → SFU → quantize → write quantized to DRAM
        o_dram_bytes = M * N * w_bits / 8 * count
    else:
        # Spill FP32 to DRAM (write) + read back + write quantized
        fp32_bytes = M * N * accel.acc_bits / 8 * count
        o_dram_bytes = 2 * fp32_bytes + M * N * w_bits / 8 * count

    # -- Cycles --
    total_dram_bytes = w_bytes_dram + a_bytes_dram + o_dram_bytes
    dram_bytes_per_cycle = accel.dram_bw_bytes_per_sec / accel.freq_hz
    dram_cycles = total_dram_bytes / dram_bytes_per_cycle
    total_cycles = max(compute_cycles, dram_cycles)

    # -- Core energy (power-based from RTL synthesis) --
    core_energy = compute_cycles / accel.freq_hz * accel.power_mW * 1e9  # pJ

    # -- SRAM energy (separate read/write) --
    o_sram_bytes = M * N * accel.acc_bits / 8 * count  # 1 read + 1 write per output element

    # SRAM writes = DRAM reads (every DRAM fetch → SRAM write) + output accumulator writes
    sram_write_bytes = w_bytes_dram + a_bytes_dram + o_sram_bytes

    # SRAM reads: stationary data re-read per non-stationary tile arrival
    if matrix_level_fits:
        if w_matrix_fits:
            sram_read_bytes = w_bytes * num_m_tiles + a_bytes
        else:
            sram_read_bytes = a_bytes * num_n_tiles + w_bytes
    elif n_outer_chosen:
        # n-outer: weight stays per n_tile, re-read for each m_tile
        sram_read_bytes = w_bytes * k_mult_w * num_m_tiles + a_bytes_dram
    else:
        # m-outer: activation stays per m_tile, re-read for each n_tile
        sram_read_bytes = a_bytes * k_mult_a * num_n_tiles + w_bytes_dram
    sram_read_bytes += o_sram_bytes

    sram_energy = (sram_write_bytes * accel.sram_wr_energy_pJ_per_byte +
                   sram_read_bytes * accel.sram_rd_energy_pJ_per_byte)

    # -- DRAM energy --
    dram_energy = total_dram_bytes * accel.dram_energy_pJ_per_byte

    return LayerResult(
        name="linear", compute_cycles=compute_cycles,
        dram_cycles=dram_cycles, total_cycles=total_cycles,
        core_energy_pJ=core_energy, sram_energy_pJ=sram_energy,
        dram_energy_pJ=dram_energy,
        total_energy_pJ=core_energy + sram_energy + dram_energy,
    )


def _sim_bmm(
    M_q: int, M_kv: int, num_heads: int, head_dim: int, count: int,
    q_bits: float, kv_bits: float,
    accel: AcceleratorConfig,
) -> LayerResult:
    """Simulate attention BMM (Q@K^T + attn@V) for `count` layers.

    Decomposes into two _sim_linear calls with proper tiling.
    """
    total_count = num_heads * count

    # Q @ K^T: (M_q, head_dim) × (head_dim, M_kv)
    lr_qk = _sim_linear(M_q, head_dim, M_kv, total_count,
                         kv_bits, q_bits, accel)

    # attn @ V: (M_q, M_kv) × (M_kv, head_dim)
    lr_sv = _sim_linear(M_q, M_kv, head_dim, total_count,
                         kv_bits, q_bits, accel)

    return LayerResult(
        name="bmm",
        compute_cycles=lr_qk.compute_cycles + lr_sv.compute_cycles,
        dram_cycles=lr_qk.dram_cycles + lr_sv.dram_cycles,
        total_cycles=lr_qk.total_cycles + lr_sv.total_cycles,
        core_energy_pJ=lr_qk.core_energy_pJ + lr_sv.core_energy_pJ,
        sram_energy_pJ=lr_qk.sram_energy_pJ + lr_sv.sram_energy_pJ,
        dram_energy_pJ=lr_qk.dram_energy_pJ + lr_sv.dram_energy_pJ,
        total_energy_pJ=lr_qk.total_energy_pJ + lr_sv.total_energy_pJ,
    )


def simulate_task(
    model: ModelProfile,
    accel: AcceleratorConfig,
    task: TaskConfig,
) -> SimResult:
    """Simulate a full VLM inference for one task on one accelerator.

    Phases:
    1. Vision encoder (per image): M = num_vision_patches
    2. Projector: M = num_vision_patches / spatial_merge_size^2
    3. LLM prefill: M = vision_tokens + text_tokens
    4. LLM decode: M = 1 for max_output_tokens steps
    """
    result = SimResult(accel_name=accel.name, task_name=task.name)

    # --- Phase 1: Vision encoder (run once per image) ---
    M_v = model.num_vision_patches
    for layer in model.vision_linears:
        lr = _sim_linear(M_v, layer.in_features, layer.out_features,
                         layer.count * task.num_images,
                         accel.w_eff_bits, accel.a_eff_bits, accel)
        result.add(lr)
    for bmm in model.vision_bmms:
        lr = _sim_bmm(M_v, M_v, bmm.num_heads, bmm.head_dim,
                      bmm.count * task.num_images,
                      accel.bmm_q_eff_bits, accel.bmm_kv_eff_bits, accel)
        result.add(lr)

    # --- Phase 2: Projector / Merger ---
    merge_factor = model.spatial_merge_size ** 2
    M_p = (model.num_vision_patches // merge_factor) * task.num_images
    for layer in model.proj_linears:
        lr = _sim_linear(M_p, layer.in_features, layer.out_features, layer.count,
                         accel.w_eff_bits, accel.a_eff_bits, accel)
        result.add(lr)

    # --- Phase 3: LLM prefill ---
    M_l = task.prefill_seq_len
    for layer in model.llm_linears:
        lr = _sim_linear(M_l, layer.in_features, layer.out_features, layer.count,
                         accel.w_eff_bits, accel.a_eff_bits, accel)
        result.add(lr)
    for bmm in model.llm_bmms:
        lr = _sim_bmm(M_l, M_l, bmm.num_heads, bmm.head_dim, bmm.count,
                      accel.bmm_q_eff_bits, accel.bmm_kv_eff_bits, accel)
        result.add(lr)

    # --- Phase 4: LLM decode ---
    for t in range(task.max_output_tokens):
        M_q = 1
        M_kv = M_l + t  # KV cache grows each step

        for layer in model.llm_linears:
            lr = _sim_linear(M_q, layer.in_features, layer.out_features, layer.count,
                             accel.w_eff_bits, accel.a_eff_bits, accel)
            result.add(lr)

        for bmm in model.llm_bmms:
            lr = _sim_bmm(M_q, M_kv, bmm.num_heads, bmm.head_dim, bmm.count,
                          accel.bmm_q_eff_bits, accel.bmm_kv_eff_bits, accel)
            result.add(lr)

    return result
