"""Extract layer shapes from VLM model architectures.

Each model is described as a list of LayerOp (linear GEMM or attention BMM).
The 'M' dimension (tokens) is filled in at simulation time based on WorkloadConfig.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LinearOp:
    """A linear layer GEMM: output = input @ weight.T   (M, K) @ (K, N)."""
    name: str
    in_features: int   # K
    out_features: int  # N
    count: int         # how many instances (e.g., 28 decoder layers)


@dataclass
class AttentionBMM:
    """Attention BMM pair: Q@K^T and attn@V.
    Dimensions per head:
      Q@K^T : (M_q, head_dim, M_kv)  — M_q query tokens, M_kv key tokens
      attn@V: (M_q, M_kv, head_dim)
    Total heads = num_heads.
    """
    name: str
    num_heads: int
    head_dim: int
    count: int  # how many attention layers


@dataclass
class ModelProfile:
    """Complete model profile with vision, projector, and LLM components."""
    name: str
    vision_linears: List[LinearOp]
    vision_bmms: List[AttentionBMM]
    proj_linears: List[LinearOp]
    llm_linears: List[LinearOp]
    llm_bmms: List[AttentionBMM]
    # Vision-specific: number of patches (M for vision encoder)
    num_vision_patches: int
    # Spatial merge factor for projector/merger (M_proj = num_vision_patches / spatial_merge_size^2)
    # Qwen2/3-VL: 2 (merges 2x2 patches). LLaVA: 1 (no spatial merge in projector).
    spatial_merge_size: int = 1


# ---------------------------------------------------------------------------
# Qwen2-VL-7B-Instruct
# ---------------------------------------------------------------------------
def qwen2_vl_7b() -> ModelProfile:
    """Qwen2-VL-7B architecture dimensions.

    From HF config (Qwen/Qwen2-VL-7B-Instruct):
      Vision: depth=32, embed_dim=1280, num_heads=16, mlp_ratio=4
      Merger: spatial_merge_size=2, hidden_size=3584
      LLM: 28 layers, hidden=3584, intermediate=18944, heads=28, kv_heads=4
    """
    # --- Vision encoder (Qwen2VisionTransformerPretrainedModel) ---
    # 32 layers, embed_dim=1280, num_heads=16, head_dim=80, mlp_ratio=4
    v_depth = 32
    v_dim = 1280
    v_heads = 16
    v_head_dim = 80
    v_mlp_dim = v_dim * 4  # 5120

    vision_linears = [
        LinearOp("vision.qkv_proj", v_dim, v_dim * 3, v_depth),  # fused QKV
        LinearOp("vision.out_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.mlp.fc1", v_dim, v_mlp_dim, v_depth),
        LinearOp("vision.mlp.fc2", v_mlp_dim, v_dim, v_depth),
    ]
    vision_bmms = [
        AttentionBMM("vision.attn", v_heads, v_head_dim, v_depth),
    ]

    # --- Projector (PatchMerger) ---
    # Merges 2×2 patches: in = 1280*4 = 5120, out = 3584
    proj_linears = [
        LinearOp("projector.mlp.0", 5120, 5120, 1),
        LinearOp("projector.mlp.2", 5120, 3584, 1),
    ]

    # --- LLM (Qwen2, 28 layers) ---
    # hidden=3584, intermediate=18944, heads=28, kv_heads=4, head_dim=128
    l_depth = 28
    l_dim = 3584
    l_inter = 18944
    l_heads = 28
    l_kv_heads = 4
    l_head_dim = 128

    llm_linears = [
        LinearOp("llm.q_proj", l_dim, l_heads * l_head_dim, l_depth),
        LinearOp("llm.k_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.v_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.o_proj", l_heads * l_head_dim, l_dim, l_depth),
        LinearOp("llm.gate_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.up_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.down_proj", l_inter, l_dim, l_depth),
    ]
    llm_bmms = [
        AttentionBMM("llm.attn", l_heads, l_head_dim, l_depth),
    ]

    # patch_size=14: for ~448x448 image → 32x32 = 1024 patches
    # After spatial merge (2x2): 1024/4 = 256 vision tokens
    return ModelProfile(
        name="Qwen2-VL-7B",
        vision_linears=vision_linears,
        vision_bmms=vision_bmms,
        proj_linears=proj_linears,
        llm_linears=llm_linears,
        llm_bmms=llm_bmms,
        num_vision_patches=1024,
        spatial_merge_size=2,
    )


# ---------------------------------------------------------------------------
# Qwen3-VL-8B
# ---------------------------------------------------------------------------
def qwen3_vl_8b() -> ModelProfile:
    """Qwen3-VL-8B architecture dimensions.

    From HF config (Qwen/Qwen3-VL-8B-Instruct):
      Vision: depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304
      Merger: spatial_merge_size=2, out_hidden_size=4096
      Deepstack: visual_indexes=[8, 16, 24] — 3 extra mergers
      LLM: 36 layers, hidden=4096, intermediate=12288, heads=32, kv_heads=8
    """
    # --- Vision encoder (27 layers, NOT 32 like Qwen2-VL) ---
    v_depth = 27
    v_dim = 1152
    v_heads = 16
    v_head_dim = 72   # 1152 / 16
    v_mlp_dim = 4304  # from config intermediate_size

    vision_linears = [
        LinearOp("vision.qkv_proj", v_dim, v_dim * 3, v_depth),
        LinearOp("vision.out_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.mlp.fc1", v_dim, v_mlp_dim, v_depth),
        LinearOp("vision.mlp.fc2", v_mlp_dim, v_dim, v_depth),
    ]
    vision_bmms = [
        AttentionBMM("vision.attn", v_heads, v_head_dim, v_depth),
    ]

    # --- Projector: main merger + 3 deepstack mergers ---
    # spatial_merge_size=2: merges 2x2 patches → input dim = 1152 * 4 = 4608
    # out_hidden_size=4096 (matches LLM hidden_size)
    merger_in = v_dim * 4  # 4608
    merger_out = 4096
    proj_linears = [
        # Main merger (runs on output of layer 27)
        LinearOp("merger.fc1", merger_in, merger_in, 1),
        LinearOp("merger.fc2", merger_in, merger_out, 1),
        # Deepstack mergers at layers [8, 16, 24] — 3 identical mergers
        LinearOp("deepstack_merger.fc1", merger_in, merger_in, 3),
        LinearOp("deepstack_merger.fc2", merger_in, merger_out, 3),
    ]

    # --- LLM (Qwen3, 36 layers) ---
    # hidden=4096, intermediate=12288, heads=32, kv_heads=8, head_dim=128
    l_depth = 36
    l_dim = 4096
    l_inter = 12288
    l_heads = 32
    l_kv_heads = 8
    l_head_dim = 128

    llm_linears = [
        LinearOp("llm.q_proj", l_dim, l_heads * l_head_dim, l_depth),
        LinearOp("llm.k_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.v_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.o_proj", l_heads * l_head_dim, l_dim, l_depth),
        LinearOp("llm.gate_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.up_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.down_proj", l_inter, l_dim, l_depth),
    ]
    llm_bmms = [
        AttentionBMM("llm.attn", l_heads, l_head_dim, l_depth),
    ]

    # patch_size=16: for ~448x448 image → 28x28 = 784 patches
    # After spatial merge (2x2): 784/4 = 196 vision tokens
    return ModelProfile(
        name="Qwen3-VL-8B",
        vision_linears=vision_linears,
        vision_bmms=vision_bmms,
        proj_linears=proj_linears,
        llm_linears=llm_linears,
        llm_bmms=llm_bmms,
        num_vision_patches=784,
        spatial_merge_size=2,
    )


# ---------------------------------------------------------------------------
# LLaVA-OneVision-7B
# ---------------------------------------------------------------------------
def llava_onevision_7b() -> ModelProfile:
    """LLaVA-OneVision-7B architecture dimensions.

    From HF config (llava-hf/llava-onevision-qwen2-7b-ov-hf):
      Vision (Siglip): num_hidden_layers=26, hidden=1152, heads=16, intermediate=4304
      LLM (Qwen2): 28 layers, hidden=3584, intermediate=18944, heads=28, kv_heads=4
    """
    # --- Vision encoder (SiglipVisionModel) ---
    # 26 layers (verified from model.vision_tower.vision_model.encoder.layers)
    v_depth = 26
    v_dim = 1152
    v_heads = 16
    v_head_dim = 72
    v_mlp_dim = 4304  # Siglip intermediate_size

    vision_linears = [
        LinearOp("vision.q_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.k_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.v_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.out_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.mlp.fc1", v_dim, v_mlp_dim, v_depth),
        LinearOp("vision.mlp.fc2", v_mlp_dim, v_dim, v_depth),
    ]
    vision_bmms = [
        AttentionBMM("vision.attn", v_heads, v_head_dim, v_depth),
    ]

    # --- Projector (MLP) ---
    # 2-layer MLP: 1152 → 3584 → 3584
    proj_linears = [
        LinearOp("projector.linear_1", v_dim, 3584, 1),
        LinearOp("projector.linear_2", 3584, 3584, 1),
    ]

    # --- LLM (Qwen2, 28 layers) ---
    l_depth = 28
    l_dim = 3584
    l_inter = 18944
    l_heads = 28
    l_kv_heads = 4
    l_head_dim = 128

    llm_linears = [
        LinearOp("llm.q_proj", l_dim, l_heads * l_head_dim, l_depth),
        LinearOp("llm.k_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.v_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.o_proj", l_heads * l_head_dim, l_dim, l_depth),
        LinearOp("llm.gate_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.up_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.down_proj", l_inter, l_dim, l_depth),
    ]
    llm_bmms = [
        AttentionBMM("llm.attn", l_heads, l_head_dim, l_depth),
    ]

    # Siglip: 384×384 image / 14 patch = 27×27 + 1 CLS = 730 patches (approx)
    return ModelProfile(
        name="LLaVA-OneVision-7B",
        vision_linears=vision_linears,
        vision_bmms=vision_bmms,
        proj_linears=proj_linears,
        llm_linears=llm_linears,
        llm_bmms=llm_bmms,
        num_vision_patches=729,
    )


# ---------------------------------------------------------------------------
# MiniCPM-V-2.6
# ---------------------------------------------------------------------------
def minicpm_v_2_6() -> ModelProfile:
    """MiniCPM-V-2.6 architecture dimensions.

    From HF config (openbmb/MiniCPM-V-2_6):
      Vision (Siglip-400M-384): 27 layers, hidden=1152, heads=16, intermediate=4304
      Resampler: cross-attention with 64 learnable queries; 1152 -> 3584
      LLM (Qwen2-7B): 28 layers, hidden=3584, intermediate=18944, heads=28, kv_heads=4
    """
    # --- Vision encoder (Siglip-400M-384) ---
    # 27 layers, embed_dim=1152, num_heads=16, head_dim=72, mlp_ratio=3.736
    v_depth = 27
    v_dim = 1152
    v_heads = 16
    v_head_dim = 72
    v_mlp_dim = 4304  # Siglip intermediate_size

    vision_linears = [
        LinearOp("vision.q_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.k_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.v_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.out_proj", v_dim, v_dim, v_depth),
        LinearOp("vision.mlp.fc1", v_dim, v_mlp_dim, v_depth),
        LinearOp("vision.mlp.fc2", v_mlp_dim, v_dim, v_depth),
    ]
    vision_bmms = [
        AttentionBMM("vision.attn", v_heads, v_head_dim, v_depth),
    ]

    # --- Resampler (cross-attention with 64 queries) ---
    # Modeled as Q/K/V/O projections + a single cross-attention BMM.
    # Inputs: 64 query tokens (Q) cross-attending to 729 vision keys/values.
    # Output: 64 tokens at the LLM hidden size (3584).
    l_dim = 3584
    proj_linears = [
        LinearOp("resampler.q_proj", l_dim, l_dim, 1),
        LinearOp("resampler.k_proj", v_dim, l_dim, 1),
        LinearOp("resampler.v_proj", v_dim, l_dim, 1),
        LinearOp("resampler.out_proj", l_dim, l_dim, 1),
    ]

    # --- LLM (Qwen2-7B, 28 layers) ---
    l_depth = 28
    l_inter = 18944
    l_heads = 28
    l_kv_heads = 4
    l_head_dim = 128

    llm_linears = [
        LinearOp("llm.q_proj", l_dim, l_heads * l_head_dim, l_depth),
        LinearOp("llm.k_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.v_proj", l_dim, l_kv_heads * l_head_dim, l_depth),
        LinearOp("llm.o_proj", l_heads * l_head_dim, l_dim, l_depth),
        LinearOp("llm.gate_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.up_proj", l_dim, l_inter, l_depth),
        LinearOp("llm.down_proj", l_inter, l_dim, l_depth),
    ]
    llm_bmms = [
        AttentionBMM("llm.attn", l_heads, l_head_dim, l_depth),
    ]

    # Siglip-384 -> 27x27 = 729 patches; resampler reduces to 64 tokens.
    # num_vision_patches drives vision encoder seq_len; the LLM sees only
    # 64 vision tokens (set via task.vision_tokens in MINICPM_TASKS).
    return ModelProfile(
        name="MiniCPM-V-2.6",
        vision_linears=vision_linears,
        vision_bmms=vision_bmms,
        proj_linears=proj_linears,
        llm_linears=llm_linears,
        llm_bmms=llm_bmms,
        num_vision_patches=729,
        spatial_merge_size=1,  # resampler is cross-attention, not spatial merge
    )


MODEL_REGISTRY = {
    "qwen2vl": qwen2_vl_7b,
    "qwen3vl": qwen3_vl_8b,
    "llava": llava_onevision_7b,
    "minicpm_v": minicpm_v_2_6,
}


def get_model_profile(name: str) -> ModelProfile:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
