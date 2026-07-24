"""
Low precision attention modules
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Callable

from transformers.models.llama.configuration_llama import LlamaConfig
from src.module.base import _QBase, _QBaseLinear
from src.module.ops import FloatMatMul, BatchIntMatMul, BatchHeadIntMatMul
from src.module.fuse import MulShift
from src.quantization.smoothquant import SmoothQuantizer
from src.quantization.mxint import MXINTActivationQuantizer
from src.quantization.inverted_mxint import InvertedMXINTActivationQuantizer
from src.quantization.mxfp4 import (
    MXFP4ActivationQuantizer,
    MXFP4CeilActivationQuantizer,
    MXFP4PlusActivationQuantizer,
    MXFP4PlusCeilActivationQuantizer,
    AMXFP4ActivationQuantizer,
)
from src.quantization.nvfp4 import NVFP4ActivationQuantizer


from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, eager_attention_forward
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.siglip.modeling_siglip import SiglipAttention
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


SUPPORTED_BMM_QTYPES = (
    "smooth_quant",
    "mxint",
    "inverted_mx",
    "mixed_mxint_invmx",
    "mixed_invmx_mxint",
    "mxfp4",
    "mxfp4_ceil",
    "mxfp4_plus",
    "mxfp4_plus_ceil",
    "amxfp4",
    "nvfp4",
    "mixed_mxfp4_invmx",
    "mixed_invmx_mxfp4",
    "mixed_mxfp4_ceil_invmx",
    "mixed_invmx_mxfp4_ceil",
)


# Quantizers whose statistics never cross tokens: every scale/exponent is
# computed within blocks along the last (head_dim) axis. For these, quantizing
# K/V token-by-token before the KV-cache insert is bit-identical to
# re-quantizing the full cached sequence at every decode step, and quantizing
# before repeat_kv is bit-identical to after (it merely duplicates heads).
# NVFP4 (dynamic per-tensor scale) and SmoothQuant (cross-token smoothing) are
# excluded and keep the original re-quantize-every-step path.
_TOKEN_LOCAL_BMM_QUANTIZERS = (
    MXINTActivationQuantizer,
    InvertedMXINTActivationQuantizer,
    MXFP4ActivationQuantizer,
    MXFP4CeilActivationQuantizer,
    MXFP4PlusActivationQuantizer,
    MXFP4PlusCeilActivationQuantizer,
    AMXFP4ActivationQuantizer,
)


def _token_local(*quantizers) -> bool:
    return all(isinstance(q, _TOKEN_LOCAL_BMM_QUANTIZERS) for q in quantizers)


def _k_smooth_mean(attn, key_states, cache_position):
    """Per-channel Key mean to subtract before K quantization (K-smoothing).

    Subtracting a constant per-channel vector from K is exact for attention:
    softmax(Q @ K^T) == softmax(Q @ (K - mean)^T), so this only reduces the
    quantization error of the (often outlier-heavy) Key tensor.

    Two flows, selected by ``attn.k_smooth_static``:
    - Dynamic (default, ``False``): recompute the mean over the *full current
      sequence* every forward. Faithful but hardware-costly — every decode step
      must re-reduce over the whole KV cache.
    - Static (``True``): compute the mean *once at prefill* (the forward whose
      ``cache_position`` starts at 0), freeze it, and reuse it for every decode
      step. This is the hardware-efficient flow: an FP32 accumulator + divider at
      prefill, then one constant subtractor per output channel at decode — no
      KV-cache re-reduction. The frozen mean auto-resets each sample because every
      new sample begins with a prefill at position 0.
    """
    if not getattr(attn, "k_smooth_static", False):
        return key_states.mean(dim=-2, keepdim=True)
    is_prefill = cache_position is None or int(cache_position[0]) == 0
    if is_prefill or getattr(attn, "_k_static_mean", None) is None:
        attn._k_static_mean = key_states.mean(dim=-2, keepdim=True).detach()
    return attn._k_static_mean


def _create_attention_bmm_quantizers(
    bmm_qtype: str,
    bmm_bits: int,
    bmm_ebit: int,
    bmm_block_size: int = None,
    bmm_q_bits: int = None,
    bmm_kv_bits: int = None,
    bmm_kv_ebit: int = None,
    bmm_q_ebit: int = None,
    bmm_q_block_size: int = None,
    bmm_kv_block_size: int = None,
    bmm_kv_shared_exp_bits: int = None,
    bmm_kv_shared_exp_relative: bool = False,
):
    bmm_q_bits = bmm_q_bits if bmm_q_bits is not None else bmm_bits
    bmm_kv_bits = bmm_kv_bits if bmm_kv_bits is not None else bmm_bits
    bmm_kv_ebit = bmm_kv_ebit if bmm_kv_ebit is not None else bmm_ebit
    bmm_q_ebit = bmm_q_ebit if bmm_q_ebit is not None else bmm_ebit
    # K/V MXINT shared-exponent precision/mode (for mixed_invmx_mxint). None -> the
    # MXINTActivationQuantizer default (5, absolute), so existing configs are unchanged.
    bmm_kv_shared_exp_bits = bmm_kv_shared_exp_bits if bmm_kv_shared_exp_bits is not None else 5

    def _blk(specific, shared, side=""):
        if specific is not None: return specific
        if shared is not None: return shared
        raise ValueError(
            f"bmm_block_size must be set in config for bmm_qtype='{bmm_qtype}' "
            f"({side} side). Set bmm_block_size or bmm_{side}_block_size in YAML."
        )

    if bmm_qtype in ("smooth_quant", "smooth"):
        q_quant = SmoothQuantizer(nbit=bmm_bits, unsigned=False)
        k_quant = SmoothQuantizer(nbit=bmm_bits, unsigned=False)
        v_quant = SmoothQuantizer(nbit=bmm_bits, unsigned=False)
        attn_quant = SmoothQuantizer(nbit=bmm_bits, unsigned=True)
        return q_quant, k_quant, v_quant, attn_quant

    # All non-smooth qtypes require block sizes — resolve now or raise
    q_bs = _blk(bmm_q_block_size, bmm_block_size, "q")
    kv_bs = _blk(bmm_kv_block_size, bmm_block_size, "kv")

    if bmm_qtype == "mxint":
        q_quant = MXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = MXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "inverted_mx":
        q_quant = InvertedMXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=False, block_size=q_bs, ebit=bmm_ebit)
        k_quant = InvertedMXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_ebit)
        v_quant = InvertedMXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_ebit)
        attn_quant = InvertedMXINTActivationQuantizer(nbit=bmm_bits, train_flag=True, unsigned=True, block_size=q_bs, ebit=bmm_ebit)
    elif bmm_qtype == "mixed_mxint_invmx":
        q_quant = MXINTActivationQuantizer(nbit=bmm_q_bits, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = InvertedMXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_kv_ebit)
        v_quant = InvertedMXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_kv_ebit)
        attn_quant = MXINTActivationQuantizer(nbit=bmm_q_bits, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mixed_invmx_mxint":
        q_quant = InvertedMXINTActivationQuantizer(nbit=bmm_q_bits, train_flag=True, unsigned=False, block_size=q_bs, ebit=bmm_q_ebit)
        k_quant = MXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, shared_exp_bits=bmm_kv_shared_exp_bits, shared_exp_relative=bmm_kv_shared_exp_relative)
        v_quant = MXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, shared_exp_bits=bmm_kv_shared_exp_bits, shared_exp_relative=bmm_kv_shared_exp_relative)
        attn_quant = MXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=True, block_size=kv_bs, shared_exp_bits=bmm_kv_shared_exp_bits, shared_exp_relative=bmm_kv_shared_exp_relative)
    elif bmm_qtype == "mxfp4":
        q_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mxfp4_ceil":
        q_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mxfp4_plus":
        q_quant = MXFP4PlusActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = MXFP4PlusActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXFP4PlusActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXFP4PlusActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mxfp4_plus_ceil":
        q_quant = MXFP4PlusCeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = MXFP4PlusCeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXFP4PlusCeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXFP4PlusCeilActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "amxfp4":
        q_quant = AMXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = AMXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = AMXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = AMXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "nvfp4":
        q_quant = NVFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = NVFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = NVFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = NVFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mixed_mxfp4_invmx":
        q_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = InvertedMXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_kv_ebit)
        v_quant = InvertedMXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_kv_ebit)
        attn_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mixed_invmx_mxfp4":
        q_quant = InvertedMXINTActivationQuantizer(nbit=bmm_q_bits, train_flag=True, unsigned=False, block_size=q_bs, ebit=bmm_q_ebit)
        k_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXFP4ActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=kv_bs)
    elif bmm_qtype == "mixed_mxfp4_ceil_invmx":
        q_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=q_bs)
        k_quant = InvertedMXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_kv_ebit)
        v_quant = InvertedMXINTActivationQuantizer(nbit=bmm_kv_bits, train_flag=True, unsigned=False, block_size=kv_bs, ebit=bmm_kv_ebit)
        attn_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=q_bs)
    elif bmm_qtype == "mixed_invmx_mxfp4_ceil":
        q_quant = InvertedMXINTActivationQuantizer(nbit=bmm_q_bits, train_flag=True, unsigned=False, block_size=q_bs, ebit=bmm_q_ebit)
        k_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        v_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=False, block_size=kv_bs)
        attn_quant = MXFP4CeilActivationQuantizer(nbit=4, train_flag=True, unsigned=True, block_size=kv_bs)
    else:
        raise ValueError(
            f"Unsupported bmm_qtype '{bmm_qtype}'. Supported values: {', '.join(SUPPORTED_BMM_QTYPES)}"
        )

    return q_quant, k_quant, v_quant, attn_quant


class QSiglipAttention(SiglipAttention):
    """SiglipAttention with BMM quantization support.

    Subclasses SiglipAttention to preserve isinstance checks required by
    transformers' @check_model_inputs decorator on SiglipVisionModel.forward.

    Replaces nn.Linear projections with _QBaseLinear and adds BMM quantization
    on Q, K, V, and attention weights.
    """
    def __init__(
            self,
            config,
            quantize_bmm_input: bool = False,
            bmm_qtype: str = "smooth_quant",
            bmm_bits: int = 8,
            bmm_ebit: int = None,
            bmm_block_size: int = None,
            bmm_q_bits: int = None,
            bmm_kv_bits: int = None,
            bmm_kv_ebit: int = None,
            bmm_q_ebit: int = None,
            bmm_q_block_size: int = None,
            bmm_kv_block_size: int = None,
            bmm_kv_shared_exp_bits: int = None,
            bmm_kv_shared_exp_relative: bool = False,
    ):
        super().__init__(config)

        # Replace nn.Linear projections with _QBaseLinear
        self.q_proj = _QBaseLinear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = _QBaseLinear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = _QBaseLinear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = _QBaseLinear(self.embed_dim, self.embed_dim, bias=True)

        # Attention scale (applied after matmul)
        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(self.scale)

        # BMM quantization
        self.quantize_bmm_input = quantize_bmm_input
        self.q_quant, self.k_quant, self.v_quant, self.attn_quant = \
            _create_attention_bmm_quantizers(
                bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size,
                bmm_q_bits, bmm_kv_bits, bmm_kv_ebit, bmm_q_ebit,
                bmm_q_block_size, bmm_kv_block_size,
                bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative
            )

        self.train_flag = True

    def inference(self):
        self.train_flag = False
        self.q_proj.inference()
        self.k_proj.inference()
        self.v_proj.inference()
        self.out_proj.inference()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel

        Uses SDPA for memory efficiency (avoids materializing full NxN attention
        matrix). BMM quantization is applied to Q, K, V before SDPA.
        """
        bsz, tgt_len, embed_dim = hidden_states.size()

        # Project Q, K, V (weight quantization via _QBaseLinear)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply BMM quantization on Q, K, V
        if self.quantize_bmm_input:
            # K-mean smoothing (SageAttention): remove channel-wise bias
            key_states = key_states - key_states.mean(dim=-2, keepdim=True)
            query_states = self.q_quant(query_states)
            key_states = self.k_quant(key_states)
            value_states = self.v_quant(value_states)

        # Use memory-efficient SDPA (avoids materializing full attention matrix)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape back: (bsz, num_heads, tgt_len, head_dim) -> (bsz, tgt_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim).contiguous()

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, None


class QLlamaAttention(LlamaAttention):
    """
    Llama Attention with Low precision operations
    
    Args:
        config: LlamaConfig
        layer_idx: Layer index
        dtype: Data type for the layer
        rescale_out: Whether to rescale output in linear layers
        quantize_bmm_input: Whether to quantize Q, K, V for attention score computation
        bmm_qtype: Quantization type for attention BMMs.
        bmm_bits: Bit precision for Q/K/V quantization (used by mxint and inverted_mx)
        bmm_ebit: Exponent bits for inverted_mx quantization (None for unlimited)
        bmm_q_bits: Bit precision for Q (for mixed quantization, defaults to bmm_bits)
        bmm_kv_bits: Bit precision for K/V (for mixed quantization, defaults to bmm_bits)
        bmm_kv_ebit: Exponent bits for K/V inverted_mx (for mixed quantization, defaults to bmm_ebit)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int, dtype=torch.float16, rescale_out:bool=False,
                 quantize_bmm_input:bool=False, bmm_qtype:str="smooth_quant", bmm_bits:int=8, bmm_ebit:int=None,
                 bmm_q_bits:int=None, bmm_kv_bits:int=None, bmm_kv_ebit:int=None, bmm_q_ebit:int=None,
                 bmm_block_size:int=None, bmm_q_block_size:int=None, bmm_kv_block_size:int=None,
                 bmm_kv_shared_exp_bits:int=None, bmm_kv_shared_exp_relative:bool=False):
        super().__init__(config, layer_idx)

        # t2c base layer
        self.q_proj = _QBaseLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, rescale_out=rescale_out).to(torch.float16)
        self.k_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, rescale_out=rescale_out).to(torch.float16)
        self.v_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, rescale_out=rescale_out).to(torch.float16)
        self.o_proj = _QBaseLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, rescale_out=rescale_out).to(torch.float16)

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Whether to quantize BMM inputs (Q, K, V) for attention computation
        self.quantize_bmm_input = quantize_bmm_input
        self.bmm_qtype = bmm_qtype
        self.bmm_block_size = bmm_block_size

        # Mixed quantization parameters (defaults to shared params if not specified)
        self.bmm_q_bits = bmm_q_bits if bmm_q_bits is not None else bmm_bits
        self.bmm_kv_bits = bmm_kv_bits if bmm_kv_bits is not None else bmm_bits
        self.bmm_kv_ebit = bmm_kv_ebit if bmm_kv_ebit is not None else bmm_ebit
        self.bmm_q_ebit = bmm_q_ebit if bmm_q_ebit is not None else bmm_ebit
        self.bmm_kv_shared_exp_bits = bmm_kv_shared_exp_bits
        self.bmm_kv_shared_exp_relative = bmm_kv_shared_exp_relative

        # Create Q, K, V quantizers based on bmm_qtype
        self.q_quant, self.k_quant, self.v_quant, self.attn_quant = self._create_bmm_quantizers(
            bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size, self.bmm_q_bits, self.bmm_kv_bits, self.bmm_kv_ebit, self.bmm_q_ebit,
            bmm_q_block_size, bmm_kv_block_size, bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative
        )

        # Attention scale for dequantization
        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(self.head_dim ** (-0.5))

        # batch matmul operators for INT8 computation
        self.qk = BatchHeadIntMatMul(nbit=8)
        self.attnv = BatchHeadIntMatMul(nbit=8)  # INT8 matmul for Attn×V
        
        # training flag
        self.train_flag = True
    
    def _create_bmm_quantizers(self, bmm_qtype: str, bmm_bits: int, bmm_ebit: int, bmm_block_size: int = None,
                                  bmm_q_bits: int = None, bmm_kv_bits: int = None, bmm_kv_ebit: int = None,
                                  bmm_q_ebit: int = None, bmm_q_block_size: int = None, bmm_kv_block_size: int = None,
                                  bmm_kv_shared_exp_bits: int = None, bmm_kv_shared_exp_relative: bool = False):
        return _create_attention_bmm_quantizers(
            bmm_qtype,
            bmm_bits,
            bmm_ebit,
            bmm_block_size,
            bmm_q_bits,
            bmm_kv_bits,
            bmm_kv_ebit,
            bmm_q_ebit,
            bmm_q_block_size,
            bmm_kv_block_size,
            bmm_kv_shared_exp_bits,
            bmm_kv_shared_exp_relative,
        )

    def inference(self):
        """Switch to inference mode"""
        self.train_flag = False
        # Note: q_proj, k_proj, v_proj, o_proj inference() is called by the fuser
        # For Q/K/V quantizers, we keep dequantize=True so they produce float outputs
        # (fake quantization approach - matching original SmoothQuant paper)

    def manual_sdpa(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            # out-of-place so that 4D masks ([B, 1, L, S], padded batches)
            # broadcast against the 2D attn_bias
            if attn_mask.dtype == torch.bool:
                mask_bias = torch.zeros(attn_mask.shape, dtype=query.dtype, device=query.device)
                mask_bias = mask_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
                attn_bias = attn_bias + mask_bias
            else:
                attn_bias = attn_bias + attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight

    def quantized_sdpa(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """
        Quantized scaled dot-product attention for inference.
        Full INT8 attention computation matching SmoothQuant paper:
        1. Q×K^T: Both Q and K are quantized to INT8 (fake quant with dequantize)
        2. Scale + Softmax: Float operations
        3. Quantize attention weights to INT8
        4. Attn×V: Both attention weights and V are quantized to INT8
        """
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            # out-of-place so that 4D masks ([B, 1, L, S], padded batches)
            # broadcast against the 2D attn_bias
            if attn_mask.dtype == torch.bool:
                mask_bias = torch.zeros(attn_mask.shape, dtype=query.dtype, device=query.device)
                mask_bias = mask_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
                attn_bias = attn_bias + mask_bias
            else:
                attn_bias = attn_bias + attn_mask

        # Q×K^T - Q and K are already quantized+dequantized
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        
        # Quantize attention weights for Attn×V matmul
        attn_weight_q = self.attn_quant(attn_weight)
        
        # Attn×V - both attention weights and V are quantized
        attn_output = attn_weight_q @ value
        
        return attn_output, attn_weight

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Token-local quantizers: quantize the new V tokens once, before the
        # cache insert, so decode steps stop re-quantizing the whole V history.
        # K cannot be pre-quantized (its centering below spans tokens), but it
        # is centered/quantized on the un-expanded KV heads: repeat_kv merely
        # duplicates heads, so this is bit-identical at 1/num_groups the cost.
        kv_token_local = self.quantize_bmm_input and _token_local(self.k_quant, self.v_quant)
        if kv_token_local:
            value_states = self.v_quant(value_states)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if kv_token_local:
            # center K before quantization (softmax-invariant, shrinks the
            # dynamic range) — same treatment as QQWen2Attention/QMistralAttention
            key_states = key_states - key_states.mean(dim=-2, keepdim=True)
            key_states = self.k_quant(key_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # Quantize Q (and K/V on the legacy path for non-token-local quantizers)
        if self.quantize_bmm_input:
            if not kv_token_local:
                # center K before quantization (softmax-invariant, shrinks the
                # dynamic range) — same treatment as QQWen2Attention/QMistralAttention
                key_states = key_states - key_states.mean(dim=-2, keepdim=True)
                key_states = self.k_quant(key_states)
                value_states = self.v_quant(value_states)
            query_states = self.q_quant(query_states)

        # Use quantized_sdpa only when quantize_bmm_input is enabled
        # Otherwise use manual_sdpa to avoid unwanted quantization
        if self.quantize_bmm_input and not self.train_flag:
            attn_output, attn_weight = self.quantized_sdpa(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=causal_mask is None and q_len > 1,
            )
        else:
            attn_output, attn_weight = self.manual_sdpa(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=causal_mask is None and q_len > 1,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weight


class QQWen2Attention(Qwen2Attention):
    def __init__(self, config, layer_idx,
                 quantize_bmm_input: bool = False,
                 bmm_qtype: str = "smooth_quant",
                 bmm_bits: int = 8,
                 bmm_ebit: int = None,
                 bmm_block_size: int = None,
                 bmm_q_bits: int = None,
                 bmm_kv_bits: int = None,
                 bmm_kv_ebit: int = None,
                 bmm_q_ebit: int = None,
                 bmm_q_block_size: int = None,
                 bmm_kv_block_size: int = None,
                 k_smooth_static: bool = False,
                 bmm_kv_shared_exp_bits: int = None, bmm_kv_shared_exp_relative: bool = False):
        super().__init__(config, layer_idx)

        self.q_proj = _QBaseLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = _QBaseLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # BMM quantization
        self.quantize_bmm_input = quantize_bmm_input
        self.k_smooth_static = k_smooth_static
        self._k_static_mean = None
        self.q_quant, self.k_quant, self.v_quant, self.attn_quant = \
            _create_attention_bmm_quantizers(
                bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size,
                bmm_q_bits, bmm_kv_bits, bmm_kv_ebit, bmm_q_ebit,
                bmm_q_block_size, bmm_kv_block_size,
                bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Token-local quantizers: quantize the new V tokens once, before the
        # cache insert, so decode steps stop re-quantizing the whole V history.
        # K cannot be pre-quantized (its mean-smoothing below spans tokens),
        # but it is centered/quantized on the un-expanded KV heads: repeat_kv
        # merely duplicates heads, so this is bit-identical at lower cost.
        kv_token_local = self.quantize_bmm_input and _token_local(self.k_quant, self.v_quant)
        if kv_token_local:
            value_states = self.v_quant(value_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # BMM quantization requires explicit GQA expansion before quantizing
        if self.quantize_bmm_input:
            if kv_token_local:
                key_states = key_states - _k_smooth_mean(self, key_states, cache_position)
                key_states = self.k_quant(key_states)
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                key_states = key_states - _k_smooth_mean(self, key_states, cache_position)
                key_states = self.k_quant(key_states)
                value_states = self.v_quant(value_states)
            query_states = self.q_quant(query_states)

            # Use SDPA directly since KV is already expanded for GQA
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            is_causal = causal_mask is None and query_states.shape[-2] > 1
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=causal_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=is_causal,
                scale=self.scaling,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # Use transformers attention dispatch (handles GQA and transpose internally)
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class QMistralAttention(MistralAttention):
    """Mistral attention with BMM quantization, mirroring QQWen2Attention.

    Identical structure to Qwen2 except the projections carry no bias.
    """
    def __init__(self, config, layer_idx,
                 quantize_bmm_input: bool = False,
                 bmm_qtype: str = "smooth_quant",
                 bmm_bits: int = 8,
                 bmm_ebit: int = None,
                 bmm_block_size: int = None,
                 bmm_q_bits: int = None,
                 bmm_kv_bits: int = None,
                 bmm_kv_ebit: int = None,
                 bmm_q_ebit: int = None,
                 bmm_q_block_size: int = None,
                 bmm_kv_block_size: int = None,
                 k_smooth_static: bool = False,
                 bmm_kv_shared_exp_bits: int = None, bmm_kv_shared_exp_relative: bool = False):
        super().__init__(config, layer_idx)

        self.q_proj = _QBaseLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = _QBaseLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = _QBaseLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # BMM quantization
        self.quantize_bmm_input = quantize_bmm_input
        self.k_smooth_static = k_smooth_static
        self._k_static_mean = None
        self.q_quant, self.k_quant, self.v_quant, self.attn_quant = \
            _create_attention_bmm_quantizers(
                bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size,
                bmm_q_bits, bmm_kv_bits, bmm_kv_ebit, bmm_q_ebit,
                bmm_q_block_size, bmm_kv_block_size,
                bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Token-local quantizers: quantize the new V tokens once, before the
        # cache insert, so decode steps stop re-quantizing the whole V history.
        # K cannot be pre-quantized (its mean-smoothing below spans tokens),
        # but it is centered/quantized on the un-expanded KV heads: repeat_kv
        # merely duplicates heads, so this is bit-identical at lower cost.
        kv_token_local = self.quantize_bmm_input and _token_local(self.k_quant, self.v_quant)
        if kv_token_local:
            value_states = self.v_quant(value_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # BMM quantization requires explicit GQA expansion before quantizing
        if self.quantize_bmm_input:
            if kv_token_local:
                key_states = key_states - _k_smooth_mean(self, key_states, cache_position)
                key_states = self.k_quant(key_states)
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                key_states = key_states - _k_smooth_mean(self, key_states, cache_position)
                key_states = self.k_quant(key_states)
                value_states = self.v_quant(value_states)
            query_states = self.q_quant(query_states)

            # Use SDPA directly since KV is already expanded for GQA
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            is_causal = causal_mask is None and query_states.shape[-2] > 1
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=causal_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=is_causal,
                scale=self.scaling,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # Use transformers attention dispatch (handles GQA and transpose internally)
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class QQwen2VLVisionAttention(nn.Module):
    """Qwen2-VL Vision Attention with BMM quantization support.

    Drop-in replacement for VisionAttention (Qwen2-VL vision encoder) that adds:
    - Weight quantization via _QBaseLinear for qkv/proj projections
    - BMM quantization via activation quantizers on Q, K, V

    Uses fused qkv (single Linear for Q/K/V) matching original architecture.
    Processes each chunk separately via cu_seqlens (matching non-flash path).
    """
    def __init__(
            self,
            config,
            quantize_bmm_input: bool = False,
            bmm_qtype: str = "smooth_quant",
            bmm_bits: int = 8,
            bmm_ebit: int = None,
            bmm_block_size: int = None,
            bmm_q_bits: int = None,
            bmm_kv_bits: int = None,
            bmm_kv_ebit: int = None,
            bmm_q_ebit: int = None,
            bmm_q_block_size: int = None,
            bmm_kv_block_size: int = None,
            bmm_kv_shared_exp_bits: int = None,
            bmm_kv_shared_exp_relative: bool = False,
    ):
        super().__init__()
        self.config = config
        # dim source is overridable: Qwen2-VL vision config uses embed_dim,
        # Qwen2.5-VL uses hidden_size (see QQwen2_5_VLVisionAttention).
        self.dim = self._embed_dim(config)
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0
        self.is_causal = False

        # Quantized projections
        self.qkv = _QBaseLinear(self.dim, self.dim * 3, bias=True)
        self.proj = _QBaseLinear(self.dim, self.dim, bias=True)

        # BMM quantization
        self.quantize_bmm_input = quantize_bmm_input
        self.q_quant, self.k_quant, self.v_quant, self.attn_quant = \
            _create_attention_bmm_quantizers(
                bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size,
                bmm_q_bits, bmm_kv_bits, bmm_kv_ebit, bmm_q_ebit,
                bmm_q_block_size, bmm_kv_block_size,
                bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative
            )

        self.train_flag = True

    def inference(self):
        self.train_flag = False
        self.qkv.inference()
        self.proj.inference()

    def _embed_dim(self, config):
        return config.embed_dim

    @staticmethod
    def _apply_rotary_pos_emb_vision(q, k, cos, sin):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision
        return apply_rotary_pos_emb_vision(q, k, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        # Fused QKV projection
        qkv_out = self.qkv(hidden_states)
        query_states, key_states, value_states = (
            qkv_out.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        # Apply rotary embeddings (rotary fn overridable by subclasses, e.g. Qwen2.5-VL)
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Reshape: (seq, heads, head_dim) -> (1, heads, seq, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Process each chunk separately (matching non-flash path of original)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2)
            for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = []
        for q, k, v in zip(*splits):
            # Apply BMM quantization on Q, K, V
            if self.quantize_bmm_input:
                # K-mean smoothing (SageAttention): remove channel-wise bias
                k = k - k.mean(dim=-2, keepdim=True)
                q = self.q_quant(q)
                k = self.k_quant(k)
                v = self.v_quant(v)

            # Use memory-efficient SDPA
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_outputs.append(attn_output)

        attn_output = torch.cat(attn_outputs, dim=2)
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class QQwen2_5_VLVisionAttention(QQwen2VLVisionAttention):
    """Qwen2.5-VL vision attention.

    Identical to Qwen2-VL's QQwen2VLVisionAttention except:
    - the vision config exposes the embedding dim as ``hidden_size`` (Qwen2-VL
      uses ``embed_dim``), so projections are sized from ``config.hidden_size``;
    - rotary embeddings are imported from the qwen2_5_vl module (functionally
      identical to qwen2_vl's, but matched to the model for correctness).

    Window vs full attention is handled by the parent vision transformer's
    forward (which feeds the appropriate ``cu_seqlens`` per block), so the
    inherited chunked-attention forward needs no change.
    """
    def _embed_dim(self, config):
        # Qwen2.5-VL vision config exposes the embed dim as hidden_size
        return config.hidden_size

    @staticmethod
    def _apply_rotary_pos_emb_vision(q, k, cos, sin):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
        return apply_rotary_pos_emb_vision(q, k, cos, sin)


class QQwen2VLAttention(nn.Module):
    """Qwen2-VL LLM Attention with quantized projections and BMM quantization.

    Standalone replacement for Qwen2VLAttention that adds:
    - Weight quantization via _QBaseLinear for q/k/v/o projections
    - BMM quantization via activation quantizers on Q, K, V
    - Preserves multimodal rotary pos emb and sliding window attention
    """
    def __init__(
            self,
            config,
            layer_idx,
            quantize_bmm_input: bool = False,
            bmm_qtype: str = "smooth_quant",
            bmm_bits: int = 8,
            bmm_ebit: int = None,
            bmm_block_size: int = None,
            bmm_q_bits: int = None,
            bmm_kv_bits: int = None,
            bmm_kv_ebit: int = None,
            bmm_q_ebit: int = None,
            bmm_q_block_size: int = None,
            bmm_kv_block_size: int = None,
            k_smooth_static: bool = False,
            bmm_kv_shared_exp_bits: int = None,
            bmm_kv_shared_exp_relative: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim ** -0.5

        # Quantized projections
        self.q_proj = _QBaseLinear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = _QBaseLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = _QBaseLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = _QBaseLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        # BMM quantization
        self.quantize_bmm_input = quantize_bmm_input
        self.k_smooth_static = k_smooth_static
        self._k_static_mean = None
        self.q_quant, self.k_quant, self.v_quant, self.attn_quant = \
            _create_attention_bmm_quantizers(
                bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size,
                bmm_q_bits, bmm_kv_bits, bmm_kv_ebit, bmm_q_ebit,
                bmm_q_block_size, bmm_kv_block_size,
                bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # Token-local quantizers: quantize the new V tokens once, before the
        # cache insert, so decode steps stop re-quantizing the whole V history.
        # K cannot be pre-quantized: its mean-smoothing below spans the token
        # dimension, so quantized K depends on the full sequence every step.
        v_prequant = self.quantize_bmm_input and _token_local(self.v_quant)
        if v_prequant:
            value_states = self.v_quant(value_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Apply BMM quantization on Q, K, V (before attention computation)
        if self.quantize_bmm_input:
            # K-mean smoothing (SageAttention, Eq.6): subtract per-channel mean
            # across token dimension. This removes channel-wise bias that causes
            # extreme outliers (|K|>1000 in Qwen2-VL), while preserving attention
            # scores exactly: softmax(Q@K^T) = softmax(Q@(K-mean(K))^T)
            key_states = key_states - _k_smooth_mean(self, key_states, cache_position)
            query_states = self.q_quant(query_states)
            key_states = self.k_quant(key_states)
            if not v_prequant:
                value_states = self.v_quant(value_states)

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        is_causal = causal_mask is None and q_len > 1
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


