"""
NVIDIA FP4 (NVFP4) format for Blackwell architecture

Two-level scaling:
  - Per-tensor FP32 scale (normalizes tensor into E4M3 block-scale range)
  - Per-block E4M3 (FP8) scale
  - Per-element E2M1 (FP4) values

Reconstruction: x_hat = tensor_scale * block_scale * E2M1_value

Key difference from MXFP4 (OCP):
  MXFP4 uses E8M0 (power-of-2) block scale with unlimited range.
  NVFP4 uses E4M3 (FP8) block scale with per-tensor normalization
  to guarantee block scales fit within E4M3 range [2^-6, 448].
"""

import os
import torch
import torch.nn as nn
from src.module.base import _QBase
from src.quantization.mxfp4 import (
    quantize_to_e2m1, E2M1_MAX,
    _reshape_to_blocks, _undo_reshape_to_blocks,
)

# E4M3 constants for block scale
E4M3_MAX = 448.0         # max normal (1111 111 = NaN, max = 1111 110 = 448)
E4M3_MIN_NORMAL = 2.0 ** -6  # min normal = 2^emin, emin = -(2^(4-1)) + 2 = -6
E4M3_EMAX = 8            # 2^(ebit-1) = max unbiased exponent
E4M3_EMIN = -6           # -(2^(ebit-1)) + 2


def quantize_to_e4m3(x: torch.Tensor, use_ceil: bool = True) -> torch.Tensor:
    """Quantize positive values to E4M3 format.

    Used for block scale factors. Input must be positive (scales are always >= 0).
    Ceil rounding (default) ensures block_scale >= ideal_scale, preventing
    element overflow during E2M1 quantization.
    """
    mant_max = 7   # 2^3 - 1
    mant_shift = 8  # 2^3

    x_is_zero = x == 0

    # Exponent
    full_exp = torch.floor(torch.log2(x + x_is_zero.to(x.dtype)))
    exp = full_exp.clamp(min=E4M3_EMIN, max=E4M3_EMAX)

    # Mantissa: (x / 2^exp - 1) * 2^mbit for normal numbers
    mant = (x / torch.exp2(exp) - 1.0) * mant_shift

    if use_ceil:
        mant = torch.ceil(mant)
    else:
        mant = torch.floor(mant + 0.5)

    # Overflow: mantissa > max → bump exponent, reset mantissa
    overflow_mask = (mant > mant_max) & (exp != E4M3_EMAX)
    mant = mant.clamp(min=0, max=mant_max)
    mant = torch.where(overflow_mask, torch.zeros_like(mant), mant)
    exp = torch.where(overflow_mask, exp + 1, exp)

    # Reconstruct
    x_q = (1.0 + mant / mant_shift) * torch.exp2(exp)
    x_q = torch.where(x_is_zero, torch.zeros_like(x_q), x_q)
    x_q = x_q.clamp(max=E4M3_MAX)

    return x_q


def _nvfp4_core(xg: torch.Tensor, block_axis: int, tensor_scale: torch.Tensor) -> torch.Tensor:
    """Core NVFP4 quantize-dequantize on reshaped block tensor.

    Args:
        xg: Block-reshaped tensor (..., block_size, ...)
        block_axis: Dimension of the block elements
        tensor_scale: Per-tensor scale (scalar)

    Returns:
        Fake-quantized tensor in original dtype
    """
    xg_float = xg.float()
    ts_float = tensor_scale.float()

    # Per-block E4M3 scale
    block_max = xg_float.abs().max(dim=block_axis, keepdim=True)[0]
    block_max = torch.where(block_max == 0, torch.ones_like(block_max), block_max)
    block_scale = block_max / E2M1_MAX
    block_scale = block_scale.clamp(min=E4M3_MIN_NORMAL, max=E4M3_MAX)
    block_scale = quantize_to_e4m3(block_scale, use_ceil=True)

    # E2M1 element quantization
    xg_scaled = xg_float / block_scale
    signs = torch.sign(xg_scaled)
    xg_q = quantize_to_e2m1(torch.abs(xg_scaled))

    # Dequantize with both scales
    xg_deq = signs * xg_q * block_scale * ts_float

    return xg_deq.to(xg.dtype)


def _compute_tensor_scale(x: torch.Tensor) -> torch.Tensor:
    """Compute per-tensor scale: amax / (E4M3_MAX * E2M1_MAX).

    Normalizes the tensor so that per-block scales always fit within
    E4M3 representable range.

    Ablation toggle (rebuttal, MiX-vs-NVFP4 novelty): when the environment
    variable NVFP4_NO_TENSOR_SCALE=1 is set, this returns 1.0, removing the
    per-tensor FP32 scale. NVFP4 then degenerates to a single-block-scale
    format (E4M3 block scale + E2M1 elements only), strictly matched to MiX
    which has no global FP32 scale. Default (unset) is byte-identical to the
    standard two-level NVFP4 — existing nvfp4 runs/results are unaffected.
    """
    if os.environ.get("NVFP4_NO_TENSOR_SCALE", "0") == "1":
        return torch.ones((), dtype=x.dtype, device=x.device)
    amax = x.abs().max()
    tensor_scale = amax / (E4M3_MAX * E2M1_MAX)
    tensor_scale = torch.where(
        tensor_scale == 0, torch.ones_like(tensor_scale), tensor_scale
    )
    return tensor_scale


class NVFP4ChannelWiseWeightQuantizer(_QBase):
    """
    NVIDIA FP4 format for channel-wise weight quantization.

    Per-tensor FP32 scale + per-block E4M3 scale + E2M1 elements.
    Default block_size=16 per NVIDIA Blackwell spec.

    Args:
        nbit: Number of bits (fixed at 4, kept for interface consistency)
        train_flag: Whether in training mode
        unsigned: Whether to use unsigned quantization (False for weights)
        block_size: Size of each quantization block (default: 16)
    """
    def __init__(self, nbit: int = 4, train_flag: bool = True,
                 unsigned: bool = False, block_size: int = 16):
        super().__init__(nbit, train_flag, unsigned)
        self.block_size = block_size

    def register_qparams(self):
        super().register_qparams()

    def reshape(self, x: torch.Tensor):
        if len(x.shape) == 4:
            return _reshape_to_blocks(x, [0], block_size=self.block_size)
        elif len(x.shape) == 2:
            return _reshape_to_blocks(x, [1], block_size=self.block_size)
        else:
            raise NotImplementedError(
                f"Non-supported Layer Type with the shape of {list(x.size())}!"
            )

    def q(self, x: torch.Tensor):
        # Per-tensor scale
        tensor_scale = _compute_tensor_scale(x)
        x_norm = x / tensor_scale

        # Reshape to blocks
        xg, axes, orig_shape, padded_shape = self.reshape(x_norm)
        block_axis = axes[0] + 1

        # Core NVFP4 quantization
        xg_deq = _nvfp4_core(xg, block_axis, tensor_scale)

        # Reshape back
        xg_deq = _undo_reshape_to_blocks(xg_deq, padded_shape, orig_shape, axes)
        return xg_deq

    def trainFunc(self, x: torch.Tensor):
        return self.q(x)

    def evalFunc(self, x: torch.Tensor):
        return self.q(x)

    def inference(self):
        self.train_flag = False
        self.dequantize = True


class NVFP4ActivationQuantizer(_QBase):
    """
    NVIDIA FP4 format for block-wise activation quantization.

    Blocks are created along the feature dimension (last dim), with
    per-tensor FP32 scale + per-block E4M3 scale + E2M1 elements.

    Args:
        nbit: Number of bits (fixed at 4, kept for interface consistency)
        train_flag: Whether in training mode
        unsigned: Whether to use unsigned quantization
        block_size: Size of each quantization block (default: 16)
    """
    def __init__(self, nbit: int = 4, train_flag: bool = True,
                 unsigned: bool = False, block_size: int = 16):
        super().__init__(nbit, train_flag, unsigned)
        self.block_size = block_size

    def register_qparams(self):
        super().register_qparams()

    def reshape(self, x: torch.Tensor):
        if len(x.shape) >= 2:
            return _reshape_to_blocks(x, [-1], block_size=self.block_size)
        else:
            raise NotImplementedError(
                f"Non-supported activation shape: {list(x.size())}!"
            )

    def q(self, x: torch.Tensor):
        # Per-tensor scale
        tensor_scale = _compute_tensor_scale(x)
        x_norm = x / tensor_scale

        # Reshape to blocks along feature dim
        xg, axes, orig_shape, padded_shape = self.reshape(x_norm)
        block_axis = axes[0] + 1

        # Core NVFP4 quantization
        xg_deq = _nvfp4_core(xg, block_axis, tensor_scale)

        # Reshape back
        xg_deq = _undo_reshape_to_blocks(xg_deq, padded_shape, orig_shape, axes)
        return xg_deq

    def trainFunc(self, x: torch.Tensor):
        return self.q(x)

    def evalFunc(self, x: torch.Tensor):
        return self.q(x)

    def inference(self):
        self.train_flag = False
        self.dequantize = True
