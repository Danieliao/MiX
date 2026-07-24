"""
Inverted Micro Scaling Integer format (MiX)

Three format variants at block size B=32:

  Type 1 — MiX-4.25b:  1 shared mantissa per block, 3-bit element exponents.
      Bits: 5 (E_max) + 3 (mantissa) + 32×4 (elements) = 136 → 4.25b/elem.
      Config: mantissa_group_size=32, ebit=3, sub_ebit=None

  Type 2 — MiX-4.5b:   4 sub-group mantissas (b=8), 3-bit element exponents.
      Bits: 5 (E_max) + 4×3 (mantissas) + 32×4 (elements) = 145 → 4.53b/elem.
      Config: mantissa_group_size=8, ebit=3, sub_ebit=None

  Type 3 — MiX-3.9b:   4 sub-group mantissas (b=8), 3-bit sub-block exponents,
                        2-bit element exponents (hierarchical).
      Bits: 5 (E_max) + 4×3 (mantissas) + 4×3 (sub-exp) + 32×3 (elements) = 125 → 3.91b/elem.
      Config: mantissa_group_size=8, ebit=2, sub_ebit=3

Reconstruction:
  Type 1/2: x̂_i = (-1)^s_i × (1.m_shared)_2 × 2^(E_max - Δe_i)
  Type 3:   x̂_i = (-1)^s_i × (1.m_shared,k)_2 × 2^(E_max - ΔE_sub,k - δe_i)

Proposal: https://gemini.google.com/share/132b7308b2a2
"""

import torch
import torch.nn as nn
import numpy as np
from src.module.base import _QBase
from src.quantization.observer import BaseObserver

def ste_round(x):
    return (torch.round(x) - x).detach() + x

def _reshape_to_blocks(A, axes, block_size):
    """
    Adopte the reshape function from microscaling
    https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py#L95
    """
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape

def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

class InvertedMXINTObserver(BaseObserver):
    """Observer for all MiX format variants.

    Args:
        nbit: Shared mantissa bits (default 3).
        unsigned: Whether to use unsigned quantization.
        ebit: Per-element exponent bits. 3 for Type 1/2, 2 for Type 3.
        mantissa_strategy: How to select shared mantissa ("max", "mean", etc.).
        mantissa_group_size: Sub-block size for mantissa sharing.
            32 (=block_size) for Type 1, 8 for Type 2/3.
        sub_ebit: Sub-block exponent bits for hierarchical encoding (Type 3).
            None for Type 1/2, 3 for Type 3.
    """
    VALID_STRATEGIES = ["mean", "magnitude_weighted", "scale_weighted", "mse_optimal", "max"]

    def __init__(self, nbit: int, unsigned: bool = True, ebit: int = None,
                 mantissa_strategy: str = "max", mantissa_group_size: int = 8,
                 sub_ebit: int = None):
        super().__init__(nbit, unsigned)
        self.ebit = ebit
        self.mantissa_strategy = mantissa_strategy
        self.mantissa_group_size = mantissa_group_size
        self.sub_ebit = sub_ebit

        if mantissa_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid mantissa_strategy '{mantissa_strategy}'. "
                             f"Must be one of {self.VALID_STRATEGIES}")

    def get_bound(self, x:torch.Tensor):
        pass

    def _compute_shared_mantissa(self, m_ideal: torch.Tensor, y: torch.Tensor,
                                 exponents: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute the shared mantissa based on the selected strategy."""
        if self.mantissa_strategy == "mean":
            m_shared = torch.mean(m_ideal, dim=axis, keepdim=True)
        elif self.mantissa_strategy == "magnitude_weighted":
            weights = y
            weights_sum = torch.sum(weights, dim=axis, keepdim=True)
            weights_sum = torch.where(weights_sum == 0, torch.ones_like(weights_sum), weights_sum)
            m_shared = torch.sum(weights * m_ideal, dim=axis, keepdim=True) / weights_sum
        elif self.mantissa_strategy == "scale_weighted":
            scale = 2.0 ** exponents
            scale_sum = torch.sum(scale, dim=axis, keepdim=True)
            scale_sum = torch.where(scale_sum == 0, torch.ones_like(scale_sum), scale_sum)
            m_shared = torch.sum(scale * m_ideal, dim=axis, keepdim=True) / scale_sum
        elif self.mantissa_strategy == "mse_optimal":
            scale_sq = 2.0 ** (2.0 * exponents)
            scale_sq_sum = torch.sum(scale_sq, dim=axis, keepdim=True)
            scale_sq_sum = torch.where(scale_sq_sum == 0, torch.ones_like(scale_sq_sum), scale_sq_sum)
            m_shared = torch.sum(scale_sq * m_ideal, dim=axis, keepdim=True) / scale_sq_sum
        elif self.mantissa_strategy == "max":
            max_idx = torch.argmax(y, dim=axis, keepdim=True)
            m_shared = torch.gather(m_ideal, dim=axis, index=max_idx)
        else:
            raise ValueError(f"Unknown mantissa_strategy: {self.mantissa_strategy}")
        return m_shared

    def calculate_qparam(self, x: torch.Tensor, axis: int = -1):
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        # 1. Absolute values (replace exact zeros to avoid log2(0))
        y = torch.abs(x_f32)
        y = torch.where(y == 0, torch.full_like(y, 1e-9), y)

        # 2. Per-element exponents: floor(log2(y)) puts mantissa in [1.0, 2.0)
        exponents = torch.floor(torch.log2(y))

        # Precompute mantissa quantization scale
        scale_factor = 2.0 ** self.nbit
        pos_axis = axis if axis >= 0 else x_f32.ndim + axis

        # 3. Exponent range computation
        # upper_exp_bound / min_allowed_exp are per-element broadcastable tensors
        # that control exponent refinement bounds.
        if self.ebit is not None:
            max_exp = torch.max(exponents, dim=axis, keepdim=True)[0]
            max_exp = torch.clamp(max_exp, max=15)  # Limit E_max to 15 to ensure it fits in 5 bits after biasing

            # Overflow check: if block-max mantissa rounds to >= 2.0, bump max_exp
            y_max = torch.max(y, dim=axis, keepdim=True)[0]
            m_max_q = torch.round(y_max / torch.exp2(max_exp) * scale_factor) / scale_factor
            max_exp = torch.where(m_max_q >= 2.0, max_exp + 1, max_exp)

            if self.sub_ebit is not None:
                # ── TYPE 3: Hierarchical exponents ──
                # Reshape to sub-blocks for per-sub-block exponent computation
                block_dim_size = exponents.shape[pos_axis]
                num_groups = block_dim_size // self.mantissa_group_size
                block_shape = list(exponents.shape)
                group_shape = (block_shape[:pos_axis]
                               + [num_groups, self.mantissa_group_size]
                               + block_shape[pos_axis + 1:])
                group_elem_axis = pos_axis + 1  # axis of elements within each group

                exp_grouped = exponents.reshape(group_shape)

                # Per-sub-block max exponent
                sub_max_exp = exp_grouped.max(dim=group_elem_axis, keepdim=True)[0]

                # ΔE_sub,k: coarse offset from E_max, clamped to sub_ebit range
                max_exp_g = max_exp.unsqueeze(group_elem_axis)
                delta_E_sub = (max_exp_g - sub_max_exp).clamp(0, 2 ** self.sub_ebit - 1)
                sub_base_exp = max_exp_g - delta_E_sub

                # Per-element fine offset within sub-block (asymmetric encoding)
                min_allowed_exp_g = sub_base_exp - (2 ** self.ebit - 1)
                underflow_grouped = exp_grouped < min_allowed_exp_g
                exp_grouped = exp_grouped.clamp(min=min_allowed_exp_g, max=sub_base_exp)

                # Flatten back to block shape
                exponents = exp_grouped.reshape(block_shape)
                underflow_mask = underflow_grouped.reshape(block_shape)
                # Per-element bounds for refinement step
                upper_exp_bound = sub_base_exp.expand_as(
                    torch.empty(group_shape)).reshape(block_shape)
                min_allowed_exp = min_allowed_exp_g.expand_as(
                    torch.empty(group_shape)).reshape(block_shape)
            else:
                # ── TYPE 1 / TYPE 2: Flat per-block exponent range ──
                min_allowed_exp = max_exp - (2 ** self.ebit - 1)
                underflow_mask = exponents < min_allowed_exp
                exponents = torch.clamp(exponents, min=min_allowed_exp, max=max_exp)
                upper_exp_bound = max_exp  # broadcastable scalar per block
        else:
            underflow_mask = None
            upper_exp_bound = None
            min_allowed_exp = None

        # 4. Ideal mantissas
        pow2_exp = torch.exp2(exponents)
        m_ideal = y / pow2_exp

        # 5. Grouped shared mantissa
        block_dim_size = m_ideal.shape[pos_axis]
        use_groups = (block_dim_size > self.mantissa_group_size
                      and block_dim_size % self.mantissa_group_size == 0)

        if use_groups:
            num_groups = block_dim_size // self.mantissa_group_size
            block_shape = list(m_ideal.shape)
            group_shape = (block_shape[:pos_axis]
                           + [num_groups, self.mantissa_group_size]
                           + block_shape[pos_axis + 1:])

            m_shared_float = self._compute_shared_mantissa(
                m_ideal.reshape(group_shape),
                y.reshape(group_shape),
                exponents.reshape(group_shape),
                pos_axis + 1,
            )
        else:
            block_shape = None
            m_shared_float = self._compute_shared_mantissa(m_ideal, y, exponents, axis)

        # Quantize shared mantissa to nbit (round-to-nearest)
        m_shared_final = ste_round(m_shared_float * scale_factor) / scale_factor

        # Hardware overflow: m rounds to 2.0 → normalize to 1.0
        m_shared_final = torch.where(m_shared_final >= 2.0,
                                     torch.ones_like(m_shared_final),
                                     m_shared_final)

        # Expand grouped mantissa back to block shape
        if use_groups:
            m_shared_final = m_shared_final.expand_as(
                m_ideal.reshape(group_shape)
            ).reshape(block_shape)

        # 6. Exponent refinement: check e, e+1, e-1
        rec = m_shared_final * pow2_exp
        err_best = torch.abs(y - rec)
        final_exponents = exponents

        # Check e+1
        err_p1 = torch.abs(y - rec * 2)
        mask_p1 = err_p1 < err_best
        if upper_exp_bound is not None:
            mask_p1 = mask_p1 & (exponents + 1 <= upper_exp_bound)
        err_best = torch.where(mask_p1, err_p1, err_best)
        final_exponents = torch.where(mask_p1, exponents + 1, final_exponents)

        # Check e-1
        err_m1 = torch.abs(y - rec * 0.5)
        mask_m1 = err_m1 < err_best
        if min_allowed_exp is not None:
            mask_m1 = mask_m1 & (exponents - 1 >= min_allowed_exp)
        final_exponents = torch.where(mask_m1, exponents - 1, final_exponents)

        # 7. Sparsification: underflow elements → zero.
        # Asymmetric encoding: negative values at min_allowed_exp are also zeroed
        # (sign=1, offset=max is the zero code).
        if underflow_mask is not None:
            is_negative = x_f32 < 0
            neg_at_min = is_negative & (final_exponents <= min_allowed_exp)
            zero_mask = underflow_mask | neg_at_min
            final_exponents = torch.where(zero_mask,
                                          torch.full_like(final_exponents, -126.0),
                                          final_exponents)

        return m_shared_final.to(orig_dtype), final_exponents.to(orig_dtype)

    def forward(self, x:torch.Tensor, axis:int=-1):
        return self.calculate_qparam(x, axis)


class InvertedMXChannelWiseWeightQuantizer(_QBase):
    """Inverted Micro-scaling format for channel-wise weight quantization.

    Args:
        nbit: Number of bits for the shared mantissa (default 3).
        train_flag: Whether in training mode.
        unsigned: Whether to use unsigned quantization.
        block_size: Size of each quantization block (default 32).
        ebit: Per-element exponent bits. 3 for Type 1/2, 2 for Type 3.
        mantissa_group_size: Sub-block size for mantissa sharing.
            Set to block_size for Type 1 (single mantissa), 8 for Type 2/3.
        sub_ebit: Sub-block exponent bits (None for Type 1/2, 3 for Type 3).
    """
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False,
                 block_size: int = 32, ebit: int = None,
                 mantissa_group_size: int = 8, sub_ebit: int = None):
        super().__init__(nbit, train_flag, unsigned)

        self.block_size = block_size
        self.ebit = ebit
        self.mantissa_group_size = mantissa_group_size
        self.sub_ebit = sub_ebit
        self.observer = InvertedMXINTObserver(
            nbit=self.nbit, unsigned=self.unsigned, ebit=self.ebit,
            mantissa_group_size=mantissa_group_size, sub_ebit=sub_ebit,
        )

    def register_qparams(self):
        super().register_qparams()
        pass

    def reshape(self, x:torch.Tensor):
        if len(x.shape) == 4:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [0], block_size=self.block_size
            )
        elif len(x.shape) == 2:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [1], block_size=self.block_size
            )
        else:
            raise NotImplementedError(f"Non-supported Layer Type with the shape of {list(x.size())}!")
        return x, axes, orig_shape, padded_shape

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]
        block_axis = shared_exp_axes[0]

        if self.train_flag:
            with torch.no_grad():
                m_shared, exponents = self.observer(xg, axis=block_axis)
                xq = torch.sign(xg) * m_shared * (2.0 ** exponents)
                xq = _undo_reshape_to_blocks(xq, padded_shape, orig_shape, axes)
            # STE
            xq = (xq - x).detach() + x
            return xq
        else:
            m_shared, exponents = self.observer(xg, axis=block_axis)
            xq = torch.sign(xg) * m_shared * (2.0 ** exponents)
            xq = _undo_reshape_to_blocks(xq, padded_shape, orig_shape, axes)
            return xq

    def trainFunc(self, x:torch.Tensor):
        return self.q(x)

    def evalFunc(self, x: torch.Tensor):
        return self.q(x)

    def inference(self):
        self.train_flag = False
        self.dequantize = True


class InvertedMXINTActivationQuantizer(_QBase):
    """Inverted Micro-scaling format for block-wise activation quantization.

    Blocks are created along the feature dimension (last dimension).

    Args:
        nbit: Number of bits for the shared mantissa (default 3).
        train_flag: Whether in training mode.
        unsigned: Whether to use unsigned quantization.
        block_size: Size of each quantization block (default 32).
        ebit: Per-element exponent bits. 3 for Type 1/2, 2 for Type 3.
        mantissa_group_size: Sub-block size for mantissa sharing.
            Set to block_size for Type 1 (single mantissa), 8 for Type 2/3.
        sub_ebit: Sub-block exponent bits (None for Type 1/2, 3 for Type 3).
    """
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False,
                 block_size: int = 32, ebit: int = None,
                 mantissa_group_size: int = 8, sub_ebit: int = None):
        super().__init__(nbit, train_flag, unsigned)

        self.block_size = block_size
        self.ebit = ebit
        self.mantissa_group_size = mantissa_group_size
        self.sub_ebit = sub_ebit
        self.observer = InvertedMXINTObserver(
            nbit=self.nbit, unsigned=self.unsigned, ebit=self.ebit,
            mantissa_group_size=mantissa_group_size, sub_ebit=sub_ebit,
        )

    def register_qparams(self):
        super().register_qparams()
        pass

    def reshape(self, x:torch.Tensor):
        if len(x.shape) >= 2:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        else:
            raise NotImplementedError(f"Non-supported activation shape: {list(x.size())}!")
        return x, axes, orig_shape, padded_shape

    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [a + 1 for a in axes]
        block_axis = shared_exp_axes[0]

        if self.train_flag:
            with torch.no_grad():
                m_shared, exponents = self.observer(xg, axis=block_axis)
                xq = torch.sign(xg) * m_shared * (2.0 ** exponents)
                xq = _undo_reshape_to_blocks(xq, padded_shape, orig_shape, axes)
            # STE
            xq = (xq - x).detach() + x
            return xq
        else:
            m_shared, exponents = self.observer(xg, axis=block_axis)
            xq = torch.sign(xg) * m_shared * (2.0 ** exponents)
            xq = _undo_reshape_to_blocks(xq, padded_shape, orig_shape, axes)
            return xq

    def trainFunc(self, x:torch.Tensor):
        return self.q(x)

    def evalFunc(self, x: torch.Tensor):
        return self.q(x)

    def inference(self):
        self.train_flag = False
        self.dequantize = True
