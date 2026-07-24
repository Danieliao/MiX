"""
Micro Scaling Integer format by MicroSoft

Paper: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

import torch
import torch.nn as nn
from src.module.base import _QBase
from src.quantization.observer import BaseObserver

def _reshape_to_blocks(A, axes, block_size):
    """
    Adopte the reshape function from microscaling
    https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py#L95
    """
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension to apply block size to"
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

class MXINTObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, shared_exp_bits: int = 5, shared_exp_relative: bool = False):
        super().__init__(nbit, unsigned)
        self.shared_exp_bits = shared_exp_bits
        # When True, the per-block shared exponent is encoded as an unsigned offset
        # (0 .. 2^shared_exp_bits-1) BELOW a per-row reference exponent, instead of an
        # absolute signed exponent clamped to [-2^(b-1), 2^(b-1)-1]. This lets a small
        # shared_exp_bits (e.g. 4) track the data range (per output-channel / per token)
        # rather than nailing the window to 2^-8 .. 2^7. The reference is one exponent
        # per row (~8 bits / row -> negligible bit overhead).
        self.shared_exp_relative = shared_exp_relative

    def get_bound(self, x:torch.Tensor):
        max_norm = float(2**(self.nbit-1) - 1) / 2**(self.nbit-2)

        # update bound
        self.lb.copy_(-max_norm)
        self.ub.copy_(max_norm)

    def get_shared_exp(self, x:torch.Tensor, axis=0) -> torch.Tensor:
        shared_exp, _ = torch.max(torch.abs(x), dim=axis, keepdim=True)
        # Handle zero values to avoid log2(0) = -inf
        shared_exp = torch.where(shared_exp == 0, torch.ones_like(shared_exp), shared_exp)
        shared_exp = torch.floor(torch.log2(shared_exp))
        if self.shared_exp_relative:
            # Relative encoding: a single per-TENSOR reference exponent + an unsigned
            # per-block offset (0 .. 2^shared_exp_bits-1) below it. Using one reference
            # for the whole tensor (the global max block-exponent) keeps the extra scale
            # overhead at ~8/(numel) bit/element, i.e. ~0, so the format stays exactly at
            # the 4.5b budget (a per-row reference would add ~8/in_features bit/element).
            ref = shared_exp.amax()
            max_offset = 2**self.shared_exp_bits - 1
            offset = torch.clamp(ref - shared_exp, min=0, max=max_offset)
            shared_exp = ref - offset
            return shared_exp
        # Clamp to signed exponent range per OCP MX spec
        # Use configurable shared_exp_bits (default 5 for standard MXINT)
        max_exp = 2**(self.shared_exp_bits - 1) - 1
        min_exp = -2**(self.shared_exp_bits - 1)
        shared_exp = torch.clamp(shared_exp, min=min_exp, max=max_exp)
        return shared_exp

    def calculate_qparam(self, x:torch.Tensor, axis:int=2):
        scale = torch.tensor(2**(self.nbit-2), device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_exp = self.get_shared_exp(x, axis=axis)
        return scale, zero_point, shared_exp
    
    def forward(self, x:torch.Tensor, axis:int=2):
        self.get_bound(x)
        scale, zero_point, shared_exp = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_exp


class MXChannelWiseWeightQuantizer(_QBase):
    """
    Micro-scaling format for channel-wise weight quantization

    [WARNING]: The current implementation of MXINT8 cannot be directly employed to the INT8 matmul kernel of t2c_gemm.
    """
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, block_size:int=32, shared_exp_bits: int = 5, shared_exp_relative: bool = False):
        super().__init__(nbit, train_flag, unsigned)

        assert nbit in [4, 5, 6, 8], "MX INT format supports 4-bit, 5-bit, 6-bit, and 8-bit precision."

        self.block_size = block_size
        self.shared_exp_bits = shared_exp_bits
        self.shared_exp_relative = shared_exp_relative
        self.observer = MXINTObserver(nbit=self.nbit, unsigned=self.unsigned, shared_exp_bits=self.shared_exp_bits, shared_exp_relative=self.shared_exp_relative)

    def register_qparams(self):
        super().register_qparams()
        self.register_buffer("shared_exp", torch.tensor(1.0))

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

        # Always compute scale and shared_exp from input
        # (scale is fixed based on nbit, shared_exp depends on input magnitude)
        scale, zero_point, shared_exp = self.observer(xg, shared_exp_axes[0])

        if self.train_flag:
            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            # Note: shared_exp is per-block and can't be stored in a scalar buffer
            # It must be recomputed each forward pass

        # Use float32 for intermediate computation to avoid float16 overflow/underflow
        xg_float = xg.float()
        shared_exp_float = shared_exp.float()
        
        xg_float = xg_float / (2**shared_exp_float)
        xg_float = xg_float * scale  # scale = 64 for 8-bit

        # round to nearest
        xg_float = torch.sign(xg_float) * torch.floor(torch.abs(xg_float) + 0.5)
        xg_float = xg_float.clamp(self.qlb, self.qub)

        # rescale back
        xg_float = xg_float * (2**shared_exp_float)

        # reshape the tensor
        xg_float = _undo_reshape_to_blocks(xg_float, padded_shape, orig_shape, axes)

        if self.dequantize:
            xg_float = xg_float / scale

        # Convert back to original dtype
        xg = xg_float.to(x.dtype)

        # Debug: check for any numerical issues
        if torch.isnan(xg).any() or torch.isinf(xg).any():
            print(f"[DEBUG] NaN/Inf detected in quantization output!")
            print(f"  - xg has NaN: {torch.isnan(xg).any()}, has Inf: {torch.isinf(xg).any()}")
            print(f"  - scale: {scale}")
            print(f"  - shared_exp min: {shared_exp.min()}, max: {shared_exp.max()}")
            print(f"  - Input x min: {x.min()}, max: {x.max()}")
            import pdb; pdb.set_trace()

        return xg

    def trainFunc(self, x:torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq
    
    def inference(self):
        """
        Override inference mode for MXINT.
        MXINT always needs dequantization to output valid float values.
        Setting dequantize=False would output integer-scale values which 
        cause NaN in downstream computations.
        """
        self.train_flag = False
        self.dequantize = True  # Keep dequantization enabled for MXINT


class MXINTActivationQuantizer(_QBase):
    """
    Micro-scaling format for block-wise activation quantization along feature dimension
    Blocks are created along the feature dimension, with one shared exponent per block
    Compatible with block-wise weight quantization along the channel dimension
    """
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, block_size:int=32, shared_exp_bits: int = 5, shared_exp_relative: bool = False):
        super().__init__(nbit, train_flag, unsigned)

        assert nbit in [4, 5, 6, 8], "MX INT format supports 4-bit, 5-bit, 6-bit, and 8-bit precision."

        self.block_size = block_size
        self.shared_exp_bits = shared_exp_bits
        self.shared_exp_relative = shared_exp_relative
        self.observer = MXINTObserver(nbit=self.nbit, unsigned=self.unsigned, shared_exp_bits=self.shared_exp_bits, shared_exp_relative=self.shared_exp_relative)

    def register_qparams(self):
        super().register_qparams()
        # Note: shared_exp is per-block and cannot be stored as a scalar buffer
        # It must be recomputed each forward pass

    def reshape(self, x:torch.Tensor):
        # Apply block-wise quantization along the feature dimension (last dimension)
        if len(x.shape) == 4:
            # [batch, heads, seq_len, head_dim] -> block along head_dim (axis=-1)
            # This is for attention Q/K/V tensors
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        elif len(x.shape) == 3:
            # [batch, seq_len, hidden_dim] -> block along hidden_dim (axis=-1)
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        elif len(x.shape) == 2:
            # [batch, hidden_dim] -> block along hidden_dim (axis=-1)
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [-1], block_size=self.block_size
            )
        else:
            raise NotImplementedError(f"Non-supported activation shape: {list(x.size())}!")
        return x, axes, orig_shape, padded_shape
    
    def q(self, x:torch.Tensor):
        """
        Block-wise quantization along feature dimension
        One shared exponent per block of features
        """
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [a + 1 for a in axes]

        # Always compute scale and shared_exp from input
        # (scale is fixed based on nbit, shared_exp depends on input magnitude)
        scale, zero_point, shared_exp = self.observer(xg, shared_exp_axes[0])

        if self.train_flag:
            self.scale.data = 1 / scale
            self.zero_point.data = zero_point

        # Use float32 for intermediate computation to avoid float16 overflow/underflow
        xg_float = xg.float()
        shared_exp_float = shared_exp.float()
        
        xg_float = xg_float / (2**shared_exp_float)
        xg_float = xg_float * scale  # scale = 64 for 8-bit

        # round to nearest
        xg_float = torch.sign(xg_float) * torch.floor(torch.abs(xg_float) + 0.5)
        xg_float = xg_float.clamp(self.qlb, self.qub)

        # rescale back
        xg_float = xg_float * (2**shared_exp_float)

        # reshape the tensor
        xg_float = _undo_reshape_to_blocks(xg_float, padded_shape, orig_shape, axes)

        if self.dequantize:
            xg_float = xg_float / scale

        # Convert back to original dtype
        xg = xg_float.to(x.dtype)

        return xg

    def trainFunc(self, x:torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq
    
    def inference(self):
        """
        Override inference mode for MXINT activation.
        MXINT always needs dequantization to output valid float values.
        """
        self.train_flag = False
        self.dequantize = True  # Keep dequantization enabled for MXINT