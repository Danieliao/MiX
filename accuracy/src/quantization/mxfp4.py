"""
Micro Scaling FP4 (MXFP4) format by MicroSoft

MXFP4 uses E2M1 format: 2-bit exponent, 1-bit mantissa
Paper: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

E2M1 format:
- 1 sign bit
- 2 exponent bits (bias = 1)
- 1 mantissa bit
- Total: 4 bits

Special values:
- Exponent = 0, Mantissa = 0: ±0 (subnormal)
- Exponent = 0, Mantissa = 1: ±0.5 (subnormal)
- Exponent = 1-3: Normal numbers
- No Inf or NaN in this format

Representable values (positive):
- 0.0, 0.5 (subnormal)
- 1.0, 1.5 (exp=1: 2^0 * 1.m)
- 2.0, 3.0 (exp=2: 2^1 * 1.m)
- 4.0, 6.0 (exp=3: 2^2 * 1.m)
"""

import torch
import torch.nn as nn
from src.module.base import _QBase
from src.quantization.observer import BaseObserver


# E2M1 quantization levels (positive values)
# Format: (exp, mantissa) -> value = 2^(exp-bias) * (1 + m/2) for normal, 2^(1-bias) * m/2 for subnormal
# Bias = 1
# exp=0: subnormal -> 2^(1-1) * (0.m) = 0.0 or 0.5
# exp=1: 2^(1-1) * 1.m = 1.0 or 1.5
# exp=2: 2^(2-1) * 1.m = 2.0 or 3.0
# exp=3: 2^(3-1) * 1.m = 4.0 or 6.0
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0

# Midpoint thresholds between consecutive E2M1 levels for bucketization
# These are the decision boundaries for rounding to nearest level
E2M1_THRESHOLDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


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


def quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Quantize floating point values to E2M1 format.
    
    This maps each value to the nearest E2M1 representable value.
    Input values are assumed to be pre-scaled by the shared exponent,
    so they should be in the representable range [0, 6].
    
    Uses torch.bucketize for memory-efficient quantization without
    expanding the tensor dimensions.
    
    Args:
        x: Input tensor (absolute values, pre-scaled)
        
    Returns:
        Quantized tensor with E2M1 representable values
    """
    # Get the E2M1 levels and thresholds on the correct device
    levels = E2M1_LEVELS.to(x.device, dtype=x.dtype)
    thresholds = E2M1_THRESHOLDS.to(x.device, dtype=x.dtype)
    
    # Clamp to valid range
    x_clamped = torch.clamp(x, min=0.0, max=E2M1_MAX)
    
    # Use bucketize to find which bucket each value falls into
    # bucketize returns indices such that thresholds[i-1] < x <= thresholds[i]
    # This maps to the nearest E2M1 level index
    bucket_indices = torch.bucketize(x_clamped, thresholds)
    
    # Map bucket indices to E2M1 level values
    quantized = levels[bucket_indices]
    
    return quantized


class MXFP4Observer(BaseObserver):
    """
    Observer for MXFP4 (E2M1) format.
    
    Computes shared exponent per block to scale values into E2M1 representable range.
    """
    def __init__(self, nbit: int = 4, unsigned: bool = True):
        super().__init__(nbit, unsigned)
        # nbit is fixed at 4 for FP4, but we keep it for interface consistency

    def get_bound(self, x: torch.Tensor):
        # E2M1 max value is 6.0 after scaling
        max_norm = E2M1_MAX
        self.lb.copy_(-max_norm)
        self.ub.copy_(max_norm)
    
    def get_shared_exp(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """
        Compute shared exponent per block.
        
        The shared exponent scales the block maximum to fit within E2M1 range.
        """
        # Find max absolute value in each block
        max_abs, _ = torch.max(torch.abs(x), dim=axis, keepdim=True)
        
        # Handle zero values to avoid log2(0) = -inf
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
        
        # Compute exponent such that max_abs / 2^exp <= E2M1_MAX
        # exp = ceil(log2(max_abs / E2M1_MAX))
        # We use floor(log2(max_abs)) - floor(log2(E2M1_MAX)) for simplicity
        # This ensures values are scaled into [0, E2M1_MAX] range
        shared_exp = torch.floor(torch.log2(max_abs)) - torch.floor(torch.log2(torch.tensor(E2M1_MAX, device=x.device, dtype=x.dtype)))
        
        return shared_exp

    def calculate_qparam(self, x: torch.Tensor, axis: int = 2):
        """
        Calculate quantization parameters for MXFP4.
        
        Returns:
            scale: Scaling factor (for interface compatibility, not directly used)
            zero_point: Zero point (always 0 for symmetric quantization)
            shared_exp: Per-block shared exponent
        """
        shared_exp = self.get_shared_exp(x, axis=axis)
        # Scale is derived from shared_exp: scale = 2^shared_exp
        scale = 2.0 ** shared_exp
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return scale, zero_point, shared_exp
    
    def forward(self, x: torch.Tensor, axis: int = 2):
        self.get_bound(x)
        scale, zero_point, shared_exp = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_exp


class MXFP4ChannelWiseWeightQuantizer(_QBase):
    """
    Micro-scaling FP4 (MXFP4, E2M1) format for channel-wise weight quantization.
    
    Uses a shared 8-bit exponent per block and E2M1 format for each element.
    
    Args:
        nbit: Number of bits (fixed at 4 for FP4)
        train_flag: Whether in training mode
        unsigned: Whether to use unsigned quantization (False for weights)
        block_size: Size of each quantization block (default: 32)
    """
    def __init__(self, nbit: int = 4, train_flag: bool = True, unsigned: bool = False, block_size: int = 32):
        super().__init__(nbit, train_flag, unsigned)
        
        # nbit should be 4 for FP4, but we accept it for interface consistency
        if nbit != 4:
            print(f"[Warning] MXFP4 uses 4-bit format, ignoring nbit={nbit}")
        
        self.block_size = block_size
        self.observer = MXFP4Observer(nbit=4, unsigned=self.unsigned)

    def register_qparams(self):
        super().register_qparams()
        self.register_buffer("shared_exp", torch.tensor(1.0))

    def reshape(self, x: torch.Tensor):
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
    
    def q(self, x: torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [a + 1 for a in axes]

        # Compute shared exponent and scale
        scale, zero_point, shared_exp = self.observer(xg, shared_exp_axes[0])

        if self.train_flag:
            self.scale.data = scale.mean()  # Store average for monitoring
            self.zero_point.data = zero_point

        # Use float32 for intermediate computation
        xg_float = xg.float()
        shared_exp_float = shared_exp.float()
        
        # Scale down by shared exponent
        xg_scaled = xg_float / (2 ** shared_exp_float)
        
        # Store sign and work with absolute values
        signs = torch.sign(xg_scaled)
        xg_abs = torch.abs(xg_scaled)
        
        # Quantize to E2M1 levels
        xg_quantized = quantize_to_e2m1(xg_abs)
        
        # Restore sign and scale back up
        xg_float = signs * xg_quantized * (2 ** shared_exp_float)
        
        # Reshape back
        xg_float = _undo_reshape_to_blocks(xg_float, padded_shape, orig_shape, axes)

        # Convert back to original dtype
        xg = xg_float.to(x.dtype)

        # Debug: check for numerical issues
        if torch.isnan(xg).any() or torch.isinf(xg).any():
            print(f"[DEBUG] NaN/Inf detected in MXFP4 quantization output!")
            print(f"  - xg has NaN: {torch.isnan(xg).any()}, has Inf: {torch.isinf(xg).any()}")
            print(f"  - shared_exp min: {shared_exp.min()}, max: {shared_exp.max()}")
            print(f"  - Input x min: {x.min()}, max: {x.max()}")

        return xg

    def trainFunc(self, x: torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq
    
    def inference(self):
        """
        Override inference mode for MXFP4.
        MXFP4 always needs dequantization to output valid float values.
        """
        self.train_flag = False
        self.dequantize = True


class MXFP4ActivationQuantizer(_QBase):
    """
    Micro-scaling FP4 (MXFP4, E2M1) format for block-wise activation quantization.
    
    Blocks are created along the feature dimension, with one shared exponent per block.
    
    Args:
        nbit: Number of bits (fixed at 4 for FP4)
        train_flag: Whether in training mode
        unsigned: Whether to use unsigned quantization
        block_size: Size of each quantization block (default: 32)
    """
    def __init__(self, nbit: int = 4, train_flag: bool = True, unsigned: bool = False, block_size: int = 32):
        super().__init__(nbit, train_flag, unsigned)
        
        # nbit should be 4 for FP4, but we accept it for interface consistency
        if nbit != 4:
            print(f"[Warning] MXFP4 uses 4-bit format, ignoring nbit={nbit}")
        
        self.block_size = block_size
        self.observer = MXFP4Observer(nbit=4, unsigned=self.unsigned)

    def register_qparams(self):
        super().register_qparams()

    def reshape(self, x: torch.Tensor):
        # Apply block-wise quantization along the feature dimension (last dimension)
        if len(x.shape) == 4:
            # [batch, heads, seq_len, head_dim] -> block along head_dim (axis=-1)
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
    
    def q(self, x: torch.Tensor):
        """
        Block-wise MXFP4 quantization along feature dimension.
        One shared exponent per block of features.
        """
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [a + 1 for a in axes]

        # Compute shared exponent and scale
        scale, zero_point, shared_exp = self.observer(xg, shared_exp_axes[0])

        if self.train_flag:
            self.scale.data = scale.mean()
            self.zero_point.data = zero_point

        # Use float32 for intermediate computation
        xg_float = xg.float()
        shared_exp_float = shared_exp.float()
        
        # Scale down by shared exponent
        xg_scaled = xg_float / (2 ** shared_exp_float)
        
        # Store sign and work with absolute values
        signs = torch.sign(xg_scaled)
        xg_abs = torch.abs(xg_scaled)
        
        # Quantize to E2M1 levels
        xg_quantized = quantize_to_e2m1(xg_abs)
        
        # Restore sign and scale back up
        xg_float = signs * xg_quantized * (2 ** shared_exp_float)
        
        # Reshape back
        xg_float = _undo_reshape_to_blocks(xg_float, padded_shape, orig_shape, axes)

        # Convert back to original dtype
        xg = xg_float.to(x.dtype)

        return xg

    def trainFunc(self, x: torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq
    
    def inference(self):
        """
        Override inference mode for MXFP4 activation.
        MXFP4 always needs dequantization to output valid float values.
        """
        self.train_flag = False
        self.dequantize = True


# =============================================================================
# Variant 1: Ceiling-based shared exponent (no clipping)
# =============================================================================

class MXFP4CeilObserver(BaseObserver):
    """
    MXFP4 observer using ceil(log2(max_abs / E2M1_MAX)) for shared exponent.

    Guarantees no clipping: scaled values always fit within [0, E2M1_MAX].
    Trade-off: scaled max lands in [E2M1_MAX/2, E2M1_MAX] = [3, 6], slightly
    lower range utilization than floor-based approach.
    """
    def __init__(self, nbit: int = 4, unsigned: bool = True):
        super().__init__(nbit, unsigned)

    def get_bound(self, x: torch.Tensor):
        max_norm = E2M1_MAX
        self.lb.copy_(-max_norm)
        self.ub.copy_(max_norm)

    def get_shared_exp(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        max_abs, _ = torch.max(torch.abs(x), dim=axis, keepdim=True)
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
        # ceil(log2(max_abs / E2M1_MAX)) guarantees max_abs / 2^exp <= E2M1_MAX
        shared_exp = torch.ceil(torch.log2(max_abs / E2M1_MAX))
        return shared_exp

    def calculate_qparam(self, x: torch.Tensor, axis: int = 2):
        shared_exp = self.get_shared_exp(x, axis=axis)
        scale = 2.0 ** shared_exp
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return scale, zero_point, shared_exp

    def forward(self, x: torch.Tensor, axis: int = 2):
        self.get_bound(x)
        scale, zero_point, shared_exp = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_exp


class MXFP4CeilChannelWiseWeightQuantizer(MXFP4ChannelWiseWeightQuantizer):
    """MXFP4 weight quantizer with ceiling-based shared exponent (no clipping)."""
    def __init__(self, nbit: int = 4, train_flag: bool = True, unsigned: bool = False, block_size: int = 32):
        super().__init__(nbit, train_flag, unsigned, block_size)
        self.observer = MXFP4CeilObserver(nbit=4, unsigned=self.unsigned)


class MXFP4CeilActivationQuantizer(MXFP4ActivationQuantizer):
    """MXFP4 activation quantizer with ceiling-based shared exponent (no clipping)."""
    def __init__(self, nbit: int = 4, train_flag: bool = True, unsigned: bool = False, block_size: int = 32):
        super().__init__(nbit, train_flag, unsigned, block_size)
        self.observer = MXFP4CeilObserver(nbit=4, unsigned=self.unsigned)


# =============================================================================
# Variant 2: 3/4 scaling (Tseng et al., AISTATS 2025)
# =============================================================================

# Hardware approximation: 16/9 ≈ 1.78125 (via shift-add: 2 - 1/4 + 1/32)
# Per-tensor compensation = sqrt(1.78125) so that A_fake @ B_fake uses 1.78125 total.
_HW_16_OVER_9 = 1.78125
_HW_COMPENSATE = _HW_16_OVER_9 ** 0.5  # 1.334634...


def _build_e2m3_levels():
    """E2M3 positive representable levels (2 exp bits, 3 mantissa bits)."""
    levels = set()
    # subnormal (exp field 0): 0.mmm * 2^0
    for m in range(8):
        levels.add(m / 8.0)
    # normal binades matching E2M1's exp in {0,1,2}: (1 + m/8) * 2^e
    for e in range(3):
        for m in range(8):
            levels.add((1.0 + m / 8.0) * (2 ** e))
    return torch.tensor(sorted(levels))

E2M3_LEVELS = _build_e2m3_levels()                 # 32 levels, max 7.5
E2M3_MAX = float(E2M3_LEVELS[-1].item())           # 7.5
E2M3_THRESHOLDS = (E2M3_LEVELS[1:] + E2M3_LEVELS[:-1]) / 2.0


def quantize_to_e2m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize absolute, pre-scaled values to the nearest E2M3 level."""
    levels = E2M3_LEVELS.to(x.device, dtype=x.dtype)
    thresholds = E2M3_THRESHOLDS.to(x.device, dtype=x.dtype)
    x_clamped = torch.clamp(x, min=0.0, max=E2M3_MAX)
    return levels[torch.bucketize(x_clamped, thresholds)]


def _mxplus_q(self, x: torch.Tensor) -> torch.Tensor:
    """MX+ fake-quant: E2M1 for all elements, E2M3 for the per-block maximum.

    Shared (floor/ceil) exponent comes from self.observer, so subclassing only
    needs to swap the observer to get the ceil-based variant.
    """
    xg, axes, orig_shape, padded_shape = self.reshape(x)
    block_axis = [a + 1 for a in axes][0]

    scale, zero_point, shared_exp = self.observer(xg, block_axis)
    if self.train_flag:
        self.scale.data = scale.mean()
        self.zero_point.data = zero_point

    xg_float = xg.float()
    se = shared_exp.float()
    xg_scaled = xg_float / (2 ** se)
    signs = torch.sign(xg_scaled)
    xg_abs = torch.abs(xg_scaled)

    q_e2m1 = quantize_to_e2m1(xg_abs)
    q_e2m3 = quantize_to_e2m3(xg_abs)
    # Block-maximum element(s): the largest |value| along the block axis gets E2M3.
    is_bm = xg_abs == xg_abs.max(dim=block_axis, keepdim=True).values
    q = torch.where(is_bm, q_e2m3, q_e2m1)

    xg_float = signs * q * (2 ** se)
    xg_float = _undo_reshape_to_blocks(xg_float, padded_shape, orig_shape, axes)
    return xg_float.to(x.dtype)


class MXFP4PlusChannelWiseWeightQuantizer(MXFP4ChannelWiseWeightQuantizer):
    """MX+ weight quantizer (floor shared exponent, like base MX)."""
    def q(self, x: torch.Tensor):
        return _mxplus_q(self, x)


class MXFP4PlusActivationQuantizer(MXFP4ActivationQuantizer):
    """MX+ activation quantizer (floor shared exponent)."""
    def q(self, x: torch.Tensor):
        return _mxplus_q(self, x)


class MXFP4PlusCeilChannelWiseWeightQuantizer(MXFP4ChannelWiseWeightQuantizer):
    """MX+ weight quantizer with ceil shared exponent (no NBM clipping)."""
    def __init__(self, nbit: int = 4, train_flag: bool = True, unsigned: bool = False, block_size: int = 32):
        super().__init__(nbit, train_flag, unsigned, block_size)
        self.observer = MXFP4CeilObserver(nbit=4, unsigned=self.unsigned)
    def q(self, x: torch.Tensor):
        return _mxplus_q(self, x)


class MXFP4PlusCeilActivationQuantizer(MXFP4ActivationQuantizer):
    """MX+ activation quantizer with ceil shared exponent (no NBM clipping)."""
    def __init__(self, nbit: int = 4, train_flag: bool = True, unsigned: bool = False, block_size: int = 32):
        super().__init__(nbit, train_flag, unsigned, block_size)
        self.observer = MXFP4CeilObserver(nbit=4, unsigned=self.unsigned)
    def q(self, x: torch.Tensor):
        return _mxplus_q(self, x)


# =============================================================================
# Variant 4: AMXFP4 (asymmetric MXFP4)
# Lee et al., "AMXFP4: Taming Activation Outliers ...", 2025 (github.com/aiha-lab/MX-QLLM).
#
# E2M1 elements with TWO asymmetric FP8 (E5M2) shared scales per block: one for
# positive values, one for negative. Each element is quantized with the scale
# matching its sign, fitting asymmetric outlier distributions. The scales map the
# per-sign max exactly to E2M1_MAX, so AMXFP4 is non-clipping by construction
# (no separate floor/ceil variant needed). Two FP8 scales = +8b/block -> 4.5b.
# =============================================================================

def quantize_to_e5m2(s: torch.Tensor) -> torch.Tensor:
    """Round a positive scale tensor to the nearest FP8 E5M2 value (2 mantissa bits)."""
    s = torch.clamp(s, min=1e-12)
    e = torch.floor(torch.log2(s))
    mant = s / (2 ** e)                      # in [1, 2)
    mant_q = torch.round((mant - 1.0) * 4.0) / 4.0 + 1.0   # 2-bit mantissa
    return mant_q * (2 ** e)


def _amxfp4_q(self, x: torch.Tensor) -> torch.Tensor:
    """AMXFP4 fake-quant: per-block, independent E5M2 scales for + and - values."""
    xg, axes, orig_shape, padded_shape = self.reshape(x)
    block_axis = [a + 1 for a in axes][0]

    xg_float = xg.float()
    pos = torch.clamp(xg_float, min=0.0)          # positive magnitudes (0 elsewhere)
    neg = torch.clamp(-xg_float, min=0.0)         # negative magnitudes (0 elsewhere)

    pos_max = pos.max(dim=block_axis, keepdim=True).values
    neg_max = neg.max(dim=block_axis, keepdim=True).values
    scale_pos = quantize_to_e5m2(torch.clamp(pos_max, min=1e-12) / E2M1_MAX)
    scale_neg = quantize_to_e5m2(torch.clamp(neg_max, min=1e-12) / E2M1_MAX)

    if self.train_flag:
        self.scale.data = scale_pos.mean()
        self.zero_point.data = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    q_pos = quantize_to_e2m1(pos / scale_pos) * scale_pos
    q_neg = quantize_to_e2m1(neg / scale_neg) * scale_neg
    xg_q = q_pos - q_neg                            # +ve stay +ve, -ve become -ve

    xg_q = _undo_reshape_to_blocks(xg_q, padded_shape, orig_shape, axes)
    return xg_q.to(x.dtype)


class AMXFP4ChannelWiseWeightQuantizer(MXFP4ChannelWiseWeightQuantizer):
    """AMXFP4 weight quantizer (asymmetric +/- E5M2 scales)."""
    def q(self, x: torch.Tensor):
        return _amxfp4_q(self, x)


class AMXFP4ActivationQuantizer(MXFP4ActivationQuantizer):
    """AMXFP4 activation quantizer (asymmetric +/- E5M2 scales)."""
    def q(self, x: torch.Tensor):
        return _amxfp4_q(self, x)


