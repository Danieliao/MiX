"""
VLM (Vision-Language Model) converters for quantization.

Converts vanilla VLM modules to quantized (_QBaseLinear) modules.
Supports LLaVA-1.5 (CLIP vision encoder + Llama LLM), Qwen2-VL, and Qwen3-VL.
"""
import torch
import torch.nn as nn

from src.module.base import _QBaseLinear
from src.module.attention import QSiglipAttention, QQwen2VLVisionAttention, QQwen2VLAttention, QQwen2_5_VLVisionAttention
from src.module.mlp import QQwen2MLP
from src.t2c.convert import Vanilla4Compress, Llama4Compress, QWen4Compress, get_parent_name


class Projector4Compress(Vanilla4Compress):
    """Convert VLM projector (e.g., LlavaMultiModalProjector) Linear layers."""
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8):
        super().__init__(model, wbit, abit)

    def convert(self):
        """Replace all nn.Linear in the projector with _QBaseLinear."""
        model = self.model
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, nn.Linear):
                parent_name, name = get_parent_name(n)
                new_layer = self.linear(m)
                setattr(modules[parent_name], name, new_layer)

        return model


class VLM4Compress:
    """Composite VLM converter that orchestrates per-component conversion.

    Concrete subclasses (per model) convert:
        - vision encoder  -> a Vanilla4Compress-derived vision converter
        - multi_modal_projector / resampler -> Projector4Compress
        - language_model  -> Llama4Compress / QWen4Compress

    Each component gets its own quantization config (wbit, abit, etc.).
    Subclasses override ``_get_vision_converter`` and ``convert``.
    """
    def __init__(self, model: nn.Module, vision_config: dict, projector_config: dict, llm_config: dict):
        self.model = model
        self.vision_config = vision_config
        self.projector_config = projector_config
        self.llm_config = llm_config

    def _get_vision_converter(self, vision_module):
        raise NotImplementedError(
            "Subclasses must implement _get_vision_converter for their vision encoder."
        )

    def _get_projector_converter(self, projector_module):
        return Projector4Compress(
            model=projector_module,
            wbit=self.projector_config.get("wbit", 8),
            abit=self.projector_config.get("abit", 8),
        )

    def _get_llm_converter(self, llm_module):
        return Llama4Compress(
            model=llm_module,
            wbit=self.llm_config.get("wbit", 8),
            abit=self.llm_config.get("abit", 8),
            quantize_bmm_input=self.llm_config.get("quantize_bmm_input", False),
            bmm_qtype=self.llm_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.llm_config.get("bmm_bits", 8),
            bmm_ebit=self.llm_config.get("bmm_ebit", None),
            bmm_block_size=self.llm_config.get("bmm_block_size", None),
            bmm_q_bits=self.llm_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.llm_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.llm_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.llm_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.llm_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.llm_config.get("bmm_kv_block_size", None),
        )

    def convert(self):
        """Convert all VLM components to quantized modules."""
        # LLaVA-1.5 structure:
        #   model.vision_tower — CLIP ViT
        #   model.multi_modal_projector — MLP projector
        #   model.language_model — Llama LLM

        # 1. Convert vision encoder
        vision_tower = self.model.vision_tower
        vision_converter = self._get_vision_converter(vision_tower)
        self.model.vision_tower = vision_converter.convert()

        # 2. Convert projector
        projector = self.model.multi_modal_projector
        proj_converter = self._get_projector_converter(projector)
        self.model.multi_modal_projector = proj_converter.convert()

        # 3. Convert LLM backbone
        llm = self.model.language_model
        llm_converter = self._get_llm_converter(llm)
        self.model.language_model = llm_converter.convert()

        return self.model


class SiglipVision4Compress(Vanilla4Compress):
    """Convert SiglipVisionModel to quantized modules.

    When quantize_bmm_input is True, replaces SiglipAttention with
    QSiglipAttention (which subclasses SiglipAttention to preserve isinstance
    checks for the @check_model_inputs decorator), then replaces remaining
    nn.Linear (MLP layers) with _QBaseLinear.

    When quantize_bmm_input is False, all nn.Linear layers (including
    q/k/v/out_proj inside attention) are replaced with _QBaseLinear.
    """
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8,
                 quantize_bmm_input: bool = False, bmm_qtype: str = "smooth_quant",
                 bmm_bits: int = 8, bmm_ebit: int = None, bmm_block_size: int = None,
                 bmm_q_bits: int = None, bmm_kv_bits: int = None,
                 bmm_kv_ebit: int = None, bmm_q_ebit: int = None,
                 bmm_q_block_size: int = None, bmm_kv_block_size: int = None,
                 bmm_kv_shared_exp_bits: int = None, bmm_kv_shared_exp_relative: bool = False):
        super().__init__(model, wbit, abit)
        self.quantize_bmm_input = quantize_bmm_input
        self.bmm_qtype = bmm_qtype
        self.bmm_bits = bmm_bits
        self.bmm_ebit = bmm_ebit
        self.bmm_block_size = bmm_block_size
        self.bmm_q_bits = bmm_q_bits
        self.bmm_kv_bits = bmm_kv_bits
        self.bmm_kv_ebit = bmm_kv_ebit
        self.bmm_q_ebit = bmm_q_ebit
        self.bmm_q_block_size = bmm_q_block_size
        self.bmm_kv_block_size = bmm_kv_block_size
        self.bmm_kv_shared_exp_bits = bmm_kv_shared_exp_bits
        self.bmm_kv_shared_exp_relative = bmm_kv_shared_exp_relative

    def _create_qsiglip_attention(self, orig_attn):
        """Create QSiglipAttention from an existing SiglipAttention, copying weights."""
        new_attn = QSiglipAttention(
            config=orig_attn.config,
            quantize_bmm_input=self.quantize_bmm_input,
            bmm_qtype=self.bmm_qtype,
            bmm_bits=self.bmm_bits,
            bmm_ebit=self.bmm_ebit,
            bmm_block_size=self.bmm_block_size,
            bmm_q_bits=self.bmm_q_bits,
            bmm_kv_bits=self.bmm_kv_bits,
            bmm_kv_ebit=self.bmm_kv_ebit,
            bmm_q_ebit=self.bmm_q_ebit,
            bmm_q_block_size=self.bmm_q_block_size,
            bmm_kv_block_size=self.bmm_kv_block_size,
            bmm_kv_shared_exp_bits=self.bmm_kv_shared_exp_bits,
            bmm_kv_shared_exp_relative=self.bmm_kv_shared_exp_relative,
        )
        # Copy weights and biases from original projections
        new_attn.q_proj.weight = orig_attn.q_proj.weight
        new_attn.q_proj.bias = orig_attn.q_proj.bias
        new_attn.k_proj.weight = orig_attn.k_proj.weight
        new_attn.k_proj.bias = orig_attn.k_proj.bias
        new_attn.v_proj.weight = orig_attn.v_proj.weight
        new_attn.v_proj.bias = orig_attn.v_proj.bias
        new_attn.out_proj.weight = orig_attn.out_proj.weight
        new_attn.out_proj.bias = orig_attn.out_proj.bias
        return new_attn

    def convert(self):
        """Convert SiglipVisionModel to quantized modules.

        Pass 1: If BMM enabled, replace SiglipAttention -> QSiglipAttention.
        Pass 2: Replace remaining nn.Linear (MLP layers, etc.) -> _QBaseLinear.
        """
        from transformers.models.siglip.modeling_siglip import SiglipAttention

        model = self.model

        # Pass 1: Replace SiglipAttention with QSiglipAttention (if BMM enabled)
        # Match by isinstance OR class name to support custom SiglipAttention
        # (e.g., MiniCPM-V's NaViT variant from trust_remote_code)
        if self.quantize_bmm_input:
            modules = dict(model.named_modules(remove_duplicate=True))
            for n, m in modules.items():
                if isinstance(m, SiglipAttention) or type(m).__name__ == "SiglipAttention":
                    parent_name, name = get_parent_name(n)
                    new_attn = self._create_qsiglip_attention(m)
                    setattr(modules[parent_name], name, new_attn)

        # Pass 2: Replace remaining nn.Linear with _QBaseLinear
        modules = dict(model.named_modules(remove_duplicate=True))
        for n, m in modules.items():
            if isinstance(m, nn.Linear) and not isinstance(m, _QBaseLinear):
                parent_name, name = get_parent_name(n)
                new_layer = self.linear(m)
                setattr(modules[parent_name], name, new_layer)

        return model


class MiniCPMV4Compress(VLM4Compress):
    """Composite converter for MiniCPM-V-2.6.

    MiniCPM-V-2.6 structure:
        - model.vpm — SiglipVisionModel (from timm)
        - model.resampler — Cross-attention resampler (projector)
        - model.llm — Qwen2ForCausalLM
    """
    def _get_vision_converter(self, vision_module):
        return SiglipVision4Compress(
            model=vision_module,
            wbit=self.vision_config.get("wbit", 8),
            abit=self.vision_config.get("abit", 8),
            quantize_bmm_input=self.vision_config.get("quantize_bmm_input", False),
            bmm_qtype=self.vision_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.vision_config.get("bmm_bits", 8),
            bmm_ebit=self.vision_config.get("bmm_ebit", None),
            bmm_block_size=self.vision_config.get("bmm_block_size", None),
            bmm_q_bits=self.vision_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.vision_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.vision_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.vision_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.vision_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.vision_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.vision_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.vision_config.get("bmm_kv_shared_exp_relative", False),
        )

    def _get_llm_converter(self, llm_module):
        return QWen4Compress(
            model=llm_module,
            wbit=self.llm_config.get("wbit", 8),
            abit=self.llm_config.get("abit", 8),
            quantize_bmm_input=self.llm_config.get("quantize_bmm_input", False),
            bmm_qtype=self.llm_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.llm_config.get("bmm_bits", 8),
            bmm_ebit=self.llm_config.get("bmm_ebit", None),
            bmm_block_size=self.llm_config.get("bmm_block_size", None),
            bmm_q_bits=self.llm_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.llm_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.llm_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.llm_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.llm_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.llm_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.llm_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.llm_config.get("bmm_kv_shared_exp_relative", False),
            k_smooth_static=self.llm_config.get("k_smooth_static", False),
        )

    def convert(self):
        """Convert all MiniCPM-V-2.6 components."""
        # 1. Convert vision encoder
        vpm = self.model.vpm
        vision_converter = self._get_vision_converter(vpm)
        self.model.vpm = vision_converter.convert()

        # 2. Convert resampler (projector)
        resampler = self.model.resampler
        proj_converter = self._get_projector_converter(resampler)
        self.model.resampler = proj_converter.convert()

        # 3. Convert LLM backbone
        llm = self.model.llm
        llm_converter = self._get_llm_converter(llm)
        self.model.llm = llm_converter.convert()

        return self.model


class LlavaOnevision4Compress(VLM4Compress):
    """Composite converter for LLaVA-OneVision.

    LLaVA-OneVision structure:
        - model.vision_tower — SiglipVisionModel
        - model.multi_modal_projector — linear_1 + GELU + linear_2
        - model.language_model — Qwen2 decoder
    """
    def _get_vision_converter(self, vision_module):
        return SiglipVision4Compress(
            model=vision_module,
            wbit=self.vision_config.get("wbit", 8),
            abit=self.vision_config.get("abit", 8),
            quantize_bmm_input=self.vision_config.get("quantize_bmm_input", False),
            bmm_qtype=self.vision_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.vision_config.get("bmm_bits", 8),
            bmm_ebit=self.vision_config.get("bmm_ebit", None),
            bmm_block_size=self.vision_config.get("bmm_block_size", None),
            bmm_q_bits=self.vision_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.vision_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.vision_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.vision_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.vision_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.vision_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.vision_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.vision_config.get("bmm_kv_shared_exp_relative", False),
        )

    def _get_llm_converter(self, llm_module):
        return QWen4Compress(
            model=llm_module,
            wbit=self.llm_config.get("wbit", 8),
            abit=self.llm_config.get("abit", 8),
            quantize_bmm_input=self.llm_config.get("quantize_bmm_input", False),
            bmm_qtype=self.llm_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.llm_config.get("bmm_bits", 8),
            bmm_ebit=self.llm_config.get("bmm_ebit", None),
            bmm_block_size=self.llm_config.get("bmm_block_size", None),
            bmm_q_bits=self.llm_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.llm_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.llm_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.llm_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.llm_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.llm_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.llm_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.llm_config.get("bmm_kv_shared_exp_relative", False),
            k_smooth_static=self.llm_config.get("k_smooth_static", False),
        )


class Qwen2VLVision4Compress(Vanilla4Compress):
    """Convert Qwen2-VL vision encoder (visual.blocks) to quantized modules.

    Two-pass conversion:
    - Pass 1: Replace VisionAttention -> QQwen2VLVisionAttention (with BMM params)
    - Pass 2: Replace remaining nn.Linear (MLP, merger) -> _QBaseLinear
    """
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8,
                 quantize_bmm_input: bool = False, bmm_qtype: str = "smooth_quant",
                 bmm_bits: int = 8, bmm_ebit: int = None, bmm_block_size: int = None,
                 bmm_q_bits: int = None, bmm_kv_bits: int = None,
                 bmm_kv_ebit: int = None, bmm_q_ebit: int = None,
                 bmm_q_block_size: int = None, bmm_kv_block_size: int = None,
                 bmm_kv_shared_exp_bits: int = None, bmm_kv_shared_exp_relative: bool = False):
        super().__init__(model, wbit, abit)
        self.quantize_bmm_input = quantize_bmm_input
        self.bmm_qtype = bmm_qtype
        self.bmm_bits = bmm_bits
        self.bmm_ebit = bmm_ebit
        self.bmm_block_size = bmm_block_size
        self.bmm_q_bits = bmm_q_bits
        self.bmm_kv_bits = bmm_kv_bits
        self.bmm_kv_ebit = bmm_kv_ebit
        self.bmm_q_ebit = bmm_q_ebit
        self.bmm_q_block_size = bmm_q_block_size
        self.bmm_kv_block_size = bmm_kv_block_size
        self.bmm_kv_shared_exp_bits = bmm_kv_shared_exp_bits
        self.bmm_kv_shared_exp_relative = bmm_kv_shared_exp_relative

    def _create_qvision_attention(self, orig_attn):
        """Create QQwen2VLVisionAttention from original, copying weights."""
        new_attn = QQwen2VLVisionAttention(
            config=orig_attn.config,
            quantize_bmm_input=self.quantize_bmm_input,
            bmm_qtype=self.bmm_qtype,
            bmm_bits=self.bmm_bits,
            bmm_ebit=self.bmm_ebit,
            bmm_block_size=self.bmm_block_size,
            bmm_q_bits=self.bmm_q_bits,
            bmm_kv_bits=self.bmm_kv_bits,
            bmm_kv_ebit=self.bmm_kv_ebit,
            bmm_q_ebit=self.bmm_q_ebit,
            bmm_q_block_size=self.bmm_q_block_size,
            bmm_kv_block_size=self.bmm_kv_block_size,
            bmm_kv_shared_exp_bits=self.bmm_kv_shared_exp_bits,
            bmm_kv_shared_exp_relative=self.bmm_kv_shared_exp_relative,
        )
        new_attn.qkv.weight = orig_attn.qkv.weight
        new_attn.qkv.bias = orig_attn.qkv.bias
        new_attn.proj.weight = orig_attn.proj.weight
        new_attn.proj.bias = orig_attn.proj.bias
        return new_attn

    def convert(self):
        """Convert Qwen2-VL vision encoder.

        Pass 1: If BMM enabled, replace VisionAttention -> QQwen2VLVisionAttention.
        Pass 2: Replace remaining nn.Linear (MLP, merger) -> _QBaseLinear.
        """
        from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention

        model = self.model

        # Pass 1: Replace VisionAttention with QQwen2VLVisionAttention (if BMM enabled)
        if self.quantize_bmm_input:
            modules = dict(model.named_modules(remove_duplicate=True))
            for n, m in modules.items():
                if isinstance(m, VisionAttention):
                    parent_name, name = get_parent_name(n)
                    new_attn = self._create_qvision_attention(m)
                    setattr(modules[parent_name], name, new_attn)

        # Pass 2: Replace remaining nn.Linear with _QBaseLinear
        modules = dict(model.named_modules(remove_duplicate=True))
        for n, m in modules.items():
            if isinstance(m, nn.Linear) and not isinstance(m, _QBaseLinear):
                parent_name, name = get_parent_name(n)
                new_layer = self.linear(m)
                setattr(modules[parent_name], name, new_layer)

        return model


class Qwen2VLLLM4Compress(QWen4Compress):
    """Convert Qwen2-VL LLM decoder layers to quantized modules.

    Qwen2-VL uses Qwen2VLAttention (NOT Qwen2Attention) for its decoder layers.
    Qwen2VLAttention has multimodal rotary embeddings and sliding window support.
    QWen4Compress (with attn_cls=Qwen2Attention) would never match these layers,
    leaving the LLM entirely unquantized.
    """
    def __init__(self, model, wbit=8, abit=8, state_dict=None,
                 quantize_bmm_input=False, bmm_qtype="smooth_quant",
                 bmm_bits=8, bmm_ebit=None, bmm_block_size=None,
                 bmm_q_bits=None, bmm_kv_bits=None,
                 bmm_kv_ebit=None, bmm_q_ebit=None,
                 bmm_q_block_size=None, bmm_kv_block_size=None,
                 bmm_kv_shared_exp_bits=None, bmm_kv_shared_exp_relative=False,
                 k_smooth_static=False):
        super().__init__(model, wbit, abit, state_dict)
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, Qwen2MLP as Qwen2VLMlp
        self.attn_cls = Qwen2VLAttention
        self.qattn_cls = QQwen2VLAttention
        # Qwen2-VL re-exports Qwen2MLP in its own module — different class object
        # from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
        self.mlp_cls = Qwen2VLMlp
        self.quantize_bmm_input = quantize_bmm_input
        self.bmm_qtype = bmm_qtype
        self.bmm_bits = bmm_bits
        self.bmm_ebit = bmm_ebit
        self.bmm_block_size = bmm_block_size
        self.bmm_q_bits = bmm_q_bits
        self.bmm_kv_bits = bmm_kv_bits
        self.bmm_kv_ebit = bmm_kv_ebit
        self.bmm_q_ebit = bmm_q_ebit
        self.bmm_q_block_size = bmm_q_block_size
        self.bmm_kv_block_size = bmm_kv_block_size
        self.bmm_kv_shared_exp_bits = bmm_kv_shared_exp_bits
        self.bmm_kv_shared_exp_relative = bmm_kv_shared_exp_relative
        self.k_smooth_static = k_smooth_static

    def attn(self, attn):
        """Create QQwen2VLAttention from original, copying weights."""
        new_attn = self.qattn_cls(
            attn.config, attn.layer_idx,
            quantize_bmm_input=self.quantize_bmm_input,
            bmm_qtype=self.bmm_qtype,
            bmm_bits=self.bmm_bits,
            bmm_ebit=self.bmm_ebit,
            bmm_block_size=self.bmm_block_size,
            bmm_q_bits=self.bmm_q_bits,
            bmm_kv_bits=self.bmm_kv_bits,
            bmm_kv_ebit=self.bmm_kv_ebit,
            bmm_q_ebit=self.bmm_q_ebit,
            bmm_q_block_size=self.bmm_q_block_size,
            bmm_kv_block_size=self.bmm_kv_block_size,
            bmm_kv_shared_exp_bits=self.bmm_kv_shared_exp_bits,
            bmm_kv_shared_exp_relative=self.bmm_kv_shared_exp_relative,
            k_smooth_static=self.k_smooth_static,
        )
        new_attn.load_state_dict(attn.state_dict(), strict=False)
        new_attn = self.to_half(new_attn)
        return new_attn


class Qwen2VL4Compress(VLM4Compress):
    """Composite converter for Qwen2-VL.

    Qwen2-VL structure:
        - model.visual — ViT with blocks + merger (PatchMerger)
        - model.model — Qwen2 decoder layers
    """
    def __init__(self, model: nn.Module, vision_config: dict, projector_config: dict, llm_config: dict):
        super().__init__(model, vision_config, projector_config, llm_config)

    def _get_vision_converter(self, vision_module):
        return Qwen2VLVision4Compress(
            model=vision_module,
            wbit=self.vision_config.get("wbit", 8),
            abit=self.vision_config.get("abit", 8),
            quantize_bmm_input=self.vision_config.get("quantize_bmm_input", False),
            bmm_qtype=self.vision_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.vision_config.get("bmm_bits", 8),
            bmm_ebit=self.vision_config.get("bmm_ebit", None),
            bmm_block_size=self.vision_config.get("bmm_block_size", None),
            bmm_q_bits=self.vision_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.vision_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.vision_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.vision_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.vision_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.vision_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.vision_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.vision_config.get("bmm_kv_shared_exp_relative", False),
        )

    def _get_llm_converter(self, llm_module):
        """Qwen2-VL uses Qwen2VLAttention (not Qwen2Attention) in its decoder."""
        return Qwen2VLLLM4Compress(
            model=llm_module,
            wbit=self.llm_config.get("wbit", 8),
            abit=self.llm_config.get("abit", 8),
            quantize_bmm_input=self.llm_config.get("quantize_bmm_input", False),
            bmm_qtype=self.llm_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.llm_config.get("bmm_bits", 8),
            bmm_ebit=self.llm_config.get("bmm_ebit", None),
            bmm_block_size=self.llm_config.get("bmm_block_size", None),
            bmm_q_bits=self.llm_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.llm_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.llm_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.llm_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.llm_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.llm_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.llm_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.llm_config.get("bmm_kv_shared_exp_relative", False),
            k_smooth_static=self.llm_config.get("k_smooth_static", False),
        )

    def convert(self):
        """Convert all Qwen2-VL components."""
        # Qwen2-VL structure:
        #   model.visual — ViT blocks + merger
        #   model.model — Qwen2 decoder

        # 1. Convert vision encoder (includes PatchMerger as projector)
        visual = self.model.visual
        vision_converter = self._get_vision_converter(visual)
        self.model.visual = vision_converter.convert()

        # 2. Convert LLM backbone
        # Qwen2-VL's LLM is at model.model (the Qwen2 decoder)
        llm = self.model.model
        llm_converter = self._get_llm_converter(llm)
        self.model.model = llm_converter.convert()

        return self.model


class Qwen2_5_VLVision4Compress(Qwen2VLVision4Compress):
    """Convert Qwen2.5-VL vision encoder (visual.blocks) to quantized modules.

    Same two-pass scheme as Qwen2-VL, but Pass 1 matches Qwen2.5-VL's vision
    attention class and instantiates QQwen2_5_VLVisionAttention (which sizes
    projections from config.hidden_size). Pass 2's generic nn.Linear ->
    _QBaseLinear swap covers the gated (SwiGLU) vision MLP and the patch merger
    automatically. Window attention is preserved because the parent vision
    transformer forward (untouched) feeds the right cu_seqlens to each block.
    """
    def _create_qvision_attention(self, orig_attn):
        new_attn = QQwen2_5_VLVisionAttention(
            config=orig_attn.config,
            quantize_bmm_input=self.quantize_bmm_input,
            bmm_qtype=self.bmm_qtype,
            bmm_bits=self.bmm_bits,
            bmm_ebit=self.bmm_ebit,
            bmm_block_size=self.bmm_block_size,
            bmm_q_bits=self.bmm_q_bits,
            bmm_kv_bits=self.bmm_kv_bits,
            bmm_kv_ebit=self.bmm_kv_ebit,
            bmm_q_ebit=self.bmm_q_ebit,
            bmm_q_block_size=self.bmm_q_block_size,
            bmm_kv_block_size=self.bmm_kv_block_size,
            bmm_kv_shared_exp_bits=self.bmm_kv_shared_exp_bits,
            bmm_kv_shared_exp_relative=self.bmm_kv_shared_exp_relative,
        )
        new_attn.qkv.weight = orig_attn.qkv.weight
        new_attn.qkv.bias = orig_attn.qkv.bias
        new_attn.proj.weight = orig_attn.proj.weight
        new_attn.proj.bias = orig_attn.proj.bias
        return new_attn

    def convert(self):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention

        model = self.model

        # Pass 1: Replace Qwen2_5_VLVisionAttention -> QQwen2_5_VLVisionAttention
        if self.quantize_bmm_input:
            modules = dict(model.named_modules(remove_duplicate=True))
            for n, m in modules.items():
                if isinstance(m, Qwen2_5_VLVisionAttention):
                    parent_name, name = get_parent_name(n)
                    new_attn = self._create_qvision_attention(m)
                    setattr(modules[parent_name], name, new_attn)

        # Pass 2: Replace remaining nn.Linear (SwiGLU MLP, merger) -> _QBaseLinear
        modules = dict(model.named_modules(remove_duplicate=True))
        for n, m in modules.items():
            if isinstance(m, nn.Linear) and not isinstance(m, _QBaseLinear):
                parent_name, name = get_parent_name(n)
                new_layer = self.linear(m)
                setattr(modules[parent_name], name, new_layer)

        return model


class Qwen2_5_VLLLM4Compress(Qwen2VLLLM4Compress):
    """Convert Qwen2.5-VL LLM decoder layers to quantized modules.

    Qwen2.5-VL's decoder attention (Qwen2_5_VLAttention) is architecturally
    identical to Qwen2-VL's (M-RoPE, no QK-norm), so the quantized attention
    QQwen2VLAttention is reused directly. Only the matched original classes
    differ: Qwen2_5_VLAttention and the qwen2_5_vl-local Qwen2MLP.
    """
    def __init__(self, model, wbit=8, abit=8, state_dict=None,
                 quantize_bmm_input=False, bmm_qtype="smooth_quant",
                 bmm_bits=8, bmm_ebit=None, bmm_block_size=None,
                 bmm_q_bits=None, bmm_kv_bits=None,
                 bmm_kv_ebit=None, bmm_q_ebit=None,
                 bmm_q_block_size=None, bmm_kv_block_size=None,
                 bmm_kv_shared_exp_bits=None, bmm_kv_shared_exp_relative=False):
        super().__init__(model, wbit, abit, state_dict,
                         quantize_bmm_input, bmm_qtype, bmm_bits, bmm_ebit, bmm_block_size,
                         bmm_q_bits, bmm_kv_bits, bmm_kv_ebit, bmm_q_ebit,
                         bmm_q_block_size, bmm_kv_block_size,
                         bmm_kv_shared_exp_bits, bmm_kv_shared_exp_relative)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLAttention, Qwen2MLP as Qwen2_5_VLMlp,
        )
        self.attn_cls = Qwen2_5_VLAttention
        self.qattn_cls = QQwen2VLAttention   # reused — Qwen2.5-VL LLM attn == Qwen2-VL
        self.mlp_cls = Qwen2_5_VLMlp         # qwen2_5_vl-local Qwen2MLP (distinct class)
        # qmlp_cls inherited (QQwen2MLP)


class Qwen2_5_VL4Compress(VLM4Compress):
    """Composite converter for Qwen2.5-VL.

    Qwen2.5-VL structure (nested, like Qwen3-VL):
        - model.model.visual — ViT blocks (window attention) + merger
        - model.model.language_model — Qwen2.5 text decoder
    No separate projector (merger is part of visual).
    """
    def __init__(self, model: nn.Module, vision_config: dict, projector_config: dict, llm_config: dict):
        super().__init__(model, vision_config, projector_config, llm_config)

    def _get_vision_converter(self, vision_module):
        return Qwen2_5_VLVision4Compress(
            model=vision_module,
            wbit=self.vision_config.get("wbit", 8),
            abit=self.vision_config.get("abit", 8),
            quantize_bmm_input=self.vision_config.get("quantize_bmm_input", False),
            bmm_qtype=self.vision_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.vision_config.get("bmm_bits", 8),
            bmm_ebit=self.vision_config.get("bmm_ebit", None),
            bmm_block_size=self.vision_config.get("bmm_block_size", None),
            bmm_q_bits=self.vision_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.vision_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.vision_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.vision_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.vision_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.vision_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.vision_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.vision_config.get("bmm_kv_shared_exp_relative", False),
        )

    def _get_llm_converter(self, llm_module):
        return Qwen2_5_VLLLM4Compress(
            model=llm_module,
            wbit=self.llm_config.get("wbit", 8),
            abit=self.llm_config.get("abit", 8),
            quantize_bmm_input=self.llm_config.get("quantize_bmm_input", False),
            bmm_qtype=self.llm_config.get("bmm_qtype", "smooth_quant"),
            bmm_bits=self.llm_config.get("bmm_bits", 8),
            bmm_ebit=self.llm_config.get("bmm_ebit", None),
            bmm_block_size=self.llm_config.get("bmm_block_size", None),
            bmm_q_bits=self.llm_config.get("bmm_q_bits", None),
            bmm_kv_bits=self.llm_config.get("bmm_kv_bits", None),
            bmm_kv_ebit=self.llm_config.get("bmm_kv_ebit", None),
            bmm_q_ebit=self.llm_config.get("bmm_q_ebit", None),
            bmm_q_block_size=self.llm_config.get("bmm_q_block_size", None),
            bmm_kv_block_size=self.llm_config.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=self.llm_config.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=self.llm_config.get("bmm_kv_shared_exp_relative", False),
        )

    def convert(self):
        # 1. Convert vision encoder (includes merger as projector)
        visual = self.model.model.visual
        vision_converter = self._get_vision_converter(visual)
        self.model.model.visual = vision_converter.convert()

        # 2. Convert LLM backbone
        llm = self.model.model.language_model
        llm_converter = self._get_llm_converter(llm)
        self.model.model.language_model = llm_converter.convert()

        return self.model
