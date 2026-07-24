"""
VLM Post-Training Quantization.

Composite PTQ with per-component quantizer injection:
- Vision encoder: direct quantizer injection on _QBaseLinear modules
- Projector: direct quantizer injection on _QBaseLinear modules
- LLM backbone: SmoothQuant-style activation profiling + quantizer injection

Calibration runs the full model (image+text) so each component sees realistic distributions.
"""
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from src.module.base import _QBaseLinear, _QBase
from src.module.attention import QLlamaAttention
from src.module.mlp import QLlamaMLP
from src.t2c.convert import get_parent_name
from src.trainer.llm.ptq import (
    WEIGHT_QUANTIZER_MAP, INPUT_QUANTIZER_MAP, MinMaxHook
)


class ComponentQuantizerMixin:
    """Helper methods to create quantizers from a component config dict."""

    @staticmethod
    def get_weight_quantizer(config):
        wqtype = config.get("wqtype", "identity")
        wbit = config.get("wbit", 8)

        if wqtype == "inverted_mxint_quant":
            block_size = config.get("w_block_size", config.get("block_size", 32))
            ebit = config.get("w_ebit", config.get("ebit", None))
            mantissa_group_size = config.get("w_mantissa_group_size", config.get("mantissa_group_size", 8))
            sub_ebit = config.get("w_sub_ebit", config.get("sub_ebit", None))
            return WEIGHT_QUANTIZER_MAP[wqtype](nbit=wbit, train_flag=True, unsigned=False, block_size=block_size, ebit=ebit, mantissa_group_size=mantissa_group_size, sub_ebit=sub_ebit)
        elif wqtype == "mxint_quant":
            block_size = config.get("w_block_size", config.get("block_size", 32))
            shared_exp_bits = config.get("shared_exp_bits", 5)
            shared_exp_relative = config.get("shared_exp_relative", False)
            return WEIGHT_QUANTIZER_MAP[wqtype](nbit=wbit, train_flag=True, unsigned=False, block_size=block_size, shared_exp_bits=shared_exp_bits, shared_exp_relative=shared_exp_relative)
        elif wqtype in ("mxfp4_quant", "mxfp4_ceil_quant", "mxfp4_plus_quant", "mxfp4_plus_ceil_quant", "amxfp4_quant"):
            block_size = config.get("w_block_size", config.get("block_size", 32))
            return WEIGHT_QUANTIZER_MAP[wqtype](nbit=4, train_flag=True, unsigned=False, block_size=block_size)
        elif wqtype == "nvfp4_quant":
            block_size = config.get("w_block_size", config.get("block_size", 16))
            return WEIGHT_QUANTIZER_MAP[wqtype](nbit=4, train_flag=True, unsigned=False, block_size=block_size)
        elif wqtype == "identity":
            return WEIGHT_QUANTIZER_MAP[wqtype](nbit=wbit, train_flag=True, unsigned=False)
        else:
            return WEIGHT_QUANTIZER_MAP[wqtype](nbit=wbit, train_flag=True, unsigned=False)

    @staticmethod
    def get_input_quantizer(config):
        xqtype = config.get("xqtype", "identity")
        abit = config.get("abit", 8)

        if xqtype == "inverted_mxint_quant":
            block_size = config.get("a_block_size", config.get("block_size", 32))
            ebit = config.get("a_ebit", config.get("ebit", None))
            mantissa_group_size = config.get("a_mantissa_group_size", config.get("mantissa_group_size", 8))
            sub_ebit = config.get("a_sub_ebit", config.get("sub_ebit", None))
            return INPUT_QUANTIZER_MAP[xqtype](nbit=abit, train_flag=True, unsigned=False, block_size=block_size, ebit=ebit, mantissa_group_size=mantissa_group_size, sub_ebit=sub_ebit)
        elif xqtype == "mxint_quant":
            block_size = config.get("a_block_size", config.get("block_size", 32))
            shared_exp_bits = config.get("shared_exp_bits", 5)
            shared_exp_relative = config.get("shared_exp_relative", False)
            return INPUT_QUANTIZER_MAP[xqtype](nbit=abit, train_flag=True, unsigned=False, block_size=block_size, shared_exp_bits=shared_exp_bits, shared_exp_relative=shared_exp_relative)
        elif xqtype in ("mxfp4_quant", "mxfp4_ceil_quant", "mxfp4_plus_quant", "mxfp4_plus_ceil_quant", "amxfp4_quant"):
            block_size = config.get("a_block_size", config.get("block_size", 32))
            return INPUT_QUANTIZER_MAP[xqtype](nbit=4, train_flag=True, unsigned=False, block_size=block_size)
        elif xqtype == "nvfp4_quant":
            block_size = config.get("a_block_size", config.get("block_size", 16))
            return INPUT_QUANTIZER_MAP[xqtype](nbit=4, train_flag=True, unsigned=False, block_size=block_size)
        elif xqtype == "identity":
            return INPUT_QUANTIZER_MAP[xqtype](nbit=abit, train_flag=True, unsigned=False)
        else:
            return INPUT_QUANTIZER_MAP[xqtype](nbit=abit, train_flag=True, unsigned=False)


class VLMPTQ:
    """Composite VLM PTQ calibrator.

    Runs calibration on the full model with multimodal inputs,
    then injects quantizers per-component with separate configs.

    Args:
        model: the converted VLM model (with _QBaseLinear modules)
        calib_loader: DataLoader yielding processed VLM inputs
        vision_config: quantization config dict for vision encoder
        projector_config: quantization config dict for projector
        llm_config: quantization config dict for LLM backbone
        logger: logging.Logger instance
        run_dir: directory to save artifacts
        smooth_alpha: SmoothQuant alpha for LLM backbone
    """
    def __init__(self, model, calib_loader, vision_config, projector_config, llm_config,
                 logger, run_dir, smooth_alpha=0.85, smooth_flag=True):
        self.model = model
        self.calib_loader = calib_loader
        self.vision_config = vision_config
        self.projector_config = projector_config
        self.llm_config = llm_config
        self.logger = logger
        self.run_dir = run_dir
        self.smooth_alpha = smooth_alpha
        self.smooth_flag = smooth_flag
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _inject_direct_quantizers(self, module, config, component_name):
        """Inject weight and activation quantizers on all _QBaseLinear in a module."""
        count = 0
        for n, m in module.named_modules():
            if isinstance(m, _QBaseLinear):
                wq = ComponentQuantizerMixin.get_weight_quantizer(config).to(torch.float16).to(self.device)
                aq = ComponentQuantizerMixin.get_input_quantizer(config).to(torch.float16).to(self.device)
                setattr(m, "wq", wq)
                setattr(m, "aq", aq)
                count += 1
        self.logger.info(f"[{component_name}] Injected quantizers on {count} _QBaseLinear layers")

    @torch.no_grad()
    def _calibration_forward(self):
        """Run calibration forward passes through the full model."""
        import gc
        self.model.eval()
        for idx, batch in enumerate(tqdm(self.calib_loader, desc="VLM Calibration")):
            # Move inputs to device (recursive for nested structures like MiniCPM-V)
            def _move(x):
                if isinstance(x, torch.Tensor):
                    return x.to(self.device)
                elif isinstance(x, list):
                    return [_move(v) for v in x]
                return x
            batch = {k: _move(v) for k, v in batch.items()}
            self.model(**batch)
            del batch
        gc.collect()
        torch.cuda.empty_cache()

    def _profile_llm_activations(self, llm_module):
        """Profile activation scales on LLM decoder layers using hooks."""
        hooks = {}
        handles = {}

        for n, m in llm_module.named_modules():
            if isinstance(m, _QBaseLinear):
                hook = MinMaxHook()
                handle = m.register_forward_hook(hook)
                hooks[n] = hook
                handles[n] = handle

        # Run calibration forward through full model
        self._calibration_forward()

        # Collect activation scales
        act_scales = OrderedDict()
        for n, hook in hooks.items():
            if "down_proj" not in n:
                xmax = hook.max
                if self.smooth_flag:
                    act_scales[n] = xmax
                else:
                    act_scales[n] = torch.ones_like(xmax)
            handles[n].remove()

        del handles, hooks

        # Save scales
        smooth_path = os.path.join(self.run_dir, "vlm_llm_act_scales.pt")
        torch.save(act_scales, smooth_path)
        self.logger.info(f"[LLM] Saved activation scales to {smooth_path}")

        return act_scales

    @torch.no_grad()
    def _smooth_fcs(self, fcs, act_scales, config):
        """Apply SmoothQuant smoothing and inject quantizers on linear layers."""
        if not isinstance(fcs, list):
            fcs = [fcs]

        for fc in fcs:
            assert isinstance(fc, _QBaseLinear)
            fc.wq = ComponentQuantizerMixin.get_weight_quantizer(config).to(torch.float16)
            fc.aq = ComponentQuantizerMixin.get_input_quantizer(config).to(torch.float16)

        device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)

        weight_scales = torch.cat(
            [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
        )
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        scales = (
            (act_scales.pow(self.smooth_alpha) / weight_scales.pow(1 - self.smooth_alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

        return scales

    @torch.no_grad()
    def _inject_llm_quantizers(self, llm_module, act_scales):
        """Inject quantizers into LLM decoder layers with SmoothQuant smoothing."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        decoder_layer_types = (LlamaDecoderLayer, Qwen2DecoderLayer)
        # Try importing Qwen2VLDecoderLayer for Qwen2-VL support
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
            decoder_layer_types = decoder_layer_types + (Qwen2VLDecoderLayer,)
        except ImportError:
            pass
        # Try importing Qwen3VLTextDecoderLayer for Qwen3-VL support
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
            decoder_layer_types = decoder_layer_types + (Qwen3VLTextDecoderLayer,)
        except ImportError:
            pass
        # Try importing Qwen2_5_VLDecoderLayer for Qwen2.5-VL support
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
            decoder_layer_types = decoder_layer_types + (Qwen2_5_VLDecoderLayer,)
        except ImportError:
            pass

        config = self.llm_config
        count = 0
        for n, m in llm_module.named_modules():
            if isinstance(m, decoder_layer_types):
                # Attention: smooth q/k/v, direct inject o_proj
                attn = m.self_attn
                qkv = [attn.q_proj, attn.k_proj, attn.v_proj]
                qkv_scale_key = n + ".self_attn.q_proj"
                if qkv_scale_key in act_scales:
                    qkv_smooth = self._smooth_fcs(qkv, act_scales[qkv_scale_key], config)
                    m.input_layernorm.weight.div_(qkv_smooth)
                else:
                    # Fallback: direct inject without smoothing
                    for fc in qkv:
                        fc.wq = ComponentQuantizerMixin.get_weight_quantizer(config).to(self.device)
                        fc.aq = ComponentQuantizerMixin.get_input_quantizer(config).to(self.device)

                attn.o_proj.wq = ComponentQuantizerMixin.get_weight_quantizer(config).to(self.device)
                attn.o_proj.aq = ComponentQuantizerMixin.get_input_quantizer(config).to(self.device)

                # MLP: smooth gate/up, direct inject down
                mlp = m.mlp
                fcs = [mlp.gate_proj, mlp.up_proj]
                mlp_scale_key = n + ".mlp.gate_proj"
                if mlp_scale_key in act_scales:
                    mlp_smooth = self._smooth_fcs(fcs, act_scales[mlp_scale_key], config)
                    m.post_attention_layernorm.weight.div_(mlp_smooth)
                else:
                    for fc in fcs:
                        fc.wq = ComponentQuantizerMixin.get_weight_quantizer(config).to(self.device)
                        fc.aq = ComponentQuantizerMixin.get_input_quantizer(config).to(self.device)

                mlp.down_proj.wq = ComponentQuantizerMixin.get_weight_quantizer(config).to(self.device)
                mlp.down_proj.aq = ComponentQuantizerMixin.get_input_quantizer(config).to(self.device)
                count += 1

        self.logger.info(f"[LLM] Injected quantizers on {count} decoder layers")

    def run(self, vision_module, projector_module, llm_module):
        """Run full VLM PTQ pipeline.

        Args:
            vision_module: the vision encoder submodule
            projector_module: the projector submodule
            llm_module: the LLM backbone submodule

        Returns:
            the quantized model
        """
        self.logger.info("VLM PTQ Start!")
        self.model.eval()

        # Step 1: Profile LLM activation scales via full-model calibration
        self.logger.info("Step 1: Profiling LLM activation scales...")
        act_scales = self._profile_llm_activations(llm_module)

        # Step 2: Inject quantizers per component
        self.logger.info("Step 2: Injecting vision encoder quantizers...")
        self._inject_direct_quantizers(vision_module, self.vision_config, "Vision")

        if projector_module is not None:
            self.logger.info("Step 3: Injecting projector quantizers...")
            self._inject_direct_quantizers(projector_module, self.projector_config, "Projector")
        else:
            self.logger.info("Step 3: No separate projector module (skipping)")

        self.logger.info("Step 4: Injecting LLM quantizers with SmoothQuant...")
        self._inject_llm_quantizers(llm_module, act_scales)

        self.logger.info("VLM PTQ Complete!")
        return self.model
