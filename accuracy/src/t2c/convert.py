"""
Vanilla to low precision modules
"""
import torch
import copy
import torch.nn as nn

from typing import Tuple
from src.module.base import _QBaseLinear, _QBaseConv2d, _QBase
from src.module.fuse import MulQuant, MulShift
from src.module.attention import QLlamaAttention, QQWen2Attention, QMistralAttention
from src.module.mlp import QLlamaMLP, QQwen2MLP
from src.quantization.minmax import MinMaxQuantizer, MinMaxTokenWiseQuantizer, MinMaxChannelWiseWeightQuantizer, MinMaxChannelWiseActQuantizer
from src.quantization.observer import BaseObserver, BaseChannelWiseObserver, BaseTokenWiseObserver
from src.quantization.smoothquant import SmoothQuantizer, SmoothQuantChannelWiseWeightQuantizer, SmoothQuantTokenWiseQuantizer
from src.quantization.mxfp4 import (
    MXFP4ChannelWiseWeightQuantizer, MXFP4ActivationQuantizer,
    MXFP4CeilChannelWiseWeightQuantizer, MXFP4CeilActivationQuantizer,
    MXFP4PlusChannelWiseWeightQuantizer, MXFP4PlusActivationQuantizer,
    MXFP4PlusCeilChannelWiseWeightQuantizer, MXFP4PlusCeilActivationQuantizer,
    AMXFP4ChannelWiseWeightQuantizer, AMXFP4ActivationQuantizer,
)
from src.quantization.nvfp4 import NVFP4ChannelWiseWeightQuantizer, NVFP4ActivationQuantizer

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralMLP

from typing import Union, Dict

weight_quantizer = {
    "minmax": MinMaxQuantizer,
    "minmax_channel": MinMaxChannelWiseWeightQuantizer,
    "smooth": SmoothQuantizer,
    "smooth_channel": SmoothQuantChannelWiseWeightQuantizer,
    "mxfp4_quant": MXFP4ChannelWiseWeightQuantizer,
    "mxfp4_ceil_quant": MXFP4CeilChannelWiseWeightQuantizer,
    "mxfp4_plus_quant": MXFP4PlusChannelWiseWeightQuantizer,
    "mxfp4_plus_ceil_quant": MXFP4PlusCeilChannelWiseWeightQuantizer,
    "amxfp4_quant": AMXFP4ChannelWiseWeightQuantizer,
    "nvfp4_quant": NVFP4ChannelWiseWeightQuantizer,
    "identity": _QBase
}

input_quantizer = {
    "minmax": MinMaxQuantizer,
    "minmax_token": MinMaxTokenWiseQuantizer,
    "minmax_channel": MinMaxChannelWiseActQuantizer,
    "smooth": SmoothQuantizer,
    "smooth_token": SmoothQuantTokenWiseQuantizer,
    "mxfp4_quant": MXFP4ActivationQuantizer,
    "mxfp4_ceil_quant": MXFP4CeilActivationQuantizer,
    "mxfp4_plus_quant": MXFP4PlusActivationQuantizer,
    "mxfp4_plus_ceil_quant": MXFP4PlusCeilActivationQuantizer,
    "amxfp4_quant": AMXFP4ActivationQuantizer,
    "nvfp4_quant": NVFP4ActivationQuantizer,
    "identity": _QBase
}

def get_parent_name(target:str) -> Tuple[str, str]:
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]

class Vanilla4Compress(object):
    def __init__(self, model:nn.Module, wbit:int=8, abit:int=8, state_dict:Dict=None) -> None:
        self.model = model
        self.wbit = wbit
        self.abit = abit
        self.state_dict = state_dict

    def to_half(self, module:nn.Module):
        for param in module.parameters():
            param.data = param.data.to(torch.float16)

        return module

    def conv(self, layer:nn.Conv2d):
        has_bias = layer.bias is not None

        new_layer = _QBaseConv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            bias = has_bias,
            wbit=self.wbit,
            abit=self.abit
        )
        
        # copy the weights and bias to the new layer
        new_layer.weight.data[:] = layer.weight
        
        if has_bias:
            new_layer.bias.data[:] = layer.bias

        return new_layer

    def linear(self, layer:nn.Linear):
        has_bias = layer.bias is not None

        new_layer = _QBaseLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=has_bias,
            wbit=self.wbit,
            abit=self.abit
        )

        new_layer.weight.data[:] = layer.weight

        if has_bias:
            new_layer.bias.data[:] = layer.bias
        return new_layer

    def assign_quantizer(self, model, wqtype, xqtype):
        model = copy.deepcopy(model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, (_QBaseConv2d, _QBaseLinear)):
                if wqtype == "adaround":
                    m.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False, weights=m.weight)
                else:
                    m.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False)
                
                if isinstance(m, _QBaseConv2d):
                    if m.in_channels != 3:
                        m.aq = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=True)
                else:
                    m.aq = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=True)

                m = self.reshape_quantizer(m, n)

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, m)

            elif isinstance(m, (MulQuant, MulShift)):
                parent_name, name = get_parent_name(n)
                m = self.reshape_quantizer(m, n)
                setattr(modules[parent_name], name, m)

        return model
    
    def convert(self):
        model = copy.deepcopy(self.model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            parent_name, name = get_parent_name(n)

            if isinstance(m, nn.Conv2d):
                new_layer = self.conv(m)
                setattr(modules[parent_name], name, new_layer)
            
            elif isinstance(m, nn.Linear):
                new_layer = self.linear(m)
                setattr(modules[parent_name], name, new_layer)

        return model
    
    def reshape_quantizer(self, layer:Union[_QBaseLinear, _QBaseConv2d], layer_name:str):
        
        if isinstance(layer, _QBaseLinear):
            layer.wq.num_channels = layer.out_features
            layer.aq.num_channels = layer.in_features

            layer.wq.register_qparams()
            layer.aq.register_qparams()

            layer.wq.observer.num_channels = layer.out_features
            layer.aq.observer.num_channels = layer.in_features

            layer.wq.observer.register_range()
            layer.aq.observer.register_range()
        
        elif isinstance(layer, _QBaseConv2d):
            layer.wq.num_channels = layer.out_channels
            layer.aq.num_channels = layer.in_channels
            
            layer.wq.register_qparams()
            layer.aq.register_qparams()

            layer.wq.observer.num_channels = layer.out_channels
            layer.aq.observer.num_channels = layer.in_channels

            layer.wq.observer.register_range()
            layer.aq.observer.register_range()

            if isinstance(layer.wq.observer, BaseChannelWiseObserver):
                layer.wq.scale.unsqueeze_(2).unsqueeze_(3)
                layer.wq.zero_point.unsqueeze_(2).unsqueeze_(3)

            if isinstance(layer.aq.observer, BaseChannelWiseObserver):
                layer.aq.scale.unsqueeze_(2).unsqueeze_(3)
                layer.aq.zero_point.unsqueeze_(2).unsqueeze_(3)

        elif isinstance(layer, (MulQuant, MulShift)):
            layer.scale.data = torch.ones_like(self.state_dict[layer_name+".scale"])
            layer.bias.data = torch.ones_like(self.state_dict[layer_name+".bias"])

            if isinstance(layer, MulQuant):
                layer.zero_point.data = torch.ones_like(self.state_dict[layer_name+".zero_point"])
        
        return layer

    def reload_fake_quant(self, wqtype, xqtype):
        qmodel = self.convert()
        qmodel = self.assign_quantizer(qmodel, wqtype=wqtype, xqtype=xqtype)
        return qmodel


class Llama4Compress(Vanilla4Compress):
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8, state_dict: Dict = None,
                 quantize_bmm_input: bool = False, bmm_qtype: str = "smooth_quant",
                 bmm_bits: int = 8, bmm_ebit: int = None, bmm_block_size: int = None,
                 bmm_q_bits: int = None, bmm_kv_bits: int = None, bmm_kv_ebit: int = None,
                 bmm_q_ebit: int = None,
                 bmm_q_block_size: int = None, bmm_kv_block_size: int = None,
                 bmm_kv_shared_exp_bits: int = None, bmm_kv_shared_exp_relative: bool = False) -> None:
        super().__init__(model, wbit, abit, state_dict)
        self.attn_cls = LlamaAttention
        self.mlp_cls = LlamaMLP

        self.qattn_cls = QLlamaAttention
        self.qmlp_cls = QLlamaMLP

        # Whether to quantize BMM inputs (Q, K, V) for attention computation
        self.quantize_bmm_input = quantize_bmm_input
        # BMM quantization type: "smooth_quant", "mxint", "inverted_mx", "mixed_mxint_invmx", or one of the FP4 mixed modes
        self.bmm_qtype = bmm_qtype
        # Bit precision for BMM quantization
        self.bmm_bits = bmm_bits
        # Exponent bits for inverted_mx (None for unlimited)
        self.bmm_ebit = bmm_ebit
        # Block size for MX-style quantization
        self.bmm_block_size = bmm_block_size
        # Mixed quantization parameters
        self.bmm_q_bits = bmm_q_bits
        self.bmm_kv_bits = bmm_kv_bits
        self.bmm_kv_ebit = bmm_kv_ebit
        self.bmm_q_ebit = bmm_q_ebit
        self.bmm_q_block_size = bmm_q_block_size
        self.bmm_kv_block_size = bmm_kv_block_size
        self.bmm_kv_shared_exp_bits = bmm_kv_shared_exp_bits
        self.bmm_kv_shared_exp_relative = bmm_kv_shared_exp_relative
    
    def attn(self, attn):
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
            bmm_kv_shared_exp_relative=self.bmm_kv_shared_exp_relative
        )
        new_attn.load_state_dict(attn.state_dict(), strict=False)

        new_attn = self.to_half(new_attn)
        return new_attn

    def mlp(self, mlp):
        new_module = self.qmlp_cls(config=mlp.config)
        new_module.load_state_dict(mlp.state_dict(), strict=False)

        new_module = self.to_half(new_module)
        return new_module

    def convert(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        # avoid mem explosion
        self.model = self.model.to(torch.device("cpu"))

        for n, m in modules.items():
            if isinstance(m, self.attn_cls):
                parent_name, name = get_parent_name(n)
                new_module = self.attn(m)
                setattr(modules[parent_name], name, new_module)

            elif isinstance(m, self.mlp_cls):
                parent_name, name = get_parent_name(n)
                new_module = self.mlp(m)
                setattr(modules[parent_name], name, new_module)

        return self.model

    def reload(self, wqtype, xqtype):
        qmodel = self.convert()
        qmodel = self.assign_quantizer(qmodel, wqtype=wqtype, xqtype=xqtype)
        return qmodel
    
class QWen4Compress(Llama4Compress):
    def __init__(self, model, wbit=8, abit=8, state_dict=None,
                 quantize_bmm_input=False, bmm_qtype="smooth_quant",
                 bmm_bits=8, bmm_ebit=None, bmm_block_size=None,
                 bmm_q_bits=None, bmm_kv_bits=None,
                 bmm_kv_ebit=None, bmm_q_ebit=None,
                 bmm_q_block_size=None, bmm_kv_block_size=None,
                 k_smooth_static=False,
                 bmm_kv_shared_exp_bits=None, bmm_kv_shared_exp_relative=False):
        super().__init__(model, wbit, abit, state_dict)
        self.attn_cls = Qwen2Attention
        self.mlp_cls = Qwen2MLP

        self.qattn_cls = QQWen2Attention
        self.qmlp_cls = QQwen2MLP

        self.k_smooth_static = k_smooth_static
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

    def attn(self, attn):
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
            k_smooth_static=self.k_smooth_static,
            bmm_kv_shared_exp_bits=self.bmm_kv_shared_exp_bits,
            bmm_kv_shared_exp_relative=self.bmm_kv_shared_exp_relative,
        )
        new_attn.load_state_dict(attn.state_dict(), strict=False)
        new_attn = self.to_half(new_attn)
        return new_attn


class Mistral4Compress(QWen4Compress):
    """Mistral converter: same llama-like structure as Qwen2, but the
    attention projections carry no bias (QMistralAttention) and the MLP
    reuses the generic gate/up/down QQwen2MLP."""
    def __init__(self, model, wbit=8, abit=8, state_dict=None,
                 quantize_bmm_input=False, bmm_qtype="smooth_quant",
                 bmm_bits=8, bmm_ebit=None, bmm_block_size=None,
                 bmm_q_bits=None, bmm_kv_bits=None,
                 bmm_kv_ebit=None, bmm_q_ebit=None,
                 bmm_q_block_size=None, bmm_kv_block_size=None,
                 bmm_kv_shared_exp_bits=None, bmm_kv_shared_exp_relative=False):
        super().__init__(model, wbit, abit, state_dict,
                         quantize_bmm_input, bmm_qtype,
                         bmm_bits, bmm_ebit, bmm_block_size,
                         bmm_q_bits, bmm_kv_bits,
                         bmm_kv_ebit, bmm_q_ebit,
                         bmm_q_block_size, bmm_kv_block_size,
                         bmm_kv_shared_exp_bits=bmm_kv_shared_exp_bits,
                         bmm_kv_shared_exp_relative=bmm_kv_shared_exp_relative)
        self.attn_cls = MistralAttention
        self.mlp_cls = MistralMLP

        self.qattn_cls = QMistralAttention
        self.qmlp_cls = QQwen2MLP


CONVERTNN = {
    "meta-llama/Llama-2-7b-hf": Llama4Compress,
    "meta-llama/Llama-3.2-1B-Instruct": Llama4Compress,
    "meta-llama/Llama-3.2-3B-Instruct": Llama4Compress,
    "meta-llama/Llama-3.2-3B": Llama4Compress,
    "meta-llama/Llama-3.1-8B-Instruct": Llama4Compress,
    "meta-llama/Llama-3.1-8B": Llama4Compress,
    "Qwen/Qwen2.5-1.5B": QWen4Compress,
    "Qwen/Qwen2.5-1.5B-Instruct": QWen4Compress,
    "Qwen/Qwen2.5-7B": QWen4Compress,
    "Qwen/Qwen2.5-14B": QWen4Compress,
    "mistralai/Mistral-7B-v0.3": Mistral4Compress
}