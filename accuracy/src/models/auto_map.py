"""
Load model architecture from different sources
"""

import torch
import timm
import torchvision

from transformers import AutoModelForCausalLM
from src.models.lm.retnet import RetNetForCausalLM

# Lazy imports for VLM models to avoid import errors when transformers doesn't have them
def _load_llava(model_name):
    from transformers import LlavaForConditionalGeneration
    return LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

def _load_qwen2_vl(model_name):
    from transformers import Qwen2VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

def _load_llava_onevision(model_name):
    from transformers import LlavaOnevisionForConditionalGeneration
    # Load to CPU first, then move to GPU after conversion.
    # device_map="auto" uses meta tensors which are incompatible with module replacement.
    return LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

def _load_qwen3_vl(model_name):
    from transformers import Qwen3VLForConditionalGeneration
    # Load to CPU first; the entry script moves the converted model to GPU.
    # (device_map="auto" offloads part of larger checkpoints (e.g. 32B) to CPU
    # and the later .to(device) then OOMs against the pre-filled GPU.)
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

def _load_qwen2_5_vl(model_name):
    from transformers import Qwen2_5_VLForConditionalGeneration
    # Load to CPU first; the entry script moves the converted model to GPU.
    # (device_map="auto" offloads part of larger checkpoints and the later
    # .to(device) then OOMs against the pre-filled GPU.)
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

def _load_minicpmv(model_name):
    from transformers import AutoModel
    # Load to CPU first, then move to GPU after conversion.
    # device_map="auto" uses meta tensors which are incompatible with module replacement.
    return AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

# TODO: expand this list to support more model architectures
MODEL_LIBRARY_MAP = {
    'vit_tiny_patch16_224': ('timm', 'vision_transformer'),
    'vit_small_patch16_224': ('timm', 'vision_transformer'),
    'vit_base_patch16_224': ('timm', 'vision_transformer'),
    'swin_tiny_patch4_window7_224': ('timm', 'swin_transformer'),
    'swin_small_patch4_window7_224': ('timm', 'swin_transformer'),
    'swin_base_patch4_window7_224': ('timm', 'swin_transformer'),
    'resnet18': ('torchvision', 'models'),
    'resnet34': ('torchvision', 'models'),
    'resnet50': ('torchvision', 'models'),
    'vgg16_bn': ('torchvision', 'models'),
    'Spiral-AI/Spiral-RetNet-3b-base': ('retnet', 'RetNetForCausalLM'),
    'meta-llama/Llama-2-7b-hf': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-1B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-3B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-3B': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B': ('transformers', 'AutoModelForCausalLM'),
    "Qwen/Qwen2.5-1.5B-Instruct": ('transformers', 'AutoModelForCausalLM'),
    'Qwen/Qwen2.5-1.5B': ('transformers', 'AutoModelForCausalLM'),
    'Qwen/Qwen2.5-7B': ('transformers', 'AutoModelForCausalLM'),
    'Qwen/Qwen2.5-14B': ('transformers', 'AutoModelForCausalLM'),
    'mistralai/Mistral-7B-v0.3': ('transformers', 'AutoModelForCausalLM'),
    # VLM models
    'llava-hf/llava-1.5-7b-hf': ('transformers_vlm', 'llava'),
    'llava-hf/llava-onevision-qwen2-7b-ov-hf': ('transformers_vlm', 'llava_onevision'),
    'NCSOFT/VARCO-VISION-14B-HF': ('transformers_vlm', 'llava_onevision'),
    'Qwen/Qwen2-VL-7B-Instruct': ('transformers_vlm', 'qwen2_vl'),
    'Qwen/Qwen2-VL-2B-Instruct': ('transformers_vlm', 'qwen2_vl'),
    'Qwen/Qwen3-VL-8B-Instruct': ('transformers_vlm', 'qwen3_vl'),
    'Qwen/Qwen3-VL-32B-Instruct': ('transformers_vlm', 'qwen3_vl'),
    'Qwen/Qwen2.5-VL-3B-Instruct': ('transformers_vlm', 'qwen2_5_vl'),
    'Qwen/Qwen2.5-VL-7B-Instruct': ('transformers_vlm', 'qwen2_5_vl'),
    'Qwen/Qwen2.5-VL-32B-Instruct': ('transformers_vlm', 'qwen2_5_vl'),
    'Qwen/Qwen2.5-VL-72B-Instruct': ('transformers_vlm', 'qwen2_5_vl'),
    'openbmb/MiniCPM-V-2_6': ('transformers_vlm', 'minicpm_v'),
}

TORCH_WEIGHTS_MAP = {
    'resnet18': 'ResNet18_Weights',
    'resnet34': 'ResNet34_Weights',
    'resnet50': 'ResNet50_Weights',
    'vgg16_bn': 'VGG16_BN_Weights',
}

class ModelMap:
    def __init__(self, model_name:str):
        self.model_name = model_name
        
    def fetch(self):
        if self.model_name not in MODEL_LIBRARY_MAP:
            raise ValueError(f"Model: {self.model_name} is unknown! Available models: {MODEL_LIBRARY_MAP.keys()}")

        lib_name, sub_name = MODEL_LIBRARY_MAP[self.model_name]

        if lib_name == "transformers_vlm":
            if sub_name == "llava":
                model = _load_llava(self.model_name)
            elif sub_name == "llava_onevision":
                model = _load_llava_onevision(self.model_name)
            elif sub_name == "qwen2_vl":
                model = _load_qwen2_vl(self.model_name)
            elif sub_name == "qwen3_vl":
                model = _load_qwen3_vl(self.model_name)
            elif sub_name == "qwen2_5_vl":
                model = _load_qwen2_5_vl(self.model_name)
            elif sub_name == "minicpm_v":
                model = _load_minicpmv(self.model_name)
            else:
                raise ValueError(f"Unknown VLM sub_name: {sub_name}")

        elif lib_name == "transformers":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        elif lib_name == "timm":
            model_lib = getattr(timm, "models")
            sub_lib = getattr(model_lib, sub_name)
            model_func = getattr(sub_lib, self.model_name)

            model = model_func(pretrained=True)

        elif lib_name == "torchvision":
            model_func = getattr(torchvision.models, self.model_name)
            model_weights = getattr(torchvision.models, TORCH_WEIGHTS_MAP[self.model_name])

            if hasattr(model_weights, "IMAGENET1K_V2"):
                model = model_func(weights=model_weights.IMAGENET1K_V2)
            else:
                model = model_func(weights=model_weights.IMAGENET1K_V1)
        
        elif lib_name == "t2c_models":
            if "RetNet" in self.model_name:
                model = RetNetForCausalLM.from_pretrained(
                    self.model_name
                )
        else:
            raise ValueError(f"Unknown model library {lib_name}")

        return model