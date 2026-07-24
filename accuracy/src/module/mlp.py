import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.base import _QBaseLinear
from transformers.activations import ACT2FN

class QLlamaMLP(nn.Module):
    def __init__(self, config, rescale_out:bool=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, rescale_out=rescale_out)
        self.up_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, rescale_out=rescale_out)
        self.down_proj = _QBaseLinear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, rescale_out=rescale_out)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class QQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = _QBaseLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


