"""vLLM's Qwen3 with ReasoningMLPs on the outputs of the decoder layers in config.reasoning_mlp_layers.

vLLM keeps the residual stream split as (hidden_states, residual) and fuses their sum into the next
norm, so an adapted layer returns the summed stream with the MLP applied and a zero residual -- the
same function as the Transformers forward hook in arms.py.
"""
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer


class ReasoningMLP(nn.Module):
    def __init__(self, hidden, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck, bias=False, dtype=torch.float32)
        self.up = nn.Linear(bottleneck, hidden, bias=False, dtype=torch.float32)

    def forward(self, hidden):
        return hidden + self.up(F.silu(self.down(hidden.float()))).to(hidden.dtype)


class ReasoningQwen3DecoderLayer(Qwen3DecoderLayer):
    def forward(self, positions, hidden_states, residual):
        hidden_states, residual = super().forward(positions, hidden_states, residual)
        return self.reasoning_mlp(hidden_states + residual), torch.zeros_like(residual)


class Qwen3ReasoningMLPForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        for index in config.reasoning_mlp_layers:
            layer = self.model.layers[index]
            if isinstance(layer, PPMissingLayer):
                continue
            layer.__class__ = ReasoningQwen3DecoderLayer
            # weights come from the checkpoint; vLLM fails the load if any are missing
            layer.reasoning_mlp = ReasoningMLP(config.hidden_size, config.reasoning_mlp_bottleneck)
