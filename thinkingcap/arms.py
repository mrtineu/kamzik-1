"""The non-LoRA arms: reasoning MLPs on the residual stream, or one extra decoder block.

Both start as an exact identity, so step 0 is the base model:
  - a ReasoningMLP adds up(silu(down(h))) to the output of a decoder layer, with `up` zero-initialized;
  - the extra block is a copy of the layer before it with o_proj and mlp.down_proj zeroed, so both of
    its residual branches add exactly 0.

A block arm with "train": "mlp" keeps the attention branch at exactly 0 (frozen) and trains only the
block's MLP and its norm: a reasoning MLP that is still a plain Qwen3 layer. "mlp_width": N keeps only
the first N of its intermediate channels alive (the rest stay zero and get no gradient), a bottleneck
close to the ReasoningMLP's without leaving the standard architecture.

Adapters live at model.layers.{i}.reasoning_mlp, the same names the vLLM plugin (vllm_plugin/) loads.
The extra block exports as a plain 29-layer Qwen3 checkpoint; the MLP arms export with
architectures=["Qwen3ReasoningMLPForCausalLM"], which only loads with the plugin installed.
"""
import copy
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

MLP_ARCHITECTURE = 'Qwen3ReasoningMLPForCausalLM'
TOKENIZER_FILES = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']


class ReasoningMLP(nn.Module):
    """Residual bottleneck MLP; fp32 parameters and compute, model-dtype interface."""

    def __init__(self, hidden, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck, bias=False, dtype=torch.float32)
        self.up = nn.Linear(bottleneck, hidden, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        # fp32 even under bf16 autocast, so training computes what vLLM serves
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            delta = self.up(F.silu(self.down(hidden.float())))
        return hidden + delta.to(hidden.dtype)


def _apply_reasoning_mlp(layer, args, output):
    if isinstance(output, tuple):
        return (layer.reasoning_mlp(output[0]),) + output[1:]
    return layer.reasoning_mlp(output)


def attach_mlps(model, layers, bottleneck):
    """Freeze the model and put a ReasoningMLP on the output of each listed decoder layer."""
    model.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    for index in layers:
        layer = model.model.layers[index]
        layer.reasoning_mlp = ReasoningMLP(model.config.hidden_size, bottleneck).to(device)
        layer.register_forward_hook(_apply_reasoning_mlp)
    model.config.reasoning_mlp_layers = list(layers)
    model.config.reasoning_mlp_bottleneck = bottleneck
    return [p for index in layers for p in model.model.layers[index].reasoning_mlp.parameters()]


def _keep_channels(width):
    """Gradient hooks that zero everything outside the first `width` intermediate channels."""
    def rows(grad):
        grad = grad.clone()
        grad[width:] = 0
        return grad

    def columns(grad):
        grad = grad.clone()
        grad[:, width:] = 0
        return grad
    return rows, columns


def insert_block(model, after, train='block', mlp_width=None):
    """Freeze the model and insert an identity copy of layer `after` right behind it.

    train='block' trains the whole block; train='mlp' only its MLP and post-attention norm, with the
    attention branch frozen at 0. mlp_width (train='mlp' only) narrows the MLP to its first channels.
    """
    if train not in ('block', 'mlp') or (mlp_width and train != 'mlp'):
        raise ValueError(f'bad block arm: train={train!r}, mlp_width={mlp_width!r}')
    model.requires_grad_(False)
    layers = model.model.layers
    block = copy.deepcopy(layers[after])
    nn.init.zeros_(block.self_attn.o_proj.weight)
    nn.init.zeros_(block.mlp.down_proj.weight)
    layers.insert(after + 1, block)
    for index, layer in enumerate(layers):      # KV-cache slots follow the new numbering
        layer.self_attn.layer_idx = index
    config = model.config
    config.layer_types = config.layer_types[:after + 1] + [config.layer_types[after]] + config.layer_types[after + 1:]
    config.num_hidden_layers = len(layers)
    config.max_window_layers = len(layers)
    if train == 'block':
        block.requires_grad_(True)
        return list(block.parameters())
    mlp = block.mlp
    trainable = [*mlp.parameters(), *block.post_attention_layernorm.parameters()]
    for p in trainable:
        p.requires_grad_(True)
    if mlp_width:
        with torch.no_grad():
            mlp.gate_proj.weight[mlp_width:] = 0
            mlp.up_proj.weight[mlp_width:] = 0
        rows, columns = _keep_channels(mlp_width)
        mlp.gate_proj.weight.register_hook(rows)
        mlp.up_proj.weight.register_hook(rows)
        mlp.down_proj.weight.register_hook(columns)
    return trainable


def build(arm, model):
    """Apply an arm spec from arms.json in place; returns the trainable parameters."""
    if arm['kind'] == 'mlp':
        return attach_mlps(model, arm['layers'], arm['bottleneck'])
    if arm['kind'] == 'block':
        return insert_block(model, arm['after'], arm.get('train', 'block'), arm.get('mlp_width'))
    raise ValueError(f"unknown arm kind {arm['kind']!r}")


def trainable_state(arm, model):
    """The weights an arm adds or trains, under their full model names."""
    if arm['kind'] == 'mlp':
        prefixes = tuple(f'model.layers.{i}.reasoning_mlp.' for i in arm['layers'])
    else:
        prefixes = (f"model.layers.{arm['after'] + 1}.",)
    return {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
            if name.startswith(prefixes)}


def load_trainable_state(model, state):
    missing = set(state) - set(model.state_dict())
    if missing:
        raise KeyError(f'unexpected adapter weights: {sorted(missing)[:5]}')
    model.load_state_dict(state, strict=False)


def copy_tokenizer(source_dir, out):
    for name in TOKENIZER_FILES:
        if (Path(source_dir) / name).exists():
            shutil.copyfile(Path(source_dir) / name, Path(out) / name)


def export(arm, model, base_dir, out):
    """A checkpoint evaluate.py takes via --model: weights, config and the base model's tokenizer files.

    base_dir is the base model's snapshot directory (config, model.safetensors, tokenizer files).
    """
    from safetensors.torch import load_file, save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base_dir = Path(base_dir)
    config = json.loads((base_dir / 'config.json').read_text())
    if arm['kind'] == 'mlp':
        # base weights untouched (bf16) plus the fp32 adapters, in one file
        weights = load_file(str(base_dir / 'model.safetensors'))
        weights.update(trainable_state(arm, model))
        save_file(weights, str(out / 'model.safetensors'), metadata={'format': 'pt'})
        config.update(architectures=[MLP_ARCHITECTURE], reasoning_mlp_layers=arm['layers'],
                      reasoning_mlp_bottleneck=arm['bottleneck'])
    else:
        dtype = getattr(torch, config.get('torch_dtype') or config['dtype'])
        weights = {name: tensor.detach().to('cpu', dtype).contiguous() for name, tensor in model.state_dict().items()
                   if name != 'lm_head.weight' or not config.get('tie_word_embeddings')}
        save_file(weights, str(out / 'model.safetensors'), metadata={'format': 'pt'})
        config.update(num_hidden_layers=model.config.num_hidden_layers,
                      max_window_layers=model.config.num_hidden_layers,
                      layer_types=list(model.config.layer_types))
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    copy_tokenizer(base_dir, out)
    return out


def load_export(path, dtype=torch.bfloat16, device='cpu'):
    """An exported arm as a Transformers model (MLP arms re-attach their adapters from the file)."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation='sdpa').to(device)
    if config['architectures'] == [MLP_ARCHITECTURE]:
        arm = {'kind': 'mlp', 'layers': config['reasoning_mlp_layers'], 'bottleneck': config['reasoning_mlp_bottleneck']}
        attach_mlps(model, arm['layers'], arm['bottleneck'])
        weights = load_file(str(path / 'model.safetensors'))
        load_trainable_state(model, {k: v for k, v in weights.items() if '.reasoning_mlp.' in k})
    model.eval()
    return model


def perturb(arm, model, seed):
    """Non-identity weights of a realistic size, to prove an export path actually applies the arm."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if arm['kind'] == 'mlp':
            for index in arm['layers']:
                mlp = model.model.layers[index].reasoning_mlp
                for weight, scale in ((mlp.down.weight, 1.0), (mlp.up.weight, 0.1)):
                    weight.copy_(torch.randn(weight.shape, generator=generator) * scale * weight.shape[1] ** -0.5)
        else:
            source = model.model.layers[arm['after']]
            block = model.model.layers[arm['after'] + 1]
            if arm.get('train', 'block') == 'block':
                block.self_attn.o_proj.weight.copy_(source.self_attn.o_proj.weight * 0.5)
            width = arm.get('mlp_width') or source.mlp.down_proj.weight.shape[1]
            # the narrow arm only has its first channels alive, so scale them up to a comparable effect
            scale = 0.5 * (source.mlp.down_proj.weight.shape[1] / width) ** 0.5
            block.mlp.down_proj.weight[:, :width].copy_(source.mlp.down_proj.weight[:, :width] * scale)
