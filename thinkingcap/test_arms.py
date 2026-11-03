"""CPU tests for arms.py on a tiny random 28-layer Qwen3 (needs torch + transformers, no GPU, no download).

    python -m unittest thinkingcap/test_arms.py

TC_ARMS_MODEL=/path/to/Qwen3-0.6B-snapshot runs the same checks on the real model as well.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arms  # noqa: E402

ARMS = json.loads((Path(__file__).resolve().parent / 'arms.json').read_text())['arms']


def tiny_base(directory):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=28,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
                         max_position_embeddings=256, dtype='bfloat16')
    Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(directory)
    return directory


class Arms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base_dir = os.environ.get('TC_ARMS_MODEL') or tiny_base(Path(cls.tmp.name) / 'base')
        vocab = json.loads((Path(cls.base_dir) / 'config.json').read_text())['vocab_size']
        generator = torch.Generator().manual_seed(1)
        cls.ids = torch.randint(0, min(vocab, 5000), (2, 48), generator=generator)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(self.base_dir, dtype=torch.bfloat16).eval()

    def logits(self, model):
        with torch.no_grad():
            return model(input_ids=self.ids).logits

    def test_step0_is_base_model(self):
        reference = self.logits(self.load())
        for name, spec in ARMS.items():
            with self.subTest(arm=name):
                model = self.load()
                trainable = arms.build(spec, model)
                self.assertTrue(torch.equal(self.logits(model), reference))
                self.assertEqual({p.requires_grad for p in model.parameters() if p.requires_grad}, {True})
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 sum(p.numel() for p in trainable))

    def test_trainable_counts_and_numbering(self):
        model = self.load()
        hidden = model.config.hidden_size
        self.assertEqual(sum(p.numel() for p in arms.build(ARMS['mlp-quarters'], model)), 4 * 2 * hidden * 64)
        model = self.load()
        layer = sum(p.numel() for p in model.model.layers[13].parameters())
        self.assertEqual(sum(p.numel() for p in arms.build(ARMS['extra-block'], model)), layer)
        self.assertEqual(model.config.num_hidden_layers, 29)
        self.assertEqual(len(model.config.layer_types), 29)
        self.assertEqual([l.self_attn.layer_idx for l in model.model.layers], list(range(29)))

    def test_perturbed_export_roundtrip(self):
        reference = self.logits(self.load())
        for name, spec in ARMS.items():
            with self.subTest(arm=name):
                model = self.load()
                arms.build(spec, model)
                arms.perturb(spec, model, 0)
                expected = self.logits(model)
                self.assertFalse(torch.equal(expected, reference))
                with tempfile.TemporaryDirectory() as out:
                    arms.export(spec, model, self.base_dir, out)
                    config = json.loads((Path(out) / 'config.json').read_text())
                    if spec['kind'] == 'block':
                        self.assertEqual(config['architectures'], ['Qwen3ForCausalLM'])
                        self.assertEqual(config['num_hidden_layers'], 29)
                    else:
                        self.assertEqual(config['architectures'], [arms.MLP_ARCHITECTURE])
                    self.assertTrue(torch.equal(self.logits(arms.load_export(out)), expected))

    def test_trainable_state_restores(self):
        for name, spec in ARMS.items():
            with self.subTest(arm=name):
                model = self.load()
                arms.build(spec, model)
                arms.perturb(spec, model, 0)
                state = arms.trainable_state(spec, model)
                expected = self.logits(model)
                fresh = self.load()
                arms.build(spec, fresh)
                arms.load_trainable_state(fresh, state)
                self.assertTrue(torch.equal(self.logits(fresh), expected))

    def test_mlp_only_block_trains_only_its_mlp(self):
        for name in ('extra-mlp', 'extra-mlp-64'):
            with self.subTest(arm=name):
                spec = ARMS[name]
                model = self.load()
                trainable = arms.build(spec, model)
                hidden, inter = model.config.hidden_size, model.config.intermediate_size
                self.assertEqual(sum(p.numel() for p in trainable), 3 * hidden * inter + hidden)
                block = model.model.layers[spec['after'] + 1]
                self.assertFalse(any(p.requires_grad for p in block.self_attn.parameters()))
                for p in trainable:
                    p.data = p.data.float()
                optimizer = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.0)
                model.train()
                for _ in range(3):
                    with torch.autocast('cpu', dtype=torch.bfloat16):   # as pipeline.fit trains
                        loss = model(input_ids=self.ids, labels=self.ids).loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                self.assertTrue(torch.equal(block.self_attn.o_proj.weight, torch.zeros_like(block.self_attn.o_proj.weight)))
                self.assertGreater(float(block.mlp.down_proj.weight.abs().sum()), 0)
                width = spec.get('mlp_width')
                if width:
                    mlp = block.mlp
                    for dead in (mlp.gate_proj.weight[width:], mlp.up_proj.weight[width:], mlp.down_proj.weight[:, width:]):
                        self.assertEqual(float(dead.abs().max()), 0.0)
                    self.assertGreater(float(mlp.gate_proj.weight[:width].abs().sum()), 0)

    def test_mlp_names_dodge_vllm_stacked_mapping(self):
        model = self.load()
        arms.build(ARMS['mlp-quarters'], model)
        for name in arms.trainable_state(ARMS['mlp-quarters'], model):
            self.assertTrue(all(s not in name for s in ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')), name)


if __name__ == '__main__':
    unittest.main()
