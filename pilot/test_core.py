from fractions import Fraction
from types import SimpleNamespace
import torch
from torch import nn
from common import ReasoningMLP,attach_adapter,extract_answer,numeric,score_tokens,summarize


def test_final_answer_only():
    assert extract_answer('I think 8') is None
    assert extract_answer('8</think>We used 8 operations.') is None
    assert extract_answer('wrong 99</think>\\boxed{42}')==42
    assert extract_answer('</think>\\boxed{\\frac{3}{2}}')==Fraction(3,2)
    assert numeric('1,234.50')==Fraction(2469,2)


def test_adapter_identity_and_gradient_flow():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(hidden_size=8)
            self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Linear(8,8),nn.Linear(8,8)])
        @property
        def device(self):return next(self.parameters()).device
        def forward(self,x):
            for layer in self.model.layers:x=layer(x)
            return x
    torch.manual_seed(1);model=Toy();x=torch.randn(2,8)
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():expected=model(x)
    adapter,handle=attach_adapter(model,0,3)
    assert torch.equal(model(x),expected)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    assert adapter.up.weight.grad is not None and adapter.up.weight.grad.abs().sum()>0
    assert adapter.down.weight.grad is not None and adapter.down.weight.grad.abs().sum()>0
    assert not torch.equal(model(x),expected)
    for name,p in model.named_parameters():
        if name in original:
            assert torch.equal(p,original[name])
            assert p.grad is None
    handle.remove()
    assert torch.equal(model(x),expected)


def test_truncation_cannot_count_as_correct():
    class Tokenizer:
        eos_token_id=9
        def convert_tokens_to_ids(self,x):return {'</think>':3,'<|im_end|>':9}[x]
        def decode(self,ids,**kwargs):return 'reason</think>\\boxed{42}'
    t=Tokenizer()
    assert not score_tokens([1,3,4],t,'42',3)['correct']
    result=score_tokens([1,3,4,9,9],t,'42',3)
    assert result['correct'] and result['thinking_tokens']==1 and result['final_tokens']==1


def test_compute_metrics_include_first_and_final_attempts():
    common=dict(correct=True,thinking_closed=True,thinking_tokens=2,total_tokens=4,
                truncated=False,finished=True,parsed_answer='42',seconds=3)
    rows=[{**common,'initial_attempt':{'seconds':2}},
          {**common,'total_compute_seconds':7,'initial_attempt':{'seconds':4}}]
    metrics=summarize(rows)
    assert metrics['total_compute_seconds']==12
    assert metrics['mean_seconds']==3
