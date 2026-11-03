"""Verify actual-model backward with memory-saving non-reentrant checkpointing."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['XDG_CACHE_HOME']=str(ROOT/'.cache')
os.environ['HIP_VISIBLE_DEVICES']='0'
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from common import attach_adapter
from peft import get_peft_model,LoraConfig

torch.set_num_threads(8);torch.manual_seed(20261018)
t=AutoTokenizer.from_pretrained(ROOT/'models/student',local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(ROOT/'models/student',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
x=t('Solve 4(x-3)=28. Divide by 4, then add 3. The answer is 10.',return_tensors='pt').to('cuda')
adapter,handle=attach_adapter(m,13,64)
m.train();plain=m(**x,labels=x.input_ids,use_cache=False).loss
plain.backward();grad=adapter.up.weight.grad.detach().clone();m.zero_grad(set_to_none=True)
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
checkpointed=m(**x,labels=x.input_ids,use_cache=False).loss
checkpointed.backward()
assert torch.allclose(plain,checkpointed,atol=1e-6)
assert torch.allclose(grad,adapter.up.weight.grad,atol=1e-5,rtol=1e-4)
print('MLP checkpointed loss and gradient match ordinary backward',float(checkpointed.detach()),flush=True)
m.zero_grad(set_to_none=True);handle.remove();delattr(m,'reasoning_adapter')
m=get_peft_model(m,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});m.train()
loss=m(**x,labels=x.input_ids,use_cache=False).loss;loss.backward()
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'lora_B' in n)
assert all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)
print('LoRA checkpointed backward passed',float(loss.detach()),flush=True)
