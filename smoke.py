import experiment as e
import torch
from transformers import AutoTokenizer
from common import attach_adapter

e.TOKENIZER=AutoTokenizer.from_pretrained(e.ROOT/'models/student',local_files_only=True,padding_side='left')
e.TOKENIZER.pad_token=e.TOKENIZER.eos_token
model=e.load_student()
p=e.prompt('What is 4(x-3)=28 solved for x?')
x=e.TOKENIZER(p,return_tensors='pt').to('cuda')
with torch.inference_mode():before=model(**x).logits
adapter,handle=attach_adapter(model,e.C['layer'],e.C['bottleneck'])
with torch.inference_mode():after=model(**x).logits
assert torch.equal(before,after),'Identity adapter changed logits'
del before,after
text=p+'4(x-3)=28 -> x-3=7 -> x=10.\n</think>\n\\boxed{10}<|im_end|>'
x=e.TOKENIZER(text,return_tensors='pt').to('cuda')
labels=x.input_ids.clone();labels[:,:len(e.TOKENIZER.encode(p,add_special_tokens=False))]=-100
loss=model(**x,labels=labels,use_cache=False).loss
loss.backward()
assert adapter.up.weight.grad.abs().sum()>0
assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('reasoning_adapter.'))
print('REAL MODEL IDENTITY + BACKWARD PASSED; loss',float(loss.detach()),'peak GB',torch.cuda.max_memory_allocated()/1e9,flush=True)
