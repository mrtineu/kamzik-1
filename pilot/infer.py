"""Run the frozen student, concise prompting, or a saved MLP/LoRA adapter."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--question',required=True)
p.add_argument('--arm',choices=['baseline','concise','mlp','lora'],default='mlp')
p.add_argument('--run',default='quick')
p.add_argument('--max-new-tokens',type=int,default=4096)
p.add_argument('--answer-type',choices=['numeric','math','choice'],default='numeric')
p.add_argument('--choice-labels',nargs='+',help='Labels appearing in the options supplied with --question')
a=p.parse_args()
if a.answer_type=='choice' and not a.choice_labels:p.error('--choice-labels is required for choice answers')
run=ROOT/'runs'/a.run
if not (run/'config.json').exists():raise SystemExit('Run config does not exist: '+str(run))
os.environ['QRL_RUN']=a.run
os.environ['QRL_CONFIG']=str(run/'config.json')
import experiment as e
import torch
from transformers import AutoTokenizer
from common import attach_adapter

e.log=lambda message: print(message,flush=True)
torch.set_num_threads(e.C.get('cpu_threads',8))
e.TOKENIZER=AutoTokenizer.from_pretrained(ROOT/'models/student',local_files_only=True,padding_side='left')
e.TOKENIZER.pad_token=e.TOKENIZER.eos_token
model=e.load_student()
if a.arm in ['mlp','lora']:
 checkpoint=run/f'{a.arm}.pt'
 if not checkpoint.exists():raise SystemExit('Checkpoint not trained yet: '+str(checkpoint))
 if a.arm=='mlp':adapter,handle=attach_adapter(model,e.C['layer'],e.C['bottleneck'])
 else:
  from peft import LoraConfig,get_peft_model
  model=get_peft_model(model,LoraConfig(r=e.C['lora_rank'],lora_alpha=2*e.C['lora_rank'],lora_dropout=0,
      target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
 weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
 parameters=dict(model.named_parameters())
 for name,value in weights.items():parameters[name].data.copy_(value.to(parameters[name].device))
 model.eval()
example={'id':'interactive','question':a.question,'gold':'0','difficulty':'unrated',
         'answer_type':a.answer_type,'choice_labels':a.choice_labels}
row=e.generate_batch(model,[example],[e.concise_instruction(example)] if a.arm=='concise' else None,cap=a.max_new_tokens)[0]
print(row['text'])
print(json.dumps({k:row[k] for k in ['thinking_tokens','final_tokens','total_tokens','finished','truncated','seconds']},indent=2))
