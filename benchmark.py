"""ROCm attention, batch-size, and decode-overhead benchmark (no training/run mutation).

Each attention mode runs in a fresh process because the AOTriton flag is read at
runtime initialization. Fixed decode windows use real saved TRAINING traces;
--end-to-end additionally measures ordinary sampled answers from original prompts.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def summarize_benchmark(directory):
    import statistics
    results=[json.loads(p.read_text()) for p in sorted(directory.glob('attention_*.json'))]
    lines=['# Inference optimization benchmark','',
           'Fixed decode windows continue saved training traces at the stated context length. '
           'They force a fixed number of output tokens to compare kernels; end-to-end rows allow normal termination. '
           'VRAM is peak allocated memory, not total driver memory.','',
           '| AOTriton flag | Context | Batch | Median tokens/s | Peak VRAM GiB | GPU busy % |',
           '|---|---:|---:|---:|---:|---:|']
    for result in results:
        groups={}
        for row in result['measurements']:
            if row['kind']=='decode_window':groups.setdefault((row['context_tokens'],row['batch_size']),[]).append(row)
        for (context,batch),rows in sorted(groups.items()):
            valid=[r for r in rows if r.get('status')!='oom']
            if not valid:
                lines.append(f'| {int(result["experimental_attention"])} | {context} | {batch} | OOM | — | — |');continue
            rate=statistics.median(r['tokens_per_second'] for r in valid)
            memory=max(r['peak_allocated_bytes'] for r in valid)/2**30
            busy=[r['gpu_busy_percent_mean'] for r in valid if r['gpu_busy_percent_mean'] is not None]
            busy_text=f'{statistics.mean(busy):.1f}' if busy else 'unavailable'
            lines.append(f'| {int(result["experimental_attention"])} | {context} | {batch} | {rate:.2f} | {memory:.2f} | {busy_text} |')
    lines+=['','## Complete answers','',
            '| AOTriton flag | Cap | Batch | Seconds | Useful tokens/s | Decode slot efficiency | Correct | Finished |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for result in results:
        for row in result['measurements']:
            if row['kind']=='end_to_end' and row.get('status')!='oom':
                lines.append(f'| {int(result["experimental_attention"])} | {row["max_new_tokens"]} | {row["batch_size"]} | '
                    f'{row["seconds"]:.2f} | {row["tokens_per_second"]:.2f} | {row["decode_slot_efficiency"]:.1%} | '
                    f'{row["correct"]} | {row["finished"]} |')
    lines+=['','## Correctness and profiling','',
            'See `attention_1.json` for BF16 versus FP32 math error, math/flash token agreement, padded auto-dispatch, '
            'MLP gradient comparison, and checkpointed LoRA backward. These finite checks do not prove equal sampled accuracy. '
            'The mode flag permits experimental dispatch; actual kernels depend on shape and padding. '
            'Profiler tables and Chrome traces show the selected attention operations and CPU/device overhead.','']
    for result in results:
        if result.get('optimized_training_eligible') is False:
            lines.append('Optimized attention did **not** meet the training-gradient tolerance. '
                         'The recommended configuration keeps training on forced math attention.\n')
    (directory/'RESULTS.md').write_text('\n'.join(lines))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--source-run', default='pilot')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[4,8,16])
    p.add_argument('--contexts', type=int, nargs='+', default=[2048,4096])
    p.add_argument('--decode-tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--modes', type=int, nargs='+', choices=[0, 1], default=[0,1],
                   help='Compare disabled/enabled experimental attention')
    p.add_argument('--vram-fraction',type=float,default=0.8)
    p.add_argument('--vram-reserve-gib',type=float,default=2.0)
    p.add_argument('--timeout-seconds', type=int, default=1200,
                   help='Stop a worker after this wall-time limit; do not start further cases')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--end-to-end', action='store_true')
    p.add_argument('--worker', type=int, choices=[0, 1], help=argparse.SUPPRESS)
    return p


def worker(a):
    for key, directory in [('HF_HOME', '.cache/huggingface'), ('XDG_CACHE_HOME', '.cache'),
                           ('TORCH_HOME', '.cache/torch'), ('TRITON_CACHE_DIR', '.cache/triton')]:
        os.environ[key] = str(ROOT / directory)
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = str(a.worker)
    from memory_guard import configure_allocator,apply_budget,check_headroom,VRAMBudgetExceeded
    configure_allocator(a.vram_fraction)
    import contextlib
    import gc
    import threading
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed
    from common import format_instruction, attach_adapter, read_jsonl, score_tokens, write_json

    if not torch.cuda.is_available():
        raise RuntimeError('AMD GPU unavailable')
    budget=apply_budget(torch,fraction=a.vram_fraction,reserve_gib=a.vram_reserve_gib)
    torch.set_num_threads(a.threads)
    t = AutoTokenizer.from_pretrained(ROOT/'models/student', local_files_only=True, padding_side='left')
    t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(ROOT/'models/student', torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    source = read_jsonl(ROOT/'runs'/a.source_run/'train_baseline.jsonl')
    if not source:
        raise RuntimeError('Saved training baseline traces are required')
    def prompt(row):
        return t.apply_chat_template([{'role':'user','content':row['question']+'\n'+format_instruction(row)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
    traces = sorted([(r, t.encode(prompt(r)+r['text'], add_special_tokens=False)) for r in source],
                    key=lambda pair: len(pair[1]), reverse=True)
    def inputs(context, batch):
        available = [(r, ids) for r, ids in traces if len(ids) >= context]
        if not available:
            raise RuntimeError(f'No saved training trace reaches {context} tokens')
        chosen = [available[i % len(available)] for i in range(batch)]
        return {'input_ids':torch.tensor([ids[:context] for _, ids in chosen], device='cuda'),
                'attention_mask':torch.ones((batch, context), dtype=torch.long, device='cuda')}, [r for r, _ in chosen]

    result = dict(experimental_attention=bool(a.worker), torch=torch.__version__, hip=torch.version.hip,
                  gpu=torch.cuda.get_device_name(0), threads=a.threads, measurements=[], correctness=[],memory_budget=budget)
    destination = a.output/f'attention_{a.worker}.json'
    def save():
        write_json(destination, result)
    def emit(row):
        result['measurements'].append(row); save()
        print(json.dumps(row), flush=True)

    # Check the actual model, with left padding, against forced math SDPA. A
    # successful forced-flash call proves availability rather than trusting a flag.
    if a.worker:
        for context in a.contexts:
            x, _ = inputs(context, 2)
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                m.float()
                fp32 = m(**x, logits_to_keep=8, use_cache=False).logits
                m.bfloat16()
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                reference = m(**x, logits_to_keep=8, use_cache=False).logits.float()
            with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                actual = m(**x, logits_to_keep=8, use_cache=False).logits.float()
            error = actual-reference
            relative_rms = float(error.square().mean().sqrt()/reference.square().mean().sqrt())
            agreement = float((actual.argmax(-1)==reference.argmax(-1)).float().mean())
            math_fp32_rms=float((reference-fp32).square().mean().sqrt()/fp32.square().mean().sqrt())
            flash_fp32_rms=float((actual-fp32).square().mean().sqrt()/fp32.square().mean().sqrt())
            kl=float(torch.nn.functional.kl_div(actual.log_softmax(-1),reference.softmax(-1),reduction='none').sum(-1).mean())
            check = dict(context_tokens=context, max_abs_error=float(error.abs().max()),
                         relative_rms=relative_rms, greedy_token_agreement=agreement,
                         math_fp32_relative_rms=math_fp32_rms,flash_fp32_relative_rms=flash_fp32_rms,
                         mean_kl_divergence=kl,
                         finite=bool(torch.isfinite(actual).all()), forced_flash=True)
            # BF16 kernels need not be bit-identical. Keep measured deviations
            # visible, and reject material distribution/token changes.
            check['passed'] = (check['finite'] and flash_fp32_rms <= max(.005,1.5*math_fp32_rms)
                               and kl < .02 and agreement >= .90)
            result['correctness'].append(check); save(); print(json.dumps(check), flush=True)
            if not check['passed']:
                raise RuntimeError('Optimized attention failed model correctness check')
            # ROCm flash cannot accept this padding mask. Auto dispatch must
            # select a supported backend and remain correct for real batches.
            x['attention_mask'][0, :17] = 0
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                reference = m(**x, logits_to_keep=8, use_cache=False).logits.float()
            with torch.inference_mode():
                actual = m(**x, logits_to_keep=8, use_cache=False).logits.float()
            error=actual-reference
            relative_rms=float(error.square().mean().sqrt()/reference.square().mean().sqrt())
            agreement=float((actual.argmax(-1)==reference.argmax(-1)).float().mean())
            check=dict(context_tokens=context,left_padding=True,relative_rms=relative_rms,
                       greedy_token_agreement=agreement,passed=bool(torch.isfinite(actual).all()) and relative_rms<.03 and agreement>=.90)
            result['correctness'].append(check);save()
            if not check['passed']:raise RuntimeError('Padded attention correctness check failed')
            del x, reference, actual, error, fp32
        # Validate adapter gradients against math, then checkpointed LoRA backward.
        x, _ = inputs(min(512, min(a.contexts)), 1)
        set_seed(20261018)
        adapter, handle = attach_adapter(m, 13, 64)
        grads = []
        losses = []
        for backend in [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]:
            m.zero_grad(set_to_none=True)
            with sdpa_kernel(backend):
                loss = m(**x, labels=x['input_ids'], use_cache=False).loss
                loss.backward()
            grads.append(adapter.up.weight.grad.detach().clone()); losses.append(float(loss.detach()))
        relative = float((grads[1]-grads[0]).norm()/grads[0].norm())
        check = dict(adapter_backward=True, losses=losses, gradient_relative_error=relative,
                     passed=all(torch.isfinite(g).all().item() for g in grads) and relative < .05)
        result['correctness'].append(check); save()
        result['optimized_training_eligible']=check['passed']
        m.zero_grad(set_to_none=True); handle.remove(); delattr(m, 'reasoning_adapter')
        from peft import LoraConfig, get_peft_model
        lora = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
                    target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
        lora.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        lora.train()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            loss = lora(**x, labels=x['input_ids'], use_cache=False).loss
            loss.backward()
        passed = bool(torch.isfinite(loss)) and any(p.grad is not None and bool(p.grad.abs().sum()>0)
                    for n, p in lora.named_parameters() if 'lora_B' in n)
        passed = passed and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in lora.parameters())
        passed = passed and all(p.grad is None for p in lora.parameters() if not p.requires_grad)
        result['correctness'].append(dict(lora_checkpointed_backward=True, passed=passed)); save()
        result['optimized_training_eligible']=result['optimized_training_eligible'] and passed
        lora.zero_grad(set_to_none=True); lora.gradient_checkpointing_disable()
        m = lora.unload(); m.eval(); m.config.use_cache=True
        del x, adapter, grads, lora, loss
        gc.collect(); torch.cuda.empty_cache()

    # GPU busy percentage is sampled from the AMD driver's read-only sysfs file.
    busy_paths = list(Path('/sys/class/drm').glob('card*/device/gpu_busy_percent'))
    @contextlib.contextmanager
    def utilization():
        values=[]; stop=threading.Event()
        def poll():
            while not stop.is_set():
                try:
                    if busy_paths:values.append(float(busy_paths[0].read_text().strip()))
                except (OSError,ValueError):pass
                stop.wait(.2)
        thread=threading.Thread(target=poll,daemon=True);thread.start()
        try:yield values
        finally:stop.set();thread.join()

    def measure(x, rows, cap, fixed, label, repeat):
        check_headroom(torch,reserve_gib=a.vram_reserve_gib)
        set_seed(20261018)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        cpu_start=time.process_time();start=time.monotonic()
        class Progress(StoppingCriteria):
            last=0
            def __call__(self,input_ids,scores,**kwargs):
                steps=input_ids.shape[1]-x['input_ids'].shape[1]
                if steps%16==0:check_headroom(torch,reserve_gib=a.vram_reserve_gib)
                if steps>=self.last+256:
                    print(json.dumps(dict(progress=label,steps=steps,cap=cap,seconds=time.monotonic()-start)),flush=True)
                    self.last=steps
                return False
        with utilization() as busy, torch.inference_mode():
            out=m.generate(**x, max_new_tokens=cap, min_new_tokens=cap if fixed else 0,
                    do_sample=True, temperature=.6, top_p=.95, top_k=20,
                    pad_token_id=t.pad_token_id, use_cache=True,
                    stopping_criteria=StoppingCriteriaList([Progress()]))
            torch.cuda.synchronize()
        elapsed=time.monotonic()-start;cpu=time.process_time()-cpu_start
        generated=out[:, x['input_ids'].shape[1]:].tolist()
        scored=[score_tokens(ids,t,r['gold'],cap,answer_type=r.get('answer_type','numeric'),
                 choice_labels=r.get('choice_labels'),accepted_answers=r.get('accepted_answers')) for ids,r in zip(generated,rows)]
        slots=len(rows)*len(generated[0])
        useful=slots if fixed else sum(r['total_tokens']+int(r['finished']) for r in scored)
        row=dict(kind=label, repeat=repeat, batch_size=len(rows), context_tokens=x['input_ids'].shape[1],
                 max_new_tokens=cap, generated_steps=len(generated[0]), seconds=elapsed, cpu_seconds=cpu,
                 useful_tokens=useful, tokens_per_second=useful/elapsed,
                 decode_slot_efficiency=useful/slots if slots else 1,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 gpu_busy_percent_mean=sum(busy)/len(busy) if busy else None,
                 unique_training_examples=len({r['id'] for r in rows}))
        if not fixed:
            row.update(correct=sum(r['correct'] for r in scored), finished=sum(r['finished'] for r in scored),
                       ids=[r['id'] for r in rows])
            write_json(a.output/f'answers_{a.worker}_{len(rows)}_{cap}_{repeat}.json',
                       [dict(id=r['id'],**s) for r,s in zip(rows,scored)])
        return row

    # Warm up outside measurements, including per-shape prefill/kernel setup.
    for context in a.contexts:
        for batch in a.batch_sizes:
            x, rows=inputs(context,batch)
            try:
                measure(x,rows,4,True,'warmup',0)
                for repeat in range(a.repeats):
                    emit(measure(x,rows,a.decode_tokens,True,'decode_window',repeat))
            except (torch.OutOfMemoryError,VRAMBudgetExceeded):
                emit(dict(kind='decode_window',batch_size=batch,context_tokens=context,status='oom'))
                result['complete']=False;result['error']='out_of_memory';save()
                raise SystemExit('Out of GPU memory; stopping benchmark without further GPU work')
            del x
            gc.collect();torch.cuda.empty_cache()
    if a.end_to_end:
        # Same original training prompts in both modes, varied answer lengths.
        batch=max(a.batch_sizes)
        rows=source[:batch]
        x=t([prompt(r) for r in rows],padding=True,return_tensors='pt').to('cuda')
        for cap in a.contexts:
            try:emit(measure(x,rows,cap,False,'end_to_end',0))
            except (torch.OutOfMemoryError,VRAMBudgetExceeded):
                emit(dict(kind='end_to_end',batch_size=batch,max_new_tokens=cap,status='oom'))
                result['complete']=False;result['error']='out_of_memory';save()
                raise SystemExit('Out of GPU memory; stopping benchmark without further GPU work')
        del x
    if a.profile:
        x, rows=inputs(max(a.contexts),min(a.batch_sizes))
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                measure(x,rows,min(32,a.decode_tokens),True,'profile',0)
            prof.export_chrome_trace(str(a.output/f'trace_{a.worker}.json'))
            events=prof.key_averages()
            result['profile_top_ops']=[dict(name=e.key,calls=e.count,cpu_self_us=e.self_cpu_time_total,
                    device_self_us=e.self_device_time_total) for e in sorted(events,key=lambda e:e.self_device_time_total,reverse=True)[:25]]
            result['attention_ops']=[dict(name=e.key,calls=e.count,cpu_us=e.cpu_time_total,
                    device_us=e.device_time_total) for e in events if any(s in e.key.lower() for s in ['attention','flash','softmax'])]
            (a.output/f'profile_{a.worker}.txt').write_text(events.table(sort_by='self_device_time_total',row_limit=40))
        except (torch.OutOfMemoryError,VRAMBudgetExceeded):
            result['complete']=False;result['error']='out_of_memory_during_profile';save()
            raise SystemExit('Out of GPU memory during profiling; stopping benchmark')
        except RuntimeError as error:
            result['profile_error']=str(error)
    result['complete']=True;save()


def main():
    a=parser().parse_args()
    if not 0<a.vram_fraction<1 or a.vram_reserve_gib<=0:
        raise SystemExit('Use a VRAM fraction below one and a positive reserve')
    if any(v<1 for v in a.batch_sizes+a.contexts+[a.decode_tokens,a.repeats,a.threads,a.timeout_seconds]):
        raise SystemExit('Batch sizes, contexts, decode tokens, repeats, and threads must be positive')
    if a.worker is not None:
        if a.output is None:raise SystemExit('Worker needs output directory')
        worker(a);return
    a.output=a.output or ROOT/'runs'/('benchmark_'+time.strftime('%Y%m%d_%H%M%S'))
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'arguments.json').write_text(json.dumps({**vars(a),'output':str(a.output)},indent=2))
    print(f'Benchmark artifacts: {a.output}',flush=True)
    for mode in a.modes:
        cmd=[sys.executable,'-u',str(ROOT/'benchmark.py'),'--worker',str(mode),'--output',str(a.output),
             '--source-run',a.source_run,'--threads',str(a.threads),'--decode-tokens',str(a.decode_tokens),
             '--repeats',str(a.repeats),'--vram-fraction',str(a.vram_fraction),
             '--vram-reserve-gib',str(a.vram_reserve_gib),
             '--batch-sizes',*map(str,a.batch_sizes),'--contexts',*map(str,a.contexts)]
        if a.profile:cmd.append('--profile')
        if a.end_to_end:cmd.append('--end-to-end')
        with (a.output/f'attention_{mode}.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=a.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                (a.output/'failure.json').write_text(json.dumps(dict(
                    status='timeout',mode=mode,timeout_seconds=a.timeout_seconds,worker_pid=process.pid)))
                raise SystemExit('Benchmark worker timed out; no further modes will run')
        if code:raise SystemExit(f'Attention mode {mode} failed ({code}); see {a.output}/attention_{mode}.log')
    summarize_benchmark(a.output)
    print('Benchmark complete: '+str(a.output),flush=True)


if __name__=='__main__':main()
