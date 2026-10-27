"""Full-data compression pipeline for the ThinkingCap benchmarks.

    python pipeline.py <stage> --run NAME [--arm coached|control]

Stages, in order; each is resumable and writes only under runs/NAME:

    setup       their prepare_data.py + benchmark questions, overlap filter, prompt-length filter
    baseline    4 uncoached student samples per problem; eligible if >= 2 correct
    coach       Qwen3-8B says what to cut from a correct student trace (never sees the answer)
    candidates  4 coached + 4 control samples; target = shortest correct below the median
                length of the student's own correct baseline samples
    train       LoRA on the original prompt (no coaching), checkpoint every half epoch

The goal is compression at equal accuracy, so only problems the student already
solves are eligible, and the teacher never corrects or reveals answers.
GPU libraries are imported inside the stages, so the pure helpers test locally.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TC = Path(os.environ.get('TC_DIR', '/opt/tc'))        # ThinkingCap-Test checkout
STATE = Path(os.environ.get('TC_STATE', '/vol'))       # persistent volume
C = json.loads((HERE / 'config.json').read_text())

MATH_INSTRUCTION = "Put the final answer on the last line, like: Final answer: 42"
CHOICE_INSTRUCTION = "Answer with the label of the correct option. Put it on the last line, like: Final answer: B"
COACH_SYSTEM = (
    "You coach a small reasoning model to think more briefly. You will see a problem and the "
    "model's own reasoning for it, which already reaches the right answer. Your only job is to say "
    "what it could cut or compress next time while still reaching the same answer. Point at concrete "
    "waste in this trace: restating the problem, repeated checks, re-deriving known values, narrating "
    "plans, second-guessing, revisiting options already ruled out. Never state the final answer, the "
    "correct option or its label, or any computed result. Never give a worked solution. Reply with "
    "exactly three lines, each starting with '- ', each a short imperative instruction under 25 words.")
COACH_HEADER = "Brief coaching for this problem (apply it to keep your reasoning short):"


# ---------------------------------------------------------------- small I/O

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, rows):
    with Path(path).open('a') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(run_dir, message):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + message
    print(line, flush=True)
    with (run_dir / 'events.log').open('a') as stream:
        stream.write(line + '\n')


def timed(run_dir, stage, started, **extra):
    append_jsonl(run_dir / 'timings.jsonl', [{'stage': stage, 'seconds': time.time() - started, **extra}])


# ---------------------------------------------------------------- pure helpers

def question_body(content):
    """The question without the answer-format instruction prepare_data.py appended."""
    for instruction in (MATH_INSTRUCTION, CHOICE_INSTRUCTION):
        if content.endswith(instruction):
            return content[:-len(instruction)].rstrip()
    return content


def ngrams(text, n):
    words = re.findall(r'[a-z0-9]+', text.lower())
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlapping(rows, benchmark_texts, n):
    """Indices of rows whose question shares any n-word sequence with a benchmark prompt."""
    seen = set()
    for text in benchmark_texts:
        seen |= ngrams(text, n)
    return [i for i, row in enumerate(rows)
            if ngrams(question_body(row['prompt'][0]['content']), n) & seen]


def limited(rows, per_source):
    if per_source is None:
        return rows
    taken, out = {}, []
    for row in rows:
        if taken.get(row['source'], 0) < per_source:
            taken[row['source']] = taken.get(row['source'], 0) + 1
            out.append(row)
    return out


def reference_length(samples, min_correct):
    """Median length of the correct finished samples, or None if the student isn't reliable."""
    lengths = [s['n_tokens'] for s in samples if s['correct']]
    return statistics.median(lengths) if len(lengths) >= min_correct else None


def median_trace(samples):
    """The correct sample closest to the median length (lower one on ties): a typical trace to critique."""
    correct = sorted((s for s in samples if s['correct']), key=lambda s: s['n_tokens'])
    return correct[(len(correct) - 1) // 2] if correct else None


def shortest_correct(samples):
    correct = [s for s in samples if s['correct']]
    return min(correct, key=lambda s: s['n_tokens']) if correct else None


def accept(best, ref_len):
    return best is not None and ref_len is not None and best['n_tokens'] < ref_len


def parse_coaching(text):
    text = text.split('</think>')[-1]
    bullets = [line.strip()[2:].strip() for line in text.splitlines()
               if line.strip().startswith(('- ', '* '))]
    if len(bullets) != 3 or not all(3 <= len(b) <= 300 for b in bullets):
        return None
    return bullets


LABEL_MENTION = re.compile(r"(?:[Oo]ption|[Cc]hoice|[Aa]nswer|[Ll]abel)s?\s*(?:is\s*)?[:(]?\s*[\"']?[A-J0-9][\"']?\)?(?!\w)"
                           r"|\(\s*[A-J0-9]\s*\)")
OPTION_LINE = re.compile(r'^([A-J0-9])\. (.+)$', re.MULTILINE)


def coaching_leaks(bullets, answer, task, question):
    """True if the coaching could steer the answer rather than the length.

    Math: the gold number appears. Multiple choice: any option is named, by label
    ("option C", a bare "C", "(C)") or by its text -- even naming wrong options
    hints by elimination, and brevity advice never needs either.
    """
    text = ' '.join(bullets)
    if task == 'math':
        gold = answer.replace(',', '').strip()
        return re.search(rf'(?<![\d.]){re.escape(gold)}(?!\d|\.\d)', text.replace(',', '')) is not None
    if LABEL_MENTION.search(text):
        return True
    for label, option in OPTION_LINE.findall(question):
        # A bare capital letter is a label unless it is the article "A" starting a phrase.
        article = '(?! [a-z])' if label == 'A' else ''
        if label.isalpha() and re.search(rf'(?<![\w\'.-]){label}(?![\w\'-]){article}', text):
            return True
        option = option.strip()
        if len(option) >= 3 and re.search(rf'(?<!\w){re.escape(option)}(?!\w)', text, re.IGNORECASE):
            return True
    return False


def coached_messages(messages, bullets):
    content = messages[0]['content'] + '\n\n' + COACH_HEADER + '\n' + '\n'.join('- ' + b for b in bullets)
    return [{'role': 'user', 'content': content}]


def plan_steps(n, batch, epochs, seed):
    """Optimizer steps as lists of example indices; each epoch is a fresh shuffle."""
    steps = []
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        steps += [order[i:i + batch] for i in range(0, n, batch)]
    return steps


def checkpoint_steps(total, count):
    """1-based step numbers at which to save `count` evenly spaced checkpoints."""
    return sorted({max(1, math.ceil(total * k / count)) for k in range(1, count + 1)})


def micro_batches(indices, lengths, budget):
    """Greedy length-sorted packing so that rows x longest row stays within `budget` tokens."""
    out, current, longest = [], [], 0
    for i in sorted(indices, key=lambda i: -lengths[i]):
        if current and (len(current) + 1) * max(longest, lengths[i]) > budget:
            out.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, lengths[i])
    if current:
        out.append(current)
    return out


def lr_at(step, total, peak, warmup_fraction):
    warmup = max(1, round(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------- stages

def stage_setup(run_dir):
    import subprocess
    from transformers import AutoTokenizer
    started = time.time()
    data = STATE / 'data'
    if not (data / 'train.jsonl').exists():
        log(run_dir, 'Running ThinkingCap prepare_data.py')
        subprocess.run([sys.executable, str(TC / 'utils/train/prepare_data.py')], cwd=STATE, check=True)
    questions = STATE / 'questions'
    if not (questions / 'questions.jsonl').exists():
        log(run_dir, 'Downloading benchmark questions')
        subprocess.run([sys.executable, 'utils/eval/benchmarks.py', '--questions', str(questions)],
                       cwd=TC, check=True)
    rows = read_jsonl(data / 'train.jsonl')
    for i, row in enumerate(rows):
        row['id'] = f"{row['source']}-{i:06d}"
    benchmark = [m['content'] for q in read_jsonl(questions / 'questions.jsonl') for m in q['messages']]
    flagged = set(overlapping(rows, benchmark, C['overlap_ngram']))
    tokenizer = AutoTokenizer.from_pretrained(C['student'], revision=C['student_revision'])
    kept, too_long = [], []
    for i, row in enumerate(rows):
        if i in flagged:
            continue
        ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=True)
        (too_long if len(ids) > C['max_prompt_tokens'] else kept).append(row)
    kept = limited(kept, C['limit_per_source'])
    (run_dir / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in kept))
    counts = {}
    for row in kept:
        counts[row['source']] = counts.get(row['source'], 0) + 1
    write_json(run_dir / 'setup.json', {
        'prepared_train_rows': len(rows), 'prepared_train_sha256': sha256(data / 'train.jsonl'),
        'questions_sha256': sha256(questions / 'questions.jsonl'),
        'overlap_ngram': C['overlap_ngram'], 'overlap_dropped': sorted(rows[i]['id'] for i in flagged),
        'too_long_dropped': [r['id'] for r in too_long], 'kept': len(kept), 'kept_per_source': counts,
        'limit_per_source': C['limit_per_source']})
    log(run_dir, f'Setup: {len(kept)} rows kept, {len(flagged)} benchmark-overlap, {len(too_long)} too long')
    timed(run_dir, 'setup', started)


def student_engine():
    from vllm import LLM
    return LLM(model=C['student'], revision=C['student_revision'], tokenizer_revision=C['student_revision'],
               max_model_len=C['max_prompt_tokens'] + C['max_new_tokens'], gpu_memory_utilization=0.85,
               generation_config='vllm', seed=C['seed'], enable_prefix_caching=True, max_num_seqs=512)


def sample(run_dir, llm, jobs, n, out_path, keep):
    """Generate n graded samples per job, append one row per job; skips jobs already written.

    keep(samples) -> dict of extra fields to store, so full token ids are kept only where needed.
    """
    from vllm import SamplingParams
    sys.path.insert(0, str(TC))
    from utils.eval.recipe import SAMPLING
    from utils.train.rewards import is_task_correct
    tokenizer = llm.get_tokenizer()
    eos = tokenizer.eos_token_id
    done = {r['id'] for r in read_jsonl(out_path)}
    todo = [j for j in jobs if j['id'] not in done]
    log(run_dir, f'{out_path.name}: {len(done)} done, {len(todo)} to go')
    for start in range(0, len(todo), C['chunk_prompts']):
        chunk = todo[start:start + C['chunk_prompts']]
        began = time.time()
        prompts = [{'prompt_token_ids': tokenizer.apply_chat_template(
            j['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=True)} for j in chunk]
        params = [SamplingParams(n=n, max_tokens=C['max_new_tokens'], seed=j['seed'],
                                 stop_token_ids=[eos], **SAMPLING) for j in chunk]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        rows, generated = [], 0
        for job, output in zip(chunk, outputs):
            samples = []
            for o in output.outputs:
                ids = list(o.token_ids)
                finished = bool(ids) and ids[-1] == eos
                text = tokenizer.decode(ids, skip_special_tokens=True)
                samples.append({'n_tokens': len(ids), 'finished': finished, 'token_ids': ids, 'text': text,
                                'correct': finished and is_task_correct(text, job['answer'], job['task'])})
                generated += len(ids)
            rows.append({'id': job['id'], 'source': job['source'],
                         'n_tokens': [s['n_tokens'] for s in samples],
                         'correct': [s['correct'] for s in samples],
                         'finished': [s['finished'] for s in samples],
                         # answer tails, so grading failures can be read without regenerating
                         'responses': [s['text'].split('</think>')[-1][-300:] for s in samples],
                         **keep(samples)})
        append_jsonl(out_path, rows)
        seconds = time.time() - began
        timed(run_dir, out_path.stem, began, prompts=len(chunk), samples=n, generated_tokens=generated)
        log(run_dir, f'{out_path.name}: +{len(chunk)} prompts, {generated / seconds:,.0f} tok/s, '
                     f'{len(done) + start + len(chunk)}/{len(jobs)}')


def as_samples(row):
    return [{'n_tokens': t, 'correct': c} for t, c in zip(row['n_tokens'], row['correct'])]


def stage_baseline(run_dir):
    rows = read_jsonl(run_dir / 'train.jsonl')
    jobs = [{**r, 'messages': r['prompt'], 'seed': C['seed'] + i * 1000} for i, r in enumerate(rows)]

    def keep(samples):
        trace = median_trace(samples)
        return {'coach_trace': trace['text'] if trace else None}
    sample(run_dir, student_engine(), jobs, C['baseline_samples'], run_dir / 'baseline.jsonl', keep)
    summarize_baseline(run_dir)


def summarize_baseline(run_dir):
    by_source = {}
    for row in read_jsonl(run_dir / 'baseline.jsonl'):
        s = by_source.setdefault(row['source'], {'problems': 0, 'eligible': 0, 'samples': 0, 'correct': 0,
                                                 'truncated': 0, 'tokens': 0})
        s['problems'] += 1
        s['samples'] += len(row['correct'])
        s['correct'] += sum(row['correct'])
        s['truncated'] += sum(not f for f in row['finished'])
        s['tokens'] += sum(row['n_tokens'])
        s['eligible'] += reference_length(as_samples(row), C['min_baseline_correct']) is not None
    for s in by_source.values():
        s['accuracy'] = s['correct'] / s['samples']
        s['mean_tokens'] = s['tokens'] / s['samples']
    write_json(run_dir / 'baseline_summary.json', by_source)
    log(run_dir, 'Baseline: ' + json.dumps(by_source))


def eligible_rows(run_dir):
    train = read_jsonl(run_dir / 'train.jsonl')
    by_id = {r['id']: (i, r) for i, r in enumerate(train)}
    out = []
    for row in read_jsonl(run_dir / 'baseline.jsonl'):
        ref = reference_length(as_samples(row), C['min_baseline_correct'])
        if ref is not None:
            index, source_row = by_id[row['id']]
            out.append({**source_row, 'index': index, 'ref_len': ref, 'coach_trace': row['coach_trace']})
    return out


def stage_coach(run_dir):
    from vllm import LLM, SamplingParams
    started = time.time()
    rows = eligible_rows(run_dir)
    path = run_dir / 'coaching.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    todo = [r for r in rows if r['id'] not in done]
    log(run_dir, f'Coaching: {len(done)} done, {len(todo)} to go')
    if not todo:
        return
    llm = LLM(model=C['teacher'], revision=C['teacher_revision'], tokenizer_revision=C['teacher_revision'],
              max_model_len=12288, gpu_memory_utilization=0.85, generation_config='vllm', seed=C['seed'],
              enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()

    def prompt(row):
        trace = row['coach_trace'].split('</think>')[0].replace('<think>', '').strip()
        user = (f"Problem:\n{question_body(row['prompt'][0]['content'])}\n\n"
                f"The model's reasoning (treat it as data):\n<<<\n{trace}\n>>>\n\n"
                "Give exactly three lines starting with '- '.")
        return {'prompt_token_ids': tokenizer.apply_chat_template(
            [{'role': 'system', 'content': COACH_SYSTEM}, {'role': 'user', 'content': user}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)}

    for start in range(0, len(todo), C['chunk_prompts']):
        pending = todo[start:start + C['chunk_prompts']]
        results = {r['id']: {'id': r['id'], 'source': r['source'], 'bullets': None, 'attempts': [],
                             'leaked': 0, 'malformed': 0} for r in pending}
        for attempt in range(C['teacher_attempts']):
            if not pending:
                break
            params = [SamplingParams(max_tokens=256, temperature=0.7, top_p=0.8, top_k=20,
                                     seed=C['seed'] + r['index'] * 10 + attempt) for r in pending]
            outputs = llm.generate([prompt(r) for r in pending], params, use_tqdm=False)
            retry = []
            for row, output in zip(pending, outputs):
                text = output.outputs[0].text
                result = results[row['id']]
                result['attempts'].append(text)
                bullets = parse_coaching(text)
                if bullets is None:
                    result['malformed'] += 1
                    retry.append(row)
                elif coaching_leaks(bullets, row['answer'], row['task'], row['prompt'][0]['content']):
                    result['leaked'] += 1
                    retry.append(row)
                else:
                    result['bullets'] = bullets
            pending = retry
        append_jsonl(path, list(results.values()))
        log(run_dir, f'Coaching: {len(done) + start + len(results)}/{len(rows)}')
    coached = read_jsonl(path)
    write_json(run_dir / 'coaching_summary.json', {
        'rows': len(coached), 'usable': sum(r['bullets'] is not None for r in coached),
        'rows_with_leak_attempt': sum(r['leaked'] > 0 for r in coached),
        'rows_with_malformed_attempt': sum(r['malformed'] > 0 for r in coached)})
    timed(run_dir, 'coach', started, rows=len(todo))


def stage_candidates(run_dir):
    rows = eligible_rows(run_dir)
    coaching = {r['id']: r['bullets'] for r in read_jsonl(run_dir / 'coaching.jsonl')}

    def keep(samples):
        best = shortest_correct(samples)
        return {'best_ids': best['token_ids'] if best else None}
    llm = student_engine()
    control = [{**r, 'messages': r['prompt'], 'seed': C['seed'] + 500_000_000 + r['index'] * 1000} for r in rows]
    sample(run_dir, llm, control, C['candidate_samples'], run_dir / 'candidates_control.jsonl', keep)
    coached = [{**r, 'messages': coached_messages(r['prompt'], coaching[r['id']]),
                'seed': C['seed'] + 1_000_000_000 + r['index'] * 1000}
               for r in rows if coaching.get(r['id'])]
    sample(run_dir, llm, coached, C['candidate_samples'], run_dir / 'candidates_coached.jsonl', keep)
    select_targets(run_dir, llm.get_tokenizer(), rows)


def select_targets(run_dir, tokenizer, rows):
    by_id = {r['id']: r for r in rows}
    summary = {}
    for arm in ('control', 'coached'):
        targets, stats = [], {}
        for cand in read_jsonl(run_dir / f'candidates_{arm}.jsonl'):
            row = by_id[cand['id']]
            s = stats.setdefault(row['source'], {'eligible': 0, 'any_correct': 0, 'accepted': 0,
                                                 'ref_tokens': 0, 'target_tokens': 0, 'candidate_correct': 0,
                                                 'candidate_samples': 0})
            s['eligible'] += 1
            s['candidate_samples'] += len(cand['correct'])
            s['candidate_correct'] += sum(cand['correct'])
            best = {'n_tokens': len(cand['best_ids'])} if cand['best_ids'] else None
            s['any_correct'] += best is not None
            if accept(best, row['ref_len']):
                s['accepted'] += 1
                s['ref_tokens'] += row['ref_len']
                s['target_tokens'] += best['n_tokens']
                prompt_ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                                                           add_generation_prompt=True, enable_thinking=True)
                targets.append({'id': row['id'], 'source': row['source'], 'prompt_ids': prompt_ids,
                                'completion_ids': cand['best_ids'], 'ref_len': row['ref_len']})
        for s in stats.values():
            s['token_ratio_accepted'] = s['target_tokens'] / s['ref_tokens'] if s['ref_tokens'] else None
            s['candidate_accuracy'] = s['candidate_correct'] / s['candidate_samples'] if s['candidate_samples'] else None
        (run_dir / f'targets_{arm}.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in targets))
        summary[arm] = {'targets': len(targets), 'per_source': stats}
        log(run_dir, f'{arm}: {len(targets)} targets')
    write_json(run_dir / 'selection_summary.json', summary)


def stage_train(run_dir, arm):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    targets = read_jsonl(run_dir / f'targets_{arm}.jsonl')
    if not targets:
        log(run_dir, f'{arm}: no accepted targets, nothing to train')
        return
    examples = [t['prompt_ids'] + t['completion_ids'] for t in targets]
    prompt_lengths = [len(t['prompt_ids']) for t in targets]
    lengths = [len(e) for e in examples]
    steps = plan_steps(len(examples), C['batch_sequences'], C['epochs'], C['seed'])
    saves = checkpoint_steps(len(steps), C['epochs'] * C['checkpoints_per_epoch'])
    out = run_dir / f'lora_{arm}'
    out.mkdir(exist_ok=True)
    torch.manual_seed(C['seed'])
    model = AutoModelForCausalLM.from_pretrained(C['student'], revision=C['student_revision'],
                                                 dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    model.config.use_cache = False
    lora = C['lora']
    model = get_peft_model(model, LoraConfig(r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
                                             target_modules=lora['target_modules'], bias='none',
                                             task_type='CAUSAL_LM'))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=C['learning_rate'], weight_decay=0.0)
    start = 0
    finished = sorted(int(p.name.split('-')[1]) for p in out.glob('step-*') if (p / 'trainer_state.pt').exists())
    if finished:
        last = out / f'step-{finished[-1]:05d}'
        set_peft_model_state_dict(model, load_file(str(last / 'adapter_model.safetensors')))
        optimizer.load_state_dict(torch.load(last / 'trainer_state.pt')['optimizer'])
        start = finished[-1]
        log(run_dir, f'{arm}: resuming after step {start}')
    log(run_dir, f'{arm}: {len(examples)} examples, {len(steps)} steps, checkpoints at {saves}, '
                 f'{sum(p.numel() for p in trainable):,} trainable parameters')
    pad = 0
    model.train()
    began = time.time()
    for step in range(start, len(steps)):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, len(steps), C['learning_rate'], C['warmup_fraction'])
        indices = steps[step]
        target_tokens = sum(lengths[i] - prompt_lengths[i] for i in indices)
        total_loss = 0.0
        for mb in micro_batches(indices, lengths, C['micro_batch_tokens']):
            width = max(lengths[i] for i in mb)
            ids = torch.full((len(mb), width), pad, dtype=torch.long)
            labels = torch.full((len(mb), width), -100, dtype=torch.long)
            mask = torch.zeros((len(mb), width), dtype=torch.long)
            for row, i in enumerate(mb):
                ids[row, :lengths[i]] = torch.tensor(examples[i])
                labels[row, prompt_lengths[i]:lengths[i]] = torch.tensor(examples[i][prompt_lengths[i]:])
                mask[row, :lengths[i]] = 1
            ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(),
                                   ignore_index=-100, reduction='sum') / target_tokens
            loss.backward()
            total_loss += loss.item()
            del logits, loss
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step + 1 in saves:
            append_jsonl(run_dir / f'train_{arm}.jsonl', [{
                'step': step + 1, 'loss': total_loss, 'grad_norm': norm, 'lr': optimizer.param_groups[0]['lr'],
                'seconds': time.time() - began}])
            log(run_dir, f'{arm} step {step + 1}/{len(steps)} loss {total_loss:.4f}')
        if step + 1 in saves:
            ckpt = out / f'step-{step + 1:05d}'
            model.save_pretrained(ckpt)
            torch.save({'optimizer': optimizer.state_dict(), 'step': step + 1}, ckpt / 'trainer_state.pt')
            log(run_dir, f'{arm}: saved {ckpt.name}')
    timed(run_dir, f'train_{arm}', began, steps=len(steps) - start)


def stage_eval(name, adapter=None, smoke=False):
    """Their evaluate.py, unmodified, into evals/NAME; skips a finished run, clears a crashed one.

    A run where a benchmark failed to grade is moved aside (kept for inspection) and redone.
    """
    import shutil
    import subprocess
    out = STATE / 'evals' / name
    if (out / 'results.json').exists():
        failed = json.loads((out / 'results.json').read_text()).get('failed_benchmarks')
        if not failed:
            print(f'{out} already finished', flush=True)
            return
        aside = out.with_name(f'{name}.failed-{int(time.time())}')
        print(f'{out} has failed benchmarks {sorted(failed)}; moving it to {aside} and rerunning', flush=True)
        out.rename(aside)
    if out.exists():
        shutil.rmtree(out)
    command = [sys.executable, 'evaluate.py', '--questions', str(STATE / 'questions'), '--out', str(out)]
    if adapter:
        command += ['--adapter', str(adapter)]
    if smoke:
        command.append('--smoke')
    started = time.time()
    subprocess.run(command, cwd=TC, check=True)
    write_json(out / 'thinkingcap_eval.json', {'adapter': str(adapter) if adapter else None, 'smoke': smoke,
                                               'seconds': time.time() - started})


def stage_compare(base='base'):
    """compare.py of every finished full eval against the base run, plus one summary file."""
    import subprocess
    evals = STATE / 'evals'
    summary = {}
    for out in sorted(evals.iterdir()):
        if out.name == base or '.failed-' in out.name or not (out / 'results.json').exists():
            continue
        if json.loads((out / 'results.json').read_text()).get('smoke'):
            continue
        report = subprocess.run([sys.executable, 'compare.py', str(evals / base), str(out),
                                 '--out', str(out / 'compare.json')], cwd=TC, capture_output=True, text=True)
        (out / 'compare.txt').write_text(report.stdout + report.stderr)
        if report.returncode:
            summary[out.name] = {'error': (report.stdout + report.stderr).strip()[-2000:]}
            continue
        comparison = json.loads((out / 'compare.json').read_text())
        summary[out.name] = {'average': comparison['average'], 'per_dataset': {
            d: {'accuracy_delta': r['accuracy_delta'], 'token_reduction': r['token_reduction']}
            for d, r in comparison['results'].items()}}
        print(f'== {out.name}\n{report.stdout}', flush=True)
    write_json(evals / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['setup', 'baseline', 'coach', 'candidates', 'train', 'eval', 'compare'])
    parser.add_argument('--run', help='Run name under runs/ (training stages)')
    parser.add_argument('--arm', choices=['coached', 'control'])
    parser.add_argument('--limit', type=int, help='Rows per source, overriding config (smoke runs)')
    parser.add_argument('--name', help='Eval name under evals/')
    parser.add_argument('--adapter', help='LoRA directory to evaluate')
    parser.add_argument('--smoke', action='store_true', help="Eval on their 30-question smoke subset")
    arguments = parser.parse_args()
    if arguments.stage == 'eval':
        return stage_eval(arguments.name, arguments.adapter, arguments.smoke)
    if arguments.stage == 'compare':
        return stage_compare()
    run_dir = STATE / 'runs' / arguments.run
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = run_dir / 'config.json'
    if arguments.limit is not None:
        C['limit_per_source'] = arguments.limit
    elif saved.exists():
        C['limit_per_source'] = json.loads(saved.read_text())['limit_per_source']
    if saved.exists() and json.loads(saved.read_text()) != C:
        raise SystemExit(f'{saved} differs from the current config; use a new --run name')
    write_json(saved, C)
    if arguments.stage == 'train':
        stage_train(run_dir, arguments.arm)
    else:
        globals()[f'stage_{arguments.stage}'](run_dir)


if __name__ == '__main__':
    main()
