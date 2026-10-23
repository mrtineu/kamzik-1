"""Auditable parsing, adapter insertion, and metrics for the pilot."""
import json
import math
import re
from fractions import Fraction
from pathlib import Path
import torch
from torch import nn

FORMAT = r"Solve the problem. Put your final numeric answer in \boxed{...}."
CONCISE = "Use the minimum sufficient reasoning. Keep essential equations and intermediate results; omit narration, repeated plans, and unnecessary checks. Preserve needed steps on difficult problems."
QA_CONCISE = "Use the minimum sufficient reasoning. Keep the facts and inferences needed to select the best option; omit repeated plans, narration, and unnecessary checks."


def format_instruction(example):
    if example.get('answer_type') == 'choice':
        labels = ', '.join(example['choice_labels'])
        return rf"Answer the question by choosing the best option. Put only its label ({labels}) in \boxed{{...}}."
    if example.get('answer_type') == 'math':
        return r"Solve the problem. Put your final answer in \boxed{...}."
    return FORMAT


def concise_instruction(example):
    return QA_CONCISE if example.get('answer_type') == 'choice' else CONCISE


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()


def numeric(value):
    value = value.strip().replace(',', '').replace('$', '').replace(' ', '')
    value = re.sub(r'\\(?:d?frac)\{(-?[\d.]+)\}\{(-?[\d.]+)\}', r'\1/\2', value)
    value = value.rstrip('.')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+(?:\.\d*)?)?', value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_math(value):
    """Conservative exact match plus rational equivalence; no symbolic evaluation."""
    value = value.strip().strip('$')
    value = re.sub(r'\\+(dfrac|tfrac|frac|sqrt|pi|text|mathrm|begin|end)\b', r'\\\1', value)
    value = value.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    for decoration in ['\\left', '\\right', '\\!', '\\,', '\\;', '\\ ', '\\(', '\\)', '\\[', '\\]']:
        value = value.replace(decoration, '')
    # A math answer like "1,2" is a list, not the integer 12.
    grouped_number = re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value)
    number = numeric(value) if ',' not in value or grouped_number else None
    if number is not None:
        return str(number)
    return re.sub(r'\s+', '', value) or None


def boxed_values(text):
    """Read nested LaTeX braces without executing model output."""
    values = []
    for match in re.finditer(r'\\boxed\{', text):
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            if text[end] == '{': depth += 1
            elif text[end] == '}': depth -= 1
            if depth == 0:
                values.append(text[start:end])
                break
    return values


def extract_answer(text, answer_type='numeric', choice_labels=None):
    # Only the final answer section counts. Never use an incidental last number in reasoning.
    if '</think>' not in text:
        return None
    final = text.rsplit('</think>', 1)[1]
    if answer_type in ('choice', 'math'):
        boxes = boxed_values(final)
        marked = re.findall(r'(?:####|final answer\s*[:=])\s*([^\n]+)', final, re.IGNORECASE)
        raw = boxes[-1] if boxes else marked[-1].strip() if marked else None
        if raw is None:
            return None
        if answer_type == 'math':
            return normalize_math(raw)
        label = raw.strip().strip('$').strip()
        label = re.sub(r'^\\(?:text|mathrm)\{([^{}]+)\}$', r'\1', label)
        label = label.strip().rstrip('.').strip('()').upper()
        labels = choice_labels or []
        return label if label in labels else None
    if answer_type != 'numeric':
        raise ValueError(f'Unknown answer type: {answer_type}')
    boxes = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', final)
    if boxes:
        return numeric(boxes[-1])
    explicit = re.findall(r'(?:####|[Ff]inal answer\s*[:=])\s*\$?\s*([+-]?[\d,]+(?:\.\d+)?(?:/\d+)?)', final)
    return numeric(explicit[-1]) if explicit else None


def score_tokens(ids, tokenizer, gold, cap, *, answer_type='numeric', choice_labels=None, accepted_answers=None):
    end = tokenizer.convert_tokens_to_ids('</think>')
    eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    stop_ids = set(eos_ids + [tokenizer.convert_tokens_to_ids('<|im_end|>')])
    ids = list(ids)
    stop_at = next((i for i, x in enumerate(ids) if x in stop_ids), len(ids))
    finished = stop_at < len(ids)
    ids = ids[:stop_at]
    think_end = ids.index(end) if end in ids else None
    text = tokenizer.decode(ids, skip_special_tokens=False)
    answer = extract_answer(text, answer_type, choice_labels)
    references = accepted_answers or [gold]
    if answer_type == 'choice':
        matches = answer in references
    elif answer_type == 'math':
        matches = answer in [normalize_math(value) for value in references]
    else:
        matches = answer in [numeric(value) for value in references]
    return dict(text=text, correct=bool(finished and answer is not None and matches),
                parsed_answer=str(answer) if answer is not None else None,
                thinking_tokens=think_end if think_end is not None else len(ids),
                final_tokens=len(ids)-think_end-1 if think_end is not None else 0,
                total_tokens=len(ids), finished=finished, thinking_closed=think_end is not None,
                truncated=not finished and len(ids) >= cap, cap=cap)


class ReasoningMLP(nn.Module):
    """Identity-initialized residual adapter; fp32 parameters, model-dtype interface."""
    def __init__(self, hidden, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck, bias=False, dtype=torch.float32)
        self.up = nn.Linear(bottleneck, hidden, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return hidden + self.up(torch.nn.functional.silu(self.down(hidden.float()))).to(hidden.dtype)


def attach_adapter(model, layer, bottleneck):
    model.requires_grad_(False)
    adapter = ReasoningMLP(model.config.hidden_size, bottleneck).to(model.device)
    model.add_module('reasoning_adapter', adapter)
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (adapter(output[0]),) + output[1:]
        return adapter(output)
    handle = model.model.layers[layer].register_forward_hook(hook)
    return adapter, handle


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.96
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,center-half),min(1,center+half)]


def summarize(rows):
    import statistics as s
    if not rows:
        return {'n': 0}
    correct = [r for r in rows if r['correct']]
    closed = [r for r in rows if r['thinking_closed']]
    return dict(n=len(rows), accuracy=sum(r['correct'] for r in rows)/len(rows),
        accuracy_wilson_95=wilson(len(correct),len(rows)),
        mean_thinking_tokens=s.mean(r['thinking_tokens'] for r in rows),
        median_thinking_tokens=s.median(r['thinking_tokens'] for r in rows),
        mean_total_tokens=s.mean(r['total_tokens'] for r in rows),
        correct_only_mean_thinking_tokens=s.mean(r['thinking_tokens'] for r in correct) if correct else None,
        closed_only_mean_thinking_tokens=s.mean(r['thinking_tokens'] for r in closed) if closed else None,
        truncated=sum(r['truncated'] for r in rows), unfinished=sum(not r['finished'] for r in rows),
        missing_answer=sum(r['parsed_answer'] is None for r in rows),
        mean_seconds=s.mean(r.get('seconds',0) for r in rows),
        total_compute_seconds=sum(r.get('total_compute_seconds',
            r.get('seconds',0)+r.get('initial_attempt',{}).get('seconds',0)) for r in rows))
