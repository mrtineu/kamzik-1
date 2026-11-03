"""Where in Qwen3-0.6B does the thinking decision live? Forward hooks on every decoder layer, two reads.

Replays real traces from an eval's records.jsonl (default: evals/base, 8 samples per prompt) and
captures each layer's output (the residual stream) with register_forward_hook. Then:

  logit lens   each layer's state through the final norm + lm_head, scored at three kinds of position:
               the one that emits </think> (the stop), the ones that emit "Wait" (the restart) and all
               thinking positions (agreement with the model's own top-1, the generic baseline).
               A layer where the stop is top-1 far more often than the baseline curve says the stop
               decision is readable there. stop_contrast (</think> logprob at the stop minus its mean over the
               trace's other positions) is the calibration-free version: does this layer tell "stop now"
               from "keep going", even before </think> wins outright?
  plan probe   the state at the <think> token, before any reasoning is written, against the log of the
               prompt's mean thinking length over its 8 samples. Nested-CV ridge, held-out Pearson r,
               twice: pooled, and within dataset (features and lengths centred per dataset on the
               training fold), because task identity alone predicts length (svamp short, mbpp long).
               The within-dataset r is the one that says the layer plans length for this question.
               Also saves per-layer long-minus-short difference-of-means directions (within dataset),
               the kind of vector the halt-vector / thinking-speed papers steer with.

Run from the ThinkingCap-Test checkout (layer_probe_app.py does this on Modal):

    python layer_probe.py --questions /vol/questions --records /vol/evals/base/records.jsonl --out /vol/probes/base
    python layer_probe.py ... --model /vol/runs/full/arms/extra-mlp-64/step-00170/model --out /vol/probes/...

Writes OUT/probe.json (per-layer table) and OUT/directions.pt.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())

HERE = Path(__file__).resolve().parent
C = json.loads((HERE / 'config.json').read_text())
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def chosen_traces(questions, records, per_dataset, max_thinking, seed):
    """Per dataset, up to per_dataset prompts with one clean trace each (closed </think>, not truncated)."""
    by_id = defaultdict(list)
    for record in records:
        by_id[record['id']].append(record)
    pools = defaultdict(list)
    for row in questions:
        samples = by_id.get(row['id'])
        if not samples:
            continue
        clean = [r for r in samples if not r['truncated'] and r['boundary_error'] is None
                 and '</think>' in r['text'] and r['reasoning_tokens'] <= max_thinking]
        if clean:
            mean = sum(r['reasoning_tokens'] for r in samples) / len(samples)
            pools[row['dataset']].append({'row': row, 'trace': min(clean, key=lambda r: r['sample_index']),
                                          'mean_reasoning': mean})
    rng = random.Random(seed)
    chosen = []
    for dataset in sorted(pools):
        pool = pools[dataset]
        rng.shuffle(pool)
        chosen += pool[:per_dataset]
    return chosen


def load_model(path):
    import torch
    if path:
        import arms
        return arms.load_export(path, device='cuda')
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(C['student'], revision=C['student_revision'], dtype=torch.bfloat16,
                                                attn_implementation='sdpa').cuda().eval()


def attach_capture(model):
    """A forward hook per decoder layer that keeps its output hidden state (the residual stream)."""
    captured = {}

    def keep(index):
        def hook(module, args, output):
            captured[index] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook
    handles = [layer.register_forward_hook(keep(i)) for i, layer in enumerate(model.model.layers)]
    return captured, handles


def replay(model, tokenizer, chosen, wait_ids, chunk):
    """One forward pass per trace; logit-lens tallies per layer and the <think>-token state per layer."""
    import torch
    open_id, close_id = (tokenizer.convert_tokens_to_ids(t) for t in ('<think>', '</think>'))
    captured, handles = attach_capture(model)
    layers = len(model.model.layers)
    norm, head = model.model.norm, model.get_output_embeddings()
    tally = {key: torch.zeros(layers, dtype=torch.float64) for key in (
        'stop_top1', 'stop_logprob', 'stop_contrast', 'wait_top1', 'wait_logprob', 'agree_top1', 'wait_n', 'agree_n')}
    plan_states, kept, worst_gap = [], [], 0.0
    for n, item in enumerate(chosen):
        prompt = tokenizer.apply_chat_template(item['row']['messages'], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=True)
        completion = tokenizer.encode(item['trace']['text'], add_special_tokens=False)
        if completion[:1] != [open_id] or close_id not in completion:
            continue
        ids = prompt + completion
        think = len(prompt)                                 # position of <think>
        stop = think + completion.index(close_id) - 1      # position whose next token is </think>
        with torch.no_grad():
            final_logits = model(input_ids=torch.tensor([ids], device='cuda')).logits[0].float()
            # thinking span: from <think> to the position that emits </think>
            positions = torch.arange(think, stop + 1, device='cuda')
            targets = torch.tensor(ids[think + 1:stop + 2], device='cuda')
            final_top = final_logits[positions].argmax(-1)
            is_wait = torch.isin(targets, torch.tensor(wait_ids, device='cuda'))
            plan_states.append(torch.stack([captured[i][0, think].float().cpu() for i in range(layers)]))
            for i in range(layers):
                h = captured[i][0, positions]
                top, logprob, close = [], [], []
                for s in range(0, len(positions), chunk):   # lm_head over 151k tokens, a chunk at a time
                    lp = torch.log_softmax(head(norm(h[s:s + chunk])).float(), -1)
                    top.append(lp.argmax(-1))
                    logprob.append(lp.gather(-1, targets[s:s + chunk, None])[:, 0])
                    close.append(lp[:, close_id])
                top, logprob, close = torch.cat(top), torch.cat(logprob), torch.cat(close)
                if i == layers - 1:                         # the last layer's lens must be the model itself
                    worst_gap = max(worst_gap, (top != final_top).float().mean().item())
                tally['stop_top1'][i] += float(top[-1] == close_id)
                tally['stop_logprob'][i] += logprob[-1].item()
                tally['stop_contrast'][i] += (close[-1] - close[:-1].mean()).item()
                tally['wait_top1'][i] += (top[is_wait] == targets[is_wait]).sum().item()
                tally['wait_logprob'][i] += logprob[is_wait].sum().item()
                tally['wait_n'][i] += is_wait.sum().item()
                tally['agree_top1'][i] += (top == final_top).sum().item()
                tally['agree_n'][i] += len(positions)
        kept.append(item)
        captured.clear()
        if (n + 1) % 50 == 0:
            print(f'replayed {n + 1}/{len(chosen)}', flush=True)
    for handle in handles:
        handle.remove()
    traces = len(kept)
    lens = [{'layer': i,
             'stop_top1': tally['stop_top1'][i].item() / traces,
             'stop_logprob': tally['stop_logprob'][i].item() / traces,
             'stop_contrast': tally['stop_contrast'][i].item() / traces,
             'wait_top1': tally['wait_top1'][i].item() / max(1, tally['wait_n'][i].item()),
             'wait_logprob': tally['wait_logprob'][i].item() / max(1, tally['wait_n'][i].item()),
             'agree_top1': tally['agree_top1'][i].item() / tally['agree_n'][i].item()} for i in range(layers)]
    return kept, torch.stack(plan_states), lens, worst_gap


def ridge_predict(train_x, train_y, test_x, alpha):
    """Dual-form ridge (n << hidden) on standardized features."""
    import torch
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    a, b = (train_x - mean) / std, (test_x - mean) / std
    y_mean = train_y.mean()
    gram = a @ a.T
    coef = torch.linalg.solve(gram + alpha * torch.eye(len(a), dtype=a.dtype), train_y - y_mean)
    return (b @ a.T) @ coef + y_mean


def centred(x, y, groups, train):
    """Subtract per-dataset means estimated on the training rows only."""
    x, y = x.clone(), y.clone()
    for g in set(groups):
        rows = [i for i, h in enumerate(groups) if h == g]
        fit = [i for i in rows if train[i]] or rows
        x[rows] -= x[fit].mean(0)
        y[rows] -= y[fit].mean()
    return x, y


def cv_r(x, y, groups, within, folds, seed):
    """Held-out Pearson r of ridge predictions, alpha picked by inner CV on each training fold."""
    import torch
    order = list(range(len(y)))
    random.Random(seed).shuffle(order)
    fold_of = {i: k % folds for k, i in enumerate(order)}
    predictions, truths = [], []
    for k in range(folds):
        train = [fold_of[i] != k for i in range(len(y))]
        xs, ys = centred(x, y, groups, train) if within else (x, y)
        tr = [i for i in range(len(y)) if train[i]]
        te = [i for i in range(len(y)) if not train[i]]

        def inner_error(alpha):
            error = 0.0
            for j in range(4):
                a = [i for n, i in enumerate(tr) if n % 4 != j]
                b = [i for n, i in enumerate(tr) if n % 4 == j]
                error += ((ridge_predict(xs[a], ys[a], xs[b], alpha) - ys[b]) ** 2).sum().item()
            return error
        alpha = min(ALPHAS, key=inner_error)
        predictions.append(ridge_predict(xs[tr], ys[tr], xs[te], alpha))
        truths.append(ys[te])
    p, t = torch.cat(predictions), torch.cat(truths)
    return torch.corrcoef(torch.stack([p, t]))[0, 1].item()


def plan_probe(states, kept, folds, seed):
    import torch
    y = torch.tensor([math.log(item['mean_reasoning']) for item in kept], dtype=torch.float64)
    groups = [item['row']['dataset'] for item in kept]
    rows, directions = [], []
    for i in range(states.shape[1]):
        x = states[:, i].double()
        # within each dataset: mean state of its longer-thinking half minus its shorter half
        diffs = []
        for g in sorted(set(groups)):
            members = sorted((j for j, h in enumerate(groups) if h == g), key=lambda j: y[j].item())
            half = len(members) // 2
            if half:
                diffs.append(x[members[-half:]].mean(0) - x[members[:half]].mean(0))
        direction = torch.stack(diffs).mean(0)
        directions.append(direction.float())
        rows.append({'layer': i, 'r_pooled': cv_r(x, y, groups, False, folds, seed),
                     'r_within_dataset': cv_r(x, y, groups, True, folds, seed),
                     'direction_norm': direction.norm().item(),
                     'state_norm': x.norm(dim=1).mean().item()})
        print(f"layer {i:2d}  r pooled {rows[-1]['r_pooled']:.3f}  within {rows[-1]['r_within_dataset']:.3f}",
              flush=True)
    return rows, torch.stack(directions)


def main():
    import torch
    from transformers import AutoTokenizer
    from utils.eval.benchmarks import paths, read_jsonl
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--records', required=True, help="an eval's records.jsonl, whose traces are replayed")
    parser.add_argument('--model', default='', help='an arm export directory; default the base student')
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-dataset', type=int, default=100)
    parser.add_argument('--max-thinking', type=int, default=4096)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(C['student'], revision=C['student_revision'])
    wait_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ('Wait', ' Wait')})
    questions = read_jsonl(paths(Path(arguments.questions))['questions'])
    chosen = chosen_traces(questions, read_jsonl(arguments.records), arguments.per_dataset,
                           arguments.max_thinking, arguments.seed)
    print(f'{len(chosen)} traces from {len(questions)} questions', flush=True)
    model = load_model(arguments.model)
    kept, states, lens, worst_gap = replay(model, tokenizer, chosen, wait_ids, arguments.chunk)
    if worst_gap > 0.01:
        raise SystemExit(f'last-layer lens disagrees with the model on {worst_gap:.1%} of a trace; hooks are wrong')
    plan, directions = plan_probe(states, kept, arguments.folds, arguments.seed)
    report = {'model': arguments.model or f"{C['student']}@{C['student_revision']}",
              'records': arguments.records, 'traces': len(kept),
              'per_dataset': {g: sum(item['row']['dataset'] == g for item in kept)
                              for g in sorted({item['row']['dataset'] for item in kept})},
              'wait_ids': wait_ids, 'last_layer_lens_mismatch': worst_gap,
              'layers': [{**a, **b} for a, b in zip(lens, plan)]}
    (out / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')
    torch.save({'directions': directions, 'ids': [item['row']['id'] for item in kept]}, out / 'directions.pt')
    print(f"{'layer':>5} {'stop@1':>7} {'stopΔ':>7} {'wait@1':>7} {'agree@1':>8} {'plan r':>7} {'within':>7}")
    for row in report['layers']:
        print(f"{row['layer']:5d} {row['stop_top1']:7.3f} {row['stop_contrast']:7.2f} {row['wait_top1']:7.3f} {row['agree_top1']:8.3f} "
              f"{row['r_pooled']:7.3f} {row['r_within_dataset']:7.3f}")


if __name__ == '__main__':
    main()
