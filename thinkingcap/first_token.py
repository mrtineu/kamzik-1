"""Forced first thinking token (Liu et al. 2026, arXiv:2610.34350) on the ThinkingCap benchmarks.

Runs their evaluate.py with the reply pre-filled as `<think>\\nWORD`, so the student's first
reasoning token is WORD instead of its natural choice (Qwen3-0.6B opens with "Okay" 99.9% of
the time). The forced tokens are put back at the front of every record, so reasoning-token counts
and grading see the whole reply, and each generation cap shrinks by their length so the total
stays within the recipe's budget. Run from the ThinkingCap-Test checkout:

    python first_token.py eval --force To -- --questions /vol/questions --out /vol/evals/force-To
    python first_token.py compare /vol/evals/base /vol/evals/force-To --out compare.json

compare is their compare.py minus the refusal to compare two runs of the same weights.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())


def forced_ids(tokenizer, word):
    ids = tokenizer.encode('<think>\n' + word, add_special_tokens=False)
    if ids[0] != tokenizer.convert_tokens_to_ids('<think>'):
        raise ValueError(f'<think> did not tokenize as one token: {ids}')
    return ids


def run_eval(word, evaluate_args):
    import evaluate

    build_jobs, to_record, token_cap = evaluate.build_jobs, evaluate.to_record, evaluate.token_cap
    forced = []

    def forced_build_jobs(questions, tokenizer, arguments):
        forced[:] = forced_ids(tokenizer, word)
        print(f'forcing {word!r} as {forced}', flush=True)
        jobs = build_jobs(questions, tokenizer, arguments)
        for job in jobs:
            job['ids'] = job['ids'] + forced
        return jobs

    def forced_to_record(job, output, lora, tokenizer, eos_ids, open_id, close_id):
        result = output.outputs[0]
        whole = SimpleNamespace(lora_request=output.lora_request,
                                outputs=[SimpleNamespace(token_ids=forced + list(result.token_ids))])
        return {**to_record(job, whole, lora, tokenizer, eos_ids, open_id, close_id), 'forced': word}

    evaluate.build_jobs = forced_build_jobs
    evaluate.to_record = forced_to_record
    # generation gets the cap minus the forced tokens; engine sizing and results.json keep the recipe's
    evaluate.token_cap = lambda dataset, arguments: token_cap(dataset, arguments) - len(forced)
    evaluate.caps_in_use = lambda questions, arguments: {
        d: token_cap(d, arguments) for d in sorted({q['dataset'] for q in questions})}
    sys.argv = ['evaluate.py', *evaluate_args]
    evaluate.main()
    out = Path(evaluate_args[evaluate_args.index('--out') + 1])
    (out / 'forced.json').write_text(json.dumps({'word': word, 'ids': forced}) + '\n')


def run_compare(base, candidate, out):
    from utils.eval import compare as tc_compare

    check = tc_compare.check

    def check_ignoring_weights(base_meta, candidate_meta):
        return check(base_meta, {**candidate_meta, 'model': f"{candidate_meta['model']} (forced first token)"})

    tc_compare.check = check_ignoring_weights
    comparison = tc_compare.compare(base, candidate)
    if out:
        Path(out).write_text(json.dumps(comparison, indent=2) + '\n')
    print(tc_compare.report(comparison))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    ev = sub.add_parser('eval')
    ev.add_argument('--force', required=True, help='First reasoning word, e.g. To, First, Okay')
    ev.add_argument('evaluate_args', nargs=argparse.REMAINDER, help='-- then evaluate.py arguments')
    cmp = sub.add_parser('compare')
    cmp.add_argument('base')
    cmp.add_argument('candidate')
    cmp.add_argument('--out')
    arguments = parser.parse_args()
    if arguments.command == 'eval':
        rest = arguments.evaluate_args
        run_eval(arguments.force, rest[1:] if rest[:1] == ['--'] else rest)
    else:
        run_compare(arguments.base, arguments.candidate, arguments.out)


if __name__ == '__main__':
    main()
