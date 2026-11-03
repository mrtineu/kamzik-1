"""Forced-first-token evals (first_token.py) on Modal, one H100 per word, compared against evals/base.

    .modal-venv/bin/modal run thinkingcap/first_token_app.py --words To --smoke
    .modal-venv/bin/modal run --detach thinkingcap/first_token_app.py --words Okay,To,First
    .modal-venv/bin/modal run --detach thinkingcap/first_token_app.py --words First --arms extra-mlp-64/step-00170

--arms forces the word on top of arm exports (runs/full/arms/ARM/step-N/model) instead of the base model.
Results: evals/force-WORD[-on-ARM-step-N]/{records,results,compare}.json and evals/summary_first_token.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

from modal_app import STATE, image, volume

image = image.add_local_python_source('modal_app')  # this module imports it inside the container too

app = modal.App('thinkingcap-first-token')
SCRIPT = '/opt/thinkingcap/first_token.py'


def out_dir(word, smoke, arm=''):
    on = f"-on-{arm.replace('/', '-')}" if arm else ''
    return Path(STATE) / 'evals' / f"force-{word}{on}{'-smoke' if smoke else ''}"


def arm_model(arm):
    return Path(STATE) / 'runs' / 'full' / 'arms' / arm / 'model'


@app.function(image=image, gpu='H100', cpu=4, memory=32768, volumes={STATE: volume}, timeout=4 * 60 * 60)
def force_eval(word: str, smoke: bool, arm: str = ''):
    volume.reload()
    out = out_dir(word, smoke, arm)
    if (out / 'forced.json').exists():
        print(f'{out} already finished', flush=True)
        return
    subprocess.run(['rm', '-rf', str(out)], check=True)
    command = [sys.executable, SCRIPT, 'eval', '--force', word, '--',
               '--questions', f'{STATE}/questions', '--out', str(out)] + (['--smoke'] if smoke else [])
    if arm:
        command += ['--model', str(arm_model(arm))]
    try:
        subprocess.run(command, cwd='/opt/tc', check=True)
    finally:
        volume.commit()
    if smoke:
        for line in (out / 'records.jsonl').read_text().splitlines()[:6]:
            record = json.loads(line)
            print(record['dataset'], record['reasoning_tokens'], record.get('correct'),
                  repr(record['text'][:160]), flush=True)


@app.function(image=image, cpu=4, memory=16384, volumes={STATE: volume}, timeout=60 * 60)
def force_compare(runs: list[tuple[str, str]]) -> str:
    volume.reload()
    summary_path = Path(STATE) / 'evals' / 'summary_first_token.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    reports = []
    for word, arm in runs:
        out = out_dir(word, False, arm)
        report = subprocess.run([sys.executable, SCRIPT, 'compare', f'{STATE}/evals/base', str(out),
                                 '--out', str(out / 'compare.json')], cwd='/opt/tc', capture_output=True, text=True)
        (out / 'compare.txt').write_text(report.stdout + report.stderr)
        reports.append(f'== {out.name}\n{report.stdout}{report.stderr[-2000:] if report.returncode else ""}')
        if report.returncode:
            summary[out.name] = {'error': (report.stdout + report.stderr).strip()[-2000:]}
            continue
        comparison = json.loads((out / 'compare.json').read_text())
        summary[out.name] = {'average': comparison['average'], 'per_dataset': {
            d: {'accuracy_delta': r['accuracy_delta'], 'token_reduction': r['token_reduction']}
            for d, r in comparison['results'].items()}}
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    volume.commit()
    return '\n'.join(reports)


@app.function(image=image, timeout=6 * 60 * 60)
def run_all(runs: list[tuple[str, str]], smoke: bool) -> str:
    list(force_eval.starmap([(word, smoke, arm) for word, arm in runs]))
    return 'smoke done' if smoke else force_compare.remote(runs)


@app.local_entrypoint()
def main(words: str, arms: str = '', smoke: bool = False):
    split = lambda text: [part.strip() for part in text.split(',') if part.strip()]
    print(run_all.remote([(word, arm) for arm in split(arms) or [''] for word in split(words)], smoke))
