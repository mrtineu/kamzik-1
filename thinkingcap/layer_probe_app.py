"""layer_probe.py on Modal: one H100 per model, base traces replayed through each.

    .modal-venv/bin/modal run thinkingcap/layer_probe_app.py --per-dataset 8           # quick check
    .modal-venv/bin/modal run thinkingcap/layer_probe_app.py
    .modal-venv/bin/modal run thinkingcap/layer_probe_app.py --arms extra-mlp-64/step-00170

Every model replays the same base-model traces (evals/base/records.jsonl), so an arm's table reads as
"what this arm does to the base model's thinking", layer by layer. Results: probes/NAME/probe.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

from modal_app import STATE, image, volume

image = image.add_local_python_source('modal_app')

app = modal.App('thinkingcap-layer-probe')
SCRIPT = '/opt/thinkingcap/layer_probe.py'


def probe_name(arm, per_dataset):
    return (arm.replace('/', '-') if arm else 'base') + ('' if per_dataset == 100 else f'-n{per_dataset}')


@app.function(image=image, gpu='H100', cpu=4, memory=32768, volumes={STATE: volume}, timeout=2 * 60 * 60)
def probe(arm: str, per_dataset: int) -> str:
    volume.reload()
    out = Path(STATE) / 'probes' / probe_name(arm, per_dataset)
    command = [sys.executable, SCRIPT, '--questions', f'{STATE}/questions',
               '--records', f'{STATE}/evals/base/records.jsonl', '--out', str(out),
               '--per-dataset', str(per_dataset)]
    if arm:
        command += ['--model', f'{STATE}/runs/full/arms/{arm}/model']
    try:
        subprocess.run(command, cwd='/opt/tc', check=True)
    finally:
        volume.commit()
    return json.dumps(json.loads((out / 'probe.json').read_text()))


@app.local_entrypoint()
def main(arms: str = '', base: bool = True, per_dataset: int = 100):
    targets = ([''] if base else []) + [a.strip() for a in arms.split(',') if a.strip()]
    local = Path(__file__).resolve().parent / 'results' / 'probes'
    for arm, report in zip(targets, probe.map(targets, kwargs={'per_dataset': per_dataset})):
        name = probe_name(arm, per_dataset)
        (local / name).mkdir(parents=True, exist_ok=True)
        (local / name / 'probe.json').write_text(json.dumps(json.loads(report), indent=2) + '\n')
        print(f'== {name}: {local / name / "probe.json"}')
