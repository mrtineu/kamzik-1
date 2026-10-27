"""Run the ThinkingCap compression pipeline on Modal (H100), staged.

    .modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage smoke
    .modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage baseline --run full

Long runs: `modal deploy thinkingcap/modal_app.py`, then add --deployed to `modal run` (no --detach):
the orchestrator is spawned on the deployed app and survives the laptop shutting down.

Stages: smoke | all | setup | eval-base | baseline | coach | candidates | train | eval-ckpts | compare.
--detach plus a remote orchestrator means closing the laptop does not stop a stage.
Everything lands on the `thinkingcap` volume: runs/NAME, evals/NAME, data/, questions/, hf/.
"""
from pathlib import Path
import re

import modal

HERE = Path(__file__).resolve().parent
TC_COMMIT = 'bd52d8d'
STATE = '/vol'
SMOKE_LIMIT = 16
app = modal.App('thinkingcap')
volume = modal.Volume.from_name('thinkingcap', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'build-essential', 'bubblewrap')
    .uv_pip_install(requirements=[str(HERE / 'requirements.lock')])
    .run_commands(f'git clone https://github.com/BottleCapAI/ThinkingCap-Test.git /opt/tc && '
                  f'cd /opt/tc && git checkout {TC_COMMIT}')
    # IFEval's grader needs punkt in every container that grades, not only where questions were built
    .run_commands('python -c "import nltk; [nltk.download(p, download_dir=\'/usr/local/share/nltk_data\', '
                  'raise_on_error=True) for p in (\'punkt_tab\', \'punkt\')]"')
    .env({'HF_HOME': f'{STATE}/hf', 'TC_DIR': '/opt/tc', 'TC_STATE': STATE,
          'VLLM_USE_FLASHINFER_SAMPLER': '0', 'TOKENIZERS_PARALLELISM': 'false',
          'HF_HUB_DISABLE_XET': '1'})
    .add_local_dir(str(HERE), '/opt/thinkingcap',
                   ignore=['__pycache__', '*.pyc', 'requirements.lock'])
)


def run_pipeline(args):
    """One pipeline invocation in a subprocess (frees the GPU between stages), committing as it goes."""
    import subprocess
    import sys
    import threading

    volume.reload()
    stop = threading.Event()

    def commit_periodically():
        while not stop.wait(300):
            try:
                volume.commit()
            except Exception as error:  # a failed checkpoint commit must not kill the stage
                print(f'volume commit failed: {error}', flush=True)

    thread = threading.Thread(target=commit_periodically, daemon=True)
    thread.start()
    print('$ pipeline.py ' + ' '.join(args), flush=True)
    try:
        result = subprocess.run([sys.executable, '/opt/thinkingcap/pipeline.py', *args])
    finally:
        stop.set()
        thread.join()
        volume.commit()
    if result.returncode:
        raise RuntimeError(f'pipeline.py {" ".join(args)} exited with {result.returncode}')


@app.function(image=image, gpu='H100', cpu=4, memory=32768, volumes={STATE: volume},
              timeout=24 * 60 * 60, max_containers=1)
def gpu_step(args: list[str]):
    run_pipeline(args)


@app.function(image=image, cpu=8, memory=32768, volumes={STATE: volume}, timeout=6 * 60 * 60,
              max_containers=1)
def cpu_step(args: list[str]):
    run_pipeline(args)


@app.function(image=image, volumes={STATE: volume}, timeout=60)
def list_checkpoints(run: str) -> list[str]:
    volume.reload()
    root = Path(STATE) / 'runs' / run
    return sorted(str(p) for p in root.glob('lora_*/step-*') if (p / 'adapter_config.json').exists())


def eval_name(run, checkpoint):
    path = Path(checkpoint)
    return f"{run}-{path.parent.name.removeprefix('lora_')}-{path.name}"


@app.function(image=image, timeout=24 * 60 * 60, max_containers=1)
def orchestrate(steps: list[tuple[str, list[str]]], eval_run: str = '', eval_smoke: bool = False):
    """Run steps in order on the right hardware; ('evals', []) expands to every saved checkpoint."""
    for kind, args in steps:
        if kind == 'evals':
            for checkpoint in list_checkpoints.remote(eval_run):
                gpu_step.remote(['eval', '--name', eval_name(eval_run, checkpoint), '--adapter', checkpoint]
                                + (['--smoke'] if eval_smoke else []))
        else:
            (gpu_step if kind == 'gpu' else cpu_step).remote(args)


@app.local_entrypoint()
def main(stage: str, run: str = 'full', arm: str = '', limit: int = 0, deployed: bool = False):
    """--deployed spawns on the `modal deploy`ed app, so nothing depends on this laptop staying up."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run):
        raise ValueError('run must be a simple name')
    r = ['--run', run]
    plans = {
        'setup': [('cpu', ['setup', *r, *(['--limit', str(limit)] if limit else [])])],
        'eval-base': [('gpu', ['eval', '--name', 'base'])],
        'baseline': [('gpu', ['baseline', *r])],
        'coach': [('gpu', ['coach', *r])],
        'candidates': [('gpu', ['candidates', *r])],
        'train': [('gpu', ['train', *r, '--arm', a]) for a in ([arm] if arm else ['control', 'coached'])],
        'eval-ckpts': [('evals', [])],
        'compare': [('cpu', ['compare'])],
    }
    plans['all'] = [step for name in ('setup', 'eval-base', 'baseline', 'coach', 'candidates', 'train',
                                      'eval-ckpts', 'compare') for step in plans[name]]
    if stage == 'smoke':
        s = ['--run', 'smoke']
        steps = [('cpu', ['setup', *s, '--limit', str(SMOKE_LIMIT)]),
                 ('gpu', ['eval', '--name', 'smoke-base', '--smoke']),
                 ('gpu', ['baseline', *s]), ('gpu', ['coach', *s]), ('gpu', ['candidates', *s]),
                 ('gpu', ['train', *s, '--arm', 'control']), ('gpu', ['train', *s, '--arm', 'coached']),
                 ('evals', [])]
        call = (steps, 'smoke', True)
    elif stage in plans:
        call = (plans[stage], run, False)
    else:
        raise ValueError(f'unknown stage {stage!r}; choose smoke or one of {sorted(plans)}')
    if deployed:
        spawned = modal.Function.from_name('thinkingcap', 'orchestrate').spawn(*call)
        print(f'spawned {stage} on the deployed app: {spawned.object_id}')
    else:
        orchestrate.remote(*call)
