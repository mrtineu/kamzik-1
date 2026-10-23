"""Run the existing experiment on Modal; no local GPU or model upload needed."""
import json
import os
from pathlib import Path
import re

import modal

ROOT = Path(__file__).resolve().parent
APP_ROOT = Path('/opt/qrl')
STATE = Path('/qrl-state')
LLAMA_REVISION = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
app = modal.App('qwen-reasoning-lab')
volume = modal.Volume.from_name('qwen-reasoning-lab', create_if_missing=True)

image = (
    modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')
    .entrypoint([])
    .apt_install('git', 'cmake', 'build-essential', 'libcurl4-openssl-dev')
    .run_commands(
        f'git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && '
        f'cd /opt/llama.cpp && git checkout {LLAMA_REVISION}',
        'cmake -S /opt/llama.cpp -B /opt/llama.cpp/build '
        '-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=ON '
        '-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF',
        'cmake --build /opt/llama.cpp/build --target llama-server -j 4',
    )
    .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install_from_requirements(str(ROOT / 'requirements.txt'))
    .env({'QRL_PORTABLE_CUDA': '1', 'TOKENIZERS_PARALLELISM': 'false',
          'OMP_NUM_THREADS': '4', 'HF_HUB_DISABLE_XET': '1'})
)
# Explicit allowlist: never send local credentials, models, datasets or old runs.
for source in sorted(ROOT.iterdir()):
    if source.is_file() and (source.suffix == '.py' or
                            source.name == 'manifest.json' or
                            re.fullmatch(r'config(?:_[a-z]+)?\.json', source.name)):
        image = image.add_local_file(str(source), str(APP_ROOT / source.name))


def validate(config, run, mode):
    if not re.fullmatch(r'config(?:_[a-z]+)?\.json', config):
        raise ValueError('Use a checked-in config*.json filename')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run):
        raise ValueError('Run must be a simple directory name (letters, numbers, _ and -)')
    if mode not in ('smoke', 'splits', 'experiment'):
        raise ValueError('Mode must be smoke, splits or experiment')


@app.function(image=image, gpu='A10G', cpu=4, memory=16384,
              volumes={str(STATE): volume}, timeout=24 * 60 * 60,
              max_containers=1)
def execute(config: str, run: str, mode: str):
    import fcntl
    import shutil
    import subprocess
    import sys

    validate(config, run, mode)
    volume.reload()
    # Protect shared downloads and the manifest as well as individual runs.
    with (STATE / '.runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('models', 'data', '.cache', 'runs'):
            (STATE / name).mkdir(exist_ok=True)
            target = APP_ROOT / name
            if not target.is_symlink():
                target.symlink_to(STATE / name, target_is_directory=True)
        manifest = STATE / 'manifest.json'
        if not manifest.exists():
            shutil.copy2(APP_ROOT / 'manifest.json', manifest)
        (APP_ROOT / 'manifest.json').unlink()
        (APP_ROOT / 'manifest.json').symlink_to(manifest)
        settings = json.loads((APP_ROOT / config).read_text())
        settings.update(llama_server='/opt/llama.cpp/build/bin/llama-server',
                        experimental_attention=False, cpu_threads=4)
        remote_config = APP_ROOT / 'config_modal.json'
        remote_config.write_text(json.dumps(settings, indent=2) + '\n')
        env = dict(os.environ, QRL_CONFIG=remote_config.name, QRL_RUN=run)
        try:
            subprocess.run([sys.executable, '-u', 'prepare.py', '--config', remote_config.name],
                           cwd=APP_ROOT, env=env, check=True)
            volume.commit()
            command = ([sys.executable, '-u', 'smoke.py'] if mode == 'smoke' else
                       [sys.executable, '-u', 'experiment.py'] +
                       (['--prepare-splits-only'] if mode == 'splits' else []))
            subprocess.run(command, cwd=APP_ROOT, env=env, check=True)
        finally:
            volume.commit()
    return f'Saved to Modal volume qwen-reasoning-lab: runs/{run}'


@app.local_entrypoint()
def main(config: str = 'config_generalized.json', run: str = 'modal-generalized',
         mode: str = 'experiment'):
    validate(config, run, mode)
    print(execute.spawn(config, run, mode).get())
