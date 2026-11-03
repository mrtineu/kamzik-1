# Run on Modal

The laptop only launches the job. Models download on Modal, and all GPU work runs
there. The runner uses one A10G (24 GB), CUDA PyTorch, and a pinned CUDA llama.cpp
teacher. It starts a fresh `modal-generalized` run; historical laptop results are
not mixed into the NVIDIA run.

## Launch from GitHub

1. Create a Modal API token in your Modal workspace settings.
2. In this private repository, open **Settings → Secrets and variables → Actions**
   and add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as repository secrets.
3. Open **Actions → Run on Modal → Run workflow**. First choose `smoke` mode.
4. Check the Modal dashboard for the result. Then launch `experiment` mode with
   `config_generalized.json` and run name `modal-generalized`.

The smoke job downloads the models and checks student adapter identity and
backward gradients. It does not run the full teacher/coaching experiment.
`experiment` runs the complete pipeline; `splits` downloads assets and validates
splits. GPU compatibility and the teacher startup need verification on Modal.

The GitHub job launches a detached Modal job. GitHub success confirms submission;
check Modal for actual experiment completion or errors. GPU time and persistent
storage use your Modal account's credits/billing. There is no scheduled run or
always-on deployment. Each invocation has a 24-hour limit; resume incomplete work
with the same config, run name, and code revision. Completed generation and final
adapter checkpoints are reused; interrupted training restarts that arm.
Only one job should run at a time because model downloads and the manifest share
storage. The runner rejects overlapping execution with a file lock.

## Launch from a terminal

Use a separate small client environment; the laptop's ROCm installation is unused:

```bash
uv venv .modal-venv --python 3.12
uv pip install --python .modal-venv/bin/python 'modal==1.6.2'
.modal-venv/bin/modal setup
.modal-venv/bin/modal run --detach pilot/modal_app.py --mode smoke
.modal-venv/bin/modal run --detach pilot/modal_app.py \
  --config config_generalized.json --run modal-generalized --mode experiment
```

For a smaller pipeline check, use `--config config_quick.json --run modal-quick`.
Changing settings requires a new run name. Existing local runs and weights are
left on the laptop and excluded from Git and Modal uploads. Public datasets and
models are downloaded at revisions recorded in `manifest.json`. The Modal runner
overrides only the llama-server path, CPU threads, and ROCm attention flag, and
saves the effective config with each experiment.

## Results and stopping

Use the Modal dashboard for live logs and to stop a job. Outputs and checkpoints
persist in the named volume `qwen-reasoning-lab`; models, data, cache, and the
updated manifest persist there too. To download a completed run:

```bash
.modal-venv/bin/modal volume get qwen-reasoning-lab \
  runs/modal-generalized ./modal-results
```

Inspect `RESULTS.md`, `metrics.json`, `status.json`, and `ERROR.txt` if present.
Completion is recorded in `complete.json`. A GPU job can stop on timeout; rerun the
same command to resume. The periodic Volume snapshots and final explicit commit
preserve outputs, though abrupt termination can lose the latest unsynced writes.
Modal Volume storage remains allocated until you delete it in Modal.

Implementation follows the official [CUDA guide](https://modal.com/docs/guide/cuda),
[Volume guide](https://modal.com/docs/guide/volumes), and
[detached training example](https://modal.com/docs/examples/long-training).
