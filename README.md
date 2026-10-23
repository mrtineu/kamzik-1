# Qwen reasoning lab

**Run on Modal:** see [MODAL.md](MODAL.md) for cloud GPU setup and the GitHub
Actions launcher. Your laptop does not need to load the models or run training.

Inference optimization details and measurements are in [OPTIMIZATIONS.md](OPTIMIZATIONS.md).
The separate `config_optimized.json` enables queued retries, prompt-length grouping,
ordinary batches up to 16 and retries up to 8, experimental attention for generation,
and 2 CPU threads. Training stays on math attention. PyTorch now rejects allocations
above an 80% VRAM budget before reaching the ROCm driver, with at least 2 GiB of
headroom relative to memory already in use. A rejected generation batch is split
automatically, with its grouping and seeds saved for resume. To start a new run:

```bash
QRL_CONFIG=config_optimized.json QRL_RUN=optimized .venv/bin/python -u experiment.py
```

See [the first completed results](FIRST_RESULTS.md). The larger run continues under `runs/pilot/`.

A local pilot of **teacher-coached reasoning compression** in Qwen3-0.6B using an inter-layer residual MLP. Created 2026-10-18. This is independent of the Slovak train-delay project.

## Where the space goes

Everything new needed to run the experiment is under this folder: `models/`, `data/`, `.cache/`, `.venv/`, and `runs/`. Models require roughly 7 GB. There is no paid API usage. The virtual environment reuses an existing ROCm PyTorch installation from `../pytorch-workspace/.venv` via one `.pth` file; it does not modify that installation. The existing llama.cpp executable is reused from `../llama.cpp/build/bin/llama-server`.

To reclaim space after stopping the experiment, delete `models/` and `.cache/` from this folder. Adapter checkpoints require the original student weights: keep `models/student/` for inference, or download it again later. `models/teacher/` is needed only for collecting new coaching and can be removed after training finishes. Keep `runs/`, source files, and adapter `.pt` files if you want the results and trained adapter. Delete the whole folder to remove the new project. Do not delete the pre-existing PyTorch or llama.cpp folders as part of cleanup.

## Current experiment

- Student: official Qwen3-0.6B, BF16, frozen base weights.
- Teacher: official Qwen3-8B-GGUF, Q4_K_M, local llama.cpp with non-thinking coaching.
- MLP: `h + W_up(SiLU(W_down(h)))` after decoder layer index **13** (the 14th layer), bottleneck 64, no biases. `W_up` starts at zero, making the inserted adapter exactly identity initially. **131,072 parameters**.
- LoRA comparison: rank 4, attention Q/V projections, alpha 8, no dropout; parameter count is measured and saved.
- Full pilot GSM8K: 64 train / 16 validation / 48 test. A preliminary `quick` run uses the first 8 training problems, 2 validation problems, and 6 held-out test problems (2 per difficulty group); it is a pipeline check, not evidence of a small accuracy margin. Validation is a disjoint sample from official training data. Test is from the official test split and balanced across rough reference-step-count difficulty groups.
- Original baseline and concise-prompt baseline use the unmodified student. Adapter and LoRA evaluation use the exact same ordinary problem prompt as the baseline, without coaching or concise instructions.
- One seed, 3 epochs, AdamW, learning rate 3e-4, gradient accumulation 4, fixed final epoch. The larger run uses non-reentrant activation checkpointing during training to keep long examples within 16 GB; a real-model smoke test verified matching MLP loss/gradients and working LoRA backward. Dropout is zero in both runs. No held-out examples enter coaching or training. The held-out set does not choose checkpoints.
- Sampling: temperature .6, top_p .95, top_k 20 for all thinking-mode student arms. Stable batch seeds are paired across evaluation arms.

## Generalized experiment: math and QA

`config_generalized.json` trains one shared MLP and one shared LoRA on a balanced
mixture of four datasets, then evaluates each arm on held-out examples from every
dataset. It uses the optimized generation settings in a separate `generalized`
run. Existing GSM8K configurations and saved splits retain their original behavior.

| Dataset | Task | Available source pool | Pilot train / validation / test |
|---|---|---:|---:|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Simple math | 7,473 train | 64 / 16 / 48 |
| [Skywork-OR1 easy band](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) | Math with headroom | 20,762 unique eligible math problems | 64 / 16 / 48 |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | Commonsense QA | 9,741 train (9,740 unique stems) | 64 / 16 / 48 |
| [ARC Easy](https://huggingface.co/datasets/allenai/ai2_arc) | Science QA | 2,251 train (2,245 unique stems) | 64 / 16 / 48 |

The total pilot is **256 train / 64 validation / 192 test**. These are sampled
experiment sizes, not full-dataset training. Each dataset's counts can be changed
independently in the config; use a new run directory when changing settings.

Skywork's easy band is explicitly **difficulty 1–4 inclusive on
`DeepSeek-R1-Distill-Qwen-1.5B`**. Its documented scale runs from 0 to 16, with higher
values harder. Excluding 0 leaves headroom on that reference model; this is a proxy,
not a calibrated difficulty estimate for our Qwen3-0.6B student. The pinned release
contains 105,055 math rows, and this filter yields 20,762 unique questions, rather
than assuming a fixed ~28k. Change `difficulty_model`, `difficulty_min`, and
`difficulty_max` to predeclare another band. All supported math reference forms are
retained; the scorer uses normalized exact match plus rational equivalence and
does **not** perform symbolic algebra. Equivalent formulas in a different algebraic
form can therefore be marked incorrect. Raw answers and outputs remain inspectable.

CommonsenseQA's public test labels are hidden, so its official **validation** set
provides the experiment's test examples. Its experiment validation examples are
held out from official train. ARC Easy uses official validation and test. GSM8K
retains official test and holds validation out of train. Skywork has no official
validation/test here, so those are deterministic holdouts from the filtered math
pool. Split construction deduplicates normalized question stems, excludes every
official evaluation question from training across datasets, and rejects any
cross-split duplicate. It fails if the requested unique sample is unavailable.

For QA, the prompt includes all choices, preserves original labels (including ARC's
numeric labels), and requests a boxed label. Coaching asks for efficient factual
inference and option selection. Training retains this task metadata and masks the
same original task prompt. Scoring requires an explicit valid final label after
`</think>` and normal termination; unfinished traces count as wrong for every task.

The three added sources are downloaded as revision-pinned Parquet and stored as
local JSONL. `manifest.json` records revisions, shard hashes, normalized file hashes,
and raw row counts. Runs save the manifest, per-pool eligible unique counts in
`dataset_counts.json`, and exact examples in `splits.json`.

```bash
# Install the added reader if updating an existing environment:
uv pip install --python .venv/bin/python --no-deps pyarrow==26.0.0

# Download and verify datasets without downloading models:
.venv/bin/python prepare.py --config config_generalized.json --datasets-only

# Validate and save the mixed splits without GPU inference:
QRL_CONFIG=config_generalized.json QRL_RUN=generalized \
  .venv/bin/python experiment.py --prepare-splits-only

# Train/evaluate all four arms; resumable under runs/generalized/:
QRL_CONFIG=config_generalized.json QRL_RUN=generalized \
  .venv/bin/python -u experiment.py

.venv/bin/python progress.py --run generalized
```

For a fresh setup, `QRL_CONFIG=config_generalized.json QRL_RUN=generalized ./run.sh`
also prepares models. `sequence.py` still runs the original quick/pilot sequence;
it does not start the generalized experiment. Stop a foreground mixed run with
Ctrl+C and resume with the same command.

`RESULTS.md` adds per-dataset tables with accuracy, tokens, unfinished counts, and
paired baseline comparisons. `metrics.json` additionally includes per-task
breakdowns, completion counts, and macro dataset accuracy. Inspect dataset-level
results: pooled accuracy can hide a regression on QA. This experiment measures
held-out performance after mixed-domain training; it does not establish transfer
to a domain absent from training. To test that distinction, predeclare `train_n: 0`
for a domain in another configuration and train from scratch in a fresh run.

```bash
.venv/bin/python -m pytest -q test_core.py test_generation.py test_memory_guard.py test_task_data.py

# Supply options in the question when trying a trained adapter on QA:
.venv/bin/python infer.py --run generalized --arm mlp --answer-type choice \
  --choice-labels A B C --question 'Which can fly? A. stone B. bird C. table'
```

## Original GSM8K pipeline

1. Generate the student's original reasoning on training problems and evaluate baseline/concise prompting on held-out problems.
2. Give the teacher each training problem, the original student output, and the verified **training** answer. Ask for short, problem-specific coaching, rather than a replacement solution.
3. Require structured teacher feedback consisting of exactly three short coaching instructions; incomplete JSON is retried once and never used as coaching. Generate two coached student candidates per training problem. Accept only outputs that finish, have a correct explicit final numeric answer, and are shorter in **total output tokens** than that problem's original student attempt. Keep the shortest successful candidate.
4. Train the MLP and LoRA separately on the **same** accepted student completions. Input contains the original question and standard answer-format instruction; coaching is absent. Prompt tokens are masked in the causal-LM loss. No target is silently cut to fit the training context.
5. Evaluate each trained arm once on validation and test. Save raw outputs, accuracy, Wilson intervals, thinking and total tokens, unfinished counts, paired accuracy losses/gains, and difficulty breakdowns.

The objective is 50% fewer thinking tokens with at least 98% of baseline accuracy retained. These are illustrative thresholds, not a promise. The report flags a point-estimate success only when both arms have completed outputs. A 48-question pilot cannot establish a 2% relative accuracy margin with confidence.

## Run and resume

```bash
cd ~/Documents/qwen-reasoning-lab
./run.sh
```

`prepare.py` downloads the official models and public GSM8K source. `manifest.json` records exact model revisions and dataset SHA256 hashes. `experiment.py` resumes completed generation batches and final adapter checkpoints. The active sequence runs `quick` before `pilot`; only matching training rollouts, coaching, and candidates are reused, never quick-run trained weights. If interrupted during training, that arm restarts training from the frozen base, using the same examples and seed. A file lock prevents concurrent experiments in the same run directory.

```bash
# Rebuild the report from saved outputs, without GPU inference:
.venv/bin/python experiment.py --report-only
# Algorithm checks (small CPU tensors):
.venv/bin/python -m pytest -q test_core.py
# Retry scheduling and restart regression checks:
.venv/bin/python -m pytest -q test_generation.py
# Actual-model identity and backward-gradient smoke check (GPU required):
.venv/bin/python smoke.py
```

The supplied environment uses Python 3.14 and ROCm PyTorch 2.13.0+rocm7.2. `requirements.txt` lists added packages. They were installed with `uv pip install --no-deps` because the existing PyTorch environment supplies common prerequisites, avoiding installation of an NVIDIA build over the working AMD runtime. Exact discovered distribution versions are saved in `runs/pilot/environment.json`.

Changing configuration requires a **new run directory**. Do not mix outputs produced under different settings. A future clean environment should install an appropriate ROCm PyTorch wheel first, then the listed dependencies.

## Results and monitoring

The current run is detached from the terminal and continues while the app is idle. The sequence logs to `runs/supervisor.log`, and each experiment logs to `runs/<name>/console.log`. Check progress or stop it with:

```bash
.venv/bin/python progress.py
.venv/bin/python stop.py
```

The initial unstructured teacher responses are retained as `runs/quick/rejected_v1_coaching.jsonl` for inspection and never enter training.

The quick comparison is at `runs/quick/RESULTS.md`; the larger comparison is at `runs/pilot/RESULTS.md`. To run the sequence again after a normal stop:

```bash
.venv/bin/python -u sequence.py
```

To run a trained adapter yourself:

```bash
.venv/bin/python infer.py --run quick --arm mlp --question 'Solve 4(x-3)=28 for x.'
```

Use `--arm baseline`, `--arm concise`, or `--arm lora` for comparisons; use `--run pilot` after that checkpoint exists. Inference prints the generated reasoning and output-token counts. It does not need the teacher.


- `runs/pilot/RESULTS.md`: human-readable, updated after every generation batch.
- `runs/pilot/metrics.json`: machine-readable results and paired comparisons.
- `runs/pilot/status.json`: most recent progress message and PID.
- `runs/pilot/*_{baseline,concise,mlp,lora}.jsonl`: raw held-out outputs.
- `runs/pilot/coaching.jsonl`: teacher feedback and teacher token use.
- `runs/pilot/candidates_*.jsonl`: all student candidates, including rejections.
- `runs/pilot/training_targets.json`, `selection.json`: exact training data and filter decisions.
- `runs/pilot/{mlp,lora}_training.jsonl`: epoch loss.
- `runs/pilot/{mlp,lora}.pt`: only the trained parameters, no duplicate base weights.
- `runs/pilot/ERROR.txt`: traceback if an experiment error occurs.

Stop the detached sequence with `.venv/bin/python stop.py`, or use Ctrl+C if you launched the experiment in a foreground terminal. The teacher subprocess is terminated by a `finally` block when the parent is interrupted normally. Resume a stopped sequence with `.venv/bin/python -u sequence.py`, or resume only the larger experiment with `./run.sh`.

## Measurement limits

The first generation ceiling is 2048 new tokens; unfinished responses are rerun from the original prompt with a 4096-token ceiling. This is a practical pilot limit, **not** a forced concise-thinking target. Initial attempts are preserved under `initial_attempt`; the main token metrics use the final attempt, not combined retry compute. Unfinished responses always count as incorrect. Their observed thinking length is censored and cannot be interpreted as a completed reasoning length; inspect unfinished counts and completed/correct-only metrics. Retry sampling uses a separate recorded seed formula rather than continuing the original sampled prefix.

The answer scorer requires an explicit boxed or marked numeric answer **after** `</think>` and normal termination; it never extracts a convenient number from an unfinished trace. Exact rational comparison is used for GSM8K numbers. This strict format can create false negatives for nonconforming answers; raw outputs are kept for inspection.

Generation timing is amortized wall time per example within a batch, not single-request latency; a retried example's timing records its final attempt only. Teacher processing is separate. The token objective concerns inference after training, not the full expense of collecting training data.

New runs using `generation_policy: "batched_v1"` persist first attempts in `*_initial.jsonl`,
queue retries across the workload, and record an immutable `*_generation.json` plan.
`retry_batch_size` can differ from `batch_size`; `batch_order: "prompt_length"` groups
similar prompt lengths using only prompt text. Completed outputs remain in the original
example order when returned to the pipeline. Both attempts count toward
`total_compute_seconds`, while final-answer token metrics retain their original meaning.
Memory-induced splits are saved in `*_execution.json`; this file must be kept with
the initial outputs and generation plan when resuming. Compute timing excludes work
discarded by memory rejection or interruption. If even one request cannot fit, or
a partly saved batch cannot keep its original RNG assignment, the run stops cleanly.
Each generation row also records batch throughput, CPU time, decode-slot efficiency,
and peak allocated/reserved GPU memory. Existing configurations retain legacy retry
scheduling so historical runs can resume with their original seed formula.

`experimental_attention` pins the AOTriton environment flag before importing PyTorch.
It permits optimized SDPA dispatch; unsupported shapes and masks can still use other
kernels. With this flag, training defaults to math attention; `training_attention: "math"`
explicitly pins forward and backward recomputation to that backend.
`cpu_threads` controls PyTorch's CPU thread count. All these settings belong
in a new run configuration, never in the saved configuration of an existing run.

The benchmark compares batches 4/8/16 at 2048/4096-token contexts with experimental
attention disabled/enabled, using two 128-token decode windows per shape. Both
workers use the allocator budget and headroom guard. A benchmark stops on a rejected
allocation or worker timeout, preserving the requested batch size for honest comparison;
the experiment scheduler can instead split batches. To run the matrix:

```bash
.venv/bin/python -u benchmark.py
```

It uses saved **training** traces with per-shape warmups, VRAM and GPU busy measurements.
`--end-to-end` adds complete answers, and `--profile` exports CPU/device profiler data.
Explicitly enabling attention mode 1 also runs model logits, padded dispatch and
adapter backward checks. The benchmark never rewrites experiment results. Use
`--help` to inspect its options. The earlier stress run is documented as incomplete
in `runs/optimization_benchmark_final/RESULTS.md`.

`vram_fraction` and `vram_reserve_gib` configure the experiment memory budget;
benchmark equivalents are `--vram-fraction` and `--vram-reserve-gib`. The allocator
uses `throw_on_cudamalloc_oom:True` with a native allocator limit. The installed
ROCm build was verified to reject a 256 MiB request at an artificial 81.5 MiB
process limit while 15.8 GiB remained physically free; a subsequent GPU operation
succeeded. Evidence is in `runs/allocator_guard_validation.json`.

GSM8K may already be in model pretraining. Reference equation count is only an approximate difficulty proxy. Coaching plus best-of-two correctness/length filtering is a combined intervention; a no-coaching best-of-two control is needed to attribute gains specifically to coaching. More examples, multiple seeds, baseline-correct paired analysis, and a larger untouched benchmark are needed before making a research claim. A direct-teacher-distillation ablation is a useful later extension, not included here.

## Sources

- [Official student model and sampling recommendations](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Official teacher GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Official GSM8K data](https://github.com/openai/grade-school-math)
- [In Their Own Words: Reasoning Traces Tailored for Small Models Make Them Better Reasoners](https://arxiv.org/abs/2509.22230)
- [Concise Math Reasoning via Difficulty-Aware Distillation](https://aclanthology.org/2026.findings-acl.2155/)

These papers motivate studying training targets; this pilot is not a reproduction of either paper.
