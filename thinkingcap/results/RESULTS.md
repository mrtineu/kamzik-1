# Results: full run (2026-10-24)

Qwen3-0.6B + LoRA r=8 (q/v), scored with ThinkingCap-Test @ `bd52d8d` (8 samples, seed 9001).
Average over 4 benchmarks (bfcl, ifeval, mbpp, svamp) vs the base model, with 95% paired CIs from `compare.py`.

| Checkpoint | Thinking tokens saved | Accuracy change |
|---|---|---|
| coached-step-00170 | 22.5% [21.0, 24.0] | -1.58 pp [-2.24, -0.89] |
| coached-step-00339 | 13.9% [12.5, 15.3] | -1.67 pp [-2.35, -1.01] |
| coached-step-00509 | 16.2% [14.7, 17.7] | -1.90 pp [-2.57, -1.27] |
| coached-step-00678 | 15.7% [14.3, 17.1] | -1.75 pp [-2.43, -1.09] |
| control-step-00155 | 0.1% [-1.5, 1.6] | -0.12 pp [-0.68, +0.46] |
| control-step-00309 | 0.2% [-1.4, 1.7] | -0.24 pp [-0.81, +0.31] |
| control-step-00464 | 2.3% [0.8, 3.8] | +0.09 pp [-0.50, +0.67] |
| control-step-00618 | 1.2% [-0.2, 2.7] | -0.12 pp [-0.72, +0.47] |

- **Best: `coached/step-00170`** (half an epoch): 22.5% fewer tokens for -1.58 pp, i.e. 0.07 pp per 1% saved.
  The accuracy drop is small but its CI excludes zero, so this is not yet "same accuracy".
- **The teacher drives the compression.** The control arm (shortest of its own 4 samples, no coaching)
  saves 0-2% with unchanged accuracy, even though its targets were ~0.65x the usual correct length.
- Training past half an epoch compresses less at the same accuracy cost.

`coached/step-00170` per benchmark:

| Benchmark | Tokens saved | Accuracy change |
|---|---|---|
| bfcl_irrelevance | 23.6% | -0.31 pp |
| bfcl_multiple | 16.8% | -3.25 pp |
| bfcl_simple_python | 17.3% | -1.12 pp |
| ifeval | 20.6% | -0.39 pp |
| mbpp | 13.2% | -1.82 pp |
| svamp | 35.4% | -2.70 pp |

## Data funnel

46,811 training problems → 12,901 eligible (baseline ≥2/4 correct) → 12,015 with usable coaching →
accepted targets: 9,858 control, 10,825 coached. Details in `run/*_summary.json`.

## Files

- `summary.json`: compare output for every checkpoint.
- `evals/NAME/`: `results.json` from `evaluate.py`, plus `compare.txt` / `compare.json` vs `base`.
- `adapters/{coached,control}/step-N/`: PEFT LoRA adapters (`adapter_model.safetensors`) for `Qwen/Qwen3-0.6B`.
- `run/`: config, setup, baseline / coaching / selection summaries, stage timings.

Not committed (too large, still on the `thinkingcap` Modal volume): per-sample `records.jsonl` for each eval
(~65 MB each, under `evals/`), the training/candidate jsonl files and optimizer state (`trainer_state.pt`) under `runs/full/`.
