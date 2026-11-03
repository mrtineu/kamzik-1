# ThinkingCap pipeline

Goal: same accuracy, fewer generated tokens (thinking included) for Qwen3-0.6B, scored by
[ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test) @ `bd52d8d`, run unmodified.
**All results are in the [root README](../README.md).** This file covers how to run things.

## Method

1. **setup**: their `prepare_data.py` (full train set: gsm8k, skywork, commonsense_qa, arc_easy).
   Drop questions sharing a 13-word sequence with any benchmark prompt, and prompts over 4096 tokens.
2. **baseline**: 4 uncoached student samples per problem (eval sampling, 4096 cap).
   A problem is eligible only if ≥2/4 are correct; `ref_len` = median length of the correct ones.
3. **coach**: Qwen3-8B sees the problem plus the median-length *correct* trace, never the gold answer, and
   gives three brevity-only bullets. Coaching that names the gold number (math) or any option
   (label or text) is rejected and retried up to 3 times.
4. **candidates**: 4 samples per arm. *control* = plain prompt; *coached* = prompt + bullets.
   Target = shortest correct finished sample, kept only if strictly shorter than `ref_len`.
5. **train**: LoRA r=8 on q/v, trained on the **original prompt** (no coaching) → target. 2 epochs,
   checkpoint every half epoch. Each arm uses its own accepted set.
6. **eval / compare**: their `evaluate.py` (8 samples) on the base model and every checkpoint, then `compare.py`.

The non-LoRA arms (`arms.json`, `arms.py`) reuse step 5's coached targets and the same training loop.

## Running (from the repo root)

```
# LoRA run
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage smoke
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage setup        # then eval-base, baseline,
                                                                                 # coach, candidates, train,
                                                                                 # eval-ckpts, compare
# non-LoRA arms
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage arms-smoke   # verify + short train + smoke eval
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage arms-full    # train, eval, compare every arm
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage arms-extend --arm extra-mlp-64-l17,extra-mlp-64-l20

# side experiments
.modal-venv/bin/modal run --detach thinkingcap/first_token_app.py --words First,To
.modal-venv/bin/modal run --detach thinkingcap/first_token_app.py --words First --arms extra-mlp-64/step-00170
.modal-venv/bin/modal run thinkingcap/layer_probe_app.py --arms extra-mlp-64/step-00170

.modal-venv/bin/modal volume get thinkingcap evals/summary_arms.json .
python -m unittest thinkingcap/test_pipeline.py                                   # local, no GPU
```

For long runs, `modal deploy thinkingcap/modal_app.py` first, then pass `--deployed` to `modal run` (without `--detach`).
That way nothing depends on the laptop staying up. Every stage resumes where it stopped.
Settings live in `config.json`. A run refuses to continue if the config changed, so start a new `--run` name instead.
`test_arms.py` needs `torch` and `transformers`.

The MLP arms export with `architectures=["Qwen3ReasoningMLPForCausalLM"]` and need `vllm_plugin/` installed
in the eval image, which `modal_app.py` does. The block arms export as plain 29-layer Qwen3 checkpoints.

## Results files (`results/`)

- `summary.json` (LoRA), `summary_arms.json` (arms), `summary_first_token.json`: `compare.py` output for every run.
- `evals/NAME/`: `results.json` from `evaluate.py`, plus `compare.txt` / `compare.json` vs `base`
  (`forced.json` for forced-first-token runs).
- `adapters/{coached,control}/step-N/`: PEFT LoRA adapters (`adapter_model.safetensors`) for `Qwen/Qwen3-0.6B`.
- `arms_verify/`: identity / roundtrip / vLLM checks for each arm.
- `probes/{base,extra-mlp-64-step-00170}/probe.json`: per-layer logit-lens and plan-probe table.
- `run/`: config, setup, baseline / coaching / selection summaries, stage timings.

Not committed because they are too large. They are still on the `thinkingcap` Modal volume:

- Arm weights: `runs/full/arms/NAME/step-N/`, about 50 MB each.
- Per-sample `records.jsonl` for each eval: about 65 MB each, under `evals/`.
- Probe `directions.pt`.
- Training and candidate jsonl files, and optimizer state, under `runs/full/`.
