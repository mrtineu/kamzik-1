# ThinkingCap compression run

Goal: same accuracy, fewer thinking tokens for Qwen3-0.6B, scored by
[ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test) @ `bd52d8d`, run unmodified.

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

## Running (from the repo root)

```
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage smoke
.modal-venv/bin/modal run --detach thinkingcap/modal_app.py --stage setup        # then eval-base, baseline,
                                                                                 # coach, candidates, train,
                                                                                 # eval-ckpts, compare
.modal-venv/bin/modal volume get thinkingcap runs/full/selection_summary.json .
python -m unittest thinkingcap/test_pipeline.py                                   # local, no GPU
```

Every stage resumes where it stopped. Settings live in `config.json`. A run refuses to continue if
the config changed, so start a new `--run` name instead.

## Results

See [results/RESULTS.md](results/RESULTS.md). Best checkpoint: `results/adapters/coached/step-00170`
(22.5% fewer thinking tokens, -1.58 pp accuracy).
