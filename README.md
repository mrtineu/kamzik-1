# Qwen reasoning lab

The goal is to make Qwen3-0.6B **think in fewer tokens at the same accuracy**. Raising accuracy is not a goal.
The model learns from a Qwen3-8B teacher that coaches it to be brief, and is scored by
[ThinkingCap-Test](https://github.com/BottleCapAI/ThinkingCap-Test) @ `bd52d8d`, run unmodified.

- `thinkingcap/` holds the pipeline, the arms and the side experiments, all run on a Modal H100.
  [thinkingcap/README.md](thinkingcap/README.md) covers how to run it.
- `pilot/` is the original local pilot (2026-10-18), kept for reference. See [the pilot section](#original-pilot) below.

## Summary

| Question | Answer |
|---|---|
| Does any configuration keep accuracy equal? | **No.** Every trained configuration loses accuracy, and every accuracy CI excludes 0. The loss stays around −1.6 pp whatever the architecture. |
| Best trade-off you can submit | **extra-mlp-64 after layer 13, step 170**: 28.7% fewer generated tokens for −1.71 pp. It is a plain 29-layer Qwen3 checkpoint. |
| Does the teacher matter? | **Yes.** The same training without coaching (the control arm) saves 0–2%. |
| Which architecture? | The best checkpoint of every adapter loses about 0.06–0.08 pp per 1% of tokens saved. Architecture changes how far along that line a run goes, not the line itself. |
| Where should it go? | Placing it later compresses less. After layer 20 it saves 19.5% for −1.11 pp. That is the smallest loss measured, but its CI overlaps the others. |
| Forcing the first thinking word? | Forcing `First` saves 29% for −2.7 pp, and `To` saves 54% for −7.7 pp. This is a decoding trick, not weights, so it cannot be submitted. Stacked on an arm, the token savings compound and accuracy falls further. |

All numbers below are the macro-average over 4 benchmark groups (bfcl, ifeval, mbpp, svamp), compared with the
base model. They use 8 samples per prompt and seed 9001. Brackets are the 95% paired prompt-cluster bootstrap CIs
from `compare.py`. Tokens saved counts every generated token, thinking and answer, and averages the per-benchmark
ratios with a geometric mean. Accuracy change is in percentage points.

## Method

1. **Data**: the ThinkingCap train mix (gsm8k, skywork, commonsense_qa, arc_easy), 46,811 problems.
   Problems sharing a 13-word sequence with any benchmark prompt, and prompts over 4096 tokens, are removed.
2. **Baseline**: 4 student samples per problem. A problem is eligible if at least 2 of the 4 are correct (12,901 problems).
3. **Coaching**: Qwen3-8B sees the problem and a median-length correct trace, never the gold answer. It writes three
   bullets about brevity only. Coaching that leaks the answer is rejected (12,015 problems kept).
4. **Targets**: 4 samples per arm. The *coached* arm sees prompt + bullets, the *control* arm sees the plain prompt.
   The target is the shortest correct sample, kept only if it is shorter than the baseline median.
   This leaves 10,825 coached and 9,858 control targets.
5. **Training** uses the **plain prompt** paired with the target, so no coaching is used at inference.
   Each run is 2 epochs with a checkpoint every half epoch, and base weights are frozen.
6. **Eval**: ThinkingCap's `evaluate.py` and `compare.py` on the base model and every checkpoint.

The base model on the benchmarks:

| Benchmark | Prompts | Accuracy | Mean thinking tokens |
|---|---:|---:|---:|
| bfcl_irrelevance | 240 | 83.0% | 189 |
| bfcl_multiple | 200 | 76.7% | 201 |
| bfcl_simple_python | 400 | 82.1% | 170 |
| ifeval | 541 | 58.1% | 431 |
| mbpp | 378 | 51.9% | 2066 |
| svamp | 1000 | 89.2% | 681 |

## 1. LoRA (r=8 on q/v, 1,146,880 parameters)

| Checkpoint | Tokens saved | Accuracy change |
|---|---|---|
| coached step 170 | **22.5% [21.0, 24.0]** | **−1.58 [−2.24, −0.89]** |
| coached step 339 | 13.9% [12.5, 15.3] | −1.67 [−2.35, −1.01] |
| coached step 509 | 16.2% [14.7, 17.7] | −1.90 [−2.57, −1.27] |
| coached step 678 | 15.7% [14.3, 17.1] | −1.75 [−2.43, −1.09] |
| control step 155 | 0.1% [−1.5, 1.6] | −0.12 [−0.68, +0.46] |
| control step 309 | 0.2% [−1.4, 1.7] | −0.24 [−0.81, +0.31] |
| control step 464 | 2.3% [0.8, 3.8] | +0.09 [−0.50, +0.67] |
| control step 618 | 1.2% [−0.2, 2.7] | −0.12 [−0.72, +0.47] |

- The coaching does the compressing. The control arm's targets were also about 0.65× the usual correct length,
  yet training on them saves almost nothing.
- Training past half an epoch compresses less for the same accuracy cost.
- Adapters are in `thinkingcap/results/adapters/`. The full run cost about $75–80 on Modal.

## 2. Non-LoRA arms: reasoning MLPs and an extra transformer block

Every arm trains on the same coached targets for 678 steps (2 epochs), with checkpoints at steps 170, 339, 509 and 678.
Every arm starts as an **exact identity**, so step 0 is the base model. Definitions are in
[`thinkingcap/arms.json`](thinkingcap/arms.json) and the code is in [`thinkingcap/arms.py`](thinkingcap/arms.py).

| Arm | What is added | Trained parameters | Serving |
|---|---|---:|---|
| mlp-mid | Residual `h + up(silu(down(h)))` after layer 13, bottleneck 64, `up` zero-initialized | 131,072 | vLLM plugin |
| mlp-quarters | The same MLP after layers 6, 13, 20 and 27 | 524,288 | vLLM plugin |
| extra-block | Copy of layer 13 inserted after it, with `o_proj` and `down_proj` zeroed; the whole block is trained | 15,730,944 | plain Qwen3 (29 layers) |
| extra-mlp | The same inserted block with attention frozen at 0; only its MLP and post-attention norm are trained | 9,438,208 | plain Qwen3 (29 layers) |
| extra-mlp-64 | extra-mlp with only 64 of its intermediate channels alive | 197,632 effective | plain Qwen3 (29 layers) |

`verify-arms` checks every arm before training and again on its exported weights:

- **Identity**: at step 0, each arm matches the base model.
- **Roundtrip**: export, reload and compare again.
- **vLLM agreement**: a perturbed arm under vLLM matches HF about as closely as the base model does (0.022–0.025
  vs 0.021). Under vLLM, the step-0 arm differs from the base model by 3.8e-7 for the MLP arms and 0.015 for
  extra-block.

Results are in [`thinkingcap/results/arms_verify/`](thinkingcap/results/arms_verify/).

| Arm | Step | Tokens saved | Accuracy change |
|---|---:|---|---|
| mlp-mid | 170 | 29.6% [28.3, 31.0] | −2.37 [−3.14, −1.69] |
| | 339 | 24.1% [22.7, 25.6] | −1.90 [−2.61, −1.21] |
| | 509 | 26.5% [25.1, 28.0] | −2.52 [−3.28, −1.80] |
| | 678 | 27.8% [26.4, 29.1] | −2.41 [−3.12, −1.70] |
| mlp-quarters | 170 | 21.9% [20.6, 23.3] | −1.78 [−2.46, −1.13] |
| | 339 | 15.3% [13.9, 16.7] | −2.24 [−2.91, −1.61] |
| | 509 | 17.7% [16.4, 19.0] | −2.06 [−2.73, −1.38] |
| | 678 | 17.9% [16.5, 19.3] | −1.96 [−2.63, −1.31] |
| extra-block | 170 | 25.1% [23.8, 26.4] | −2.16 [−2.86, −1.45] |
| | 339 | 23.1% [21.7, 24.3] | −1.69 [−2.36, −1.04] |
| | 509 | 24.4% [22.9, 25.8] | −2.10 [−2.80, −1.42] |
| | 678 | 24.8% [23.6, 26.0] | −1.88 [−2.56, −1.21] |
| extra-mlp | 170 | 25.8% [24.4, 27.2] | −1.92 [−2.59, −1.25] |
| | 339 | 21.5% [20.0, 22.9] | −1.70 [−2.37, −1.04] |
| | 509 | 23.5% [22.1, 24.9] | −1.80 [−2.50, −1.12] |
| | 678 | 25.1% [23.8, 26.4] | −1.63 [−2.34, −0.94] |
| **extra-mlp-64** | **170** | **28.7% [27.5, 30.0]** | **−1.71 [−2.40, −1.02]** |
| | 339 | 23.3% [22.0, 24.6] | −1.71 [−2.36, −1.06] |
| | 509 | 25.9% [24.6, 27.2] | −1.72 [−2.40, −1.04] |
| | 678 | 25.8% [24.4, 27.0] | −1.59 [−2.28, −0.90] |

- **Every arm except mlp-quarters beats LoRA on compression** (LoRA: 22.5% at −1.58) at a similar accuracy cost.
  extra-mlp-64 is the best trade-off. It gets close to mlp-mid's compression while losing 0.7 pp less, and it needs no plugin.
- **One MLP at layer 13 beats four.** mlp-quarters has 4× the parameters and compresses less.
- **Size does not matter much.** extra-block (15.7M), extra-mlp (9.4M) and extra-mlp-64 (0.2M) all fall within a few
  points of each other.
- Every arm, like LoRA, dips at step 339 and partly recovers. Step 170 compresses most, and later steps lose slightly less accuracy.

## 3. Placement: after layer 13, 17 or 20

extra-mlp-64 at step 170 (half an epoch), placed at three depths:

| Inserted after | Tokens saved | Accuracy change | pp lost per 1% saved |
|---|---|---|---:|
| layer 13 | **28.7% [27.5, 30.0]** | −1.71 [−2.40, −1.02] | 0.060 |
| layer 17 | 21.9% [20.5, 23.3] | −1.64 [−2.27, −1.02] | 0.075 |
| layer 20 | 19.5% [18.2, 20.9] | **−1.11 [−1.75, −0.52]** | 0.057 |

Earlier placement compresses more. Layer 20 has the smallest accuracy loss of any trained configuration, but its CI still
excludes 0 and overlaps layer 13's. The pp-per-1% rate is about the same at 13 and 20.

## 4. Per benchmark

Each cell is tokens saved / accuracy change in pp.

| Configuration | bfcl_irrelevance | bfcl_multiple | bfcl_simple_python | ifeval | mbpp | svamp |
|---|---|---|---|---|---|---|
| LoRA coached step 170 | 24% / −0.3 | 17% / −3.2 | 17% / −1.1 | 21% / −0.4 | 13% / −1.8 | 35% / −2.7 |
| mlp-mid step 170 | 29% / +0.2 | 26% / −5.7 | 25% / −1.2 | 29% / −1.2 | 19% / −3.4 | 42% / −3.0 |
| mlp-quarters step 170 | 20% / −0.3 | 18% / −4.3 | 17% / −1.4 | 20% / −1.2 | 13% / −2.0 | 35% / −2.2 |
| extra-block step 339 | 18% / +0.5 | 20% / −3.2 | 17% / −0.7 | 24% / −1.6 | 12% / −2.2 | 36% / −1.9 |
| extra-mlp step 678 | 22% / +0.6 | 23% / −2.9 | 20% / −0.7 | 24% / −2.2 | 13% / −1.8 | 40% / −1.7 |
| extra-mlp-64 after 13, step 170 | 26% / +0.2 | 28% / −4.3 | 24% / −1.2 | 31% / −1.1 | 16% / −2.0 | 40% / −2.3 |
| extra-mlp-64 after 17, step 170 | 27% / +1.1 | 20% / −4.6 | 17% / −0.9 | 17% / −0.7 | 7% / −1.9 | 39% / −2.7 |
| extra-mlp-64 after 20, step 170 | 21% / −0.3 | 16% / −2.0 | 13% / −1.0 | 12% / −0.7 | 13% / −0.8 | 35% / −2.0 |
| force `First` | 20% / −0.2 | 19% / −0.7 | 20% / −0.8 | 9% / −0.8 | 21% / −1.6 | 56% / −8.0 |
| force `To` | 38% / −3.0 | 39% / −3.9 | 31% / −1.1 | 20% / −3.0 | 78% / −13.2 | 59% / −12.3 |

The losses concentrate in two places: **bfcl_multiple**, where choosing among several functions costs −2 to −6 pp, and
**svamp**, where the arithmetic is cut short. svamp also saves the most tokens. bfcl_irrelevance is free to compress.

## 5. Forced first thinking token

This follows Liu et al. 2026 ([arXiv:2610.34350](https://arxiv.org/abs/2610.34350)). The reply is pre-filled with
`<think>\nWORD`, so the student's first reasoning word is WORD instead of its own choice. Qwen3-0.6B opens with "Okay"
99.9% of the time. The forced tokens count toward the reasoning length and the generation cap.
The code is [`thinkingcap/first_token.py`](thinkingcap/first_token.py).

| Forced word | On | Tokens saved | Accuracy change |
|---|---|---|---|
| `Okay` | base | −0.1% [−0.5, 0.4] | −0.01 [−0.16, +0.14] |
| `Alright` | base | 2.7% [1.4, 3.9] | +0.04 [−0.49, +0.58] |
| `First` | base | 29.2% [27.8, 30.6] | −2.74 [−3.46, −2.05] |
| `To` | base | 53.6% [52.1, 55.0] | −7.68 [−8.58, −6.85] |
| `First` | extra-mlp-64 step 170 | 51.9% [50.6, 53.2] | −6.19 [−7.05, −5.38] |
| `First` | extra-mlp step 678 | 52.2% [51.0, 53.5] | −6.24 [−7.05, −5.45] |
| `To` | extra-mlp-64 step 170 | 63.9% [62.3, 65.3] | −8.47 [−9.44, −7.55] |
| `To` | extra-mlp step 678 | 67.3% [66.2, 68.4] | −8.14 [−9.12, −7.21] |

- **`Okay` is the harness check.** Forcing the model's natural word changes nothing, so the forcing itself is neutral.
- **`Alright` is the only free option**, saving 2.7% with no accuracy change, but that is small.
- **`First` and `To` trade accuracy for tokens.** mbpp and svamp take most of the damage: `To` costs −13 pp on mbpp and −12 pp on svamp.
- **Stacking with an arm**: token savings compound roughly multiplicatively (29% and 29% give 52%).
  For `First` the accuracy loss is worse than the sum of the parts: −6.2 pp against −2.7 and −1.7. svamp falls 14 pp.
  `To` already loses so much alone that the arm adds less than its own −1.7 pp.
- None of this can be submitted, because ThinkingCap scores weights with an unmodified prompt.

## 6. Layer probe: where does the stop decision live?

[`thinkingcap/layer_probe.py`](thinkingcap/layer_probe.py) hooks every decoder layer. It replays 600 clean base-model
traces (100 per benchmark) and measures three things:

- **Plan r**: a ridge probe on the `<think>` state, before any reasoning is written. It predicts the prompt's thinking
  length and is scored as held-out Pearson r within each dataset, because task identity alone predicts length
  (pooled r is about 0.85).
- **Stop contrast**: the logit-lens `</think>` logprob at the real stop position, minus its mean over the trace's other positions.
- **`Wait` top-1**: how often the logit lens already ranks `Wait` first at the positions where the model restarts.

The arm columns are extra-mlp-64 step 170. From layer 14 on, the arm's layers are shifted by one so that each row
compares the same original layer.

| Layer | Plan r | Stop contrast | `Wait` top-1 | Arm: stop contrast | Arm: `Wait` top-1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.19 | 6.6 | 0% | 6.6 | 0% |
| 5 | 0.31 | 5.8 | 0% | 5.8 | 0% |
| 9 | 0.40 | 5.3 | 0% | 5.3 | 0% |
| 13 | 0.41 | 4.5 | 0% | 4.5 | 0% |
| 15 | 0.44 | 4.8 | 0% | 4.6 | 0% |
| 17 | 0.39 | 2.6 | 0% | 3.0 | 0% |
| 19 | 0.40 | 7.5 | 3% | 7.4 | 3% |
| 20 | 0.42 | 11.4 | 5% | 10.8 | 4% |
| 21 | 0.41 | 17.3 | 6% | 17.2 | 5% |
| 22 | 0.43 | 16.8 | 10% | 16.9 | 7% |
| 23 | 0.41 | 17.6 | 17% | 17.4 | 12% |
| 24 | 0.38 | 18.0 | 46% | 17.4 | 36% |
| 25 | 0.38 | 17.2 | 16% | 16.0 | 16% |
| 27 (final) | 0.37 | 33.0 | 72% | 36.8 | 62% |

- **Length is planned early.** By layer 9 the `<think>` state predicts within-dataset length at r of about 0.4,
  and the prediction stays flat through layer 22.
- **The stop and restart decisions are made late.** Stop contrast jumps at layers 19–21, and `Wait` only becomes
  readable at 22–24. This matches the placement results: an insert after layer 13 sits between the plan and the
  decision, and after layer 20 it arrives as the decision is forming.
- **What the trained block changes**: almost nothing in the residual stream up to layer 21. It suppresses `Wait`
  (24: 36% vs 46%, final: 62% vs 72%) and sharpens the final stop (36.8 vs 33.0).
  The block compresses by cutting restarts, not by planning a shorter length.

Probe outputs are in [`thinkingcap/results/probes/`](thinkingcap/results/probes/).

## Caveats and open issues

- **No result meets the goal.** Equal accuracy would need an accuracy CI that includes 0 alongside real savings.
  The best is after-layer-20 at −1.11 [−1.75, −0.52].
- **Step-0 smoke divergence is unresolved.** On the 30-question smoke set, the step-0 (identity) exports of the
  arms differed from the base run by 8–10% in tokens. The CIs include 0, and the logprob checks above pass,
  but the planned decode check was never run.
- **One seed per configuration.** The CIs cover prompt sampling, not training variance.
- **Arm weights are not in git** (about 50 MB per checkpoint). They are on the `thinkingcap` Modal volume under
  `runs/full/arms/NAME/step-N/`. LoRA adapters are committed.

## Original pilot

Before ThinkingCap, the same idea was tested locally on a 16 GB ROCm GPU, with llama.cpp serving a Q4 Qwen3-8B teacher
and a 131k-parameter residual MLP after layer 13 against LoRA. On a 6-problem GSM8K check, the MLP used 13.9% fewer
thinking tokens and LoRA 32.0% fewer, both with the same 4/6 correct. That check was meant to test the pipeline,
not to measure anything. Code and notes: [pilot/README.md](pilot/README.md) and
[pilot/FIRST_RESULTS.md](pilot/FIRST_RESULTS.md).

## Layout

```
thinkingcap/
  pipeline.py        all GPU stages: setup, baseline, coach, candidates, train, train-arm, verify-arms, eval, compare
  modal_app.py       Modal launcher and stage plans (LoRA run and arms)
  arms.py            ReasoningMLP and inserted-block arms; arms.json defines them
  vllm_plugin/       vLLM model class that serves the MLP arms
  first_token.py     forced-first-token eval (+ first_token_app.py on Modal)
  layer_probe.py     logit lens + plan probe (+ layer_probe_app.py on Modal)
  config.json        all settings for the run
  results/           summaries, per-eval compare output, LoRA adapters, probes, verify logs
pilot/               the original local pilot (self-contained)
```
