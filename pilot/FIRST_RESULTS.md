# First completed experiment (2026-10-18)

The local pipeline completed: official Qwen3-8B Q4_K_M coached Qwen3-0.6B, successful shorter student completions were selected, and both an inter-layer MLP and LoRA were trained with the base model frozen. No teacher or coaching prompt is used at adapter inference.

This **small preliminary run** used 8 training problems, accepted 6 shorter correct completions, trained for 3 epochs, and evaluated on 2 validation and 6 disjoint test problems. It is a pipeline check, not a reliable estimate of accuracy retention.

| Test arm | Correct / 6 | Mean thinking tokens | Unfinished |
|---|---:|---:|---:|
| baseline | 4/6 | 1110.7 | 0 |
| concise | 5/6 | 1538.7 | 1 |
| mlp | 4/6 | 956.7 | 0 |
| lora | 4/6 | 755.0 | 0 |

The MLP reduced mean thinking tokens by **13.9%**, with the exact same four correct test answers as the baseline. LoRA reduced them by **32.0%**, also with the same four correct test answers. Neither reached the illustrative 50% reduction target. On the two validation problems, the MLP retained both correct answers; LoRA lost one. Six test problems and one seed do not support a broad claim or establish a 98% accuracy-retention margin.

Concise prompting improved the test count to 5/6, but generated more tokens overall. One concise-prompt output remained unfinished at the 4096-token ceiling; its measured length is censored. Inspect raw outputs and per-difficulty results before drawing conclusions from this mean. Token metrics count the final attempt, not the computational cost of initial attempts and retries.

The MLP trained **131,072 parameters** after the 14th transformer layer; LoRA trained **573,440 parameters** (4.375 times as many). Both used the same six training completions, 3 epochs, learning rate 0.0003, and matched evaluation sampling seeds. The final epoch was predeclared. No checkpoint was selected using held-out answers.

The larger run uses **64 train / 16 validation / 48 test** problems. It starts automatically after this preliminary run, reusing only matching generated training data, never the preliminary trained adapters. See `runs/pilot/RESULTS.md` for its accumulating results and `runs/active.json` for current state. It is not yet complete at the time this note was written.

- [Quick full report](runs/quick/RESULTS.md)
- [Larger report](runs/pilot/RESULTS.md)
- [Instructions and cleanup](README.md)

All new model weights, data, checkpoints, and caches are kept inside this project (about 7 GB). It uses the SSD, which had about 102 GB free after setup. The runtime reuses existing PyTorch and llama.cpp installations rather than duplicating them. No paid API was used.
