# Motivation

When I saw this problem, it intrigued me, because right now all the talk is about efficient inference, and as a high school student I need to save every token :). In school, teachers teach us how to think more efficiently, so I thought, why couldn't that be applied to an LLM? We already distill bigger models into smaller ones, so why not teach a model how to reason more efficiently? LoRA is a great tool and has a lot of great uses, but I wanted to try something more interesting and experimental, so I landed on putting an MLP in between two layers.

# Method of teaching

The teacher (Qwen3-8B) never writes the solution or anything for the student to learn from directly. It only gives hints about what to skip or optimize, which are added to the prompt. The teacher never sees the gold answer, just one correct reasoning trace of the student. We check that the hints do not contain the answer (the gold number for math, or any option letter or text for multiple choice), and if they do, we throw them away and ask again, so there is no data leakage. After that, we let the student answer 4 more times, and then we take the shortest correct answer and set it as the target.

# Training

The student is trained on the original question without the teacher's hints, paired with the shorter answer it
wrote with the hints. At test time there is no teacher, so the model has to learn to be concise on its own.

Instead of LoRA, I decided to try a small MLP inserted between two transformer layers, after layer 13. The rest of
the model is frozen, so only the MLP is trained. To be precise, it is a copy of layer 13 inserted after it with the attention switched off, so the final model is a normal 29-layer Qwen3 that loads without any custom code. It starts with zero output, which means at the beginning the model
behaves exactly like the original and can only learn changes on top of it. It worked better than LoRA: 28.7% fewer
tokens against LoRA's 22.5%, at a similar accuracy cost (-1.71 pp vs -1.58 pp). It also needed far fewer trainable
parameters, about 198k against LoRA's 1.1M.

What surprised me was that the smaller MLPs performed better. A full inserted layer (15.7M parameters) saved 25.1%
and a full-width MLP (9.4M) saved 25.8%, while the version with only 64 active channels (198k) saved 28.7%. I think
this is because the small MLP can learn the patterns much more quickly and does not need as much training data. It can only learn a simple general rule like "stop re-checking", not memorize the training answers. A layer probe I ran backs this up: at the points where the model restarts its thinking, the trained model picks "Wait" as the next word 62% of the time instead of 72%. This also matches the fact that every version compressed the most after only half an epoch, and training longer did not help (later checkpoints only lost slightly less accuracy). The smallest MLP I tried (131k) compressed even more but lost more accuracy, so 64 channels was the best config I found for now. That 131k version was a different design, a separate residual MLP rather than a copy of a Qwen3 layer, so stock vLLM cannot load it: I scored it through my own vLLM plugin, and it cannot be submitted.

To make sure the improvement comes from the teacher, I ran a control experiment: the same pipeline and training, but
without any hints. The student answered 4 times on the plain question and I trained on its shortest correct answer.
Those targets were still clearly shorter than usual (0.65× the normal length, against 0.44× for the coached ones), but
the model learned almost nothing from them: only 0-2% fewer tokens, compared with 22.5% for the coached LoRA and 28.7%
for the MLP. So picking short answers alone is not enough. The teacher's hints produce answers that are short in a way
the student can actually learn.

# Surprising findings

 - Adding a transformer block instead of an MLP. I tried that, and the tokens saved got worse compared to the MLP, and it also had about 80 times more trainable parameters (15.7M).
 - LoRA compressed less than almost every other experiment, which surprised me, because I expected that with almost 6 times more trainable parameters it would adapt better through plain brute force.
 - Adding more MLPs between layers: for example, I tried putting one after every quarter of the model, and it also compressed less, even with 4 times the parameters. These used the same plugin design as the 131k MLP, so they cannot be submitted either.
 - The most compression came after only half an epoch.

# Extra experiment
While experimenting with this task, I read a [paper](https://arxiv.org/pdf/2610.34350v1) which presented a crazy stupid idea: just by fixing the first output token, you can influence the length of the reasoning. I first implemented this idea on its own, prefixing the token "To", and the number of generated tokens dropped a lot (53.6%) while accuracy dropped by 7.68 pp. When I combined it with the full-width MLP (9.4M parameters), I achieved 67.3% fewer generated tokens with an 8.14 pp drop in accuracy. I still cannot believe that such a small change can change the reasoning length so drastically. This is only a decoding trick, not a change to the weights, so it cannot be submitted.

# Results

My best result was the MLP with only 64 active channels after layer 13 (198k trainable parameters), trained for half an epoch: 28.7% fewer generated tokens for -1.71 pp of accuracy. None of my runs kept the accuracy fully equal though: even for the best one, the confidence interval does not include 0, so it is a real (small) accuracy loss. Putting the same MLP later, after layer 20, lost the least accuracy of all runs (-1.11 pp) but also saved less (19.5%). All the data from every run is in [data.md](data.md).

# Hardware
Everything ran on Modal, one H100 80GB per job, about 35–38 GPU-hours and $150 of credits in total. Most of it went to the baseline sampling (11 h) and the 39 evaluations (~19 h). Training was the cheap part: under 1.5 h for all runs together. Before Modal, I tested the idea locally on a 16 GB AMD GPU.
