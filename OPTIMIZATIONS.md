# Inference optimization

## VRAM guard after the crash

The batch-16, 4096-token complete-answer benchmark did not finish. Its last log entry
reports an allocation failure with about 38 MiB free out of 15.92 GiB. The user reported
a crash. This establishes severe memory pressure; the saved log does not establish
whether the driver also reset. The disabled-attention comparison and GPU profiling
were not reached, so the partial timings do not establish an end-to-end speedup.

`config_optimized.json` retains ordinary batches up to **16**, retries up to **8**,
experimental generation attention, **2 CPU threads**, and math attention for training.
Memory handling now uses the installed ROCm allocator's
`per_process_memory_fraction:0.8,throw_on_cudamalloc_oom:True` options, set before
importing torch under all supported environment aliases. The native allocator rejects
over-budget requests in PyTorch before attempting the corresponding driver allocation.

The effective cap is the smaller of 80% of total VRAM and the current free VRAM plus
our reserved memory minus a 2 GiB reserve. Generation checks device headroom before
allocating inputs and every 16 decode steps. These controls cover this process's
PyTorch allocations; desktop applications and the separate teacher are outside its cap.
This is application-level prevention, not a Linux kernel change.

A recoverable allocator rejection or headroom check splits a generation batch in half.
The split tree and deterministic subgroup seeds are written to `*_execution.json`
before retrying. Resume replays those exact subgroups even if more VRAM is available.
Other runtime errors, a single request that cannot fit, or a partly saved group that
would require changing its RNG assignment stop the run. Token caps are never reduced.
Benchmarks retain the requested shape and stop on budget rejection; their worker
watchdog is 1200 seconds. The full stress benchmark has not been relaunched.

The installed allocator was validated with an artificial **81.5 MiB** process limit:
a **256 MiB** request raised `torch.OutOfMemoryError` while **15.8 GiB** remained free.
A subsequent GPU operation succeeded. See `runs/allocator_guard_validation.json`.
All **19 CPU tests** pass, including budget calculations, HIP environment aliases,
adaptive splitting, interruption recovery, and refusing to retry driver faults.
This verifies the controlled rejection path; a full guarded model run remains untested.

The completed pilot contains 95 individually generated retry attempts whose final-attempt
wall times sum to **111.11 minutes**. This excludes their first attempts and teacher work.
For that same saved retry set, batches of 8 would require **19 generation calls**
across the 11 workloads instead of 95 serial calls. This is a scheduling count,
not a claim that the new run will take one fifth of the time; sampling and batch
membership change generated answers, and fixed batches can still waste decode slots.

## Implemented

- `generation_policy: "batched_v1"` queues retries across the entire workload instead
  of generating each retry individually. First attempts are saved immediately, including
  incomplete responses. `retry_batch_size` is independent of ordinary `batch_size`.
- An immutable plan pins input text, extra instructions, prompt lengths, batch sizes,
  caps, order, and seed offsets. After interruption, only an incomplete batch is
  regenerated with its original members and seed; already saved rows remain intact.
  The initial/retry phases use different seed ranges. Retry groups depend on which
  outputs truncate, so retry RNG is not paired per example across different model arms.
  Memory-induced splitting also changes initial group membership; actual batch IDs
  and seeds are recorded, and the split plan makes later resumes repeatable.
- `batch_order: "prompt_length"` sorts by actual tokenized prompt length before fixed
  batching. This reduces prompt padding. It cannot predict answer length or replace
  finished requests during a generation call. Continuous batching remains a possible
  backend change after profiling; no inference backend has been replaced here.
- `experimental_attention` sets the ROCm AOTriton flag before importing PyTorch.
  SDPA still selects a backend for each shape/mask. In this environment forced flash
  rejects non-null padding masks, so real padded batches retain automatic dispatch.
  Training defaults to forced math attention when this flag is enabled; the optimized
  configuration explicitly sets `training_attention: "math"`. The flash MLP backward
  comparison exceeded the 5% gradient-error tolerance, so forward validation does not
  justify enabling it for training. The context covers backward checkpoint recomputation.
- Generation records wall time, process CPU time, useful tokens/second, peak allocated
  and reserved VRAM, and useful generated tokens divided by decode slots. Low slot
  efficiency exposes time spent processing padding after other requests finish.
  `total_compute_seconds` includes both attempts; existing final-attempt timing and
  final-answer token metrics are preserved.
  Discarded work from memory rejection and crash replay is not included in that timing.

## Validation and measurements

`test_generation.py` checks cross-batch retries, instruction alignment, stable ordering,
restart after partial initial/retry writes, duplicate prevention, plan-change rejection,
and refusal to mix new scheduling with legacy outputs. `test_core.py` covers answer
scoring and adapter identity/gradients.

Each explicitly selected attention mode runs in a fresh process. The interrupted
stress run used saved training traces at 2048/4096-token contexts, fixed decode windows
for kernel comparison, full sampled answers for effective throughput, and batch sizes
4/8/16. Repeated traces were necessary at the longest context; the number of unique
training examples is recorded. The fixed-window rate includes prefill. Per-shape
warmup and repeated runs reduce one-time setup effects.

Correctness checks compare unpadded forced-flash and padded automatic-dispatch logits
against math attention. FP32 math provides a reference for normal BF16 rounding:
flash error must be at most 1.5 times BF16 math error (or 0.5% RMS), with mean
KL divergence below 0.02 and at least 90% greedy-token agreement. Padded BF16 dispatch
must stay below 3% relative RMS with the same token-agreement threshold. MLP gradients
must agree within 5% relative norm; checkpointed LoRA backward must produce finite,
nonzero adapter gradients and leave base weights frozen. These checks establish a
bounded numerical comparison, not equal stochastic outputs or accuracy.

The optional profiler can identify attention kernels, CPU operators and device time;
it was not reached in the interrupted stress run.
GPU busy percentage is sampled from AMD driver sysfs where available. Operator times
can overlap and should not be interpreted as wall-clock utilization.

PyTorch documents backend selection with
[`sdpa_kernel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
and [ROCm attention backends](https://docs.pytorch.org/docs/stable/backends.html).
Its [allocator documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
describes preemptive rejection to avoid fatal ROCm out-of-resources aborts.
