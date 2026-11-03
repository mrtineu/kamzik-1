#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export TORCH_HOME="$PWD/.cache/torch"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export TMPDIR="$PWD/.cache/tmp"
export UV_CACHE_DIR="$PWD/.cache/uv"
export HIP_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_XET=1
mkdir -p "$TMPDIR" runs/pilot
.venv/bin/python -u prepare.py
.venv/bin/python -u experiment.py "$@"
