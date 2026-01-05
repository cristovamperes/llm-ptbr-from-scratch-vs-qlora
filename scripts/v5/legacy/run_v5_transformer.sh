#!/usr/bin/env bash
set -euo pipefail

CUDA_PIP_LIB="/usr/local/lib/python3.10/dist-packages/nvidia"
export LD_LIBRARY_PATH="${CUDA_PIP_LIB}/cudnn/lib:${CUDA_PIP_LIB}/cublas/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "${TOKENIZER_MODEL:-}" ]]; then
  TOKENIZER_MODEL="versions/v4-subword-lstm/tokenizer_v4k_bf/spm_v4k_bf.model"
fi

python3 scripts/llm_transformer_v5.py \
  --tokenizer_path "${TOKENIZER_MODEL}" \
  --seq_len 320 \
  --stride 2 \
  --batch_size 192 \
  --shuffle_buffer 400000 \
  --epochs 2 \
  --train \
  --learning_rate 3e-4 \
  --warmup_steps 2000 \
  --steps_per_epoch 41653 \
  --total_steps 10000 \
  --tensorboard_logdir versions/v5-transformer/logs/tensorboard "$@"
