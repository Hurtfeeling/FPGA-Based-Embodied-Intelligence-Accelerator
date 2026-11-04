#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
export PYTHONPATH="$task_repo_root/src:$task_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "${SMOLVLA_PYTHON:-/home/ial-mohd/miniforge3/envs/lerobot/bin/python}" \
  scripts/train_smolvla_s8_distill.py \
  --teacher outputs/train/smolvla_20261103_wrist_beside_base_200k_3gpu/checkpoints/200000/pretrained_model \
  --vlm-assets /home/ial-mohd/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --dataset-root /data/ial-dataset/ial-mohd/test_20261102_142034_cleaned_wrist_beside \
  --output outputs/train/smolvla_s8_wrist_beside_distill \
  --device cuda:0 \
  --steps 20000 \
  --batch-size 2 \
  --grad-accum 4 \
  --workers 2 \
  --lr 3e-5 \
  --warmup-steps 300 \
  --eval-every 500 \
  --eval-batches 32 \
  --save-every 2000 \
  "$@"
