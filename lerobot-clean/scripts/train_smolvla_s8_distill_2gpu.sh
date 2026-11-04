#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
export CUDA_VISIBLE_DEVICES="${SMOLVLA_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec "${SMOLVLA_PYTHON:-/home/ial-mohd/miniforge3/envs/lerobot/bin/python}" \
  -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  --no-python bash scripts/train_smolvla_s8_distill.sh \
  --output outputs/train/smolvla_s8_wrist_beside_distill_2gpu \
  --grad-accum 2 \
  "$@"
