#!/usr/bin/env bash
set -e

GPU_ID=$1
EVAL_DATA_PATH=$2
CHECKPOINT_PATH=$3
GRID_PATH=$4

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export CUDA_LAUNCH_BLOCKING=1
python train_mapd.py \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --solver "PBS" \
  --test_checkpoint "${CHECKPOINT_PATH}" \
  --grid_path "${GRID_PATH}" \
  --save_dir "${MODEL_DIR}"