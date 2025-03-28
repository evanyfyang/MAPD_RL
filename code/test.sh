#!/usr/bin/env bash
set -e
# bash test.sh 500  /local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-30-500-5.map /local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-5.task /local-scratchg/yifan/2024/MAPD/MAPD_RL/models/_20250320_2108_lr_1e-4_gamma_0.99_tau_0.1_1_0_0_16_500/checkpoints/a2c_mapd_model_48000_steps.zip 5

TASK_NUM=$1
GRID_PATH=$2
EVAL_DATA_PATH=$3
CHECKPOINT_PATH=$4
GPU_ID=$5

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

export CUDA_LAUNCH_BLOCKING=1
python -m pdb train_mapd.py \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --solver "PBS" \
  --test_checkpoint "${CHECKPOINT_PATH}" \
  --grid_path "${GRID_PATH}" \
  --task_num "${TASK_NUM}"