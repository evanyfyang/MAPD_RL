#!/usr/bin/env bash
set -e

# Usage:
#   bash eval_model_only_fixed.sh <checkpoint> <grid_path> <task_file> [episodes] [gpu_id]
#
# Example:
#   bash eval_model_only_fixed.sh \
#     "../models/xxx/checkpoints/reinforce_gnn_mapd_model_10000_steps.zip" \
#     "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-20-500-5.map" \
#     "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-2.task" \
#     5 7

CKPT_PATH="${1:-}"
GRID_PATH="${2:-}"
TASK_PATH="${3:-}"
EPISODES="${4:-5}"
GPU_ID="${5:-}"

if [[ -z "${CKPT_PATH}" || -z "${GRID_PATH}" || -z "${TASK_PATH}" ]]; then
  echo "Usage: $0 <checkpoint> <grid_path> <task_file> [episodes] [gpu_id]"
  exit 1
fi

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

python train_mapd_gnn.py \
  --test_checkpoint "${CKPT_PATH}" \
  --grid_path "${GRID_PATH}" \
  --eval_data_path "${TASK_PATH}" \
  --test_episodes "${EPISODES}" \
  --task_num 500 \
  --n_envs 1 \
  --lower_gnn_type sp_mpnn \
  --higher_gnn_type edge_node_gnn \
  --edge_combine concat \
  --use_undirected \
  --use_gumbel_hungarian \
  --model_only_eval
