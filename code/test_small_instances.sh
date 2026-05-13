#!/usr/bin/env bash
set -euo pipefail

# 单卡快速批量测试脚本
# 固定参数：
# - checkpoint: reinforce_gnn_mapd_model_18000_steps.zip
# - hidden_dim: 256
# - infer_decode_mode: hungarian
# - task_truncated_size: 2

GPU_ID="${GPU_ID:-1}"
CHECKPOINT="${CHECKPOINT:-/local-scratchg/yifan/2024/MAPD/MAPD_RL/models/gnn_trun2_kl_20260311_2244/checkpoints/reinforce_gnn_mapd_model_48000_steps.zip}"
BASE_DIR="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small"
LOG_DIR="${LOG_DIR:-/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/small_eval_logs}"

AGENTS=(10 20 30 40 50)
FREQS=(5)
SEEDS=(0)

mkdir -p "${LOG_DIR}"

# for a in "${AGENTS[@]}"; do
#   MAP_PATH="${BASE_DIR}/kiva-${a}-500-5.map"
#   for f in "${FREQS[@]}"; do
#     TASK_PATH="${BASE_DIR}/kiva-${f}.task"
#     echo "=============================="
#     echo "Running: agents=${a}, freq=${f}"
#     echo "map:  ${MAP_PATH}"
#     echo "task: ${TASK_PATH}"
#     bash test_gnn.sh \
#       -d "${TASK_PATH}" \
#       -g "${GPU_ID}" \
#       --hidden_dim 256 \
#       --grid_path "${MAP_PATH}" \
#       -c "${CHECKPOINT}" \
#       --infer_decode_mode hungarian \
#       --task_truncated_size 2
#   done
# done
for f in "${FREQS[@]}"; do
  TASK_PATH="${BASE_DIR}/kiva-${f}.task"
  for a in "${AGENTS[@]}"; do
    MAP_PATH="${BASE_DIR}/kiva-${a}-500-5.map"
    for s in "${SEEDS[@]}"; do
      LOG_FILE="${LOG_DIR}/a${a}_f${f}_seed${s}.out"
      echo "=============================="
      echo "Running: agents=${a}, freq=${f}, seed=${s}"
      echo "map:  ${MAP_PATH}"
      echo "task: ${TASK_PATH}"
      echo "log:  ${LOG_FILE}"
      bash test_gnn.sh \
        -d "${TASK_PATH}" \
        -g "${GPU_ID}" \
        --hidden_dim 256 \
        --grid_path "${MAP_PATH}" \
        -c "${CHECKPOINT}" \
        --infer_decode_mode hungarian \
        --task_truncated_size 2 \
        --test_env_seed "${s}" \
        > "${LOG_FILE}" 2>&1
    done
  done
done

echo "All small instance tests finished."

