#!/usr/bin/env bash
set -e

# ====== 你只需要改这两个 ======
GPU_ID=0
CHECKPOINT_PATH="/path/to/your/checkpoint.zip"

# ====== 固定测试集（你给的map/task） ======
TASK_FILE="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-40-500-5-n6000-f2.task"
MAP_FILE="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-40-500-5.map"

# ====== throughput配置 ======
THROUGHPUT_HORIZON=1000
THROUGHPUT_PENDING_CAP=500
EVAL_SIM_TIME=1500

cd "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code"

bash ./test_gnn.sh \
  -g "${GPU_ID}" \
  -c "${CHECKPOINT_PATH}" \
  -d "${TASK_FILE}" \
  --grid_path "${MAP_FILE}" \
  --throughput_mode \
  --throughput_horizon "${THROUGHPUT_HORIZON}" \
  --throughput_pending_cap "${THROUGHPUT_PENDING_CAP}" \
  --eval_simulation_time "${EVAL_SIM_TIME}" \
  --infer_decode_mode hungarian \
  --model_only_eval

