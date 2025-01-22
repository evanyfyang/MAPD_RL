#!/usr/bin/env bash
# 让脚本在遇到错误时退出
set -e

########################################
# 使用说明:
#   ./run_mapd.sh <gpu_id> <learning_rate> <gamma> <tau> <decay_rate>
#   例如:
#     ./run_mapd.sh 0 3e-4 0.99 1.0 1e-4
#   这表示:
#     - 使用 GPU 0
#     - learning_rate = 3e-4
#     - gamma         = 0.99
#     - tau           = 1.0
#     - decay_rate    = 1e-4
#
#   其余参数 (optimizer=Adam, ent_coef=0, hidden_size=128, checkpoint_freq=1000, global_seed=40, grid_path=... ) 固定。
#   脚本会自动在 "../models" 下创建一个带时间戳和超参数信息的目录, 并将 train_mapd.py 的产物保存在其中.
########################################

if [ $# -lt 5 ]; then
  echo ": $0 <gpu_id> <learning_rate> <gamma> <tau> <decay_rate>"
  exit 1
fi

GPU_ID=$1
LEARNING_RATE=$2
GAMMA=$3
TAU=$4
DECAY_RATE=$5
POS_REWARD_FLAG=$6
FIX_DIV_FLAG=$7
NOT_DIV_FLAG=$8

# 生成时间戳 (形如 20250110_1030)
TIMESTAMP=$(date +%Y%m%d_%H%M)

# 构造模型保存目录: ../models/lr_${LEARNING_RATE}_gamma_${GAMMA}_tau_${TAU}_decay_${DECAY_RATE}_${TIMESTAMP}
MODEL_DIR="../models/lr_${LEARNING_RATE}_gamma_${GAMMA}_tau_${TAU}_decay_${DECAY_RATE}"

# 创建目录
mkdir -p "${MODEL_DIR}"

if [ "$POS_REWARD_FLAG" = "1" ]; then
  POS_REWARD_ARG="--pos_reward"
fi

if [ "$FIX_DIV_FLAG" = "1" ]; then
  FIX_DIV_ARG="--fix_div"
fi

if [ "$NOT_DIV_FLAG" = "1" ]; then
  NOT_DIV_ARG="--not_div"
fi

# 让指定的 GPU 可见
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================="
echo "Start Training: GPU=${GPU_ID}, LR=${LEARNING_RATE}, gamma=${GAMMA}, tau=${TAU}, decay_rate=${DECAY_RATE}"
echo "Model Dir: ${MODEL_DIR}"
echo "============================================="

export CUDA_LAUNCH_BLOCKING=1
# 调用 train_mapd.py
python train_mapd.py \
  --learning_rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --tau "${TAU}" \
  --decay_rate "${DECAY_RATE}" \
  --ent_coef 0 \
  --hidden_size 128 \
  --checkpoint_freq 1000 \
  --global_seed 40 \
  --grid_path "/localhome/yya305/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map" \
  --save_dir "${MODEL_DIR}" \
  --training \
  ${POS_REWARD_ARG} \
  ${FIX_DIV_ARG} \
  ${NOT_DIV_ARG}

echo "Finished. Saved: ${MODEL_DIR}"
