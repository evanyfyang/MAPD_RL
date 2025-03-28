#!/usr/bin/env bash
# Exit script if any command fails
set -e

########################################
# Usage:
#   ./run_mapd.sh -g <gpu_id> -l <learning_rate> -m <gamma> -t <tau> -n <task_num> -p <process_num> [-r] [-f] [-d]
#   Example:
#     ./run_mapd.sh -g 0 -l 3e-4 -m 0.99 -t 1.0 -n 10 -p 5 -r -f
#   Parameters:
#     -g: GPU ID
#     -l: learning rate
#     -m: gamma (discount factor)
#     -t: tau (target network update rate)
#     -n: task number
#     -p: process number
#     -r: enable position reward (optional)
#     -f: enable fixed division (optional)
#     -d: disable division (optional)
########################################

# Initialize default values
GPU_ID=""
LEARNING_RATE=""
GAMMA=""
TAU=""
TASK_NUM=""
PROCESS_NUM=""
POS_REWARD_FLAG=0
FIX_DIV_FLAG=0
NOT_DIV_FLAG=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -g)
      GPU_ID="$2"
      shift 2
      ;;
    -l)
      LEARNING_RATE="$2"
      shift 2
      ;;
    -m)
      GAMMA="$2"
      shift 2
      ;;
    -t)
      TAU="$2"
      shift 2
      ;;
    -n)
      TASK_NUM="$2"
      shift 2
      ;;
    -p)
      PROCESS_NUM="$2"
      shift 2
      ;;
    -r)
      POS_REWARD_FLAG=1
      shift
      ;;
    -f)
      FIX_DIV_FLAG=1
      shift
      ;;
    -d)
      CAL_TYPE="$2"
      shift 2
      ;;
    -e)
      FILENAME="$2"
      shift 2
      ;;
    -h)
      HIDDEN_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check required parameters
if [ -z "$GPU_ID" ] || [ -z "$LEARNING_RATE" ] || [ -z "$GAMMA" ] || [ -z "$TAU" ] || [ -z "$TASK_NUM" ] || [ -z "$PROCESS_NUM" ]; then
  echo "Usage: $0 -g <gpu_id> -l <learning_rate> -m <gamma> -t <tau> -n <task_num> -p <process_num> [-r] [-f] [-d]"
  exit 1
fi

if [ -z "$HIDDEN_SIZE" ]; then
  HIDDEN_SIZE=128
fi
# Generate timestamp (format: 20250110_1030)
TIMESTAMP=$(date +%Y%m%d_%H%M)

# Construct model save directory
MODEL_DIR="../models/${FILENAME}_${TIMESTAMP}_lr_${LEARNING_RATE}_gamma_${GAMMA}_tau_${TAU}_${CAL_TYPE}_${POS_REWARD_FLAG}_${PROCESS_NUM}_${TASK_NUM}"

# Create directory
mkdir -p "${MODEL_DIR}"

# Set optional parameters
POS_REWARD_ARG=""
FIX_DIV_ARG=""

if [ "$POS_REWARD_FLAG" = "1" ]; then
  POS_REWARD_ARG="--pos_reward"
fi

if [ "$FIX_DIV_FLAG" = "1" ]; then
  FIX_DIV_ARG="--fix_div"
fi

# Make specified GPU visible
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================="
echo "Start Training: GPU=${GPU_ID}, LR=${LEARNING_RATE}, gamma=${GAMMA}, tau=${TAU}"
echo "Task Number: ${TASK_NUM}, Process Number: ${PROCESS_NUM}"
echo "Model Directory: ${MODEL_DIR}"
echo "============================================="

export CUDA_LAUNCH_BLOCKING=1
# Call train_mapd.py
python train_mapd.py \
  --learning_rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --tau "${TAU}" \
  --ent_coef 0 \
  --hidden_size 128 \
  --checkpoint_freq 1000 \
  --global_seed 40 \
  --grid_path "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map" \
  --save_dir "${MODEL_DIR}" \
  --training \
  --task_num "${TASK_NUM}" \
  --n_envs "${PROCESS_NUM}" \
  --hidden_size "${HIDDEN_SIZE}" \
  ${POS_REWARD_ARG} \
  ${FIX_DIV_ARG} \
  --cal_type "${CAL_TYPE}"

echo "Training completed. Model saved to: ${MODEL_DIR}"
