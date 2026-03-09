#!/bin/bash
set -e

########################################
# Two-stage training script:
#   1) pretrain stage
#   2) resume checkpoint for RL stage (pretrain_steps=0)
#
# Required:
#   -g <gpu_id> -l <learning_rate> -m <gamma> -t <task_num> -p <n_envs>
#
# Example:
#   nohup bash ./train_pretrain_then_rl.sh \
#     -g 5 -l 3e-4 -m 0.99 -t 500 -p 16 \
#     --use_gumbel_hungarian \
#     --pretrain_steps 3000 \
#     --rl_n_steps 4 \
#     --rl_total_timesteps 100000 > pretrain_then_rl.out 2>&1 &
########################################

# Required
GPU_ID=""
LEARNING_RATE=""
GAMMA=""
TASK_NUM=""
PROCESS_NUM=""

# Optional - experiment
EXPERIMENT_NAME="gnn_pretrain_then_rl"

# Optional - model hyper-params (aligned with train_gnn.sh defaults)
HIDDEN_DIM="256"
GRID_FEATURE_DIM="2"
LOWER_GNN_TYPE="sp_mpnn"
HIGHER_GNN_TYPE="edge_node_gnn"
LOWER_GNN_NUM_LAYERS="9"
HIGHER_GNN_NUM_LAYERS="3"
GNN_DROPOUT="0.1"
GNN_HEADS="4"
EDGE_COMBINE="concat"
MAX_DISTANCE="3"
SELF_ATTENTION_LAYERS="2"
TAU="0.1"
ITERATIONS="5"

# Optional - two-stage controls
PRETRAIN_STEPS="3000"
PRETRAIN_N_STEPS="1"
RL_N_STEPS="4"
PRETRAIN_TOTAL_TIMESTEPS=""     # if empty, auto-compute
RL_TOTAL_TIMESTEPS="100000"
RL_N_SAMPLES="1"
RL_CENTERED_WEIGHT="0.7"
NEAREST_TASKS_MIN_K="8"

# Flags
USE_SINKHORN_FLAG="--use_sinkhorn"
USE_HUNGARIAN_FLAG="--use_hungarian_for_deterministic"
USE_UNDIRECTED_FLAG="--use_undirected"
USE_GUMBEL_HUNGARIAN_FLAG=""
USE_GUMBEL_SINKHORN_FLAG=""
FIX_DIV_FLAG=""
NOT_DIV_FLAG=""
NORMALIZE_ADVANTAGE_FLAG=""
POS_REWARD_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -g)
      GPU_ID="$2"; shift 2 ;;
    -l)
      LEARNING_RATE="$2"; shift 2 ;;
    -m)
      GAMMA="$2"; shift 2 ;;
    -t)
      TASK_NUM="$2"; shift 2 ;;
    -p)
      PROCESS_NUM="$2"; shift 2 ;;
    -e)
      EXPERIMENT_NAME="$2"; shift 2 ;;

    --hidden_dim)
      HIDDEN_DIM="$2"; shift 2 ;;
    --lower_gnn_type)
      LOWER_GNN_TYPE="$2"; shift 2 ;;
    --higher_gnn_type)
      HIGHER_GNN_TYPE="$2"; shift 2 ;;
    --lower_gnn_num_layers)
      LOWER_GNN_NUM_LAYERS="$2"; shift 2 ;;
    --higher_gnn_num_layers)
      HIGHER_GNN_NUM_LAYERS="$2"; shift 2 ;;
    --gnn_dropout)
      GNN_DROPOUT="$2"; shift 2 ;;
    --gnn_heads)
      GNN_HEADS="$2"; shift 2 ;;
    --edge_combine)
      EDGE_COMBINE="$2"; shift 2 ;;
    --max_distance)
      MAX_DISTANCE="$2"; shift 2 ;;
    --self_attention_layers)
      SELF_ATTENTION_LAYERS="$2"; shift 2 ;;
    --tau)
      TAU="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;

    --pretrain_steps)
      PRETRAIN_STEPS="$2"; shift 2 ;;
    --pretrain_n_steps)
      PRETRAIN_N_STEPS="$2"; shift 2 ;;
    --rl_n_steps)
      RL_N_STEPS="$2"; shift 2 ;;
    --pretrain_total_timesteps)
      PRETRAIN_TOTAL_TIMESTEPS="$2"; shift 2 ;;
    --rl_total_timesteps)
      RL_TOTAL_TIMESTEPS="$2"; shift 2 ;;
    --rl_n_samples)
      RL_N_SAMPLES="$2"; shift 2 ;;
    --rl_centered_weight)
      RL_CENTERED_WEIGHT="$2"; shift 2 ;;
    --nearest_tasks_min_k)
      NEAREST_TASKS_MIN_K="$2"; shift 2 ;;

    --use_sinkhorn)
      USE_SINKHORN_FLAG="--use_sinkhorn"; shift ;;
    --no_sinkhorn)
      USE_SINKHORN_FLAG=""; shift ;;
    --use_hungarian)
      USE_HUNGARIAN_FLAG="--use_hungarian_for_deterministic"; shift ;;
    --no_hungarian)
      USE_HUNGARIAN_FLAG=""; shift ;;
    --use_undirected)
      USE_UNDIRECTED_FLAG="--use_undirected"; shift ;;
    --no_undirected)
      USE_UNDIRECTED_FLAG=""; shift ;;
    --use_gumbel_hungarian)
      USE_GUMBEL_HUNGARIAN_FLAG="--use_gumbel_hungarian"; shift ;;
    --use_gumbel_sinkhorn)
      USE_GUMBEL_SINKHORN_FLAG="--use_gumbel_sinkhorn"; shift ;;
    --fix_div)
      FIX_DIV_FLAG="--fix_div"; shift ;;
    --not_div)
      NOT_DIV_FLAG="--not_div"; shift ;;
    --normalize_advantage)
      NORMALIZE_ADVANTAGE_FLAG="--normalize_advantage"; shift ;;
    --pos_reward)
      POS_REWARD_FLAG="--pos_reward"; shift ;;
    *)
      echo "Unknown parameter: $1"
      exit 1 ;;
  esac
done

if [ -z "$GPU_ID" ] || [ -z "$LEARNING_RATE" ] || [ -z "$GAMMA" ] || [ -z "$TASK_NUM" ] || [ -z "$PROCESS_NUM" ]; then
  echo "Usage: $0 -g <gpu_id> -l <learning_rate> -m <gamma> -t <task_num> -p <n_envs> [options]"
  exit 1
fi

if [ -z "${PRETRAIN_TOTAL_TIMESTEPS}" ]; then
  PRETRAIN_TOTAL_TIMESTEPS=$(( PRETRAIN_STEPS / PROCESS_NUM ))
  if [ "${PRETRAIN_TOTAL_TIMESTEPS}" -lt 1 ]; then
    PRETRAIN_TOTAL_TIMESTEPS=1
  fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="../models/gnn_${EXPERIMENT_NAME}_${TIMESTAMP}_lr_${LEARNING_RATE}_gamma_${GAMMA}_task_${TASK_NUM}"
PRETRAIN_DIR="${BASE_DIR}/stage_pretrain"
RL_DIR="${BASE_DIR}/stage_rl"
mkdir -p "${PRETRAIN_DIR}/checkpoints"
mkdir -p "${RL_DIR}/checkpoints"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_LAUNCH_BLOCKING=1

echo "============================================="
echo "Two-stage training (Pretrain -> RL Resume)"
echo "GPU=${GPU_ID}, LR=${LEARNING_RATE}, gamma=${GAMMA}, n_envs=${PROCESS_NUM}"
echo "Pretrain: pretrain_steps=${PRETRAIN_STEPS}, n_steps=${PRETRAIN_N_STEPS}, total_timesteps=${PRETRAIN_TOTAL_TIMESTEPS}"
echo "RL: pretrain_steps=0, n_steps=${RL_N_STEPS}, total_timesteps=${RL_TOTAL_TIMESTEPS}, rl_n_samples=${RL_N_SAMPLES}"
echo "Model: lower=${LOWER_GNN_TYPE}, higher=${HIGHER_GNN_TYPE}, hidden=${HIDDEN_DIM}"
echo "Save root: ${BASE_DIR}"
echo "============================================="

COMMON_ARGS="\
  --learning_rate ${LEARNING_RATE} \
  --gamma ${GAMMA} \
  --hidden_dim ${HIDDEN_DIM} \
  --grid_feature_dim ${GRID_FEATURE_DIM} \
  --lower_gnn_type ${LOWER_GNN_TYPE} \
  --higher_gnn_type ${HIGHER_GNN_TYPE} \
  --lower_gnn_num_layers ${LOWER_GNN_NUM_LAYERS} \
  --higher_gnn_num_layers ${HIGHER_GNN_NUM_LAYERS} \
  --gnn_dropout ${GNN_DROPOUT} \
  --gnn_heads ${GNN_HEADS} \
  --edge_combine ${EDGE_COMBINE} \
  --max_distance ${MAX_DISTANCE} \
  --self_attention_layers ${SELF_ATTENTION_LAYERS} \
  --global_seed 40 \
  --grid_path /local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map \
  --training \
  --task_num ${TASK_NUM} \
  --n_envs ${PROCESS_NUM} \
  --tau ${TAU} \
  --iterations ${ITERATIONS} \
  --rl_policy row_softmax \
  --rl_centered_weight ${RL_CENTERED_WEIGHT} \
  --nearest_tasks_min_k ${NEAREST_TASKS_MIN_K} \
  ${USE_SINKHORN_FLAG} \
  ${USE_HUNGARIAN_FLAG} \
  ${USE_UNDIRECTED_FLAG} \
  ${USE_GUMBEL_HUNGARIAN_FLAG} \
  ${USE_GUMBEL_SINKHORN_FLAG} \
  ${FIX_DIV_FLAG} \
  ${NOT_DIV_FLAG} \
  ${NORMALIZE_ADVANTAGE_FLAG} \
  ${POS_REWARD_FLAG}"

echo "[Stage 1/2] Pretraining..."
python train_mapd_gnn.py \
  ${COMMON_ARGS} \
  --n_steps "${PRETRAIN_N_STEPS}" \
  --total_timesteps "${PRETRAIN_TOTAL_TIMESTEPS}" \
  --pretrain_steps "${PRETRAIN_STEPS}" \
  --ent_coef 0.001 \
  --checkpoint_freq 125 \
  --save_dir "${PRETRAIN_DIR}" \
  --rl_n_samples 1

PRETRAIN_CKPT="${PRETRAIN_DIR}/final_model.zip"
if [ ! -f "${PRETRAIN_CKPT}" ]; then
  echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}"
  exit 1
fi

echo "[Stage 2/2] RL training from pretrain checkpoint..."
python train_mapd_gnn.py \
  ${COMMON_ARGS} \
  --resume_checkpoint "${PRETRAIN_CKPT}" \
  --n_steps "${RL_N_STEPS}" \
  --total_timesteps "${RL_TOTAL_TIMESTEPS}" \
  --pretrain_steps 0 \
  --ent_coef 0.001 \
  --checkpoint_freq 125 \
  --save_dir "${RL_DIR}" \
  --rl_n_samples "${RL_N_SAMPLES}"

echo "Done."
echo "Pretrain model: ${PRETRAIN_CKPT}"
echo "RL final model: ${RL_DIR}/final_model.zip"

