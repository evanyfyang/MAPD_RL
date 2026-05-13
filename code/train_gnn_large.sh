#!/bin/bash

########################################
# Usage:
#   ./train_gnn.sh -g <gpu_id> -l <learning_rate> -m <gamma> -s <n_steps> -t <task_num> -p <process_num> [options]
#   Example:
#     ./train_gnn.sh -g 0 -l 3e-4 -m 0.99 -s 5 -t 500 -p 5 --use_sinkhorn
#   Parameters:
#     -g: GPU ID
#     -l: learning rate
#     -m: gamma (discount factor)
#     -s: n_steps for updating (REINFORCE, optional)
#     -t: task number
#     -p: process number (parallel environments)
#     -e: experiment name (optional)
#     --hidden_dim: GNN hidden dimension (default: 128)
#     --grid_feature_dim: Grid feature dimension (default: 2)
#     --lower_gnn_type: Lower level GNN type, gcn, gat, sp_mpnn or cnn_channels (default: cnn_channels)
#     --higher_gnn_type: Higher level GNN type, gat, line_graph or self_attention_gat (default: self_attention_gat)
#     --lower_gnn_num_layers: Number of lower level GNN layers (default: 3)
#     --higher_gnn_num_layers: Number of higher level GNN layers (default: 2)
#     --gnn_dropout: GNN dropout rate (default: 0.1)
#     --gnn_heads: Number of GAT attention heads (default: 4)
#     --edge_combine: Edge feature combination method, add or concat (default: add)
#     --self_attention_layers: Self-attention layers for self_attention_gat (default: 2)
#     --use_sinkhorn: Whether to use Sinkhorn normalization (default: enabled)
#     --no_sinkhorn: Disable Sinkhorn normalization
#     --use_undirected: Whether to use undirected graphs (default: enabled)
#     --no_undirected: Disable undirected graphs
#     --tau: 训练/采样温度参数（默认: 0.1）
#     --iterations: Sinkhorn iterations (default: 5)
#     --unassign_threshold: Threshold for unassigned tasks (default: 0.3679)
#     --invalid_edge_score: Score for invalid edges (default: -100.0)
#     --use_hungarian: Use Hungarian algorithm for deterministic actions (default: enabled)
#     --no_hungarian: Disable Hungarian algorithm
#     --use_gumbel_hungarian: Enable Gumbel+Hungarian mode (pretrain: sigmoid+BCE, RL: gumbel+hungarian)
#     --use_gumbel_sinkhorn: Enable Gumbel Sinkhorn mode (pretrain: sinkhorn+BCE, RL: gumbel_sinkhorn+hungarian)
#     --fix_div: Enable fixed division (flag)
#     --not_div: Disable division (flag)
#     --normalize_advantage: Normalize advantage (flag)
#     --pos_reward: Enable position reward (flag)
#     --rl_n_samples: Number of read-only samples per state for centering (default: 4)
#     --rl_policy: RL log-prob policy for REINFORCE: row_softmax or sinkhorn (default: row_softmax)
#     --rl_centered_weight: Mix weight alpha for centered/returns advantage (default: 0.7)
#     --ent_coef: 熵正则系数（默认: 0）
#     --optimizer: 优化器，adam 或 adamw（默认: adam）
#     --optimizer_weight_decay: 优化器weight decay（默认: 0）
#     --max_grad_norm: 梯度裁剪阈值（默认: 0.5）
#     --target_kl: KL gate阈值，<=0关闭（默认: 0）
#     --lr_schedule: 学习率调度，constant / linear / step（默认: constant）
#     --lr_decay_step_size: step调度中每多少env steps衰减一次（默认: 10000）
#     --lr_decay_gamma: step调度衰减比例（默认: 0.5）
#     --min_learning_rate: 学习率下界（默认: 1e-5）
#     --nearest_tasks_min_k: Unified candidate cap K (default: 100)
#     --task_truncated_size: 每个agent执行队列长度上限（默认: 1；可设为2）
#     --no_explicit_path_feature: 关闭FA/DA头的显式路径长度特征（ablation）
#     --pretrain_steps: 预训练步数（默认: 3000）
#     --n_steps: 覆盖默认n_steps。若显式传入，则自动将rl_n_samples固定为1
########################################

# Initialize default values
GPU_ID=""
LEARNING_RATE=""
GAMMA=""
N_STEPS="1"
TASK_NUM=""
PROCESS_NUM=""
EXPERIMENT_NAME="gnn_experiment"
HIDDEN_DIM="128"
GRID_FEATURE_DIM="2"
LOWER_GNN_TYPE="sp_mpnn"
HIGHER_GNN_TYPE="edge_node_gnn_complex"
TAU="1"
ITERATIONS="5"
UNASSIGN_THRESHOLD="-1"
INVALID_EDGE_SCORE="-100"
# 新增的GNN参数默认值
LOWER_GNN_NUM_LAYERS="6"
HIGHER_GNN_NUM_LAYERS="3"
GNN_DROPOUT="0.1"
GNN_HEADS="4"
EDGE_COMBINE="concat"
MAX_DISTANCE="3"
SELF_ATTENTION_LAYERS="2"
USE_SINKHORN_FLAG="--use_sinkhorn"
USE_HUNGARIAN_FLAG="--use_hungarian_for_deterministic"
USE_UNDIRECTED_FLAG="--use_undirected"
USE_GUMBEL_HUNGARIAN_FLAG=""
USE_GUMBEL_SINKHORN_FLAG=""
FIX_DIV_FLAG=""
NOT_DIV_FLAG=""
NORMALIZE_ADVANTAGE_FLAG=""
POS_REWARD_FLAG=""
RL_CENTERED_WEIGHT="0.7"
ENT_COEF="0.001"
OPTIMIZER="adam"
OPTIMIZER_WEIGHT_DECAY="0"
MAX_GRAD_NORM="0.5"
TARGET_KL="0.03"
LR_SCHEDULE="step"
LR_DECAY_STEP_SIZE="10000"
LR_DECAY_GAMMA="0.5"
MIN_LEARNING_RATE="1e-5"
NEAREST_TASKS_MIN_K="8"
TASK_TRUNCATED_SIZE="1"
EXPLICIT_PATH_FEATURE_FLAG="--use_explicit_path_feature"
RL_N_SAMPLES="4"
N_STEPS_USER_SET="0"
PRETRAIN_STEPS="3000"

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
      TASK_NUM="$2"
      shift 2
      ;;
    -s)
      N_STEPS="$2"
      N_STEPS_USER_SET="1"
      shift 2
      ;;
    -p)
      PROCESS_NUM="$2"
      shift 2
      ;;
    -e)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --hidden_dim)
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --lower_gnn_type)
      LOWER_GNN_TYPE="$2"
      shift 2
      ;;
    --higher_gnn_type)
      HIGHER_GNN_TYPE="$2"
      shift 2
      ;;
    --lower_gnn_num_layers)
      LOWER_GNN_NUM_LAYERS="$2"
      shift 2
      ;;
    --higher_gnn_num_layers)
      HIGHER_GNN_NUM_LAYERS="$2"
      shift 2
      ;;
    --gnn_dropout)
      GNN_DROPOUT="$2"
      shift 2
      ;;
    --gnn_heads)
      GNN_HEADS="$2"
      shift 2
      ;;
    --edge_combine)
      EDGE_COMBINE="$2"
      shift 2
      ;;
    --max_distance)
      MAX_DISTANCE="$2"
      shift 2
      ;;
    --self_attention_layers)
      SELF_ATTENTION_LAYERS="$2"
      shift 2
      ;;
    --tau)
      TAU="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --use_sinkhorn)
      USE_SINKHORN_FLAG="--use_sinkhorn"
      shift
      ;;
    --no_sinkhorn)
      USE_SINKHORN_FLAG=""
      shift
      ;;
    --use_hungarian)
      USE_HUNGARIAN_FLAG="--use_hungarian_for_deterministic"
      shift
      ;;
    --no_hungarian)
      USE_HUNGARIAN_FLAG=""
      shift
      ;;
    --use_undirected)
      USE_UNDIRECTED_FLAG="--use_undirected"
      shift
      ;;
    --no_undirected)
      USE_UNDIRECTED_FLAG=""
      shift
      ;;
    --fix_div)
      FIX_DIV_FLAG="--fix_div"
      shift
      ;;
    --not_div)
      NOT_DIV_FLAG="--not_div"
      shift
      ;;
    --normalize_advantage)
      NORMALIZE_ADVANTAGE_FLAG="--normalize_advantage"
      shift
      ;;
    --pos_reward)
      POS_REWARD_FLAG="--pos_reward"
      shift
      ;;
    --use_gumbel_hungarian)
      USE_GUMBEL_HUNGARIAN_FLAG="--use_gumbel_hungarian"
      shift
      ;;
    --use_gumbel_sinkhorn)
      USE_GUMBEL_SINKHORN_FLAG="--use_gumbel_sinkhorn"
      shift
      ;;
    --rl_centered_weight)
      RL_CENTERED_WEIGHT="$2"
      shift 2
      ;;
    --ent_coef)
      ENT_COEF="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
      shift 2
      ;;
    --optimizer_weight_decay)
      OPTIMIZER_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --max_grad_norm)
      MAX_GRAD_NORM="$2"
      shift 2
      ;;
    --target_kl)
      TARGET_KL="$2"
      shift 2
      ;;
    --lr_schedule)
      LR_SCHEDULE="$2"
      shift 2
      ;;
    --lr_decay_step_size)
      LR_DECAY_STEP_SIZE="$2"
      shift 2
      ;;
    --lr_decay_gamma)
      LR_DECAY_GAMMA="$2"
      shift 2
      ;;
    --min_learning_rate)
      MIN_LEARNING_RATE="$2"
      shift 2
      ;;
    --nearest_tasks_min_k)
      NEAREST_TASKS_MIN_K="$2"
      shift 2
      ;;
    --task_truncated_size)
      TASK_TRUNCATED_SIZE="$2"
      shift 2
      ;;
    --no_explicit_path_feature)
      EXPLICIT_PATH_FEATURE_FLAG="--no_explicit_path_feature"
      shift
      ;;
    --n_steps)
      N_STEPS="$2"
      N_STEPS_USER_SET="1"
      shift 2
      ;;
    --pretrain_steps)
      PRETRAIN_STEPS="$2"
      shift 2
      ;;
    *)
    echo "Unknown parameter: $1"
    exit 1
    ;;
  esac
done

# 当用户显式传入n_steps时，固定关闭K-sample（rl_n_samples=1）
if [ "${N_STEPS_USER_SET}" = "1" ]; then
  RL_N_SAMPLES="1"
fi

# Check required parameters
if [ -z "$GPU_ID" ] || [ -z "$LEARNING_RATE" ] || [ -z "$GAMMA" ] || [ -z "$TASK_NUM" ] || [ -z "$PROCESS_NUM" ]; then
  echo "Usage: $0 -g <gpu_id> -l <learning_rate> -m <gamma> -t <task_num> -p <process_num> [options]"
  exit 1
fi


# Generate timestamp (format: 20250110_1030)
TIMESTAMP=$(date +%Y%m%d_%H%M)

# Construct model save directory
MODEL_DIR="../models/gnn_${EXPERIMENT_NAME}_${TIMESTAMP}"

# Create directory
mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}/checkpoints"

# Make specified GPU visible
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================="
echo "Start Training GNN Policy with REINFORCE"
echo "GPU=${GPU_ID}, LR=${LEARNING_RATE}, gamma=${GAMMA}, n_steps=${N_STEPS}"
echo "rl_n_samples=${RL_N_SAMPLES} (auto)"
echo "optimizer=${OPTIMIZER}, lr_schedule=${LR_SCHEDULE}, max_grad_norm=${MAX_GRAD_NORM}, target_kl=${TARGET_KL}"
echo "GNN Type: ${LOWER_GNN_TYPE} and ${HIGHER_GNN_TYPE}, Hidden Dim: ${HIDDEN_DIM}"
echo "Task Number: ${TASK_NUM}, Process Number: ${PROCESS_NUM}"
echo "Task Truncated Size: ${TASK_TRUNCATED_SIZE}"
echo "Model Directory: ${MODEL_DIR}"
echo "============================================="

export CUDA_LAUNCH_BLOCKING=1
# Call train_mapd_gnn.py

echo "python train_mapd_gnn.py \
  --learning_rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --lr_schedule "${LR_SCHEDULE}" \
  --lr_decay_step_size "${LR_DECAY_STEP_SIZE}" \
  --lr_decay_gamma "${LR_DECAY_GAMMA}" \
  --min_learning_rate "${MIN_LEARNING_RATE}" \
  --n_steps "${N_STEPS}" \
  --ent_coef "${ENT_COEF}" \
  --optimizer "${OPTIMIZER}" \
  --optimizer_weight_decay "${OPTIMIZER_WEIGHT_DECAY}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --target_kl "${TARGET_KL}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --grid_feature_dim "${GRID_FEATURE_DIM}" \
  --lower_gnn_type "${LOWER_GNN_TYPE}" \
  --higher_gnn_type "${HIGHER_GNN_TYPE}" \
  --lower_gnn_num_layers "${LOWER_GNN_NUM_LAYERS}" \
  --higher_gnn_num_layers "${HIGHER_GNN_NUM_LAYERS}" \
  --gnn_dropout "${GNN_DROPOUT}" \
  --gnn_heads "${GNN_HEADS}" \
  --edge_combine "${EDGE_COMBINE}" \
  --max_distance "${MAX_DISTANCE}" \
  --self_attention_layers "${SELF_ATTENTION_LAYERS}" \
  --checkpoint_freq 125 \
  --global_seed 40 \
  --grid_path "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-large.map" \
  --save_dir "${MODEL_DIR}" \
  --training \
  --task_num "${TASK_NUM}" \
  --n_envs "${PROCESS_NUM}" \
  --tau "${TAU}" \
  --iterations "${ITERATIONS}" \
  --pretrain_steps "${PRETRAIN_STEPS}" \
     ${USE_SINKHORN_FLAG} \
   ${USE_HUNGARIAN_FLAG} \
   ${USE_UNDIRECTED_FLAG} \
   ${USE_GUMBEL_HUNGARIAN_FLAG} \
   ${USE_GUMBEL_SINKHORN_FLAG} \
   ${FIX_DIV_FLAG} \
   ${NOT_DIV_FLAG} \
   ${NORMALIZE_ADVANTAGE_FLAG} \
   ${POS_REWARD_FLAG} \
  --rl_n_samples "${RL_N_SAMPLES}" \
  --rl_policy row_softmax \
  --rl_centered_weight "${RL_CENTERED_WEIGHT}" \
  --nearest_tasks_min_k "${NEAREST_TASKS_MIN_K}" \
  --task_truncated_size "${TASK_TRUNCATED_SIZE}" \
  ${EXPLICIT_PATH_FEATURE_FLAG}"
  
python train_mapd_gnn.py \
  --learning_rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --lr_schedule "${LR_SCHEDULE}" \
  --lr_decay_step_size "${LR_DECAY_STEP_SIZE}" \
  --lr_decay_gamma "${LR_DECAY_GAMMA}" \
  --min_learning_rate "${MIN_LEARNING_RATE}" \
  --n_steps "${N_STEPS}" \
  --ent_coef "${ENT_COEF}" \
  --optimizer "${OPTIMIZER}" \
  --optimizer_weight_decay "${OPTIMIZER_WEIGHT_DECAY}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --target_kl "${TARGET_KL}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --grid_feature_dim "${GRID_FEATURE_DIM}" \
  --lower_gnn_type "${LOWER_GNN_TYPE}" \
  --higher_gnn_type "${HIGHER_GNN_TYPE}" \
  --lower_gnn_num_layers "${LOWER_GNN_NUM_LAYERS}" \
  --higher_gnn_num_layers "${HIGHER_GNN_NUM_LAYERS}" \
  --gnn_dropout "${GNN_DROPOUT}" \
  --gnn_heads "${GNN_HEADS}" \
  --edge_combine "${EDGE_COMBINE}" \
  --max_distance "${MAX_DISTANCE}" \
  --self_attention_layers "${SELF_ATTENTION_LAYERS}" \
  --checkpoint_freq 125 \
  --global_seed 40 \
  --grid_path "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-large.map" \
  --save_dir "${MODEL_DIR}" \
  --training \
  --task_num "${TASK_NUM}" \
  --n_envs "${PROCESS_NUM}" \
  --tau "${TAU}" \
  --iterations "${ITERATIONS}" \
  --pretrain_steps "${PRETRAIN_STEPS}" \
    ${USE_SINKHORN_FLAG} \
   ${USE_HUNGARIAN_FLAG} \
   ${USE_UNDIRECTED_FLAG} \
   ${USE_GUMBEL_HUNGARIAN_FLAG} \
   ${USE_GUMBEL_SINKHORN_FLAG} \
   ${FIX_DIV_FLAG} \
   ${NOT_DIV_FLAG} \
   ${NORMALIZE_ADVANTAGE_FLAG} \
   ${POS_REWARD_FLAG} \
  --rl_n_samples "${RL_N_SAMPLES}" \
  --rl_policy row_softmax \
  --rl_centered_weight "${RL_CENTERED_WEIGHT}" \
  --nearest_tasks_min_k "${NEAREST_TASKS_MIN_K}" \
  --task_truncated_size "${TASK_TRUNCATED_SIZE}" \
  ${EXPLICIT_PATH_FEATURE_FLAG} \
  --debug_every 10

echo "Training completed. Model saved to: ${MODEL_DIR}"
