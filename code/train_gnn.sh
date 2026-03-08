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
#     -s: n_steps for updating (REINFORCE)
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
#     --tau: Sinkhorn temperature parameter (default: 1.0)
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
########################################

# Initialize default values
GPU_ID=""
LEARNING_RATE=""
GAMMA=""
N_STEPS=""
TASK_NUM=""
PROCESS_NUM=""
EXPERIMENT_NAME="gnn_experiment"
HIDDEN_DIM="256"
GRID_FEATURE_DIM="2"
LOWER_GNN_TYPE="sp_mpnn"
HIGHER_GNN_TYPE="edge_node_gnn"
TAU="0.1"
ITERATIONS="5"
UNASSIGN_THRESHOLD="-1"
INVALID_EDGE_SCORE="-100"
# 新增的GNN参数默认值
LOWER_GNN_NUM_LAYERS="9"
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
    *)
    echo "Unknown parameter: $1"
    exit 1
    ;;
  esac
done

# Check required parameters
if [ -z "$GPU_ID" ] || [ -z "$LEARNING_RATE" ] || [ -z "$GAMMA" ] || [ -z "$TASK_NUM" ] || [ -z "$PROCESS_NUM" ]; then
  echo "Usage: $0 -g <gpu_id> -l <learning_rate> -m <gamma> -t <task_num> -p <process_num> [options]"
  exit 1
fi


# Generate timestamp (format: 20250110_1030)
TIMESTAMP=$(date +%Y%m%d_%H%M)

# Construct model save directory
MODEL_DIR="../models/gnn_${EXPERIMENT_NAME}_${TIMESTAMP}_lr_${LEARNING_RATE}_gamma_${GAMMA}_steps_${N_STEPS}_${LOWER_GNN_TYPE}_${HIGHER_GNN_TYPE}"

# Create directory
mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}/checkpoints"

# Make specified GPU visible
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================="
echo "Start Training GNN Policy with REINFORCE"
echo "GPU=${GPU_ID}, LR=${LEARNING_RATE}, gamma=${GAMMA}, n_steps=${N_STEPS}"
echo "GNN Type: ${LOWER_GNN_TYPE} and ${HIGHER_GNN_TYPE}, Hidden Dim: ${HIDDEN_DIM}"
echo "Task Number: ${TASK_NUM}, Process Number: ${PROCESS_NUM}"
echo "Model Directory: ${MODEL_DIR}"
echo "============================================="

export CUDA_LAUNCH_BLOCKING=1
# Call train_mapd_gnn.py

echo "python train_mapd_gnn.py \
  --learning_rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --n_steps 1 \
  --ent_coef 0.01 \
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
  --grid_path "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map" \
  --save_dir "${MODEL_DIR}" \
  --training \
  --task_num "${TASK_NUM}" \
  --n_envs "${PROCESS_NUM}" \
  --tau "${TAU}" \
  --iterations "${ITERATIONS}" \
  --pretrain_steps 1000 \
     ${USE_SINKHORN_FLAG} \
   ${USE_HUNGARIAN_FLAG} \
   ${USE_UNDIRECTED_FLAG} \
   ${USE_GUMBEL_HUNGARIAN_FLAG} \
   ${USE_GUMBEL_SINKHORN_FLAG} \
   ${FIX_DIV_FLAG} \
   ${NOT_DIV_FLAG} \
   ${NORMALIZE_ADVANTAGE_FLAG} \
   ${POS_REWARD_FLAG} \
  --rl_n_samples 4 \
  --rl_policy row_softmax"
  
python train_mapd_gnn.py \
  --learning_rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --n_steps 1 \
  --ent_coef 0.01 \
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
  --grid_path "/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map" \
  --save_dir "${MODEL_DIR}" \
  --training \
  --task_num "${TASK_NUM}" \
  --n_envs "${PROCESS_NUM}" \
  --tau "${TAU}" \
  --iterations "${ITERATIONS}" \
  --pretrain_steps 1000 \
    ${USE_SINKHORN_FLAG} \
   ${USE_HUNGARIAN_FLAG} \
   ${USE_UNDIRECTED_FLAG} \
   ${USE_GUMBEL_HUNGARIAN_FLAG} \
   ${USE_GUMBEL_SINKHORN_FLAG} \
   ${FIX_DIV_FLAG} \
   ${NOT_DIV_FLAG} \
   ${NORMALIZE_ADVANTAGE_FLAG} \
   ${POS_REWARD_FLAG} \
  --rl_n_samples 4 \
  --rl_policy row_softmax \
  --debug_every 10

echo "Training completed. Model saved to: ${MODEL_DIR}"
