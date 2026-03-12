#!/usr/bin/env bash
# Exit script if any command fails
set -e

########################################
# Usage:
#   ./test_gnn.sh -c <checkpoint_path> -d <test_data_path> [options]
#   Example:
#     ./test_gnn.sh -c ../models/model.zip -d ../test_data/test_case.txt --hidden_dim 128 --test_episodes 5 -g 0
#   Parameters:
#     -c: checkpoint路径 (必需)
#     -d: 测试数据文件路径 (必需)
#     -g: GPU ID (可选，默认使用所有可用GPU)
#     --hidden_dim: GNN隐藏层维度 (默认: 64)
#     --test_episodes: 测试回合数 (默认: 1)
#     --test_env_seed: 测试环境随机种子 (默认: 100)
#     --grid_path: 网格地图路径 (可选)
#     --task_truncated_size: 每个agent执行队列长度上限 (默认: 1)
#     --nearest_tasks_min_k: 测试时手动指定候选K（不传则自动对齐checkpoint）
#     --use_explicit_path_feature: 覆盖checkpoint开关（1开启/0关闭）
#     --use_gumbel: 启用Gumbel噪声 (flag)
#     --model_only_eval: 测试时关闭expert fallback (flag)
#     --infer_decode_mode: deterministic推理解码方式 (sequential|hungarian)
#     --verbose: 输出详细信息 (flag)
########################################

# Initialize default values
CHECKPOINT_PATH=""
TEST_DATA_PATH=""
GPU_ID=""
HIDDEN_DIM="64"
TEST_EPISODES="1"
TEST_ENV_SEED="100"
GRID_PATH=""
TASK_TRUNCATED_SIZE="1"
NEAREST_TASKS_MIN_K=""
USE_EXPLICIT_PATH_FEATURE=""
USE_GUMBEL_FLAG=""
MODEL_ONLY_EVAL_FLAG=""
INFER_DECODE_MODE="sequential"
VERBOSE_FLAG=""
HIGHER_GNN_TYPE="edge_node_gnn_complex"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    -d)
      TEST_DATA_PATH="$2"
      shift 2
      ;;
    -g)
      GPU_ID="$2"
      shift 2
      ;;
    --hidden_dim)
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --test_episodes)
      TEST_EPISODES="$2"
      shift 2
      ;;
    --test_env_seed)
      TEST_ENV_SEED="$2"
      shift 2
      ;;
    --grid_path)
      GRID_PATH="$2"
      shift 2
      ;;
    --task_truncated_size)
      TASK_TRUNCATED_SIZE="$2"
      shift 2
      ;;
    --nearest_tasks_min_k)
      NEAREST_TASKS_MIN_K="$2"
      shift 2
      ;;
    --use_explicit_path_feature)
      USE_EXPLICIT_PATH_FEATURE="$2"
      shift 2
      ;;
    --use_gumbel)
      USE_GUMBEL_FLAG="--use_gumbel"
      shift
      ;;
    --model_only_eval)
      MODEL_ONLY_EVAL_FLAG="--model_only_eval"
      shift
      ;;
    --infer_decode_mode)
      INFER_DECODE_MODE="$2"
      shift 2
      ;;
    --higher_gnn_type)
      HIGHER_GNN_TYPE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE_FLAG="--verbose"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check required parameters
if [ -z "$CHECKPOINT_PATH" ] || [ -z "$TEST_DATA_PATH" ]; then
  echo "Usage: $0 -c <checkpoint_path> -d <test_data_path> [options]"
  echo "Required parameters:"
  echo "  -c: checkpoint路径"
  echo "  -d: 测试数据文件路径"
  echo "Optional parameters:"
  echo "  -g: GPU ID"
  echo "  --hidden_dim: GNN隐藏层维度 (默认: 64)"
  echo "  --test_episodes: 测试回合数 (默认: 1)"
  echo "  --test_env_seed: 测试环境随机种子 (默认: 100)"
  echo "  --grid_path: 网格地图路径"
  echo "  --task_truncated_size: 每个agent执行队列长度上限 (默认: 1)"
  echo "  --nearest_tasks_min_k: 测试时手动指定候选K"
  echo "  --use_explicit_path_feature: 覆盖checkpoint开关（1/0）"
  echo "  --use_gumbel: 启用Gumbel噪声"
  echo "  --model_only_eval: 测试时仅使用model结果"
  echo "  --infer_decode_mode: 推理解码方式 (sequential|hungarian)"
  echo "  --verbose: 输出详细信息"
  exit 1
fi

# Set GPU if specified
if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  echo "Using GPU: ${GPU_ID}"
fi

# Activate conda environment
echo "Activating MAPD_RL environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MAPD_RL

echo "============================================="
echo "开始测试 GNN Policy"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "测试数据: ${TEST_DATA_PATH}"
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "测试回合数: ${TEST_EPISODES}"
echo "环境种子: ${TEST_ENV_SEED}"
echo "推理解码: ${INFER_DECODE_MODE}"
echo "Task Truncated Size: ${TASK_TRUNCATED_SIZE}"
if [ -n "$GRID_PATH" ]; then
  echo "Grid Path: ${GRID_PATH}"
fi
echo "============================================="

# Build the command
CMD="python -m pdb test_gnn.py \
  --checkpoint \"${CHECKPOINT_PATH}\" \
  --test_data_path \"${TEST_DATA_PATH}\" \
  --hidden_dim ${HIDDEN_DIM} \
  --test_episodes ${TEST_EPISODES} \
  --test_env_seed ${TEST_ENV_SEED} \
  --task_truncated_size ${TASK_TRUNCATED_SIZE} \
  --infer_decode_mode \"${INFER_DECODE_MODE}\""

# Add grid_path if specified
if [ -n "$GRID_PATH" ]; then
  CMD="${CMD} --grid_path \"${GRID_PATH}\""
fi

# Add nearest_tasks_min_k if specified
if [ -n "$NEAREST_TASKS_MIN_K" ]; then
  CMD="${CMD} --nearest_tasks_min_k ${NEAREST_TASKS_MIN_K}"
fi

if [ -n "$USE_EXPLICIT_PATH_FEATURE" ]; then
  CMD="${CMD} --use_explicit_path_feature ${USE_EXPLICIT_PATH_FEATURE}"
fi

# Add verbose flag if specified
if [ -n "$VERBOSE_FLAG" ]; then
  CMD="${CMD} ${VERBOSE_FLAG}"
fi

# Add use_gumbel flag if specified
if [ -n "$USE_GUMBEL_FLAG" ]; then
  CMD="${CMD} ${USE_GUMBEL_FLAG}"
fi

# Add model_only_eval flag if specified
if [ -n "$MODEL_ONLY_EVAL_FLAG" ]; then
  CMD="${CMD} ${MODEL_ONLY_EVAL_FLAG}"
fi

# Append training-aligned defaults
CMD="${CMD} \
  --lower_gnn_type \"sp_mpnn\" \
  --higher_gnn_type \"${HIGHER_GNN_TYPE}\" \
  --edge_combine \"concat\" \
  --unassign_threshold -1 \
  --use_undirected \
  --use_hungarian_for_deterministic"

# Execute the command
eval $CMD

echo "测试完成!" 