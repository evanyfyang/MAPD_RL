#!/bin/bash
set -e

# Usage:
#   nohup bash ./pretrain_spmpnn_context.sh -g 0 -e exp_name > pretrain_spmpnn.out 2>&1 &

GPU_ID=""
EXPERIMENT_NAME="spmpnn_ring_heat_pretrain"
EPOCHS="30"
BATCH_SIZE_GRAPH="24"
TRAIN_GRAPHS_PER_EPOCH="256"
VAL_GRAPHS_PER_EPOCH="64"
DATA_NUM_WORKERS="4"
PREFETCH_BATCHES="4"
SPCACHE_DIR="./spcache"
LR="1e-3"
WEIGHT_DECAY="1e-4"
HIDDEN_DIM="64"
NUM_LAYERS="3"
DROPOUT="0.1"
MAX_DISTANCE="3"
MIN_AGENTS="10"
MAX_AGENTS="50"
MIN_TASKS="20"
MAX_TASKS="100"
CLIP_MAX="4.0"
CENTERS_PER_GRAPH="48"
DELIVERING_FRACTION="0.15"
R_HEAT="3"
NEAR_RING_WEIGHT="1.5"
NEAR_WEIGHTED_RINGS="2"
VARIANTS_PER_MAP="3"
OBSTACLE_DROP_PROB_MIN="0.01"
OBSTACLE_DROP_PROB_MAX="0.08"
SEED="40"
MAP_PATHS="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map,/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-large.map"

while [[ $# -gt 0 ]]; do
  case $1 in
    -g)
      GPU_ID="$2"; shift 2 ;;
    -e)
      EXPERIMENT_NAME="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch_size_graph)
      BATCH_SIZE_GRAPH="$2"; shift 2 ;;
    --train_graphs_per_epoch)
      TRAIN_GRAPHS_PER_EPOCH="$2"; shift 2 ;;
    --val_graphs_per_epoch)
      VAL_GRAPHS_PER_EPOCH="$2"; shift 2 ;;
    --data_num_workers)
      DATA_NUM_WORKERS="$2"; shift 2 ;;
    --prefetch_batches)
      PREFETCH_BATCHES="$2"; shift 2 ;;
    --spcache_dir)
      SPCACHE_DIR="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --weight_decay)
      WEIGHT_DECAY="$2"; shift 2 ;;
    --hidden_dim)
      HIDDEN_DIM="$2"; shift 2 ;;
    --num_layers)
      NUM_LAYERS="$2"; shift 2 ;;
    --dropout)
      DROPOUT="$2"; shift 2 ;;
    --max_distance)
      MAX_DISTANCE="$2"; shift 2 ;;
    --min_agents)
      MIN_AGENTS="$2"; shift 2 ;;
    --max_agents)
      MAX_AGENTS="$2"; shift 2 ;;
    --min_tasks)
      MIN_TASKS="$2"; shift 2 ;;
    --max_tasks)
      MAX_TASKS="$2"; shift 2 ;;
    --clip_max)
      CLIP_MAX="$2"; shift 2 ;;
    --centers_per_graph)
      CENTERS_PER_GRAPH="$2"; shift 2 ;;
    --delivering_fraction)
      DELIVERING_FRACTION="$2"; shift 2 ;;
    --R_heat)
      R_HEAT="$2"; shift 2 ;;
    --near_ring_weight)
      NEAR_RING_WEIGHT="$2"; shift 2 ;;
    --near_weighted_rings)
      NEAR_WEIGHTED_RINGS="$2"; shift 2 ;;
    --variants_per_map)
      VARIANTS_PER_MAP="$2"; shift 2 ;;
    --obstacle_drop_prob_min)
      OBSTACLE_DROP_PROB_MIN="$2"; shift 2 ;;
    --obstacle_drop_prob_max)
      OBSTACLE_DROP_PROB_MAX="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --map_paths)
      MAP_PATHS="$2"; shift 2 ;;
    *)
      echo "Unknown parameter: $1"
      exit 1 ;;
  esac
done

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 -g <gpu_id> [-e <experiment_name>] [other options]"
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
SAVE_DIR="../models/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Start SP-MPNN ring+heat pretrain"
echo "GPU=${GPU_ID}, save_dir=${SAVE_DIR}"

python pretrain_spmpnn_context.py \
  --map_paths "${MAP_PATHS}" \
  --save_dir "${SAVE_DIR}" \
  --seed "${SEED}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --max_distance "${MAX_DISTANCE}" \
  --min_agents "${MIN_AGENTS}" \
  --max_agents "${MAX_AGENTS}" \
  --min_tasks "${MIN_TASKS}" \
  --max_tasks "${MAX_TASKS}" \
  --clip_max "${CLIP_MAX}" \
  --centers_per_graph "${CENTERS_PER_GRAPH}" \
  --delivering_fraction "${DELIVERING_FRACTION}" \
  --R_heat "${R_HEAT}" \
  --near_ring_weight "${NEAR_RING_WEIGHT}" \
  --near_weighted_rings "${NEAR_WEIGHTED_RINGS}" \
  --variants_per_map "${VARIANTS_PER_MAP}" \
  --obstacle_drop_prob_min "${OBSTACLE_DROP_PROB_MIN}" \
  --obstacle_drop_prob_max "${OBSTACLE_DROP_PROB_MAX}" \
  --epochs "${EPOCHS}" \
  --batch_size_graph "${BATCH_SIZE_GRAPH}" \
  --train_graphs_per_epoch "${TRAIN_GRAPHS_PER_EPOCH}" \
  --val_graphs_per_epoch "${VAL_GRAPHS_PER_EPOCH}" \
  --data_num_workers "${DATA_NUM_WORKERS}" \
  --prefetch_batches "${PREFETCH_BATCHES}" \
  --spcache_dir "${SPCACHE_DIR}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --device cuda

echo "Done. Outputs: ${SAVE_DIR}"
exit 0
#!/bin/bash
set -e

# Usage:
#   nohup bash ./pretrain_spmpnn_context.sh -g 0 -e exp_name > pretrain_spmpnn.out 2>&1 &

GPU_ID=""
EXPERIMENT_NAME="spmpnn_ring_heat_pretrain"
EPOCHS="30"
BATCH_SIZE_GRAPH="24"
TRAIN_GRAPHS_PER_EPOCH="256"
VAL_GRAPHS_PER_EPOCH="64"
DATA_NUM_WORKERS="4"
PREFETCH_BATCHES="4"
SPCACHE_DIR="./spcache"
LR="1e-3"
WEIGHT_DECAY="1e-4"
HIDDEN_DIM="64"
NUM_LAYERS="3"
DROPOUT="0.1"
MAX_DISTANCE="3"
MIN_AGENTS="10"
MAX_AGENTS="50"
MIN_TASKS="20"
MAX_TASKS="100"
CLIP_MAX="4.0"
CENTERS_PER_GRAPH="48"
DELIVERING_FRACTION="0.15"
R_HEAT="3"
NEAR_RING_WEIGHT="1.5"
NEAR_WEIGHTED_RINGS="2"
VARIANTS_PER_MAP="3"
OBSTACLE_DROP_PROB_MIN="0.01"
OBSTACLE_DROP_PROB_MAX="0.08"
SEED="40"
MAP_PATHS="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map,/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-large.map"

while [[ $# -gt 0 ]]; do
  case $1 in
    -g)
      GPU_ID="$2"; shift 2 ;;
    -e)
      EXPERIMENT_NAME="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch_size_graph)
      BATCH_SIZE_GRAPH="$2"; shift 2 ;;
    --train_graphs_per_epoch)
      TRAIN_GRAPHS_PER_EPOCH="$2"; shift 2 ;;
    --val_graphs_per_epoch)
      VAL_GRAPHS_PER_EPOCH="$2"; shift 2 ;;
    --data_num_workers)
      DATA_NUM_WORKERS="$2"; shift 2 ;;
    --prefetch_batches)
      PREFETCH_BATCHES="$2"; shift 2 ;;
    --spcache_dir)
      SPCACHE_DIR="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --weight_decay)
      WEIGHT_DECAY="$2"; shift 2 ;;
    --hidden_dim)
      HIDDEN_DIM="$2"; shift 2 ;;
    --num_layers)
      NUM_LAYERS="$2"; shift 2 ;;
    --dropout)
      DROPOUT="$2"; shift 2 ;;
    --max_distance)
      MAX_DISTANCE="$2"; shift 2 ;;
    --min_agents)
      MIN_AGENTS="$2"; shift 2 ;;
    --max_agents)
      MAX_AGENTS="$2"; shift 2 ;;
    --min_tasks)
      MIN_TASKS="$2"; shift 2 ;;
    --max_tasks)
      MAX_TASKS="$2"; shift 2 ;;
    --clip_max)
      CLIP_MAX="$2"; shift 2 ;;
    --centers_per_graph)
      CENTERS_PER_GRAPH="$2"; shift 2 ;;
    --delivering_fraction)
      DELIVERING_FRACTION="$2"; shift 2 ;;
    --R_heat)
      R_HEAT="$2"; shift 2 ;;
    --near_ring_weight)
      NEAR_RING_WEIGHT="$2"; shift 2 ;;
    --near_weighted_rings)
      NEAR_WEIGHTED_RINGS="$2"; shift 2 ;;
    --variants_per_map)
      VARIANTS_PER_MAP="$2"; shift 2 ;;
    --obstacle_drop_prob_min)
      OBSTACLE_DROP_PROB_MIN="$2"; shift 2 ;;
    --obstacle_drop_prob_max)
      OBSTACLE_DROP_PROB_MAX="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --map_paths)
      MAP_PATHS="$2"; shift 2 ;;
    *)
      echo "Unknown parameter: $1"
      exit 1 ;;
  esac
done

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 -g <gpu_id> [-e <experiment_name>] [other options]"
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
SAVE_DIR="../models/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Start SP-MPNN ring+heat pretrain"
echo "GPU=${GPU_ID}, save_dir=${SAVE_DIR}"

python pretrain_spmpnn_context.py \
  --map_paths "${MAP_PATHS}" \
  --save_dir "${SAVE_DIR}" \
  --seed "${SEED}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --max_distance "${MAX_DISTANCE}" \
  --min_agents "${MIN_AGENTS}" \
  --max_agents "${MAX_AGENTS}" \
  --min_tasks "${MIN_TASKS}" \
  --max_tasks "${MAX_TASKS}" \
  --clip_max "${CLIP_MAX}" \
  --centers_per_graph "${CENTERS_PER_GRAPH}" \
  --delivering_fraction "${DELIVERING_FRACTION}" \
  --R_heat "${R_HEAT}" \
  --near_ring_weight "${NEAR_RING_WEIGHT}" \
  --near_weighted_rings "${NEAR_WEIGHTED_RINGS}" \
  --variants_per_map "${VARIANTS_PER_MAP}" \
  --obstacle_drop_prob_min "${OBSTACLE_DROP_PROB_MIN}" \
  --obstacle_drop_prob_max "${OBSTACLE_DROP_PROB_MAX}" \
  --epochs "${EPOCHS}" \
  --batch_size_graph "${BATCH_SIZE_GRAPH}" \
  --train_graphs_per_epoch "${TRAIN_GRAPHS_PER_EPOCH}" \
  --val_graphs_per_epoch "${VAL_GRAPHS_PER_EPOCH}" \
  --data_num_workers "${DATA_NUM_WORKERS}" \
  --prefetch_batches "${PREFETCH_BATCHES}" \
  --spcache_dir "${SPCACHE_DIR}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --device cuda

echo "Done. Outputs: ${SAVE_DIR}"
#!/bin/bash
set -e

# Usage:
#   nohup bash ./pretrain_spmpnn_context.sh -g 0 -e exp_name > pretrain_spmpnn.out 2>&1 &

GPU_ID=""
EXPERIMENT_NAME="spmpnn_ctx_pretrain"
EPOCHS="30"
BATCH_SIZE_GRAPH="24"
TRAIN_GRAPHS_PER_EPOCH="256"
VAL_GRAPHS_PER_EPOCH="64"
DATA_NUM_WORKERS="4"
PREFETCH_BATCHES="4"
SPCACHE_DIR="./spcache"
LR="1e-3"
WEIGHT_DECAY="1e-4"
HIDDEN_DIM="64"
NUM_LAYERS="3"
DROPOUT="0.1"
MAX_DISTANCE="3"
MIN_AGENTS="10"
MAX_AGENTS="50"
MIN_TASKS="20"
MAX_TASKS="100"
CENTERS_PER_GRAPH="48"
NEG_PER_POS="3"
R1="2"
R2="5"
LAMBDA_OCC="0.2"
OCC_TASK_MODE="count_bins"
OCC_NUM_BINS="6"
VARIANTS_PER_MAP="3"
OBSTACLE_DROP_PROB_MIN="0.01"
OBSTACLE_DROP_PROB_MAX="0.08"
SEED="40"
MAP_PATHS="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map,/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-large.map"

while [[ $# -gt 0 ]]; do
  case $1 in
    -g)
      GPU_ID="$2"; shift 2 ;;
    -e)
      EXPERIMENT_NAME="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch_size_graph)
      BATCH_SIZE_GRAPH="$2"; shift 2 ;;
    --train_graphs_per_epoch)
      TRAIN_GRAPHS_PER_EPOCH="$2"; shift 2 ;;
    --val_graphs_per_epoch)
      VAL_GRAPHS_PER_EPOCH="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --data_num_workers)
      DATA_NUM_WORKERS="$2"; shift 2 ;;
    --prefetch_batches)
      PREFETCH_BATCHES="$2"; shift 2 ;;
    --spcache_dir)
      SPCACHE_DIR="$2"; shift 2 ;;
    --weight_decay)
      WEIGHT_DECAY="$2"; shift 2 ;;
    --hidden_dim)
      HIDDEN_DIM="$2"; shift 2 ;;
    --num_layers)
      NUM_LAYERS="$2"; shift 2 ;;
    --dropout)
      DROPOUT="$2"; shift 2 ;;
    --max_distance)
      MAX_DISTANCE="$2"; shift 2 ;;
    --min_agents)
      MIN_AGENTS="$2"; shift 2 ;;
    --max_agents)
      MAX_AGENTS="$2"; shift 2 ;;
    --min_tasks)
      MIN_TASKS="$2"; shift 2 ;;
    --max_tasks)
      MAX_TASKS="$2"; shift 2 ;;
    --centers_per_graph)
      CENTERS_PER_GRAPH="$2"; shift 2 ;;
    --neg_per_pos)
      NEG_PER_POS="$2"; shift 2 ;;
    --r1)
      R1="$2"; shift 2 ;;
    --r2)
      R2="$2"; shift 2 ;;
    --lambda_occ)
      LAMBDA_OCC="$2"; shift 2 ;;
    --occ_task_mode)
      OCC_TASK_MODE="$2"; shift 2 ;;
    --occ_num_bins)
      OCC_NUM_BINS="$2"; shift 2 ;;
    --variants_per_map)
      VARIANTS_PER_MAP="$2"; shift 2 ;;
    --obstacle_drop_prob_min)
      OBSTACLE_DROP_PROB_MIN="$2"; shift 2 ;;
    --obstacle_drop_prob_max)
      OBSTACLE_DROP_PROB_MAX="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --map_paths)
      MAP_PATHS="$2"; shift 2 ;;
    *)
      echo "Unknown parameter: $1"
      exit 1 ;;
  esac
done

if [ -z "$GPU_ID" ]; then
  echo "Usage: $0 -g <gpu_id> [-e <experiment_name>] [other options]"
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
SAVE_DIR="../models/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${SAVE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Start SP-MPNN context pretrain"
echo "GPU=${GPU_ID}, save_dir=${SAVE_DIR}"

python pretrain_spmpnn_context.py \
  --map_paths "${MAP_PATHS}" \
  --save_dir "${SAVE_DIR}" \
  --seed "${SEED}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --max_distance "${MAX_DISTANCE}" \
  --min_agents "${MIN_AGENTS}" \
  --max_agents "${MAX_AGENTS}" \
  --min_tasks "${MIN_TASKS}" \
  --max_tasks "${MAX_TASKS}" \
  --centers_per_graph "${CENTERS_PER_GRAPH}" \
  --neg_per_pos "${NEG_PER_POS}" \
  --r1 "${R1}" \
  --r2 "${R2}" \
  --lambda_occ "${LAMBDA_OCC}" \
  --occ_task_mode "${OCC_TASK_MODE}" \
  --occ_num_bins "${OCC_NUM_BINS}" \
  --variants_per_map "${VARIANTS_PER_MAP}" \
  --obstacle_drop_prob_min "${OBSTACLE_DROP_PROB_MIN}" \
  --obstacle_drop_prob_max "${OBSTACLE_DROP_PROB_MAX}" \
  --epochs "${EPOCHS}" \
  --batch_size_graph "${BATCH_SIZE_GRAPH}" \
  --train_graphs_per_epoch "${TRAIN_GRAPHS_PER_EPOCH}" \
  --val_graphs_per_epoch "${VAL_GRAPHS_PER_EPOCH}" \
  --data_num_workers "${DATA_NUM_WORKERS}" \
  --prefetch_batches "${PREFETCH_BATCHES}" \
  --spcache_dir "${SPCACHE_DIR}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --device cuda

echo "Done. Outputs: ${SAVE_DIR}"
