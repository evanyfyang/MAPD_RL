#!/usr/bin/env bash
set -euo pipefail

# 自动课程训练脚本（支持checkpoint续训 + 阶段自动晋级）
#
# 运行：
#   bash curriculum_train_auto.sh [gpu_id]
#
# 说明：
# - 每个stage按 chunk 训练，训练后做固定测试（model-only vs expert/Hungarian）
# - 当 gap 连续 PATIENCE 次 <= TARGET_GAP 时，自动进入下一stage
# - gap = (model_reward - expert_reward) / max(abs(expert_reward), 1e-6)

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

WORKDIR="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code"
MAP_PATH="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-20-500-5.map"
TASK_PATH="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-2.task"

# 模型/训练公共参数（按当前主配置）
LOWER_GNN_TYPE="sp_mpnn"
HIGHER_GNN_TYPE="edge_node_gnn"
EDGE_COMBINE="concat"
USE_UNDIRECTED_FLAG="--use_undirected"
USE_GUMBEL_HUNGARIAN_FLAG="--use_gumbel_hungarian"
LEARNING_RATE="3e-4"
GAMMA="0.99"
ENT_COEF="0.001"
N_ENVS="16"
RL_N_SAMPLES="4"
RL_CENTERED_WEIGHT="0.7"
NEAREST_TASKS_MIN_K="8"
CHECKPOINT_FREQ="125"
TEST_EPISODES="3"
TEST_SEED="100"
GLOBAL_SEED="40"

# 课程阶段（按你指定：easy(<10,<10) -> medium(<=50,<=50) -> hard(全量)）
STAGE_NAMES=("stage1_easy" "stage2_medium" "stage3_hard")
AGENT_LB=(3 10 10)
AGENT_UB=(9 50 50)
TASK_NUM=(9 50 500)
CHUNK_STEPS=(6000 8000 10000)
MAX_CHUNKS=(20 20 30)
TARGET_GAP=(0.05 0.05 0.05)     # 越小越严格
PATIENCE=(3 3 3)                 # 连续满足次数
PRETRAIN_STEPS=(0 0 0)

# 一次性 warmup（medium pretrain）
WARMUP_ENABLE=1
WARMUP_NAME="warmup_medium_pretrain"
WARMUP_AGENT_LB=10
WARMUP_AGENT_UB=50
WARMUP_TASK_NUM=50
WARMUP_TOTAL_STEPS=2000
WARMUP_PRETRAIN_STEPS=2000

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SAVE_ROOT="/local-scratchg/yifan/2024/MAPD/MAPD_RL/models/curriculum_${RUN_ID}"
mkdir -p "${SAVE_ROOT}"

extract_reward() {
  python - "$1" <<'PY'
import re, sys
txt = sys.argv[1]
ms = re.findall(r"\[Test\]\s*Episodes:\s*\d+,\s*Reward:\s*([-\d\.eE]+)", txt)
if not ms:
    print("nan")
else:
    print(ms[-1])
PY
}

calc_gap() {
  python - "$1" "$2" <<'PY'
import sys, math
m = float(sys.argv[1]); e = float(sys.argv[2])
den = max(abs(e), 1e-6)
print((m - e) / den)
PY
}

run_eval() {
  local ckpt="$1"
  local mode="$2"   # model | expert
  local extra_env=""
  if [[ "${mode}" == "expert" ]]; then
    extra_env="USE_EXPERT=True"
  fi

  local out
  out=$(cd "${WORKDIR}" && eval ${extra_env} python train_mapd_gnn.py \
    --test_checkpoint "${ckpt}" \
    --grid_path "${MAP_PATH}" \
    --eval_data_path "${TASK_PATH}" \
    --test_episodes "${TEST_EPISODES}" \
    --test_env_seed "${TEST_SEED}" \
    --task_num 500 \
    --model_only_eval \
    --lower_gnn_type "${LOWER_GNN_TYPE}" \
    --higher_gnn_type "${HIGHER_GNN_TYPE}" \
    --edge_combine "${EDGE_COMBINE}" \
    ${USE_UNDIRECTED_FLAG} \
    ${USE_GUMBEL_HUNGARIAN_FLAG} 2>&1)
  extract_reward "${out}"
}

resume_ckpt=""

echo "== Curriculum Run: ${RUN_ID} =="
echo "save_root: ${SAVE_ROOT}"

if [[ "${WARMUP_ENABLE}" == "1" ]]; then
  warmup_dir="${SAVE_ROOT}/${WARMUP_NAME}"
  mkdir -p "${warmup_dir}/checkpoints"
  echo ""
  echo "===== ENTER ${WARMUP_NAME} ====="
  echo "difficulty: agent=[${WARMUP_AGENT_LB},${WARMUP_AGENT_UB}], task_num=${WARMUP_TASK_NUM}, total_steps=${WARMUP_TOTAL_STEPS}, pretrain_steps=${WARMUP_PRETRAIN_STEPS}"

  warm_cmd=(python train_mapd_gnn.py
    --training
    --save_dir "${warmup_dir}"
    --total_timesteps "${WARMUP_TOTAL_STEPS}"
    --checkpoint_freq "${CHECKPOINT_FREQ}"
    --learning_rate "${LEARNING_RATE}"
    --gamma "${GAMMA}"
    --ent_coef "${ENT_COEF}"
    --n_steps 1
    --n_envs "${N_ENVS}"
    --global_seed "${GLOBAL_SEED}"
    --env_seed "${GLOBAL_SEED}"
    --grid_path "${MAP_PATH}"
    --agent_num_lower_bound "${WARMUP_AGENT_LB}"
    --agent_num_higher_bound "${WARMUP_AGENT_UB}"
    --task_num "${WARMUP_TASK_NUM}"
    --pretrain_steps "${WARMUP_PRETRAIN_STEPS}"
    --lower_gnn_type "${LOWER_GNN_TYPE}"
    --higher_gnn_type "${HIGHER_GNN_TYPE}"
    --edge_combine "${EDGE_COMBINE}"
    --rl_n_samples "${RL_N_SAMPLES}"
    --rl_centered_weight "${RL_CENTERED_WEIGHT}"
    --nearest_tasks_min_k "${NEAREST_TASKS_MIN_K}"
    ${USE_UNDIRECTED_FLAG}
    ${USE_GUMBEL_HUNGARIAN_FLAG}
  )
  (cd "${WORKDIR}" && "${warm_cmd[@]}")

  resume_ckpt="${warmup_dir}/final_model.zip"
  if [[ ! -f "${resume_ckpt}" ]]; then
    echo "ERROR: missing warmup checkpoint ${resume_ckpt}"
    exit 1
  fi
  echo "[${WARMUP_NAME}] done. resume_ckpt=${resume_ckpt}"
fi

for idx in "${!STAGE_NAMES[@]}"; do
  stage="${STAGE_NAMES[$idx]}"
  lb="${AGENT_LB[$idx]}"
  ub="${AGENT_UB[$idx]}"
  tn="${TASK_NUM[$idx]}"
  chunk="${CHUNK_STEPS[$idx]}"
  max_chunks="${MAX_CHUNKS[$idx]}"
  target_gap="${TARGET_GAP[$idx]}"
  patience_need="${PATIENCE[$idx]}"
  pre_steps="${PRETRAIN_STEPS[$idx]}"

  stage_dir="${SAVE_ROOT}/${stage}"
  mkdir -p "${stage_dir}/checkpoints"
  pass_count=0

  echo ""
  echo "===== ENTER ${stage} ====="
  echo "difficulty: agent=[${lb},${ub}], task_num=${tn}, chunk=${chunk}, max_chunks=${max_chunks}"

  for ((round=1; round<=max_chunks; round++)); do
    echo ""
    echo "[${stage}] round ${round}/${max_chunks} | resume=${resume_ckpt:-none}"

    cmd=(python train_mapd_gnn.py
      --training
      --save_dir "${stage_dir}"
      --total_timesteps "${chunk}"
      --checkpoint_freq "${CHECKPOINT_FREQ}"
      --learning_rate "${LEARNING_RATE}"
      --gamma "${GAMMA}"
      --ent_coef "${ENT_COEF}"
      --n_steps 1
      --n_envs "${N_ENVS}"
      --global_seed "${GLOBAL_SEED}"
      --env_seed "${GLOBAL_SEED}"
      --grid_path "${MAP_PATH}"
      --agent_num_lower_bound "${lb}"
      --agent_num_higher_bound "${ub}"
      --task_num "${tn}"
      --pretrain_steps "${pre_steps}"
      --lower_gnn_type "${LOWER_GNN_TYPE}"
      --higher_gnn_type "${HIGHER_GNN_TYPE}"
      --edge_combine "${EDGE_COMBINE}"
      --rl_n_samples "${RL_N_SAMPLES}"
      --rl_centered_weight "${RL_CENTERED_WEIGHT}"
      --nearest_tasks_min_k "${NEAREST_TASKS_MIN_K}"
      ${USE_UNDIRECTED_FLAG}
      ${USE_GUMBEL_HUNGARIAN_FLAG}
    )

    if [[ -n "${resume_ckpt}" ]]; then
      cmd+=(--resume_checkpoint "${resume_ckpt}")
    fi

    (cd "${WORKDIR}" && "${cmd[@]}")

    # 当前阶段每轮都保存 final_model.zip，直接拿它继续
    resume_ckpt="${stage_dir}/final_model.zip"
    if [[ ! -f "${resume_ckpt}" ]]; then
      echo "ERROR: missing checkpoint ${resume_ckpt}"
      exit 1
    fi

    model_reward="$(run_eval "${resume_ckpt}" model)"
    expert_reward="$(run_eval "${resume_ckpt}" expert)"
    gap="$(calc_gap "${model_reward}" "${expert_reward}")"

    echo "[${stage}] eval -> model=${model_reward}, expert=${expert_reward}, gap=${gap}, target<=${target_gap}"

    pass=$(python - "${gap}" "${target_gap}" <<'PY'
import sys
g = float(sys.argv[1]); t = float(sys.argv[2])
print(1 if g <= t else 0)
PY
)
    if [[ "${pass}" == "1" ]]; then
      pass_count=$((pass_count + 1))
    else
      pass_count=0
    fi

    echo "[${stage}] pass_streak=${pass_count}/${patience_need}"
    if (( pass_count >= patience_need )); then
      echo "[${stage}] DONE -> promote to next stage"
      break
    fi
  done
done

echo ""
echo "Curriculum finished. Last checkpoint: ${resume_ckpt}"
