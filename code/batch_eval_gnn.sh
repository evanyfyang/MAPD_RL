#!/usr/bin/env bash
set -euo pipefail

# GPU and hidden dim can be overridden via env vars
GPU_ID="${GPU_ID:-1}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"

export USE_EXPERT="False"
# Paths
CKPT_DIR="/local-scratchg/yifan/2024/MAPD/MAPD_RL/models/gnn_gnn_experiment_20260123_0352_lr_3e-4_gamma_0.99_steps__sp_mpnn_edge_node_gnn/checkpoints"
TASK_DIR="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small"
OUT_CSV="${OUT_CSV:-/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/model_eval_results_hungarian_50k.csv}"

# Optional: activate conda PFN environment if available
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate MAPD_RL || true
fi

echo "f,M,model_steps,service_time_div_500,makespan" > "$OUT_CSV"

declare -a F_VALUES=("0.2" "0.5" "1" "2" "5" "10")
declare -a M_VALUES=("10" "20" "30" "40" "50")
declare -a MODEL_STEPS=("50000")

for steps in "${MODEL_STEPS[@]}"; do
    ckpt="${CKPT_DIR}/reinforce_gnn_mapd_model_${steps}_steps.zip"
    if [[ ! -f "$ckpt" ]]; then
        echo "WARNING: checkpoint not found: $ckpt" >&2
        continue
    fi
    for f in "${F_VALUES[@]}"; do
        task="${TASK_DIR}/kiva-${f}.task"
        for M in "${M_VALUES[@]}"; do
            grid="${TASK_DIR}/kiva-${M}-500-5.map"
            echo "Running: steps=${steps} f=${f} M=${M}" >&2
            set +e
            output=$(bash /local-scratchg/yifan/2024/MAPD/MAPD_RL/code/test_gnn.sh \
                -c "$ckpt" \
                -d "$task" \
                -g "$GPU_ID" \
                --hidden_dim "$HIDDEN_DIM" \
                --grid_path "$grid" 2>&1)
            exit_code=$?
            set -e

            done_line=$(printf "%s\n" "$output" | grep -E 'done: True' | tail -n 1 || true)
            if [[ -z "$done_line" ]]; then
                echo "WARN: Could not find 'done: True' line for steps=${steps} f=${f} M=${M} (exit ${exit_code})" >&2
                continue
            fi

            service_time=$(printf "%s\n" "$done_line" | grep -oE 'service_time:\s*[0-9]+' | grep -oE '[0-9]+' || true)
            makespan=$(printf "%s\n" "$done_line" | grep -oE 'makespan[: ]+[0-9]+' | grep -oE '[0-9]+' || true)

            if [[ -z "$service_time" || -z "$makespan" ]]; then
                echo "WARN: Could not parse service_time/makespan from 'done: True' line for steps=${steps} f=${f} M=${M}" >&2
                continue
            fi

            value=$(awk -v v="$service_time" 'BEGIN{printf "%.6f", v/500}')
            echo "${f},${M},${steps},${value},${makespan}" >> "$OUT_CSV"
        done
    done
done

echo "Saved: $OUT_CSV"


