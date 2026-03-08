#!/usr/bin/env bash
set -e

PARALLEL_JOBS=4
OUTPUT_CSV="model_results.csv"
echo "agent_num,frequency,average_service_time,average_running_time" > "$OUTPUT_CSV"

agent_nums=(50)
frequencies=(0.2 0.5 1 2 5 10)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# 函数用于运行单个测试
run_test() {
  local agent_num=$1
  local frequency=$2
  local job_number=$3
  # 根据作业编号分配 GPU，每个作业编号减1后对 NUM_GPUS 取模，保证分布均匀
  local gpu_id=$(( (job_number - 1) % NUM_GPUS ))
  
  GRID_PATH="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-${agent_num}-500-5.map"
  EVAL_DATA_PATH="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-${frequency}.task"
  OUTPUT_CSV="model_results.csv"
  CHECKPOINT_PATH="/local-scratchg/yifan/2024/MAPD/MAPD_RL/models/_20250403_1529_lr_1e-4_gamma_0.99_tau_0.01_bmm_0_16_500/checkpoints/a2c_mapd_model_48000_steps.zip"
  
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  
  start_time=$(date +%s%3N)
  
  # 运行测试脚本，这里假设 test.sh 能使用传入的 GPU_ID 参数
  output=$(bash test.sh 500 "$GRID_PATH" "$EVAL_DATA_PATH" "$CHECKPOINT_PATH" "${gpu_id}")
  
  end_time=$(date +%s%3N)
  
  running_time=$((end_time - start_time))
  average_running_time=$(echo "scale=2; $running_time / 500" | bc)
  
  reward=$(echo "$output" | grep "\[Test\] Episodes: 1, Reward:" | awk '{print $5}')
  average_service_time=$(echo "scale=2; $reward / 500" | bc)
  
  echo "$agent_num,$frequency,$average_service_time,$average_running_time" >> "$OUTPUT_CSV"
}

export -f run_test
export NUM_GPUS
export OUTPUT_CSV

# 构造任务列表，每个元素是 "agent_num frequency" 的组合
tasks=()
for agent in "${agent_nums[@]}"; do
  for freq in "${frequencies[@]}"; do
    tasks+=("$agent $freq")
  done
done

# 将任务通过管道输入 GNU parallel，使用 --colsep ' ' 分割 agent_num 和 frequency，
# 并自动传递作业编号 {#} 给 run_test 来分配 GPU。
printf "%s\n" "${tasks[@]}" | parallel -j $PARALLEL_JOBS --colsep ' ' run_test {1} {2} {#}