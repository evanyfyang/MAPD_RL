from stable_baselines3.common.env_checker import check_env
from model.env import MultiAgentPickupEnv
import numpy as np
import random
import threading
import multiprocessing

import sys


def run_simulation(fre, agent_num, file_lock):
    env = MultiAgentPickupEnv(
        training=False,
        grid_path=f"/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-{agent_num}-500-5.map",
        seed=40,
        eval_data_path=f'/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-{fre}.task',
        solver="PBS",
        agent_num_lower_bound=10,
        agent_num_higher_bound=50,
        task_num=500
    )

    for i in range(1):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            task_num = obs['tasks_num']
            free_agents_num = obs['free_agents_num']
            expert_actions = obs['expert_actions']
            action = expert_actions

            assigned_tasks = []
            for t in action:
                if t != -1:
                    assigned_tasks.append(t)
            for i in range(len(action)):
                if i not in assigned_tasks:
                    for j in range(len(action)):
                        if action[j] == -1:
                            action[j] = i
                    break

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            step += 1

    with file_lock:
        with open('test_env_with_delivering.txt', 'a') as f:
            f.write(f"{agent_num} {fre} {reward/500}\n")
            f.flush()

    env.close()

def main():
    with multiprocessing.Manager() as manager:
        file_lock = manager.Lock()
        with multiprocessing.Pool(processes=30) as pool:  # 设置同时运行的进程数量上限
            tasks = [(fre, agent_num, file_lock) for fre in [0.2, 0.5, 1, 2, 5, 10] for agent_num in [10, 20, 30, 40, 50]]
            pool.starmap(run_simulation, tasks)
    
if __name__ == "__main__":
    main()

# 创建环境
# f = open('test_env_with_delivering.txt','w')

# for fre in [1]:
#     for agent_num in [10]:
#         # env = MultiAgentPickupEnv(training=False, grid_path=f"/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-{agent_num}-500-5.map", 
#         #                         seed=40, eval_data_path=f'/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-{fre}.task',
#         #                         solver="PBS", agent_num_lower_bound=10, agent_num_higher_bound=50, task_num=500, pos_reward=True)
#         env = MultiAgentPickupEnv(training=True, grid_path=f"/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map", 
#                                 seed=40, eval_data_path=None,
#                                 solver="PBS", agent_num_lower_bound=10, agent_num_higher_bound=50, task_num=500)

#         for i in range(1):
#             # breakpoint()
#             obs, _ = env.reset()
#             # obs = env.reset()
#             # print("Initial observation:", obs)

#             done = False
#             step = 0
#             while not done:
#                 task_num = obs['tasks_num']
#                 free_agents_num = obs['free_agents_num']
#                 expert_actions = obs['expert_actions']
#                 print("num", task_num, free_agents_num)
#                 action = expert_actions

#                 assigned_tasks = []
#                 for t in action:
#                     if t != -1:
#                         assigned_tasks.append(t)
#                 for i in range(len(action)):
#                     if i not in assigned_tasks:
#                         for j in range(len(action)):
#                             if action[j] == -1:
#                                 action[j] = i
#                         break
#                 print(f"Step {step}: Taking action {action}")

#                 # 执行动作
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 print(f"{agent_num} {fre} Average Serivce Time: {reward/500}, Done: {terminated or truncated}")
                

#                 # 渲染环境（可选）
#                 # env.render()

#                 done = terminated
#                 step += 1
#         # f.write(f"{agent_num} {fre} {reward/500}\n")
#         # f.flush()
#         env.close()
# f.close()