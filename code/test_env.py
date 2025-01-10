from stable_baselines3.common.env_checker import check_env
from model.env import MultiAgentPickupEnv
import numpy as np
import random

# import sys
# sys.path.insert(0, '/path/to/libstdcxx/v6')
# from libstdcxx.v6.printers import register_libstdcxx_printers
# register_libstdcxx_printers(None)

# 创建环境
env = MultiAgentPickupEnv(training=True, grid_path="/localhome/yya305/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map", 
                          seed=40, 
                        solver="PBS", agent_num_lower_bound=10, agent_num_higher_bound=50, eval_data_path=None, task_num=20)

# 重置环境
# check_env(env)

# breakpoint()
obs, info = env.reset()
# print("Initial observation:", obs)

done = False
step = 0
while not done:
    # 随机采样一个动作
    task_num = obs['tasks_num']
    free_agents_num = obs['free_agents_num']
    print("num", task_num, free_agents_num)
    action = np.zeros([50], dtype=np.int32)
    for i in range(free_agents_num):
        action[i] = i
    print(f"Step {step}: Taking action {action}")

    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, Done: {terminated or truncated}")

    # 渲染环境（可选）
    # env.render()

    done = terminated
    step += 1

env.close()
