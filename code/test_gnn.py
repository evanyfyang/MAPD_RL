#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from model.REINFORCE import REINFORCE
from model.env_gnn import MultiAgentPickupEnv
from model.gnn_policy import GNNPolicy

import argparse
import os
import numpy as np
import torch
import random

from stable_baselines3.common.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Test REINFORCE with GNN Policy on MultiAgentPickupEnv.")

    # ------------- 必需参数 -------------
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="测试样例文件路径")

    # ------------- 可选参数 -------------
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="GNN隐藏层维度 (默认: 64)")
    parser.add_argument("--test_episodes", type=int, default=1,
                        help="测试回合数 (默认: 1)")
    parser.add_argument("--test_env_seed", type=int, default=100,
                        help="测试环境随机种子 (默认: 100)")

    # ------------- 环境参数 (使用train_gnn.sh的默认值) -------------
    parser.add_argument("--grid_path", type=str, 
                        default="/local-scratchg/yifan/2024/MAPD/MAPD_RL/code/mapf_solver/maps/Instances/small/kiva-small.map",
                        help="网格地图路径")
    parser.add_argument("--solver", type=str, default="PBS",
                        help="路径规划算法")
    parser.add_argument("--agent_num_lower_bound", type=int, default=10,
                        help="agent数量下界")
    parser.add_argument("--agent_num_higher_bound", type=int, default=50,
                        help="agent数量上界")
    parser.add_argument("--task_num", type=int, default=500,
                        help="任务数量")

    # ------------- GNN Policy 参数 (使用train_gnn.sh的默认值) -------------
    parser.add_argument("--grid_feature_dim", type=int, default=2,
                        help="网格特征维度")
    parser.add_argument("--lower_gnn_type", type=str, default="cnn_channels",
                        help="低层级GNN类型")
    parser.add_argument("--higher_gnn_type", type=str, default="gat",
                        help="高层级GNN类型")
    parser.add_argument("--gnn_num_layers", type=int, default=3,
                        help="GNN层数")
    parser.add_argument("--gnn_dropout", type=float, default=0.1,
                        help="GNN dropout率")
    parser.add_argument("--gnn_heads", type=int, default=4,
                        help="GAT注意力头数")
    parser.add_argument("--edge_combine", type=str, default="add",
                        help="边特征组合方式")
    parser.add_argument("--use_undirected", action="store_true", default=True,
                        help="是否使用无向图")
    parser.add_argument("--use_sinkhorn", action="store_true", default=True,
                        help="是否使用Sinkhorn算法")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Sinkhorn温度参数")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Sinkhorn迭代次数")
    parser.add_argument("--unassign_threshold", type=float, default=0.3679,
                        help="未分配阈值")
    parser.add_argument("--invalid_edge_score", type=float, default=-100.0,
                        help="无效边分数")
    parser.add_argument("--use_hungarian_for_deterministic", action="store_true", default=True,
                        help="确定性模式下是否使用Hungarian算法")
    parser.add_argument("--use_gumbel", action="store_true", default=False,
                        help="是否使用Gumbel噪声（仅在use_sinkhorn=True时有效）")

    # ------------- 其他参数 -------------
    parser.add_argument("--global_seed", type=int, default=0,
                        help="全局随机种子")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="是否输出详细信息")
    parser.add_argument("--model_only_eval", action="store_true", default=False,
                        help="测试时仅使用model结果，不回退到expert状态")
    parser.add_argument("--infer_decode_mode", type=str, default="sequential", choices=["sequential", "hungarian"],
                        help="deterministic推理解码方式: sequential 或 hungarian")

    args = parser.parse_args()
    return args


def test_model(checkpoint_path, test_data_path, args):
    """测试模型"""
    print(f"加载模型: {checkpoint_path}")
    print(f"测试数据: {test_data_path}")
    
    # 设置随机种子
    set_random_seed(args.global_seed)

    # 先加载模型，再根据checkpoint中的observation_space对齐测试环境维度
    model = REINFORCE.load(checkpoint_path)
    if hasattr(model, "policy") and hasattr(model.policy, "infer_decode_mode"):
        model.policy.infer_decode_mode = args.infer_decode_mode

    expected_agent_cap = None
    expected_candidate_k = None
    try:
        obs_space = model.observation_space
        if hasattr(obs_space, "spaces") and "free_agents_nearest_tasks" in obs_space.spaces:
            nearest_shape = obs_space.spaces["free_agents_nearest_tasks"].shape
            if len(nearest_shape) == 3:
                expected_agent_cap = int(nearest_shape[0])
                expected_candidate_k = int(nearest_shape[1])
    except Exception:
        pass
    
    # 环境参数
    env_kwargs = dict(
        training=False,
        grid_path=args.grid_path,
        solver=args.solver,
        agent_num_lower_bound=args.agent_num_lower_bound,
        agent_num_higher_bound=args.agent_num_higher_bound,
        eval_data_path=test_data_path,
        task_num=args.task_num,
        pos_reward=False,
        model_only_eval=args.model_only_eval,
    )

    if expected_agent_cap is not None:
        env_kwargs["agent_num_higher_bound"] = expected_agent_cap
        env_kwargs["agent_num_lower_bound"] = min(env_kwargs["agent_num_lower_bound"], expected_agent_cap)
    if expected_candidate_k is not None:
        env_kwargs["nearest_tasks_min_k"] = expected_candidate_k
    
    # 创建测试环境
    test_env = MultiAgentPickupEnv(seed=args.test_env_seed, **env_kwargs)
    
    # GNN Policy 参数
    policy_kwargs = dict(
        hidden_dim=args.hidden_dim,
        grid_feature_dim=args.grid_feature_dim,
        lower_gnn_type=args.lower_gnn_type,
        higher_gnn_type=args.higher_gnn_type,
        max_agents=args.agent_num_higher_bound,
        max_tasks=args.task_num,
        pretrain_steps=0,
        fix_div=False,
        not_div=False,
        use_sinkhorn=args.use_sinkhorn,
        tau=args.tau,
        iterations=args.iterations,
        unassign_threshold=args.unassign_threshold,
        invalid_edge_score=args.invalid_edge_score,
        use_hungarian_for_deterministic=args.use_hungarian_for_deterministic,
        use_gumbel=args.use_gumbel,
        gnn_num_layers=args.gnn_num_layers,
        gnn_dropout=args.gnn_dropout,
        gnn_heads=args.gnn_heads,
        edge_combine=args.edge_combine,
        use_undirected=args.use_undirected
    )
    
    # 运行测试
    episode_rewards = []
    episode_service_times = []
    
    print(f"开始测试，共 {args.test_episodes} 个回合...")
    
    for ep in range(args.test_episodes):
        print(f"\n=== 回合 {ep + 1}/{args.test_episodes} ===")
        
        obs = test_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            
            if args.verbose:
                print(f"步骤 {step_count}: reward = {reward}")
            
            if done:
                # 在测试模式下，reward是最终的service_time
                final_service_time = reward
                total_reward = final_service_time
            else:
                total_reward += reward
            
            step_count += 1
        
        episode_rewards.append(total_reward)
        episode_service_times.append(final_service_time if done else total_reward)
        
        print(f"回合 {ep + 1} 完成:")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  服务时间: {final_service_time if done else total_reward:.2f}")
        print(f"  步数: {step_count}")
    
    # 计算统计信息
    avg_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    avg_service_time = float(np.mean(episode_service_times))
    std_service_time = float(np.std(episode_service_times))
    
    print(f"\n=== 测试结果 ===")
    print(f"测试回合数: {args.test_episodes}")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"平均服务时间: {avg_service_time:.2f} ± {std_service_time:.2f}")
    print(f"最佳服务时间: {min(episode_service_times):.2f}")
    print(f"最差服务时间: {max(episode_service_times):.2f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_service_time': avg_service_time,
        'std_service_time': std_service_time,
        'episode_rewards': episode_rewards,
        'episode_service_times': episode_service_times
    }


def main():
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 找不到checkpoint文件: {args.checkpoint}")
        return
    
    if not os.path.exists(args.test_data_path):
        print(f"错误: 找不到测试数据文件: {args.test_data_path}")
        return
    
    # 运行测试
    results = test_model(args.checkpoint, args.test_data_path, args)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 