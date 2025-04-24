#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from model.A2C import A2CMAPD
from model.env import MultiAgentPickupEnv
from model.feature_extractor import MAPDFeatureExtractor
from model.policy import MAPDActorCriticPolicy

import argparse
import os
import numpy as np
import torch
import random

from functools import partial

# SB3 常用工具
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

import pickle
from tqdm import tqdm


def make_env(
    rank: int,
    seed: int,
    env_kwargs: dict,
):
    def _init():
        env_seed = seed + rank
        env = MultiAgentPickupEnv(seed=env_seed, **env_kwargs)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train A2C on MultiAgentPickupEnv with custom classes.")

    # ------------- A2C 相关超参数 -------------
    parser.add_argument("--total_timesteps", type=int, default=100000,
                        help="训练总步数 (总采样数).")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="同时并行的环境数量 (多进程).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="初始学习率.")
    parser.add_argument("--lr_schedule", type=str, default="constant",
                        choices=["constant", "linear"],
                        help="学习率调度策略: constant 或 linear.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子 (discount factor).")
    parser.add_argument("--use_rms_prop", action="store_true", default=False,
                        help="是否使用RMSProp而非Adam优化器.")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="熵正则系数 (entropy coefficient).")

    # ------------- 环境相关超参数 -------------
    parser.add_argument("--env_seed", type=int, default=40,
                        help="环境的基础随机种子.")
    parser.add_argument("--training", action="store_true", default=True,
                        help="是否在训练模式下 (影响环境内部逻辑).")
    parser.add_argument("--grid_path", type=str, default=None,
                        help="网格地图路径 (可选).")
    parser.add_argument("--solver", type=str, default="PBS",
                        help="使用的路径规划算法 (PBS, CBS, etc).")
    parser.add_argument("--agent_num_lower_bound", type=int, default=10,
                        help="agent数量下界.")
    parser.add_argument("--agent_num_higher_bound", type=int, default=50,
                        help="agent数量上界.")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="评估数据集路径 (可选).")
    parser.add_argument("--task_num", type=int, default=500,
                        help="生成任务数量 (for MAPD).")
    parser.add_argument("--pos_reward", action="store_true", default=False,
                        help="reward是否加一.")
    parser.add_argument("--fix_div", action="store_true", default=False)
    parser.add_argument("--not_div", action="store_true", default=False)
    parser.add_argument("--grid_size", type=tuple, default=(21, 35))
                        
    # ------------- Policy 相关超参数 -------------
    parser.add_argument("--tau", type=float, default=1.0,
                        help="自定义Policy参数, 控制某些soft-update或其他逻辑.")
    parser.add_argument("--iterations", type=int, default=10,
                        help="自定义Policy参数, 可能控制迭代次数.")
    parser.add_argument("--decay_rate", type=float, default=2e-4,
                        help="自定义Policy参数.")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="传给MAPDFeatureExtractor和Policy的特征维度 (MLP维度等).")
    parser.add_argument("--cal_type", type=str, default='bmm', choices=['bmm', 'concat'])

    # ------------- Checkpoint & Seed -------------
    parser.add_argument("--checkpoint_freq", type=int, default=10000,
                        help="每隔多少step保存一次模型checkpoint.")
    parser.add_argument("--save_dir", type=str, default="./logs",
                        help="模型和日志保存的目录.")
    parser.add_argument("--global_seed", type=int, default=0,
                        help="全局随机种子 (Python, NumPy, PyTorch).")

    # ------------- 测试相关 -------------
    parser.add_argument("--test_checkpoint", type=str, default=None,
                        help="若指定某模型路径，则只测试该模型.")
    parser.add_argument("--test_env_seed", type=int, default=100,
                        help="测试环境的随机种子.")
    parser.add_argument("--test_episodes", type=int, default=1,
                        help="测试时跑多少回合.")

    parser.add_argument("--pretrain_steps", type=int, default=100000,
                       help="预训练步数（使用专家动作）")
    parser.add_argument("--pretrain_epochs", type=int, default=0,
                       help="预训练epochs（使用专家动作）")
    parser.add_argument("--pretrain_data_path", type=str, default="/local-scratchg/yifan/2024/MAPD/MAPD_RL/step_data", 
                        help="预训练数据路径")  
    args = parser.parse_args()
    return args


def linear_schedule(initial_value: float):
    """
    返回一个learning rate随训练进度线性衰减的函数:
    progress_remaining 在 [1.0 -> 0.0]之间
    """
    def func(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return func

def test_model(model_path, env_kwargs, n_episodes=5, seed=100):
    print(f"Loading model from: {model_path}")
    model = A2CMAPD.load(model_path)

    test_env = MultiAgentPickupEnv(seed=seed, **env_kwargs)

    episode_rewards = []
    for ep in range(n_episodes):
        obs = test_env.reset()
        done = False
        total_rew = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            total_rew += reward

        episode_rewards.append(total_rew)

    avg_reward = float(np.mean(episode_rewards))
    print(f"[Test] Episodes: {n_episodes}, Reward: {avg_reward:.2f}")
    return avg_reward

def pretrain_model(model, args):
    model.policy.set_training_mode(True)
    model.policy.pretrain_mode = True

    pretrain_data_path = args.pretrain_data_path
    with open(os.path.join(pretrain_data_path, "pretrain_data.pkl"), "rb") as f:
        pretrain_data = pickle.load(f)

    print(f"Load pretrain data from {pretrain_data_path}, total {len(pretrain_data)} samples")

    num_batches = (len(pretrain_data) + args.n_envs*4 - 1) // (args.n_envs * 4)

    
    for epoch in range(args.pretrain_epochs):
        total_loss = 0.0

        random.shuffle(pretrain_data)
        epoch_bar = tqdm(range(num_batches), desc=f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}")

        for batch_idx in epoch_bar:
            start_idx = batch_idx * args.n_envs * 4
            batch_samples = pretrain_data[start_idx: start_idx + args.n_envs * 4]

            batch_obs = {}

            for key in batch_samples[0]:
                if key == "expert_actions":
                    batch_obs[key] = torch.tensor(
                        [sample[key] for sample in batch_samples],
                        dtype=torch.long,
                        device=model.device
                    )
                else:
                    batch_obs[key] = torch.tensor(
                        [sample[key] for sample in batch_samples],
                        dtype=torch.float32,
                        device=model.device
                    )
            
            actions = batch_obs["expert_actions"]
            _, loss_ce, _ = model.policy.evaluate_actions(batch_obs, actions)

            loss = loss_ce.sum()/loss_ce.size(0)
            
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()

            total_loss += loss.item()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(pretrain_data)
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs} 平均Loss: {avg_loss:.4f}")

    model.policy.pretrain_mode = False

def main():
    args = parse_args()

    # --------------- 全局随机种子 ---------------
    # 让 Python、NumPy、PyTorch 的随机种子一致
    set_random_seed(args.global_seed)

    if args.test_checkpoint is not None:
        env_kwargs = dict(
            training=False,     
            grid_path=args.grid_path,
            solver=args.solver,
            agent_num_lower_bound=args.agent_num_lower_bound,
            agent_num_higher_bound=args.agent_num_higher_bound,
            eval_data_path=args.eval_data_path,
            task_num=args.task_num,
            pos_reward=args.pos_reward
        )
        test_model(args.test_checkpoint, env_kwargs, n_episodes=args.test_episodes, seed=args.test_env_seed)
        return
    
    env_kwargs = dict(
        training=args.training,
        grid_path=args.grid_path,
        solver=args.solver,
        agent_num_lower_bound=args.agent_num_lower_bound,
        agent_num_higher_bound=args.agent_num_higher_bound,
        task_num=args.task_num,
        pos_reward=args.pos_reward
    )

    if args.n_envs > 1:
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv

    def make_thunk(rank):
        return make_env(rank, args.env_seed, env_kwargs)

    env_fns = [make_thunk(i) for i in range(args.n_envs)]
    vec_env = vec_env_cls(env_fns)

    if args.lr_schedule == "constant":
        lr_func = args.learning_rate
    else:
        lr_func = linear_schedule(args.learning_rate)


    policy_kwargs = dict(
        features_extractor_class=MAPDFeatureExtractor,
        features_extractor_kwargs=dict(
            hidden_size=args.hidden_size,
            grid_size=args.grid_size
        ),
        tau=args.tau,
        iterations=args.iterations,
        decay_rate=args.decay_rate,
        fix_div=args.fix_div,
        not_div=args.not_div,
        max_task=args.task_num,
        pretrain_steps=args.pretrain_steps,
        cal_type=args.cal_type
    )

    model = A2CMAPD(
        policy=MAPDActorCriticPolicy,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_func,
        n_steps=1,               
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        use_rms_prop=args.use_rms_prop,
        verbose=1,              
    )

    if args.pretrain_epochs > 0:
        pretrain_model(model, args)
        pretrain_checkpoint_path = os.path.join(args.save_dir,"checkpoints", "pretrain_checkpoint.zip")
        model.save(pretrain_checkpoint_path)
        print(f"Pretrain finished, model saved to: {pretrain_checkpoint_path}")

  
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="a2c_mapd_model"
    )

    model.learn(
        total_timesteps=args.total_timesteps * args.n_envs,
        callback=checkpoint_callback
    )

    final_model_path = os.path.join(args.save_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Training finished, model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
