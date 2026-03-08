from typing import Any, Optional, TypeVar, Union
import sys
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

# 导入GNNPolicy
from model.gnn_policy import GNNPolicy
from model.utils import apply_hungarian_algorithm

SelfREINFORCE = TypeVar("SelfREINFORCE", bound="REINFORCE")


class REINFORCE(OnPolicyAlgorithm):
    """
    REINFORCE算法（基于策略梯度）与监督学习预训练阶段 - 专为GNNPolicy设计
    
    REINFORCE是一种直接的策略梯度方法，使用蒙特卡洛回报估计来更新策略。
    本实现在前一定步数使用监督学习进行预训练，然后切换到REINFORCE策略梯度方法。
    该实现专为GNNPolicy设计，用于多智能体任务分配问题。
    
    :param policy: GNNPolicy策略模型的实例或类
    :param env: 学习环境
    :param learning_rate: 学习率，可以是一个函数
    :param n_steps: 每次环境步数（批量大小为n_steps * n_env）
    :param gamma: 折扣因子
    :param ent_coef: 熵系数，用于鼓励探索
    :param max_grad_norm: 梯度裁剪的最大值
    :param sde_sample_freq: 使用gSDE时每n步采样一个新的噪声矩阵
    :param rollout_buffer_class: 使用的轨迹缓冲区类，如果为None则自动选择
    :param rollout_buffer_kwargs: 传递给轨迹缓冲区的关键字参数
    :param normalize_advantage: 是否标准化优势函数
    :param stats_window_size: 日志记录窗口大小
    :param tensorboard_log: tensorboard的日志位置
    :param policy_kwargs: 传递给策略的其他参数
    :param verbose: 详细程度
    :param seed: 随机种子
    :param device: 运行代码的设备（CPU或CUDA）
    :param _init_setup_model: 是否在创建实例时构建网络
    """
    
    def __init__(
        self,
        policy: Union[type[GNNPolicy], GNNPolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # 确保policy是GNNPolicy类型
        if isinstance(policy, str):
            raise ValueError(
                "REINFORCE算法需要直接使用GNNPolicy类，不支持通过字符串指定策略类型。"
                "请使用 policy=GNNPolicy 而非 policy='GNNPolicy'"
            )
        
        # 检查是否确实是GNNPolicy
        if policy != GNNPolicy and not (isinstance(policy, type) and issubclass(policy, GNNPolicy)):
            raise ValueError(
                f"REINFORCE算法只支持GNNPolicy策略类，但收到了{policy}。"
                "请使用GNNPolicy作为策略类。"
            )

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=1.0,  # REINFORCE不使用GAE，设置为1.0保持蒙特卡洛回报估计
            ent_coef=ent_coef,
            vf_coef=0.0,  # REINFORCE不使用值函数，但我们保留结构以兼容Policy类
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        
        self.normalize_advantage = normalize_advantage
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """
        初始化策略、轨迹缓冲区和全局变量
        专门为GNNPolicy设计
        """
        super()._setup_model()
        
        # 确保policy有计数器
        if not hasattr(self.policy, 'current_step'):
            self.policy.current_step = 0
            
        # 如果policy没有pretrain_steps，设置一个默认值
        if not hasattr(self.policy, 'pretrain_steps'):
            self.policy.pretrain_steps = 10000
        # 预训练阶段日志基于环境步数 current_step 触发
    
    def train(self) -> None:
        """
        使用当前收集的轨迹缓冲区更新策略（对整个数据进行一次梯度步骤）。
        如果处于预训练阶段，则使用监督学习方式；
        否则使用REINFORCE算法进行策略梯度更新。
        """
        # 切换到训练模式（影响batch norm / dropout）
        self.policy.set_training_mode(True)
        
        # 更新优化器学习率
        self._update_learning_rate(self.policy.optimizer)
        
        # 这将只循环一次（一次性获取所有数据）
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # 将离散动作从float转换为long
                actions = actions.long().flatten()
            
            # 评估动作
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, 
                actions
            )

            print("Current step: ", self.policy.current_step)
            
            # 检查是否处于预训练阶段
            if self.policy.current_step < self.policy.pretrain_steps:
                # 预训练阶段：使用监督学习
                policy_loss = log_prob  # GNNPolicy已经在evaluate_actions中计算了适当的损失
                loss = policy_loss.mean()
                print("Pre-train loss: ", loss.item())

                # 每100次update：比较模型匈牙利匹配成本与专家动作成本（按 H+W 缩放）
                if self.policy.current_step > 0 and (self.policy.current_step % 100 == 0):
                    obs = rollout_data.observations
                    (action_probs, original_scores, valid_mask, free_agents_num, free_tasks_num,
                        free_agents_nearest_tasks, device, batch_size) = self.policy._compute_policy_features(obs)

                    hungarian_mats = apply_hungarian_algorithm(
                        original_scores, free_agents_num, free_tasks_num, use_probabilities=False
                    )

                    # 取网格尺寸 H, W
                    grid = obs["grid"] if isinstance(obs, dict) else None
                    if grid is not None:
                        H, W = int(grid.shape[1]), int(grid.shape[2])
                        scale_hw = float(H + W)
                    else:
                        scale_hw = 56.0

                    def compute_costs_for_pairs(b, pairs):
                        if free_agents_nearest_tasks is None:
                            return 0.0, 0.0
                        nearest_b = free_agents_nearest_tasks[b]
                        num_agents_b = int(free_agents_num[b].item())
                        num_tasks_b = int(free_tasks_num[b].item())
                        sum_a2p = 0.0
                        sum_total = 0.0
                        for (i, j) in pairs:
                            if not (0 <= i < num_agents_b and 0 <= j < num_tasks_b):
                                sum_a2p += 1.0 * scale_hw
                                sum_total += 2.0 * scale_hw
                                continue
                            found = False
                            for k in range(nearest_b.shape[1]):
                                t_id = int(nearest_b[i, k, 0].item())
                                if t_id == j:
                                    a2p = float(nearest_b[i, k, 1].item())
                                    p2d = float(nearest_b[i, k, 2].item())
                                    sum_a2p += a2p
                                    sum_total += (a2p + p2d)
                                    found = True
                                    break
                            if not found:
                                sum_a2p += 1.0 * scale_hw
                                sum_total += 2.0 * scale_hw
                        return sum_a2p, sum_total

                    deltas_total = []
                    preds_total = []
                    exps_total = []
                    deltas_a2p = []
                    preds_a2p = []
                    exps_a2p = []
                    for b in range(len(hungarian_mats)):
                        num_agents_b = int(free_agents_num[b].item())
                        num_tasks_b_total = int(free_tasks_num[b].item()) + 1
                        mat = hungarian_mats[b]
                        pred_pairs = []
                        if mat.numel() > 0:
                            rows, cols = mat.nonzero(as_tuple=True)
                            for ri, cj in zip(rows.tolist(), cols.tolist()):
                                if 0 <= ri < num_agents_b and 0 <= cj < num_tasks_b_total - 1:
                                    pred_pairs.append((ri, cj))
                        # 专家pairs
                        exp_pairs = []
                        if isinstance(obs, dict) and "expert_actions" in obs:
                            exp_actions_b = obs["expert_actions"][b]
                            for i in range(num_agents_b):
                                a = int(exp_actions_b[i].item()) if i < exp_actions_b.shape[0] else (num_tasks_b_total - 1)
                                if 0 <= a < (num_tasks_b_total - 1):
                                    exp_pairs.append((i, a))

                        a2p_pred, total_pred = compute_costs_for_pairs(b, pred_pairs)
                        a2p_exp, total_exp = compute_costs_for_pairs(b, exp_pairs)
                        preds_a2p.append(float(a2p_pred))
                        exps_a2p.append(float(a2p_exp))
                        deltas_a2p.append(float(a2p_pred - a2p_exp))
                        preds_total.append(float(total_pred))
                        exps_total.append(float(total_exp))
                        deltas_total.append(float(total_pred - total_exp))

                    if preds_total or preds_a2p:
                        avg_pred_total = (sum(preds_total) / len(preds_total)) if preds_total else 0.0
                        avg_exp_total = (sum(exps_total) / len(exps_total)) if exps_total else 0.0
                        avg_delta_total = (sum(deltas_total) / len(deltas_total)) if deltas_total else 0.0
                        avg_pred_a2p = (sum(preds_a2p) / len(preds_a2p)) if preds_a2p else 0.0
                        avg_exp_a2p = (sum(exps_a2p) / len(exps_a2p)) if exps_a2p else 0.0
                        avg_delta_a2p = (sum(deltas_a2p) / len(deltas_a2p)) if deltas_a2p else 0.0
                        print(
                            f"[Pretrain] step {self.policy.current_step}: "
                            f"a2p(mean) pred={avg_pred_a2p:.2f}, expert={avg_exp_a2p:.2f}, delta={avg_delta_a2p:.2f} | "
                            f"a2p+p2d(mean) pred={avg_pred_total:.2f}, expert={avg_exp_total:.2f}, delta={avg_delta_total:.2f}"
                        )
            else:
                # REINFORCE阶段：使用多样本中心化策略梯度（gumbel_hungarian/row_softmax）
                # r0: 当前step即时奖励（n_steps=1下等于returns）
                r0 = rollout_data.returns
                if isinstance(r0, th.Tensor) and r0.ndim > 1:
                    r0 = r0.squeeze(-1)
                # 计算多样本 (log_prob_all, centered_all)
                # Pass cached log_prob/entropy from evaluate_actions to avoid re-sampling bias
                multi = self.policy.compute_centered_returns(
                    self.env, rollout_data.observations, actions, r0, deterministic=False,
                    cached_log_prob=log_prob, cached_entropy=entropy
                )
                if multi is not None:
                    log_prob_all, centered_all, returns_all = multi  # [B,K], [B,K], [B,K]
                    # policy_loss = -(returns_all.to(device=self.device) * log_prob_all.to(device=self.device)).mean()
                    policy_loss = -(centered_all.to(device=self.device) * log_prob_all.to(device=self.device)).mean()
                    loss = policy_loss - self.ent_coef * (entropy.mean() if entropy is not None else 0.0)
                else:
                    # 回退：单样本
                    returns = rollout_data.returns
                    policy_loss = -(returns * log_prob).mean()
                    loss = policy_loss
                
                # 打印：熵/熵损失、K个reward（中心化前后）、RL loss（单行）
                try:
                    ent_mean = None if entropy is None else entropy.mean()
                    ent_loss_val = None if ent_mean is None else float((-ent_mean).item())
                    ent_mean_val = None if ent_mean is None else float(ent_mean.item())
                except Exception:
                    ent_mean_val, ent_loss_val = None, None

                try:
                    # 打印第一个env的K个reward及其中心化（避免过长）
                    if multi is not None and returns_all is not None and centered_all is not None:
                        r_first = returns_all[0].detach().cpu().tolist()
                        c_first = centered_all[0].detach().cpu().tolist()
                        print(f"RL loss: {loss.item():.6f} | policy: {policy_loss.item():.6f} | entropy(mean): {ent_mean_val} | entropy_loss: {ent_loss_val} | rewards: {r_first} | centered: {c_first}")
                    else:
                        print(f"RL loss: {loss.item():.6f} | policy: {policy_loss.item():.6f} | entropy(mean): {ent_mean_val} | entropy_loss: {ent_loss_val}")
                except Exception:
                    print(f"RL loss: {loss.item():.6f} | policy: {policy_loss.item():.6f}")
            
            sys.stdout.flush()
            
            # 优化步骤
            self.policy.optimizer.zero_grad()
            loss.backward()
            
            # 裁剪梯度范数
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # 更新步数
            self.policy.current_step += self.n_envs * self.n_steps
        
        # 记录训练统计信息
        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", loss.item())
        if entropy is not None:
            self.logger.record("train/entropy", entropy.mean().item())
        # 记录当前训练阶段
        stage = "pretrain" if self.policy.current_step < self.policy.pretrain_steps else "policy_gradient"
        self.logger.record("train/stage", stage)
    
    def learn(
        self: SelfREINFORCE,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "REINFORCE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfREINFORCE:
        """
        训练REINFORCE模型
        
        :param total_timesteps: 学习的总时间步长
        :param callback: 在训练期间调用的回调
        :param log_interval: 日志输出的间隔步数
        :param tb_log_name: tensorboard日志的名称
        :param reset_num_timesteps: 是否重置已执行的环境步骤计数
        :param progress_bar: 是否显示进度条
        :return: self实例
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        