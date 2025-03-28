from stable_baselines3 import A2C
from typing import Any, ClassVar, Optional, TypeVar, Union
import sys
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance


class A2CMAPD(A2C):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # 正常收集数据
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            # 评估动作
            values, loss_ce, entropy = self.policy.evaluate_actions(
                rollout_data.observations, 
                actions
            )
            values = values.flatten()

            if self.policy.current_step < self.policy.pretrain_steps:
                policy_loss = loss_ce.mean()
                value_loss = F.mse_loss(rollout_data.returns, values)
                loss = policy_loss
                print("Pretraining Loss: ", policy_loss)
            else:
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                policy_loss = -(advantages * loss_ce).mean()
                value_loss = F.mse_loss(rollout_data.returns, values)
                loss = policy_loss + self.ent_coef * entropy.mean() + self.vf_coef * value_loss
                print("Loss: ", loss, "policy_loss:", policy_loss, "entropy_loss", entropy.mean(), "value_loss",value_loss)

            
            sys.stdout.flush()
            # breakpoint()
            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # 更新全局步数
            self.policy.current_step += self.n_envs

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy.mean().item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())