import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, max_agents, m_tasks, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        self.max_agents = max_agents
        self.m_tasks = m_tasks
        self.no_action = self.m_tasks  # “无动作”索引
        
        # Actor 网络：输出所有智能体的任务 logits（包括“无动作”）
        self.actor_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_agents * (m_tasks + 1))
        )
        
        # Critic 网络：为每个智能体生成单独的价值
        self.critic_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_agents)
        )

    def forward(self, obs, deterministic=False):
        # 特征提取
        latent_pi, latent_v = self.extract_features(obs)
        
        # Actor 部分：生成所有智能体的 logits
        logits = self.actor_net(latent_pi)  # [batch_size, max_agents * (m_tasks + 1)]
        logits = logits.view(-1, self.max_agents, self.m_tasks + 1)  # [batch_size, max_agents, m_tasks + 1]
        
        # Critic 部分：生成每个智能体的价值
        values = self.critic_net(latent_v)  # [batch_size, max_agents]
        
        # 创建分布对象并采样动作
        distributions = []
        actions = []
        for agent in range(self.max_agents):
            agent_logits = logits[:, agent, :]  # [batch_size, m_tasks + 1]
            distribution = Categorical(logits=agent_logits)
            distributions.append(distribution)
            if deterministic:
                action = distribution.probs.argmax(dim=1)
            else:
                action = distribution.sample()
            actions.append(action)
        
        # 合并所有智能体的动作
        actions = torch.stack(actions, dim=1)  # [batch_size, max_agents]
        
        return actions, distributions, values

    def evaluate_actions(self, obs, actions, mask):
        # 特征提取
        latent_pi, latent_v = self.extract_features(obs)
        
        # Actor 网络：生成所有智能体的 logits
        logits = self.actor_net(latent_pi).view(-1, self.max_agents, self.m_tasks + 1)  # [batch_size, max_agents, m_tasks + 1]
        
        # Critic 网络：生成每个智能体的价值
        values = self.critic_net(latent_v)  # [batch_size, max_agents]
        
        log_probs = []
        entropies = []
        for agent in range(self.max_agents):
            agent_logits = logits[:, agent, :]
            distribution = Categorical(logits=agent_logits)
            log_prob = distribution.log_prob(actions[:, agent])
            entropy = distribution.entropy()
            # 仅对有效智能体计算
            log_prob = log_prob * mask[:, agent]
            entropy = entropy * mask[:, agent]
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        # 合并 log_probs 和熵
        log_prob = torch.stack(log_probs, dim=1).sum(dim=1)  # [batch_size]
        entropy = torch.stack(entropies, dim=1).sum(dim=1)  # [batch_size]
        
        return values, log_prob, entropy
