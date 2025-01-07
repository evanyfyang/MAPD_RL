from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
    
def create_mask(num, max_valid):
    batch_size = num.size(0)
    mask = torch.zeros(batch_size, max_valid, device=num.device)
    for i in range(batch_size):
        mask[i, :num[i]] = 1
    return mask


class GumbelSinkhorn(nn.Module):
    def __init__(self, tau=1.0, iterations=5, decay_rate=2e-4):
        super(GumbelSinkhorn, self).__init__()
        self.tau = tau
        self.iterations = iterations
        self.decay_rate = decay_rate

    def forward(self, logits, free_agents_num, tasks_num, step=None):
        # 动态调整 tau
        tau = self.tau * torch.exp(-self.decay_rate * step) if step is not None else self.tau

        # 初始化分布
        distribution = torch.zeros_like(logits)

        # 逐 Batch 计算
        batch_size, _, _ = logits.size()
        for b in range(batch_size):
            num_agents = free_agents_num[b].item()
            num_tasks = tasks_num[b].item()
            if num_agents > 0 and num_tasks > 0:
                logits_b = logits[b, :num_agents, :num_tasks] / tau  # 裁剪并缩放
                for _ in range(self.iterations):
                    logits_b = F.softmax(logits_b, dim=1)
                    logits_b = F.softmax(logits_b, dim=0)
                distribution[b, :num_agents, :num_tasks] = logits_b

        return distribution


class MAPDActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, tau=1.0, iterations=5, decay_rate=2e-4, **kwargs):
        super(MAPDActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        hidden_size = self.features_dim

        self.step_sim = 0

        # Actor 部分
        self.agent_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.task_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Critic 部分
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.critic_value_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.gumbel_sinkhorn = GumbelSinkhorn(tau, iterations, decay_rate)

    def split_combined_feature(self, combined_feature, max_free_agents, max_delivering_agents, max_tasks):
        batch_size, _, hidden_size = combined_feature.size()

        free_agent_feature = combined_feature[:, :max_free_agents, :]
        delivering_agent_feature = combined_feature[:, max_free_agents:max_free_agents + max_delivering_agents, :]
        task_feature = combined_feature[:, max_free_agents + max_delivering_agents:, :]

        return free_agent_feature, delivering_agent_feature, task_feature

    def greedy_action(self, distribution, free_agents_num, tasks_num):
        """
        使用贪心策略生成动作，确保每个任务最多分配给一个 Agent。
        """
        batch_size = distribution.size(0)
        action = torch.zeros_like(distribution)
        for b in range(batch_size):
            num_agents = free_agents_num[b].item()
            num_tasks = tasks_num[b].item()
            dist = distribution[b, :num_agents, :num_tasks]  # (num_agents, num_tasks)
            for task_idx in range(num_tasks):
                agent_idx = torch.argmax(dist[:, task_idx])  # 找到分配概率最高的 agent
                action[b, agent_idx, task_idx] = 1  # 选择动作
                dist[:, task_idx] = -1  # 确保每个任务最多分配一个 agent
        return action

    def forward(self, combined_feature, free_agents_num, delivering_agents_num, tasks_num):
        # 提取有效数量
        max_free_agents = free_agents_num.max().item()
        max_delivering_agents = delivering_agents_num.max().item()
        max_tasks = tasks_num.max().item()

        # 分割特征
        free_agent_feature, delivering_agent_feature, task_feature = self.split_combined_feature(
            combined_feature, max_free_agents, max_delivering_agents, max_tasks
        )

        # 创建 mask
        free_agents_mask = create_mask(free_agents_num, max_free_agents)
        tasks_mask = create_mask(tasks_num, max_tasks)

        # Pointer Network for logits
        free_agent_mlp = self.agent_mlp(free_agent_feature)  # (batch_size, max_free_agents, hidden_size)
        task_mlp = self.task_mlp(task_feature)  # (batch_size, max_tasks, hidden_size)
        logits = torch.bmm(free_agent_mlp, task_mlp.transpose(1, 2))  # 点积计算 logits 矩阵

        # Gumbel-Sinkhorn and action
        distribution = self.gumbel_sinkhorn(logits, free_agents_num, tasks_num, self.step_sim)
        action = self.greedy_action(distribution, free_agents_mask, tasks_mask)

        # Critic
        critic_feature = self.critic_mlp(combined_feature)
        pooled_feature, _ = torch.max(critic_feature, dim=1)
        value = self.critic_value_mlp(pooled_feature)

        return distribution, action, value

    def evaluate_actions(self, combined_feature, actions, free_agents_num, delivering_agents_num, tasks_num):
        """
        评估动作的 log 概率、熵以及 value。
        """
        # 提取最大有效数量
        max_free_agents = free_agents_num.max().item()
        max_tasks = tasks_num.max().item()

        # 分割特征
        free_agent_feature, delivering_agent_feature, task_feature = self.split_combined_feature(
            combined_feature, max_free_agents, delivering_agents_num.max().item(), max_tasks
        )

        # 创建 mask
        free_agents_mask = create_mask(free_agents_num, max_free_agents)  # (batch_size, max_free_agents)
        tasks_mask = create_mask(tasks_num, max_tasks)  # (batch_size, max_tasks)

        # Pointer Network for logits
        free_agent_mlp = self.agent_mlp(free_agent_feature)  # (batch_size, max_free_agents, hidden_size)
        task_mlp = self.task_mlp(task_feature)  # (batch_size, max_tasks, hidden_size)
        logits = torch.bmm(free_agent_mlp, task_mlp.transpose(1, 2))  # (batch_size, max_free_agents, max_tasks)

        # Gumbel-Sinkhorn distribution
        distribution = self.gumbel_sinkhorn(logits, free_agents_num, tasks_num, self.step_sim)
        self.step_sim += 1

        # Log probabilities with mask
        log_probs = torch.log(distribution + 1e-10)  # 避免 log(0)
        valid_mask = torch.bmm(free_agents_mask.unsqueeze(2), tasks_mask.unsqueeze(1))  # (batch_size, max_free_agents, max_tasks)
        masked_log_probs = log_probs * valid_mask  # 屏蔽无效部分
        log_probs_action = torch.sum(masked_log_probs * actions, dim=[1, 2])  # 根据动作计算 log 概率

        # Entropy with mask
        masked_distribution = distribution * valid_mask  # 屏蔽无效部分
        entropy = -torch.sum(masked_distribution * log_probs, dim=[1, 2])

        # Critic
        critic_feature = self.critic_mlp(combined_feature)  # (batch_size, max_free_agents + max_tasks, hidden_size)
        pooled_feature, _ = torch.max(critic_feature, dim=1)  # MaxPooling
        value = self.critic_value_mlp(pooled_feature)  # (batch_size, 1)

        return log_probs_action, entropy, value
