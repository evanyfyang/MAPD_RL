from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.optimize import linear_sum_assignment
import numpy as np
    
def create_mask(num, max_valid):
    batch_size = num.size(0)
    num = num.long()
    mask = torch.zeros(batch_size, int(max_valid), device=num.device)
    for i in range(batch_size):
        mask[i, :num[i]] = 1
    return mask

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

class GumbelSinkhorn(nn.Module):
    def __init__(self, tau=1.0, iterations=5, decay_rate=2e-4):
        super(GumbelSinkhorn, self).__init__()
        self.tau = tau
        self.iterations = iterations
        self.decay_rate = decay_rate

    def row_norm(self, x):
        """Unstable implementation"""
        #y = torch.matmul(torch.matmul(x, self.ones), torch.t(self.ones))
        #return torch.div(x, y)
        """Stable, log-scale implementation"""
        return x - logsumexp(x, dim=1, keepdim=True)

    def col_norm(self, x):
        """Unstable implementation"""
        #y = torch.matmul(torch.matmul(self.ones, torch.t(self.ones)), x)
        #return torch.div(x, y)
        """Stable, log-scale implementation"""
        return x - logsumexp(x, dim=0, keepdim=True)
    
    def hungarian_sampling(self, logits):
        cost_matrix = -logits.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind

    def forward(self, logits, free_agents_num, tasks_num, step=None, deterministic=False):
        # 初始化分布
        distribution = torch.zeros_like(logits, device=logits.device)

        agent_size = logits.shape[1]
    
        # 逐 Batch 计算
        batch_size, _, _ = logits.size()
        hungarian_action = np.zeros((batch_size, agent_size), dtype=np.int32)
        for b in range(batch_size):
            num_agents = free_agents_num[b].item()
            num_tasks = tasks_num[b].item()
            if num_agents > 0 and num_tasks > 0:
                logits_b = logits[b, :num_agents, :num_tasks] / self.tau  # 裁剪并缩放
                for _ in range(self.iterations):
                    logits_b = self.row_norm(logits_b)
                    logits_b = self.col_norm(logits_b)
                logits_b += 1e-6
                logits_b = torch.exp(logits_b)

                if deterministic:
                    sample = self.hungarian_sampling(logits_b)
                    hungarian_action[b, :num_agents] = sample
                distribution[b, :num_agents, :num_tasks] = logits_b

        if not deterministic:
            return distribution, _
        else:
            return distribution, torch.from_numpy(hungarian_action).to(logits.device).int()

class MAPDActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, tau=1.0, iterations=5, decay_rate=2e-4, max_agent_num=50, max_task=500, fix_div=False, not_div=False,  **kwargs):
        super(MAPDActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        hidden_size = self.features_dim

        self.step_sim = 0
        # self.max_agent_num = torch.LongTensor([50])
        # self.max_task = torch.LongTensor([500])
        self.max_agent_num = 50
        self.max_task = 500
        # Actor 部分
        self.agent_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.task_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.bmm_mlp = nn.Sequential(
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
        self.init_net()
        self.fix_div = fix_div
        self.not_div = not_div

    def init_net(self):
        def init_weights_orthogonal2(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def init_weights_orthogonal(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def init_weights_orthogonal0(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.agent_mlp.apply(init_weights_orthogonal2)
        self.task_mlp.apply(init_weights_orthogonal2)
        self.critic_mlp.apply(init_weights_orthogonal2)
        self.bmm_mlp.apply(init_weights_orthogonal0)
        self.critic_value_mlp.apply(init_weights_orthogonal)

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

    def random_sample(self, distribution, valid_mask, free_agent_num, tasks_num):
        batch_size, x, y = distribution.shape
        masked_distribution = valid_mask * distribution

        # Sample indices from the masked and normalized distribution
        samples = torch.zeros((batch_size, distribution.shape[1]), dtype=torch.long)
        for i in range(batch_size):
            # For each batch, sample based on the valid region
            for j in range(free_agent_num[i]):
                samples[i, j] = torch.multinomial(masked_distribution[i, j, :tasks_num[i]], 1).item()

        return samples

    def unpack_obs(self, obs, batch_size):
        hidden_size = self.features_dim
        batch_size = obs.shape[0]
        combined_feature = obs[:,:-3].reshape(batch_size, -1, hidden_size)
        free_agents_num = obs[:, -3]
        delivering_agents_num = obs[:, -2]
        tasks_num = obs[:, -1]
        return combined_feature, free_agents_num, delivering_agents_num, tasks_num

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        batch_size = features.shape[0]
        combined_feature, free_agents_num, delivering_agents_num, tasks_num = self.unpack_obs(features, batch_size)
        # 提取有效数量
        max_free_agents = self.max_agent_num
        max_delivering_agents = self.max_agent_num
        max_tasks = self.max_task

        free_agents_num = free_agents_num.long()
        delivering_agent_num = delivering_agents_num.long()
        tasks_num = tasks_num.long()

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

        if deterministic:
            # Gumbel-Sinkhorn and action
            distribution, hungarian_action = self.gumbel_sinkhorn(logits, free_agents_num, tasks_num, self.step_sim, deterministic)
        else:
            distribution, _ = self.gumbel_sinkhorn(logits, free_agents_num, tasks_num, self.step_sim, deterministic)

        valid_mask = torch.bmm(free_agents_mask.unsqueeze(2), tasks_mask.unsqueeze(1)) # (batch_size, max_free_agents, max_tasks)
        action = self.random_sample(distribution, valid_mask, free_agents_num, tasks_num)
        # action = self.greedy_action(distribution, free_agents_mask, tasks_mask)
        # assert action.max() <= distribution.shape[2] and action.max() != 196
        # print(distribution.shape)
        # print(distribution.sum(dim=-1))
        # print(action)

        # Critic
        critic_feature = self.critic_mlp(combined_feature)
        pooled_feature, _ = torch.max(critic_feature, dim=1)
        value = self.critic_value_mlp(pooled_feature)

        log_probs = torch.log(distribution + 1e-10)  # 避免 log(0)
        masked_log_probs = log_probs * valid_mask  # 屏蔽无效部分
        batch_size, x = action.shape
        indices = torch.arange(batch_size).unsqueeze(1)
        if deterministic:
            log_probs = masked_log_probs[indices, torch.arange(x).unsqueeze(0), hungarian_action]
            return hungarian_action, value, log_probs.sum(dim=-1)
        else:
            log_probs = masked_log_probs[indices, torch.arange(x).unsqueeze(0), action]
            return action, value, log_probs.sum(dim=-1)
          

    def evaluate_actions(self, obs, actions):
        """
        评估动作的 log 概率、熵以及 value。
        """
        # 提取最大有效数量
        features = self.extract_features(obs)
        batch_size = actions.shape[0]
        combined_feature, free_agents_num, delivering_agents_num, tasks_num = self.unpack_obs(features, batch_size)

        max_free_agents = self.max_agent_num
        max_delivering_agents = self.max_agent_num
        max_tasks = self.max_task

        free_agents_num = free_agents_num.long()
        delivering_agent_num = delivering_agents_num.long()
        tasks_num = tasks_num.long()

        # 分割特征
        free_agent_feature, delivering_agent_feature, task_feature = self.split_combined_feature(
            combined_feature, max_free_agents, max_delivering_agents, max_tasks
        )

        # 创建 mask
        free_agents_mask = create_mask(free_agents_num, max_free_agents)  # (batch_size, max_free_agents)
        tasks_mask = create_mask(tasks_num, max_tasks)  # (batch_size, max_tasks)

        # Pointer Network for logits
        free_agent_mlp = self.agent_mlp(free_agent_feature)  # (batch_size, max_free_agents, hidden_size)
        task_mlp = self.task_mlp(task_feature)  # (batch_size, max_tasks, hidden_size)
        logits = torch.bmm(free_agent_mlp, task_mlp.transpose(1, 2))  # (batch_size, max_free_agents, max_tasks)

        # Gumbel-Sinkhorn distribution
        distribution, _ = self.gumbel_sinkhorn(logits, free_agents_num, tasks_num, self.step_sim)
        self.step_sim += 1

        # Log probabilities with mask
        log_probs = torch.log(distribution + 1e-10)  # 避免 log(0)
        valid_mask = torch.bmm(free_agents_mask.unsqueeze(2), tasks_mask.unsqueeze(1))  # (batch_size, max_free_agents, max_tasks)
        masked_log_probs = log_probs * valid_mask  # 屏蔽无效部分
        batch_size, x = actions.shape
        indices = torch.arange(batch_size).unsqueeze(1)
        r_log_probs = masked_log_probs[indices, torch.arange(x).unsqueeze(0), actions.long()]
        # breakpoint()

        # Entropy with mask
        masked_distribution = distribution * valid_mask  # 屏蔽无效部分
        if not self.not_div:
            if not self.fix_div:
                entropy = -((masked_distribution * log_probs).sum(dim=-1))/((valid_mask+1e-6).sum(dim=-1))
            else:
                entropy = -((masked_distribution * log_probs).sum(dim=-1))/50
        else:
            entropy = -((masked_distribution * log_probs).sum(dim=-1))

        # Critic
        critic_feature = self.critic_mlp(combined_feature)  # (batch_size, max_free_agents + max_tasks, hidden_size)
        pooled_feature, _ = torch.max(critic_feature, dim=1)  # MaxPooling
        value = self.critic_value_mlp(pooled_feature)  # (batch_size, 1)

        print("Step: ", self.step_sim*5)
        
        if not self.not_div:
            if not self.fix_div:
                return value, (r_log_probs/((valid_mask+1e-6).sum(dim=-1))).sum(dim=-1), entropy.sum(dim=-1)
            else:
                return value, (r_log_probs/50).sum(dim=-1), entropy.sum(dim=-1)
        else:
            return value, r_log_probs.sum(dim=-1), entropy.sum(dim=-1)

        # return value, r_log_probs.sum(dim=-1), entropy

    def _build(self, lr_schedule):
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _predict(self, observation, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def predict_values(self, obs) -> torch.Tensor:
        _, values, _ = self.forward(obs)
        return values

