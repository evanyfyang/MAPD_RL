from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.optimize import linear_sum_assignment
import numpy as np
import pygmtools as pygm
from pygmtools.linear_solvers import hungarian, sinkhorn
from model.logit_cnn import LogitCNN

pygm.set_backend('pytorch')
    
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

class Sinkhorn(nn.Module):
    def __init__(self, tau=1.0, iterations=5):
        super(Sinkhorn, self).__init__()
        self.tau = tau
        self.iterations = iterations

    def sinkhorn_log(self, matrix, iterations=20, eps=1e-8):
        row_len = matrix.shape[0] 
        col_len = matrix.shape[1] 
        desired_row_sums = torch.ones(row_len, requires_grad=False).cuda()
        desired_col_sums = torch.ones(col_len, requires_grad=False).cuda()
        desired_row_sums[-1] = col_len-1
        desired_col_sums[-1] = row_len-1

        log_M = torch.log(matrix + eps)
        for _ in range(iterations):
            # 行归一化
            row_log_sum = torch.logsumexp(log_M, dim=1, keepdim=True)
            log_M = log_M + (torch.log(desired_row_sums).unsqueeze(1) - row_log_sum)
            # 列归一化
            col_log_sum = torch.logsumexp(log_M, dim=0, keepdim=True)
            log_M = log_M + (torch.log(desired_col_sums).unsqueeze(0) - col_log_sum)
        return torch.exp(log_M)

    def forward(self, logits, free_agents_num, tasks_num):
        # 初始化分布
        distribution = torch.zeros_like(logits, device=logits.device)
 
        batch_size, _, _ = logits.size()
        for b in range(batch_size):
            num_agents = free_agents_num[b].item() + 1
            num_tasks = tasks_num[b].item() + 1
            if num_agents > 0 and num_tasks > 0:
                if self.tau > 0:
                    logits_b = logits[b, :num_agents, :num_tasks] / self.tau
                else:
                    logits_b = logits[b, :num_agents, :num_tasks]

                distribution[b, :num_agents, :num_tasks] = self.sinkhorn_log(logits_b, iterations=self.iterations)
        return distribution


class MAPDActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, 
                 tau=0.1, iterations=20, decay_rate=2e-4, 
                 max_agent_num=50, max_task=500, fix_div=False, 
                 not_div=False, pretrain_steps=10000, empty_type="fixed", cal_type='bmm', **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # 添加新参数
        self.pretrain_steps = pretrain_steps
        self.current_step = 0  # 跟踪总训练步数
        self.tau = tau
        self.iterations = iterations
        self.cal_type = cal_type
        self.pretrain_mode = False

        hidden_size = self.features_dim

        self.step_sim = 0
        # self.max_agent_num = torch.LongTensor([50])
        # self.max_task = torch.LongTensor([500])
        self.max_agent_num = max_agent_num
        self.max_task = max_task

        self.agent_task_mlp = nn.Linear(2*hidden_size, hidden_size)
        self.logit_mlp = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

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

        self.logit_cnn = LogitCNN(self.features_dim)

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

        self.sinkhorn = Sinkhorn(tau, iterations)
        self.init_net()
        self.fix_div = fix_div
        self.not_div = not_div

        self.empty_type = empty_type

        if self.empty_type == "learnable":
            self.empty_score = nn.Parameter(torch.tensor(1.))
        elif self.empty_type == "fixed":
            self.empty_score = torch.exp(torch.tensor(-1.0))

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
        free_agent_feature = combined_feature[:, :max_free_agents, :]
        delivering_agent_feature = combined_feature[:, max_free_agents:max_free_agents + max_delivering_agents, :]
        task_feature = combined_feature[:, max_free_agents + max_delivering_agents:max_free_agents + max_delivering_agents + max_tasks, :]

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
        samples = torch.zeros((batch_size, self.max_agent_num), device=distribution.device, dtype=torch.long)
        for i in range(batch_size):
            # For each batch, sample based on the valid region
            for j in range(free_agent_num[i]):
                samples[i, j] = torch.multinomial(masked_distribution[i, j, :tasks_num[i]], 1).item()

        return samples
    
    def random_sample_distinct(self, distribution, valid_mask, free_agent_num, tasks_num):
        batch_size, x, y = distribution.shape

        samples = torch.zeros((batch_size, self.max_agent_num), device=distribution.device, dtype=torch.long)

        for i in range(batch_size):
            used_tasks = set()

            for j in range(free_agent_num[i]):
                probs = distribution[i, j, :tasks_num[i]+1].clone()
                for t in used_tasks:
                    probs[t] = 0
                if probs.sum() == 0:
                    chosen = tasks_num[i].item()
                else:
                    chosen = torch.multinomial(probs, 1).item()
                samples[i,j] = chosen
                if chosen != tasks_num[i]:
                    used_tasks.add(chosen)
        return samples

    def unpack_obs(self, obs, batch_size):
        hidden_size = self.features_dim
        batch_size = obs.shape[0]
        # 使用self.max_agent_num代替硬编码的50
        total_offset = 4 + self.max_agent_num  # 3个统计量 + 专家动作维度
        combined_feature = obs[:, :-total_offset].reshape(batch_size, -1, hidden_size)
        free_agents_num = obs[:, -total_offset]  # 自由智能体数量在倒数第(total_offset)位
        delivering_agents_num = obs[:, -(total_offset-1)]  # 运输中智能体数量在倒数第(total_offset-1)位
        tasks_num = obs[:, -(total_offset-2)]  # 任务数量在倒数第(total_offset-2)位
        expert_actions = obs[:, -(self.max_agent_num+1):].long()  # 最后max_agent_num维是专家动作
        return combined_feature, free_agents_num, delivering_agents_num, tasks_num, expert_actions

    def sample_hungarian_action(self, hungarian_matrix, free_agents_num, tasks_num):
        batch_size = hungarian_matrix.shape[0]
        hungarian_action = torch.zeros([batch_size, self.max_agent_num], device=hungarian_matrix.device, dtype=torch.long)
        for i in range(batch_size):
            for j in range(free_agents_num[i]):
                if hungarian_matrix[i,j].sum() == 1:
                    hungarian_action[i, j] = torch.argmax(hungarian_matrix[i, j])
                else:
                    hungarian_action[i, j] = tasks_num[i]
        return hungarian_action

    def add_empty_scores(self, logits, free_agents_num, tasks_num):
        batch_size, max_agents, max_tasks = logits.size()
        
        new_logits = torch.full((batch_size, max_agents + 1, max_tasks + 1), 
                            self.empty_score, 
                            device=logits.device)
        
        new_logits[:, :max_agents, :max_tasks] = logits

        return new_logits

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        batch_size = features.shape[0]
        combined_feature, free_agents_num, delivering_agents_num, tasks_num, expert_actions = self.unpack_obs(features, batch_size)
        
        max_free_agents = free_agents_num.max().long().item()
        max_delivering_agents = delivering_agents_num.max().long().item()
        max_tasks = tasks_num.max().long().item()

        free_agents_num = free_agents_num.long()
        delivering_agent_num = delivering_agents_num.long()
        tasks_num = tasks_num.long()

        free_agent_feature, delivering_agent_feature, task_feature = self.split_combined_feature(
            combined_feature, max_free_agents, max_delivering_agents, max_tasks
        )

        free_agents_mask = create_mask(free_agents_num, max_free_agents+1)
        tasks_mask = create_mask(tasks_num+1, max_tasks+1)
        valid_mask = torch.bmm(free_agents_mask.unsqueeze(2), tasks_mask.unsqueeze(1))
        
        free_agent_mlp = self.agent_mlp(free_agent_feature)
        task_mlp = self.task_mlp(task_feature)

        if self.cal_type == 'bmm':
            logits = torch.bmm(free_agent_mlp, task_mlp.transpose(1, 2))
        else:
            agent_bd, task_bd = torch.broadcast_tensors(free_agent_mlp.unsqueeze(2), task_mlp.unsqueeze(1))
            logits = self.agent_task_mlp(torch.cat([agent_bd, task_bd], dim=-1))
            logits = self.logit_cnn(logits)
            logits = self.logit_mlp(logits).squeeze(-1)
        # logits = self.add_empty_scores(logits/np.sqrt(self.features_dim), free_agents_num, tasks_num)
        logits = self.add_empty_scores(logits, free_agents_num, tasks_num)

        # distribution = torch.softmax(logits, dim=-1)
        distribution = self.sinkhorn(logits, free_agents_num, tasks_num)
        batch_indices = torch.arange(batch_size)

        if deterministic:
            if self.empty_type == "learnable":
                unmatch_task = distribution[batch_indices, free_agents_num, :max_tasks+1]
                unmatch_agent = distribution[batch_indices, :free_agents_num+1, tasks_num]
                hungarian_matrix = hungarian(logits, n1=free_agents_num, n2=tasks_num, unmatch1=unmatch_agent, unmatch2=unmatch_task)
            else:
                hungarian_matrix = hungarian(logits, n1=free_agents_num, n2=tasks_num)
            hungarian_action = self.sample_hungarian_action(hungarian_matrix, free_agents_num, tasks_num)

        log_probs = torch.log(distribution + 1e-10)
        masked_log_probs = log_probs * valid_mask

        if deterministic == False and self.current_step < self.pretrain_steps or self.pretrain_mode:
            action = expert_actions.clone()
        else:
            if deterministic:
                action = hungarian_action
                # action = expert_actions.clone()
            else:
                action = self.random_sample_distinct(distribution, valid_mask, free_agents_num, tasks_num)
        
        critic_feature = self.critic_mlp(combined_feature)
        pooled_feature, _ = torch.max(critic_feature, dim=1)
        value = self.critic_value_mlp(pooled_feature)
        
        batch_indices = torch.arange(batch_size).unsqueeze(1)
        selected_log_probs = masked_log_probs[batch_indices, torch.arange(masked_log_probs.size(1)).unsqueeze(0), torch.clamp(action.long(), 0)[:,:masked_log_probs.size(1)]]
        
        return action[:,:self.max_agent_num], value, selected_log_probs.sum(dim=-1)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        batch_size = actions.shape[0]
        combined_feature, free_agents_num, delivering_agents_num, tasks_num, expert_actions = self.unpack_obs(features, batch_size)

        max_free_agents = free_agents_num.max().long().item()
        max_delivering_agents = delivering_agents_num.max().long().item()
        max_tasks = tasks_num.max().long().item()

        free_agents_num = free_agents_num.long()
        delivering_agent_num = delivering_agents_num.long()
        tasks_num = tasks_num.long()

        free_agent_feature, delivering_agent_feature, task_feature = self.split_combined_feature(
            combined_feature, max_free_agents, max_delivering_agents, max_tasks
        )

        free_agents_mask = create_mask(free_agents_num, max_free_agents+1)  # (batch_size, max_free_agents)
        tasks_mask = create_mask(tasks_num+1, max_tasks+1)  # (batch_size, max_tasks)
        valid_mask = torch.bmm(free_agents_mask.unsqueeze(2), tasks_mask.unsqueeze(1))

        # Pointer Network for logits
        free_agent_mlp = self.agent_mlp(free_agent_feature)  # (batch_size, max_free_agents, hidden_size)
        task_mlp = self.task_mlp(task_feature)  # (batch_size, max_tasks, hidden_size)

        if self.cal_type == 'bmm':
            logits = torch.bmm(free_agent_mlp, task_mlp.transpose(1, 2))
        else:
            agent_bd, task_bd = torch.broadcast_tensors(free_agent_mlp.unsqueeze(2), task_mlp.unsqueeze(1))
            logits = self.agent_task_mlp(torch.cat([agent_bd, task_bd], dim=-1))
            logits = self.logit_cnn(logits)
            logits = self.logit_mlp(logits).squeeze(-1)
        # logits = torch.bmm(free_agent_mlp, task_mlp.transpose(1, 2))  # (batch_size, max_free_agents, max_tasks)
        logits = self.add_empty_scores(logits, free_agents_num, tasks_num)
        # logits = self.add_empty_scores(logits/np.sqrt(self.features_dim), free_agents_num, tasks_num)

        # distribution = torch.softmax(logits, dim=-1)
        distribution = self.sinkhorn(logits, free_agents_num, tasks_num)

        # Log probabilities with mask
        log_probs = torch.log(distribution + 1e-10)  
        masked_log_probs = log_probs * valid_mask  

        # Entropy with mask
        masked_distribution = distribution * valid_mask  
        if not self.not_div:
            if not self.fix_div:
                entropy = -((masked_distribution * log_probs).sum(dim=-1))/((valid_mask+1e-6).sum(dim=-1))
            else:
                entropy = -((masked_distribution * log_probs).sum(dim=-1))/self.max_agent_num
        else:
            entropy = -((masked_distribution * log_probs).sum(dim=-1))

        # Critic
        critic_feature = self.critic_mlp(combined_feature)  # (batch_size, max_free_agents + max_tasks, hidden_size)
        pooled_feature, _ = torch.max(critic_feature, dim=1)  # MaxPooling
        value = self.critic_value_mlp(pooled_feature)  # (batch_size, 1)
        
        print("Step: ", self.step_sim)
        
        if self.current_step < self.pretrain_steps or self.pretrain_mode:
            expert_actions_one_hot = F.one_hot(expert_actions[:, :masked_distribution.shape[1]].long().clamp(min=0), num_classes=distribution.size(2)).float()
            # some values are out of the range [0, 1] due to the numerical error
            masked_distribution = torch.clamp(masked_distribution, min=0, max=1)
            assert torch.all((masked_distribution >= 0) & (masked_distribution <= 1)), "Some values are out of the range [0, 1]"
            bce_loss_all = F.binary_cross_entropy(masked_distribution, expert_actions_one_hot, reduction='none')
            weight = expert_actions_one_hot * 10 + (1 - expert_actions_one_hot) * 1

            # expert_actions_one_hot_clone = expert_actions_one_hot.clone()
            # expert_actions_one_hot_clone[:, :, -1] = 0
            # weight = expert_actions_one_hot_clone * 1 + (1 - expert_actions_one_hot_clone) * 0
            policy_loss = (bce_loss_all * valid_mask * weight).sum(dim=-1)
            if not self.not_div:
                if not self.fix_div:
                    policy_loss = policy_loss / (free_agents_num * (tasks_num + 1))[:, None]
                    # policy_loss = policy_loss / torch.min(free_agents_num, (tasks_num + 1))[:, None]
                else:
                    policy_loss = policy_loss / self.max_agent_num
        else:
            batch_indices = torch.arange(batch_size).unsqueeze(1)
            selected_log_probs = masked_log_probs[batch_indices, torch.arange(masked_log_probs.size(1)).unsqueeze(0), torch.clamp(actions.long(), 0)[:,:masked_log_probs.size(1)]]
            policy_loss = -(selected_log_probs.sum(dim=-1))  # (batch_size,)
            
            if not self.not_div:
                if not self.fix_div:
                    policy_loss = policy_loss / free_agents_num[:, None, None]
                else:
                    policy_loss = policy_loss / self.max_agent_num

        self.step_sim += 1 * distribution.size(0)
        return value, policy_loss, entropy

    def _build(self, lr_schedule):
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _predict(self, observation, deterministic: bool = True) -> torch.Tensor:
        actions, _, _ = self.forward(observation, True)
        return actions

    def predict_values(self, obs) -> torch.Tensor:
        return torch.tensor(0, device=obs["free_agents_num"].device)

