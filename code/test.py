import torch
import torch.nn.functional as F
import torch.nn as nn

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

# 示例
batch_size = 2
n = 3
# 输入矩阵 logits
logits = torch.randn(batch_size, n, n)
# 掩码矩阵，1 表示允许，0 表示禁止
mask = torch.tensor([
    [[1, 1, 0],
     [1, 1, 1],
     [0, 1, 1]],
    [[1, 0, 1],
     [0, 1, 1],
     [1, 1, 0]]
], dtype=torch.float32)

# 为一整行添加掩码，例如第一行
mask[:, 0, :] = 0  # 将第一行全部禁止

# 生成 Gumbel-Sinkhorn 矩阵
gumbel_sinkhorn_matrix = gumbel_sinkhorn_with_row_mask(logits, mask, n_iters=20, tau=1.0)
print(gumbel_sinkhorn_matrix)
