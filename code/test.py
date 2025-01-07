import torch
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sinkhorn_with_row_mask(log_alpha, mask, n_iters=20, tau=1.0):
    # 应用掩码，通过将不允许的位置设置为 -inf
    log_alpha = log_alpha + (mask + 1e-9).log()
    
    # 添加 Gumbel 噪声
    gumbel_noise = sample_gumbel(log_alpha.shape)
    log_alpha = log_alpha + gumbel_noise
    
    # 标记哪些行被完全掩盖
    row_valid = mask.sum(dim=2) > 0
    
    # Sinkhorn 归一化
    for _ in range(n_iters):
        # 行归一化，仅对有效行进行
        log_alpha = log_alpha.masked_fill(~row_valid.unsqueeze(-1), -float('inf'))
        log_alpha = F.log_softmax(log_alpha, dim=2)
        
        # 列归一化
        log_alpha = F.log_softmax(log_alpha, dim=1)
    
    # 对于被掩盖的行，保持为零矩阵
    log_alpha = log_alpha.masked_fill(~row_valid.unsqueeze(-1), 0.0)
    
    return torch.exp(log_alpha)

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
