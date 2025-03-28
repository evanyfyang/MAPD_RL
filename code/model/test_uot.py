import torch

def sinkhorn_sum(matrix, desired_row_sums, desired_col_sums, iterations=20, eps=1e-8):
    """直接使用sum进行行、列归一化"""
    M = matrix.clone()
    for _ in range(iterations):
        # 行归一化
        actual_row_sum = M.sum(dim=1, keepdim=True)
        M = M * (desired_row_sums.unsqueeze(1) / (actual_row_sum + eps))
        # 列归一化
        actual_col_sum = M.sum(dim=0, keepdim=True)
        M = M * (desired_col_sums.unsqueeze(0) / (actual_col_sum + eps))
    return M

def sinkhorn_log(matrix, iterations=20, eps=1e-8):
    """在对数域中使用logsumexp进行归一化"""
    # 转换到对数域
    row_len = matrix.size(0)
    col_len = matrix.size(1)
    desired_row_sums = torch.ones(row_len)
    desired_col_sums = torch.ones(col_len)
    desired_row_sums[-1] = col_len - 1
    desired_col_sums[-1] = row_len - 1
    
    log_M = torch.log(matrix + eps)
    for _ in range(iterations):
        # 行归一化
        row_log_sum = torch.logsumexp(log_M, dim=1, keepdim=True)
        log_M = log_M + (torch.log(desired_row_sums).unsqueeze(1) - row_log_sum)
        # 列归一化
        col_log_sum = torch.logsumexp(log_M, dim=0, keepdim=True)
        log_M = log_M + (torch.log(desired_col_sums).unsqueeze(0) - col_log_sum)
    return torch.exp(log_M)

if __name__ == '__main__':
    # 设定随机种子
    torch.manual_seed(0)
    
    # 构造一个随机正值矩阵（例如5行6列）
    matrix = torch.rand(5, 6)
    print("原始矩阵:")
    print(matrix)

    # 定义desired目标：
    # 默认目标是1，但最后一行设置为 col_len-1，最后一列设置为 row_len-1，用于模拟 slack 的情况
    row_len = matrix.size(0)
    col_len = matrix.size(1)
    desired_row_sums = torch.ones(row_len)
    desired_col_sums = torch.ones(col_len)
    desired_row_sums[-1] = col_len - 1
    desired_col_sums[-1] = row_len - 1

    iterations = 20

    # 分别使用两种方法计算 Sinkhorn 归一化结果
    result_sum = sinkhorn_sum(matrix, desired_row_sums, desired_col_sums, iterations)
    result_log = sinkhorn_log(matrix, iterations)

    print("\n使用直接sum归一化得到的结果:")
    print(result_sum)
    print("\n使用logsumexp归一化得到的结果:")
    print(result_log)
    print("\n两种方法结果的差异 (绝对值):")
    print(torch.abs(result_sum - result_log))
