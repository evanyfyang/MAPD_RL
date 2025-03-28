import torch
import numpy as np

tau = 0.05
pad_num = -1e6
def test_dummy_processing():
    # 测试配置
    test_cases = [
        (30, 50),   # 行 < 列
        (50, 30),   # 行 > 列
        (300, 300)    # 行 = 列
    ]
    iterations = 10

    for agent_num, task_num in test_cases:
        print(f"\n=== 测试用例: {agent_num} agents x {task_num} tasks ===")
        
        # 生成模拟logits（注意：有效区域为agent_num x task_num）
        np.random.seed(41)
        original_logits = torch.tensor(np.random.randn(agent_num, task_num), dtype=torch.float32)
        # print(original_logits)
        
        # 处理dummy节点
        processed_probs = process_dummy_rows(original_logits, agent_num, task_num, iterations)
        
        # 验证1: 输出维度恢复
        # assert processed_probs.shape == (agent_num, task_num), \
        #     f"维度错误！期望:{agent_num}x{task_num}, 实际:{processed_probs.shape}"
        
        # 验证2: 有效区域概率分布
        row_sums = processed_probs.sum(dim=1)
        col_sums = processed_probs.sum(dim=0)
        print(f"行和最大偏差: {torch.abs(row_sums - 1).max().item():.4f}")
        print(f"列和最大偏差: {torch.abs(col_sums - 1).max().item():.4f}")
        # print(processed_probs)
        print(row_sums)
        print(col_sums)
        
        # 验证3: dummy行抑制效果（当agent_num < max_dim时）
        max_dim = max(agent_num, task_num)
        if agent_num < max_dim:
            dummy_probs = process_dummy_rows(original_logits, agent_num, task_num, iterations, return_full=True)[agent_num:]
            print(f"Dummy行最大概率值: {dummy_probs.max().item():.6f}")
            # assert dummy_probs.max() < 1e-4, "Dummy行影响过大！"

        # 验证4: 概率值合理性
        assert (processed_probs >= 0).all() and (processed_probs <= 1).all(), "存在非法概率值！"
        print("测试通过！")

def process_dummy_rows(logits, agent_num, task_num, iterations, return_full=True):
    """扩展后的处理函数"""
    max_dim = max(agent_num, task_num)
    device = logits.device
    
    if logits.size(0) < max_dim:
        padded_rows = torch.full((max_dim - agent_num, task_num), pad_num, device=device)
        padded = torch.cat([logits, padded_rows], dim=0)
    else:
        padded = logits.clone()
    
    # 列填充（当task_num不足时）
    if padded.size(1) < max_dim:
        padded_cols = torch.full((max_dim, max_dim - task_num), pad_num, device=device)
        padded = torch.cat([padded, padded_cols], dim=1)
    
    padded = padded/tau
    # Sinkhorn迭代
    for _ in range(iterations):
        padded = col_normalize(padded)
        padded = row_normalize(padded)
        
        
    
    probs = torch.exp(padded)
    return probs if return_full else probs[:agent_num, :task_num]

def row_normalize(x):
    return x - x.logsumexp(dim=1, keepdim=True)

def col_normalize(x):
    return x - x.logsumexp(dim=0, keepdim=True)

if __name__ == "__main__":
    test_dummy_processing()