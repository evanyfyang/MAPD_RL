# 显存优化配置和建议
# Memory Optimization for GNN Policy

import torch
import torch.nn.functional as F
from typing import Dict, Any

class MemoryOptimizer:
    """GNN Policy显存优化器"""
    
    @staticmethod
    def get_optimized_config() -> Dict[str, Any]:
        """返回优化的配置参数"""
        return {
            # 1. 减少batch size
            'batch_size': 4,  # 从默认的8或16降低到4
            
            # 2. 减少网络规模
            'hidden_dim': 64,  # 从128降低到64
            'lower_gnn_num_layers': 2,  # 从3降低到2
            'higher_gnn_num_layers': 1,  # 从2降低到1
            'gnn_heads': 2,  # 从4降低到2
            
            # 3. 优化GNN类型选择
            'lower_gnn_type': 'gcn',  # GCN比GAT省显存
            'higher_gnn_type': 'self_attention_gat',  # 避免全连接
            
            # 4. Sinkhorn优化
            'use_sinkhorn': True,
            'use_real_domain': True,  # 预训练时使用real domain
            'iterations': 3,  # 从5降低到3
            'tau': 1.0,
            
            # 5. 限制最大实体数量
            'max_agents': 20,  # 从50降低到20
            'max_tasks': 50,   # 从500降低到50
            
            # 6. 梯度累积优化
            'gradient_accumulation_steps': 2,
            'use_gradient_checkpointing': True,
        }
    
    @staticmethod
    def optimize_edge_creation(edge_attr_list: list, device: torch.device, 
                             edge_attr_dim: int) -> torch.Tensor:
        """优化边属性创建，减少重复tensor分配"""
        if len(edge_attr_list) == 0:
            return torch.zeros((0, edge_attr_dim), device=device)
        
        # 预分配tensor而不是逐个append
        edge_attr_tensor = torch.zeros((len(edge_attr_list), edge_attr_dim), device=device)
        
        for i, attr in enumerate(edge_attr_list):
            if isinstance(attr, torch.Tensor):
                edge_attr_tensor[i] = attr
            else:
                edge_attr_tensor[i] = torch.tensor(attr, device=device)
        
        return edge_attr_tensor
    
    @staticmethod
    def optimize_sinkhorn_forward(sinkhorn_module, logits: torch.Tensor, 
                                free_agents_num: torch.Tensor, tasks_num: torch.Tensor,
                                max_agents: int = 20, max_tasks: int = 50):
        """优化Sinkhorn前向传播，减少显存占用"""
        batch_size = logits.size(0)
        device = logits.device
        
        # 预分配结果tensor
        distribution = torch.zeros((batch_size, max_agents, max_tasks + 1), 
                                 device=device, dtype=logits.dtype)
        
        for b in range(batch_size):
            num_agents = min(free_agents_num[b].item(), max_agents)
            num_tasks = min(tasks_num[b].item(), max_tasks)
            
            if num_agents > 0 and num_tasks > 0:
                # 只处理实际需要的部分，避免创建过大的矩阵
                current_logits = logits[b, :num_agents, :num_tasks]
                
                # 使用inplace操作减少内存分配
                with torch.no_grad():
                    result = sinkhorn_module.sinkhorn_real(current_logits, num_agents, num_tasks)
                    distribution[b, :num_agents, :num_tasks+1] = result
        
        return distribution
    
    @staticmethod
    def clear_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def setup_memory_efficient_training():
        """设置内存高效的训练环境"""
        # 启用内存高效的注意力机制
        torch.backends.cuda.enable_flash_sdp(True)
        
        # 设置内存映射
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 启用混合精度训练
        return torch.cuda.amp.GradScaler()

# 具体优化建议
OPTIMIZATION_TIPS = """
🚀 GNN Policy 显存优化建议：

1. **立即优化 (Emergency)**:
   - 将batch_size降到4或更小
   - 使用梯度累积: gradient_accumulation_steps=2
   - 定期调用torch.cuda.empty_cache()

2. **网络结构优化**:
   - hidden_dim: 128 → 64
   - lower_gnn_num_layers: 3 → 2  
   - higher_gnn_num_layers: 2 → 1
   - 使用'self_attention_gat'避免全连接图

3. **Sinkhorn算法优化**:
   - iterations: 5 → 3
   - 预训练时use_real_domain=True
   - 避免在训练时添加Gumbel噪声

4. **图构建优化**:
   - 限制free_agents_nearest_tasks数量
   - 使用sparse tensor存储edge_attr
   - 预分配tensor而不是动态append

5. **训练策略优化**:
   - 使用混合精度训练 (amp)
   - 启用gradient checkpointing
   - 分段训练大batch

6. **监控显存使用**:
   ```python
   import torch
   print(f"显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
   print(f"显存缓存: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
   ```

7. **紧急情况**:
   如果仍然OOM，考虑：
   - 单样本训练 (batch_size=1)
   - 使用CPU进行部分计算
   - 减少最大agent/task数量
"""

def apply_emergency_fixes():
    """应用紧急修复，立即减少显存占用"""
    print("🚨 应用紧急显存优化...")
    
    # 清理缓存
    MemoryOptimizer.clear_cache()
    
    # 返回紧急配置
    emergency_config = {
        'batch_size': 2,  # 极小的batch size
        'hidden_dim': 32,  # 极小的hidden dimension
        'lower_gnn_num_layers': 1,
        'higher_gnn_num_layers': 1,
        'max_agents': 10,
        'max_tasks': 20,
        'use_gradient_checkpointing': True,
    }
    
    print("✅ 紧急配置已生成")
    return emergency_config

if __name__ == "__main__":
    print(OPTIMIZATION_TIPS)
    print("\n" + "="*50)
    emergency_config = apply_emergency_fixes()
    print(f"紧急配置: {emergency_config}") 