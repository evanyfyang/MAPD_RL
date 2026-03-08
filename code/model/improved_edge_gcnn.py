import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class ImprovedEdgeGCNN(MessagePassing):
    """
    改进的EdgeGCNN实现，专门用于解决分配图匹配问题
    
    主要改进：
    1. 真正的边特征更新机制
    2. 边到边的消息传递
    3. 残差连接和层归一化
    4. 梯度裁剪和正则化
    5. 更稳定的训练策略
    """
    
    def __init__(self, node_dim, edge_dim=128, num_layers=3, mlp_hidden=64, 
                 dropout=0.1, num_heads=4, edge_attr_dim=3):
        super().__init__(aggr='mean', node_dim=0)
        self.node_feature_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        
        # 节点特征投影
        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 边特征编码器 - 更深的网络
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_dim // 2),
            nn.LayerNorm(edge_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim // 2, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU()
        )
        
        # 边特征更新层
        self.edge_update_layers = nn.ModuleList([
            nn.ModuleDict({
                'edge_mlp': nn.Sequential(
                    nn.Linear(3 * edge_dim, edge_dim),  # src + dst + edge
                    nn.LayerNorm(edge_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(edge_dim, edge_dim)
                ),
                'edge_norm': nn.LayerNorm(edge_dim)
            })
            for _ in range(num_layers)
        ])
        
        # 节点更新层
        self.node_update_layers = nn.ModuleList([
            nn.ModuleDict({
                'node_mlp': nn.Sequential(
                    nn.Linear(2 * edge_dim, edge_dim),  # node + aggregated_edges
                    nn.LayerNorm(edge_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(edge_dim, edge_dim)
                ),
                'node_norm': nn.LayerNorm(edge_dim)
            })
            for _ in range(num_layers)
        ])
        
        # 智能体间注意力机制
        self.agent_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=edge_dim, 
                num_heads=num_heads, 
                dropout=dropout, 
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 输出层 - 更稳定的设计
        self.output_mlp = nn.Sequential(
            nn.Linear(edge_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化 - 使用Xavier初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, node_features, edge_index, edge_attr, num_agents, num_tasks):
        """
        前向传播
        
        Args:
            node_features: [num_nodes, node_dim] - 节点特征
            edge_index: [2, num_edges] - 边索引
            edge_attr: [num_edges, edge_attr_dim] - 边属性
            num_agents: 智能体数量
            num_tasks: 任务数量
            
        Returns:
            edge_scores: [num_edges] - 边分数
        """
        device = node_features.device
        
        if node_features.size(0) == 0 or edge_index.size(1) == 0:
            return torch.zeros(0, device=device)
        
        # 投影节点特征
        x = self.node_proj(node_features)  # [num_nodes, edge_dim]
        
        # 编码边特征
        e = self.edge_encoder(edge_attr)  # [num_edges, edge_dim]
        
        # 多层消息传递
        for layer_idx in range(self.num_layers):
            # 1. 更新边特征
            x_residual = x
            e_residual = e
            
            # 边特征更新：结合源节点、目标节点和当前边特征
            src, dst = edge_index
            edge_input = torch.cat([x[src], x[dst], e], dim=-1)  # [num_edges, 3*edge_dim]
            e_new = self.edge_update_layers[layer_idx]['edge_mlp'](edge_input)
            e = self.edge_update_layers[layer_idx]['edge_norm'](e_residual + e_new)
            
            # 2. 节点特征更新：聚合相邻边的信息
            # 消息传递
            messages = self.propagate(edge_index, x=x, edge_attr=e)
            
            # 节点更新
            node_input = torch.cat([x, messages], dim=-1)  # [num_nodes, 2*edge_dim]
            x_new = self.node_update_layers[layer_idx]['node_mlp'](node_input)
            x = self.node_update_layers[layer_idx]['node_norm'](x_residual + x_new)
            
            # 3. 智能体间注意力机制
            if num_agents.item() > 1:
                agent_features = x[:num_agents.item()]  # [num_agents, edge_dim]
                
                # 多头注意力
                agent_features_input = agent_features.unsqueeze(0)  # [1, num_agents, edge_dim]
                attn_output, _ = self.agent_attention[layer_idx](
                    agent_features_input, agent_features_input, agent_features_input
                )
                attn_output = attn_output.squeeze(0)  # [num_agents, edge_dim]
                
                # 更新智能体节点特征
                x = torch.cat([attn_output, x[num_agents.item():]], dim=0)
        
        # 计算边分数
        edge_scores = self.output_mlp(e).squeeze(-1)  # [num_edges]
        
        return edge_scores

    def message(self, x_j, edge_attr):
        """
        消息函数：计算从邻居节点传递的消息
        
        Args:
            x_j: 邻居节点特征
            edge_attr: 边特征
            
        Returns:
            消息：边特征加权的邻居节点特征
        """
        # 使用边特征作为注意力权重
        attention_weights = torch.sigmoid(edge_attr.sum(dim=-1, keepdim=True))
        return attention_weights * x_j


class StabilizedTrainer:
    """
    稳定化训练器，解决训练过程中的梯度问题
    """
    
    def __init__(self, model, optimizer, device, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        # 损失平滑
        self.loss_ema = None
        self.ema_beta = 0.9
    
    def train_step(self, batch_data, criterion):
        """
        单步训练
        
        Args:
            batch_data: 批次数据
            criterion: 损失函数
            
        Returns:
            loss: 损失值
            accuracy: 准确率
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_samples = 0
        
        for data in batch_data:
            data = data.to(self.device)
            
            # 前向传播
            edge_scores = self.model(
                data.x,
                data.edge_index, 
                data.edge_attr, 
                data.num_agents, 
                data.num_tasks
            )
            
            # 计算损失
            if hasattr(data, 'edge_label'):
                labels = data.edge_label.float()
                
                # 类别平衡
                pos_weight = (labels == 0).sum().float() / (labels == 1).sum().float()
                pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0)
                
                # 使用带权重的BCE损失
                loss = F.binary_cross_entropy_with_logits(
                    edge_scores, labels, pos_weight=pos_weight
                )
                
                # 正则化项
                l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + 1e-4 * l2_reg
                
                # 计算准确率
                predictions = torch.sigmoid(edge_scores) > 0.5
                accuracy = (predictions == labels).float().mean()
                
                total_acc += accuracy.item()
            else:
                loss = criterion(edge_scores, data)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_samples += 1
        
        # 更新学习率
        self.scheduler.step()
        
        # 平滑损失
        avg_loss = total_loss / max(num_samples, 1)
        if self.loss_ema is None:
            self.loss_ema = avg_loss
        else:
            self.loss_ema = self.ema_beta * self.loss_ema + (1 - self.ema_beta) * avg_loss
        
        avg_acc = total_acc / max(num_samples, 1)
        
        return avg_loss, avg_acc
    
    def validate(self, val_data, criterion):
        """
        验证
        
        Args:
            val_data: 验证数据
            criterion: 损失函数
            
        Returns:
            val_loss: 验证损失
            val_acc: 验证准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data in val_data:
                data = data.to(self.device)
                
                # 前向传播
                edge_scores = self.model(
                    data.x,
                    data.edge_index, 
                    data.edge_attr, 
                    data.num_agents, 
                    data.num_tasks
                )
                
                # 计算损失和准确率
                if hasattr(data, 'edge_label'):
                    labels = data.edge_label.float()
                    loss = F.binary_cross_entropy_with_logits(edge_scores, labels)
                    
                    predictions = torch.sigmoid(edge_scores) > 0.5
                    accuracy = (predictions == labels).float().mean()
                    
                    total_acc += accuracy.item()
                else:
                    loss = criterion(edge_scores, data)
                
                total_loss += loss.item()
                num_samples += 1
        
        avg_loss = total_loss / max(num_samples, 1)
        avg_acc = total_acc / max(num_samples, 1)
        
        return avg_loss, avg_acc


# 使用示例
def create_improved_model(node_dim=128, edge_dim=128):
    """
    创建改进的EdgeGCNN模型
    
    Args:
        node_dim: 节点特征维度
        edge_dim: 边特征维度
        
    Returns:
        model: 改进的EdgeGCNN模型
    """
    return ImprovedEdgeGCNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_layers=3,
        mlp_hidden=64,
        dropout=0.1,
        num_heads=4,
        edge_attr_dim=3
    ) 