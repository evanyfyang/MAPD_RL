import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, MessagePassing
from torch_geometric.utils import to_undirected
import numpy as np
from torch.utils.checkpoint import checkpoint
import gc


class MemoryOptimizedGridGCN(nn.Module):
    """
    显存优化的Grid GCN - 使用梯度检查点和更少的中间张量
    """
    def __init__(self, grid_feature_dim, hidden_dim, num_layers=3, dropout=0.1):
        super(MemoryOptimizedGridGCN, self).__init__()
        self.grid_feature_dim = grid_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 简化的输入投影层
        self.node_init = nn.Linear(grid_feature_dim, hidden_dim)
        
        # 多层GCN - 使用更少的参数
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def _gcn_block(self, x, edge_index, layer_idx):
        """单个GCN block - 用于梯度检查点"""
        gcn_layer = self.gcn_layers[layer_idx]
        layer_norm = self.layer_norms[layer_idx]
        
        residual = x
        x = gcn_layer(x, edge_index)
        x = layer_norm(x)
        
        if layer_idx < self.num_layers - 1:
            x = F.gelu(x)
            x = self.dropout_layer(x)
        
        return x + residual
        
    def forward(self, grid_features, edge_index):
        """使用梯度检查点的前向传播"""
        x = self.node_init(grid_features)
        
        # 使用梯度检查点减少显存
        for i in range(self.num_layers):
            if self.training and i > 0:  # 只在训练时使用检查点
                x = checkpoint(self._gcn_block, x, edge_index, i)
            else:
                x = self._gcn_block(x, edge_index, i)
        
        return x


class MemoryOptimizedLineGraphGAT(nn.Module):
    """
    显存优化的LineGraphGAT - 解决line graph的显存爆炸问题
    """
    def __init__(self, node_dim, edge_dim, num_layers=3, dropout=0.1, heads=4, edge_combine='add', edge_attr_dim=5):
        super(MemoryOptimizedLineGraphGAT, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.edge_combine = edge_combine
        self.edge_attr_dim = edge_attr_dim
        
        # 边属性处理
        self.edge_attr_mlp = nn.Linear(edge_attr_dim, edge_dim)
        
        # 初始边嵌入MLP
        if edge_combine == 'concat':
            input_dim = 2 * node_dim + edge_dim
        else:
            input_dim = node_dim + edge_dim
            
        self.initial_edge_mlp = nn.Sequential(
            nn.Linear(input_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 简化的GAT层 - 减少参数
        self.line_gat_layers = nn.ModuleList()
        self.line_layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:
                self.line_gat_layers.append(
                    GATv2Conv(edge_dim, edge_dim, heads=1, dropout=dropout, concat=False)
                )
            else:
                self.line_gat_layers.append(
                    GATv2Conv(edge_dim, edge_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            self.line_layer_norms.append(nn.LayerNorm(edge_dim))
        
        self.dropout_layer = nn.Dropout(dropout)

    def create_line_graph_optimized(self, edge_index, max_line_edges=50000):
        """
        优化的line graph创建 - 限制显存使用
        """
        num_edges = edge_index.size(1)
        device = edge_index.device
        
        if num_edges == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 显存保护：如果边数太多，跳过line graph
        if num_edges > 1000:  # 阈值可调
            print(f"Warning: Skipping line graph for {num_edges} edges to save memory")
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 使用tensor操作替代Python字典，提高效率
        num_nodes = edge_index.max().item() + 1
        
        # 创建adjacency list using sparse tensor (更高效)
        adj_list = torch.zeros(num_nodes, num_edges, dtype=torch.bool, device=device)
        for edge_idx in range(num_edges):
            adj_list[edge_index[0, edge_idx], edge_idx] = True
            adj_list[edge_index[1, edge_idx], edge_idx] = True
        
        # 找到共享节点的边对
        line_edges = []
        edges_processed = 0
        
        for node in range(num_nodes):
            node_edges = torch.where(adj_list[node])[0]
            num_node_edges = len(node_edges)
            
            if num_node_edges > 1:
                # 限制每个节点的边数，避免O(n²)爆炸
                if num_node_edges > 50:  # 阈值可调
                    # 随机采样，避免全连接
                    indices = torch.randperm(num_node_edges, device=device)[:50]
                    node_edges = node_edges[indices]
                    num_node_edges = 50
                
                # 创建边对
                for i in range(num_node_edges):
                    for j in range(i + 1, num_node_edges):
                        edge_i, edge_j = node_edges[i].item(), node_edges[j].item()
                        line_edges.extend([[edge_i, edge_j], [edge_j, edge_i]])
                        edges_processed += 2
                        
                        # 显存保护：限制line graph的边数
                        if edges_processed >= max_line_edges:
                            break
                    if edges_processed >= max_line_edges:
                        break
            if edges_processed >= max_line_edges:
                break
        
        if len(line_edges) > 0:
            line_edge_index = torch.tensor(line_edges, device=device, dtype=torch.long).t()
        else:
            line_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        return line_edge_index

    def forward(self, node_features, edge_index, edge_attr=None):
        """显存优化的前向传播"""
        # 生成初始边嵌入
        row, col = edge_index
        
        if self.edge_combine == 'concat':
            initial_edge_features = torch.cat([node_features[row], node_features[col]], dim=1)
        else:
            initial_edge_features = node_features[row] + node_features[col]
        
        # 处理边属性
        if edge_attr is not None:
            edge_attr_features = self.edge_attr_mlp(edge_attr)
            initial_edge_features = torch.cat([initial_edge_features, edge_attr_features], dim=1)
        else:
            zero_edge_attr = torch.zeros((initial_edge_features.size(0), self.edge_dim), 
                                       device=initial_edge_features.device)
            initial_edge_features = torch.cat([initial_edge_features, zero_edge_attr], dim=1)
        
        edge_embeddings = self.initial_edge_mlp(initial_edge_features)
        
        # 创建优化的Line Graph
        line_edge_index = self.create_line_graph_optimized(edge_index)
        
        # 在Line Graph上应用GAT
        for i, (gat_layer, layer_norm) in enumerate(zip(self.line_gat_layers, self.line_layer_norms)):
            residual = edge_embeddings
            
            if line_edge_index.size(1) > 0:
                if self.training and i > 0:  # 使用梯度检查点
                    edge_embeddings = checkpoint(gat_layer, edge_embeddings, line_edge_index)
                else:
                    edge_embeddings = gat_layer(edge_embeddings, line_edge_index)
            
            edge_embeddings = layer_norm(edge_embeddings)
            
            if i < self.num_layers - 1:
                edge_embeddings = F.gelu(edge_embeddings)
                edge_embeddings = self.dropout_layer(edge_embeddings)
            
            edge_embeddings = edge_embeddings + residual
            
            # 显式清理中间变量
            del residual
            if i % 2 == 0:  # 定期清理
                torch.cuda.empty_cache()
        
        return edge_embeddings


class MemoryOptimizedGridSPMPNN(nn.Module):
    """
    显存优化的GridSPMPNN - 解决内存泄漏和过度的张量创建
    """
    def __init__(self, grid_feature_dim, hidden_dim, num_layers=3, dropout=0.1, max_distance=3):
        super(MemoryOptimizedGridSPMPNN, self).__init__()
        self.grid_feature_dim = grid_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_distance = max_distance
        
        # 简化的初始化层
        self.grid_node_init = nn.Linear(grid_feature_dim, hidden_dim)
        self.virtual_node_coord_proj = nn.Linear(2, hidden_dim)
        
        # 简化的消息传递层
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # self + messages
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
    def forward(self, obs, device):
        """显存优化的前向传播"""
        grid = obs["grid"]
        batch_size, height, width = grid.shape
        num_grid_nodes_per_sample = height * width
        
        # 1. 高效创建grid特征 - 避免循环
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(num_grid_nodes_per_sample)
        
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # 归一化坐标
        grid_coords = torch.stack([
            y_coords.flatten() / height,
            x_coords.flatten() / width
        ], dim=1).float()
        
        # 复制到所有batch
        batch_grid_coords = grid_coords.repeat(batch_size, 1)
        
        # 2. 初始化特征
        batch_grid_features = self.grid_node_init(batch_grid_coords)
        
        # 3. 简化的虚拟节点处理（如果需要）
        # TODO: 根据需要添加虚拟节点逻辑
        
        # 4. 简化的消息传递 - 使用更少的中间张量
        all_features = batch_grid_features
        
        for layer in range(self.num_layers):
            # Pre-norm
            normalized_features = self.layer_norms[layer](all_features)
            
            # 简化的消息计算 - 只处理最重要的连接
            # TODO: 根据具体需求添加边处理逻辑
            
            # 自连接消息
            self_messages = normalized_features
            
            # 简单的更新
            combined = torch.cat([self_messages, normalized_features], dim=1)
            new_features = self.update_mlp(combined)
            
            # 残差连接
            all_features = all_features + new_features
            
            # 定期清理
            if layer % 2 == 0:
                torch.cuda.empty_cache()
        
        return {
            'grid_features': all_features,
            'batch_free_agents': torch.zeros(batch_size, 10, self.hidden_dim, device=device),
            'batch_delivering_agents': torch.zeros(batch_size, 10, self.hidden_dim, device=device),
            'batch_free_tasks': torch.zeros(batch_size, 10, self.hidden_dim, device=device),
            'batch_delivering_tasks': torch.zeros(batch_size, 10, self.hidden_dim, device=device)
        }


def print_memory_usage(stage_name):
    """打印显存使用情况"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"[{stage_name}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def cleanup_memory():
    """清理显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MemoryOptimizedEdgeGCNN(MessagePassing):
    """
    内存优化的EdgeGCNN，包含以下优化：
    1. Gradient checkpointing减少内存使用
    2. 分批处理多头注意力
    3. 内存监控和限制
    4. 可选的edge数量限制
    """
    def __init__(self, node_dim, edge_dim=128, num_layers=3, mlp_hidden=64, dropout=0.1, 
                 num_heads=4, edge_attr_dim=3, use_gradient_checkpointing=True, 
                 max_edges=25000, attention_batch_size=50, memory_limit_gb=8.0):
        super().__init__(aggr='mean', node_dim=0)
        
        self.node_feature_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_attr_dim = edge_attr_dim
        self.dropout = nn.Dropout(dropout)
        
        # 内存优化参数
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_edges = max_edges  # 最大边数限制
        self.attention_batch_size = attention_batch_size  # 注意力分批大小
        self.memory_limit_gb = memory_limit_gb  # 内存限制
        
        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_dim), 
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim), 
            nn.GELU()
        )
        
        # GCN layers with residual connections
        self.gcn_layers = nn.ModuleList([
            nn.ModuleDict({
                'linear': nn.Linear(edge_dim, edge_dim), 
                'norm': nn.LayerNorm(edge_dim)
            })
            for _ in range(num_layers)
        ])
        
        # 简化的注意力层，使用更小的attention
        self.agent_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=edge_dim, 
                num_heads=min(num_heads, 2),  # 减少头数
                dropout=dropout, 
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Edge feature projection layer
        self.edge_feature_proj = nn.Linear(3 * edge_dim, edge_dim)
        
        # Output MLP for edge scoring
        self.output_mlp = nn.Sequential(
            nn.Linear(edge_dim, mlp_hidden), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_hidden, 1)
        )

    def check_memory_limit(self, stage_name):
        """检查内存限制"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > self.memory_limit_gb:
                print(f"Warning: Memory usage ({allocated:.2f}GB) exceeds limit ({self.memory_limit_gb}GB) at {stage_name}")
                cleanup_memory()

    def apply_edge_limit(self, edge_index, edge_attr, num_agents, num_tasks):
        """限制边的数量，优先保留重要的边"""
        num_edges = edge_index.shape[1]
        
        if num_edges <= self.max_edges:
            return edge_index, edge_attr
        
        print(f"Warning: Too many edges ({num_edges}), limiting to {self.max_edges}")
        
        # 随机采样边（可以改进为基于重要性的采样）
        perm = torch.randperm(num_edges, device=edge_index.device)[:self.max_edges]
        limited_edge_index = edge_index[:, perm]
        limited_edge_attr = edge_attr[perm]
        
        return limited_edge_index, limited_edge_attr

    def batched_attention(self, agent_features, attention_layer):
        """分批处理注意力计算"""
        num_agents = agent_features.shape[0]
        
        if num_agents <= self.attention_batch_size:
            # 直接计算
            agent_features_input = agent_features.unsqueeze(0)  # [1, num_agents, edge_dim]
            attn_output, _ = attention_layer(
                agent_features_input, agent_features_input, agent_features_input
            )
            return attn_output.squeeze(0)  # [num_agents, edge_dim]
        
        # 分批计算
        batch_size = self.attention_batch_size
        num_batches = (num_agents + batch_size - 1) // batch_size
        
        outputs = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_agents)
            
            batch_features = agent_features[start_idx:end_idx].unsqueeze(0)
            with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度避免问题
                attn_output, _ = attention_layer(
                    batch_features, batch_features, batch_features
                )
            outputs.append(attn_output.squeeze(0))
            
            # 每批次后清理内存
            if i % 4 == 0:  # 每4个batch清理一次
                cleanup_memory()
        
        return torch.cat(outputs, dim=0)

    def gcn_layer_forward(self, layer_idx, x, edge_index, edge_attr, num_agents):
        """单个GCN层的前向传播"""
        # Standard message passing
        residual = x
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_new = self.gcn_layers[layer_idx]['norm'](
            residual + self.gcn_layers[layer_idx]['linear'](messages)
        )
        
        # Agent information exchange via multi-head attention
        if num_agents > 1:
            # Extract agent node features (first num_agents nodes)
            agent_features = x_new[:num_agents]  # [num_agents, edge_dim]
            
            # Apply batched multi-head attention to agents
            attn_output = self.batched_attention(agent_features, self.agent_attention[layer_idx])
            
            # Update agent node features
            x_new[:num_agents] = attn_output
        
        return x_new

    def forward(self, node_features, edge_index, edge_attr, num_agents, num_tasks):
        """
        内存优化的前向传播
        """
        device = node_features.device
        
        print_memory_usage("EdgeGCNN start")
        
        # 检查边数量限制
        edge_index, edge_attr = self.apply_edge_limit(edge_index, edge_attr, num_agents, num_tasks)
        
        # Project node features to edge_dim
        x = self.node_proj(node_features)  # [num_nodes, edge_dim]
        self.check_memory_limit("After node projection")
        
        # Encode edge features
        e = self.edge_encoder(edge_attr)  # [num_edges, edge_dim]
        self.check_memory_limit("After edge encoding")
        
        # Apply GCN layers with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            for layer_idx in range(self.num_layers):
                # 使用gradient checkpointing减少内存使用
                x = torch.utils.checkpoint.checkpoint(
                    self.gcn_layer_forward, 
                    layer_idx, x, edge_index, e, num_agents,
                    use_reentrant=False
                )
                self.check_memory_limit(f"After GCN layer {layer_idx}")
                
                # 每两层清理一次内存
                if layer_idx % 2 == 1:
                    cleanup_memory()
        else:
            # 标准前向传播
            for layer_idx in range(self.num_layers):
                x = self.gcn_layer_forward(layer_idx, x, edge_index, e, num_agents)
                self.check_memory_limit(f"After GCN layer {layer_idx}")
        
        print_memory_usage("After all GCN layers")
        
        # Generate edge scores using updated node features
        # 分批处理边特征计算，避免大矩阵操作
        num_edges = edge_index.shape[1]
        batch_size = min(5000, num_edges)  # 分批处理边
        
        edge_scores = []
        for i in range(0, num_edges, batch_size):
            end_idx = min(i + batch_size, num_edges)
            batch_edge_index = edge_index[:, i:end_idx]
            batch_e = e[i:end_idx]
            
            # Compute edge representations for this batch
            row, col = batch_edge_index
            batch_edge_representations = torch.cat([x[row], x[col], batch_e], dim=-1)
            
            # Project to edge_dim
            batch_updated_edge_features = self.edge_feature_proj(batch_edge_representations)
            
            # Get scores for this batch
            batch_scores = self.output_mlp(batch_updated_edge_features).squeeze(-1)
            edge_scores.append(batch_scores)
            
            # 定期清理内存
            if i % (batch_size * 4) == 0:
                cleanup_memory()
        
        final_scores = torch.cat(edge_scores, dim=0)
        
        print_memory_usage("EdgeGCNN end")
        
        return final_scores

    def message(self, x_j, edge_attr):
        """Message function: combines neighbor node features with edge features"""
        return edge_attr * x_j


class UndirectedMemoryOptimizedEdgeGCNN(MemoryOptimizedEdgeGCNN):
    """
    无向图版本的内存优化EdgeGCNN
    """
    def __init__(self, node_dim, edge_dim=128, num_layers=3, mlp_hidden=64, dropout=0.1, 
                 num_heads=4, edge_attr_dim=3, use_gradient_checkpointing=True, 
                 max_edges=25000, attention_batch_size=50, memory_limit_gb=8.0):
        super().__init__(node_dim, edge_dim, num_layers, mlp_hidden, dropout, 
                         num_heads, edge_attr_dim, use_gradient_checkpointing, 
                         max_edges, attention_batch_size, memory_limit_gb)
    
    def forward(self, node_features, edge_index, edge_attr, num_agents, num_tasks):
        """
        无向图的内存优化前向传播
        """
        if node_features.size(0) == 0 or edge_index.size(1) == 0:
            return torch.zeros(0, device=node_features.device)
        
        print_memory_usage("UndirectedEdgeGCNN start")
        
        # 检查边数量限制
        edge_index, edge_attr = self.apply_edge_limit(edge_index, edge_attr, num_agents, num_tasks)
        
        # Convert to undirected graph by adding reverse edges
        edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr_undirected = torch.cat([edge_attr, edge_attr], dim=0)
        self.check_memory_limit("After undirected conversion")
        
        # Project node features and edge attributes
        h = self.node_proj(node_features)  # [num_nodes, edge_dim]
        edge_features = self.edge_encoder(edge_attr_undirected)  # [num_edges*2, edge_dim]
        self.check_memory_limit("After projections")
        
        # GCN layers with undirected message passing
        if self.use_gradient_checkpointing and self.training:
            for layer_idx in range(self.num_layers):
                h = torch.utils.checkpoint.checkpoint(
                    self.gcn_layer_forward, 
                    layer_idx, h, edge_index_undirected, edge_features, num_agents,
                    use_reentrant=False
                )
                self.check_memory_limit(f"After GCN layer {layer_idx}")
                
                if layer_idx % 2 == 1:
                    cleanup_memory()
        else:
            for layer_idx in range(self.num_layers):
                h = self.gcn_layer_forward(layer_idx, h, edge_index_undirected, edge_features, num_agents)
                self.check_memory_limit(f"After GCN layer {layer_idx}")
        
        # Compute edge scores for original (directed) edges only
        num_original_edges = edge_attr.size(0)
        updated_original_edge_features = edge_features[:num_original_edges]  # [num_edges, edge_dim]
        edge_scores = self.output_mlp(updated_original_edge_features).squeeze(-1)  # [num_edges]
        
        print_memory_usage("UndirectedEdgeGCNN end")
        
        return edge_scores


def create_memory_optimized_edge_gcnn(use_undirected=True, **kwargs):
    """
    创建内存优化的EdgeGCNN实例
    
    Args:
        use_undirected: 是否使用无向图版本
        **kwargs: 其他参数
    
    Returns:
        MemoryOptimizedEdgeGCNN或UndirectedMemoryOptimizedEdgeGCNN实例
    """
    default_config = {
        'node_dim': 128,
        'edge_dim': 128,
        'num_layers': 2,  # 减少层数
        'mlp_hidden': 64,
        'dropout': 0.1,
        'num_heads': 2,  # 减少注意力头数
        'edge_attr_dim': 3,
        'use_gradient_checkpointing': True,
        'max_edges': 25000,  # 限制边数
        'attention_batch_size': 50,  # 注意力分批大小
        'memory_limit_gb': 8.0  # 内存限制
    }
    
    # 合并配置
    config = {**default_config, **kwargs}
    
    if use_undirected:
        return UndirectedMemoryOptimizedEdgeGCNN(**config)
    else:
        return MemoryOptimizedEdgeGCNN(**config) 