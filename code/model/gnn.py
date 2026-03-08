import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, MessagePassing
from torch_geometric.utils import to_undirected
import numpy as np


class GridGCN(nn.Module):
    """
    优化后的Grid GCN，支持多层、dropout和layer normalization
    """
    def __init__(self, grid_feature_dim, hidden_dim, num_layers=3, dropout=0.1):
        super(GridGCN, self).__init__()
        self.grid_feature_dim = grid_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入投影层
        self.node_init = nn.Sequential(
            nn.Linear(grid_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多层GCN
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, grid_features, edge_index):
        """
        grid_features: [num_nodes, grid_feature_dim]
        edge_index: [2, num_edges] 
        """
        x = self.node_init(grid_features)
        
        # 多层GCN处理
        for i, (gcn_layer, layer_norm) in enumerate(zip(self.gcn_layers, self.layer_norms)):
            residual = x
            x = gcn_layer(x, edge_index)
            x = layer_norm(x)
            
            if i < self.num_layers - 1:  # 最后一层不加ReLU
                x = F.gelu(x)
                x = self.dropout_layer(x)
            
            # 残差连接
            x = x + residual
        
        return x


class GridGAT(nn.Module):
    """
    优化后的Grid GAT，支持多层、dropout和layer normalization
    """
    def __init__(self, grid_feature_dim, hidden_dim, num_layers=3, dropout=0.1, heads=4):
        super(GridGAT, self).__init__()
        self.grid_feature_dim = grid_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # 输入投影层
        self.node_init = nn.Sequential(
            nn.Linear(grid_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多层GAT
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:  # 最后一层
                self.gat_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
                )
            else:
                self.gat_layers.append(
                    GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, grid_features, edge_index):
        """
        grid_features: [num_nodes, grid_feature_dim]
        edge_index: [2, num_edges] - Edge indices in PyG format
        """
        x = self.node_init(grid_features)
        
        # 多层GAT处理
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            
            if i < self.num_layers - 1:  # 最后一层不加ReLU
                x = F.gelu(x)
                x = self.dropout_layer(x)
            
            # 残差连接
            x = x + residual
        
        return x


class HigherGATLayer(nn.Module):
    """
    高层级GAT，实现多层GAT然后将节点两两拼接计算边嵌入
    """
    def __init__(self, node_dim, output_dim, num_layers=3, dropout=0.1, heads=4, edge_combine='concat', edge_attr_dim=5):
        super(HigherGATLayer, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.edge_combine = edge_combine  # 'concat' or 'add'
        self.edge_attr_dim = edge_attr_dim  # 边属性维度
        
        # 节点预处理
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多层GAT - 使用GATv2Conv的edge_dim参数
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:  # 最后一层
                self.gat_layers.append(
                    GATv2Conv(output_dim, output_dim, heads=1, dropout=dropout, concat=False, edge_dim=edge_attr_dim)
                )
            else:
                self.gat_layers.append(
                    GATv2Conv(output_dim, output_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_attr_dim)
                )
            self.layer_norms.append(nn.LayerNorm(output_dim))
        
        # 边嵌入MLP
        edge_input_dim = 2 * output_dim if edge_combine == 'concat' else output_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: Node features [num_nodes, node_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge features [num_edges, edge_attr_dim] - one-hot编码的边类型
        
        Returns: Edge embeddings [num_edges, output_dim]
        """
        # 确保边索引是无向的，同时正确处理边属性
        if edge_attr is not None:
            from torch_geometric.utils import to_undirected
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=x.size(0))
        else:
            edge_index = to_undirected(edge_index)
        
        # 调用父类的forward方法，但不再调用to_undirected
        # 节点预处理
        x = self.node_mlp(x)
        
        # 多层GAT处理节点 - 传递edge_attr给GATv2Conv
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            
            x = F.gelu(x)
            x = self.dropout_layer(x)
        
            # 残差连接
            x = x + residual
        
        # 计算边嵌入：将连接的节点特征拼接或相加
        row, col = edge_index
        if self.edge_combine == 'concat':
            edge_features = torch.cat([x[row], x[col]], dim=1)
        else:  # 'add'
            edge_features = x[row] + x[col]
        
        # 通过MLP处理边特征
        edge_embeddings = self.edge_mlp(edge_features)
        
        return edge_embeddings


class LineGraphGAT(nn.Module):
    """
    将Higher图转换为Line Graph，然后用GAT对边进行学习
    """
    def __init__(self, node_dim, edge_dim, num_layers=3, dropout=0.1, heads=4, edge_combine='add', edge_attr_dim=5):
        super(LineGraphGAT, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.edge_combine = edge_combine  # 默认使用'add'以避免无向边的顺序问题
        self.edge_attr_dim = edge_attr_dim  # 边属性维度
        
        # 边属性处理MLP
        self.edge_attr_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 初始边嵌入生成 - 现在包含边属性
        initial_edge_input_dim = node_dim + edge_dim  # 节点特征 + 边属性特征
        self.initial_edge_mlp = nn.Sequential(
            nn.Linear(initial_edge_input_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Line Graph上的GAT层
        self.line_gat_layers = nn.ModuleList()
        self.line_layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:  # 最后一层
                self.line_gat_layers.append(
                    GATv2Conv(edge_dim, edge_dim, heads=1, dropout=dropout, concat=False)
                )
            else:
                self.line_gat_layers.append(
                    GATv2Conv(edge_dim, edge_dim // heads, heads=heads, dropout=dropout, concat=True)
                )
            self.line_layer_norms.append(nn.LayerNorm(edge_dim))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def create_line_graph(self, edge_index):
        """
        将原图转换为Line Graph（优化版本）
        在Line Graph中，原图的每条边成为一个节点，如果两条边共享一个节点，则在Line Graph中连接
        
        Args:
            edge_index: [2, num_edges] 原图的边索引
            
        Returns:
            line_edge_index: [2, num_line_edges] Line Graph的边索引
        """
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        # 优化方法：使用字典来快速查找共享节点的边
        # 创建节点到边的映射
        node_to_edges = {}
        for edge_idx in range(num_edges):
            for node in edge_index[:, edge_idx].tolist():
                if node not in node_to_edges:
                    node_to_edges[node] = []
                node_to_edges[node].append(edge_idx)
        
        # 创建Line Graph的边
        line_edges = []
        edge_pairs = set()  # 用于去重
        
        # 对于每个节点，连接所有通过该节点的边
        for node, edges in node_to_edges.items():
            if len(edges) > 1:
                # 连接所有通过该节点的边对
                for i in range(len(edges)):
                    for j in range(i + 1, len(edges)):
                        edge_i, edge_j = edges[i], edges[j]
                        # 确保边对的顺序一致，避免重复
                        pair = (min(edge_i, edge_j), max(edge_i, edge_j))
                        if pair not in edge_pairs:
                            edge_pairs.add(pair)
                            line_edges.append([edge_i, edge_j])
                            line_edges.append([edge_j, edge_i])  # 无向图
        
        if len(line_edges) > 0:
            line_edge_index = torch.tensor(line_edges, device=edge_index.device).t()
        else:
            # 如果没有连接，创建空的边索引
            line_edge_index = torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        return line_edge_index
    
    def forward(self, node_features, edge_index, edge_attr=None):
        """
        Args:
            node_features: [num_nodes, node_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_attr_dim] 边属性 - one-hot编码的边类型
            
        Returns:
            edge_embeddings: [num_edges, edge_dim] 边嵌入
        """
        # 生成初始边嵌入 - 使用add方式避免顺序问题
        row, col = edge_index
        initial_edge_features = node_features[row] + node_features[col]
        
        # 处理边属性
        if edge_attr is not None:
            edge_attr_features = self.edge_attr_mlp(edge_attr)
            # 将边属性特征与节点特征拼接
            initial_edge_features = torch.cat([initial_edge_features, edge_attr_features], dim=1)
        else:
            # 如果没有边属性，用零填充
            zero_edge_attr = torch.zeros((initial_edge_features.size(0), self.edge_dim), device=initial_edge_features.device)
            initial_edge_features = torch.cat([initial_edge_features, zero_edge_attr], dim=1)
        
        # 通过MLP生成初始边嵌入
        edge_embeddings = self.initial_edge_mlp(initial_edge_features)
        
        # 创建Line Graph
        line_edge_index = self.create_line_graph(edge_index)
        
        # 在Line Graph上应用GAT
        for i, (gat_layer, layer_norm) in enumerate(zip(self.line_gat_layers, self.line_layer_norms)):
            residual = edge_embeddings
            
            if line_edge_index.size(1) > 0:  # 只有当存在边时才应用GAT
                edge_embeddings = gat_layer(edge_embeddings, line_edge_index)
            
            edge_embeddings = layer_norm(edge_embeddings)
            
            if i < self.num_layers - 1:  # 最后一层不加ReLU
                edge_embeddings = F.gelu(edge_embeddings)
                edge_embeddings = self.dropout_layer(edge_embeddings)
            
            # 残差连接
            edge_embeddings = edge_embeddings + residual
        
        return edge_embeddings


class UndirectedHigherGATLayer(HigherGATLayer):
    """
    无向版本的HigherGATLayer，确保边嵌入是无向的
    """
    def __init__(self, node_dim, output_dim, num_layers=3, dropout=0.1, heads=4, edge_combine='concat', edge_attr_dim=5):
        super().__init__(node_dim, output_dim, num_layers, dropout, heads, edge_combine, edge_attr_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        x: Node features [num_nodes, node_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge features [num_edges, edge_attr_dim] - one-hot编码的边类型
        
        Returns: Edge embeddings [num_edges, output_dim]
        """
        # 确保边索引是无向的，同时正确处理边属性
        if edge_attr is not None:
            from torch_geometric.utils import to_undirected
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=x.size(0))
        else:
            edge_index = to_undirected(edge_index)
        
        # 调用父类的forward方法，但不再调用to_undirected
        # 节点预处理
        x = self.node_mlp(x)
        
        # 多层GAT处理节点 - 传递edge_attr给GATv2Conv
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = layer_norm(x)
            
            x = F.gelu(x)
            x = self.dropout_layer(x)
        
            # 残差连接
            x = x + residual
        
        # 计算边嵌入：将连接的节点特征拼接或相加
        row, col = edge_index
        if self.edge_combine == 'concat':
            edge_features = torch.cat([x[row], x[col]], dim=1)
        else:  # 'add'
            edge_features = x[row] + x[col]
        
        # 通过MLP处理边特征
        edge_embeddings = self.edge_mlp(edge_features)
        
        return edge_embeddings


class UndirectedLineGraphGAT(LineGraphGAT):
    """
    无向版本的LineGraphGAT
    """
    def __init__(self, node_dim, edge_dim, num_layers=3, dropout=0.1, heads=4, edge_combine='add', edge_attr_dim=5):
        super().__init__(node_dim, edge_dim, num_layers, dropout, heads, edge_combine, edge_attr_dim)
    
    def forward(self, node_features, edge_index, edge_attr=None):
        """
        Args:
            node_features: [num_nodes, node_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_attr_dim] 边属性 - one-hot编码的边类型
            
        Returns:
            edge_embeddings: [num_edges, edge_dim] 边嵌入
        """
        # 确保边索引是无向的，同时正确处理边属性
        if edge_attr is not None:
            from torch_geometric.utils import to_undirected
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=node_features.size(0))
        else:
            edge_index = to_undirected(edge_index)
        
        # 调用父类的forward方法
        edge_embeddings = super().forward(node_features, edge_index, edge_attr)
        
        return edge_embeddings


class SelfAttentionGATLayer(nn.Module):
    """
    使用Self-Attention在agent间交换信息的Higher GNN
    agent之间不直接连边，而是通过self-attention机制交换信息
    """
    def __init__(self, node_dim, output_dim, num_layers=2, dropout=0.1, heads=4, 
                 edge_combine='concat', edge_attr_dim=2, self_attention_layers=2):
        super(SelfAttentionGATLayer, self).__init__()
        
        self.node_dim = node_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.edge_combine = edge_combine
        self.edge_attr_dim = edge_attr_dim
        self.self_attention_layers = self_attention_layers
        
        # 节点预处理
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多层GAT - 只处理agent-task边，不处理agent-agent边
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:  # 最后一层
                self.gat_layers.append(
                    GATv2Conv(output_dim, output_dim, heads=1, dropout=dropout, concat=False, edge_dim=edge_attr_dim)
                )
            else:
                self.gat_layers.append(
                    GATv2Conv(output_dim, output_dim // heads, heads=heads, dropout=dropout, concat=True, edge_dim=edge_attr_dim)
                )
            self.layer_norms.append(nn.LayerNorm(output_dim))
        
        # Self-attention模块用于agent间交互
        self.agent_self_attention_layers = nn.ModuleList()
        self.agent_attention_norms = nn.ModuleList()
        
        for i in range(self_attention_layers):
            self.agent_self_attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=output_dim,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.agent_attention_norms.append(nn.LayerNorm(output_dim))
        
        # 边嵌入MLP
        edge_input_dim = 2 * output_dim if edge_combine == 'concat' else output_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, agent_task_mapping=None):
        """
        x: Node features [num_nodes, node_dim]
        edge_index: Edge indices [2, num_edges] - 只包含agent-task边，不包含agent-agent边
        edge_attr: Edge features [num_edges, 2] - 2维距离信息：[agent_to_pickup_dist, pickup_to_delivery_dist]
        agent_task_mapping: (agent_task_mapping_dict, num_free_agents, num_free_tasks)
        
        Returns: Edge embeddings [num_edges, output_dim]
        """
        if agent_task_mapping is None:
            raise ValueError("SelfAttentionGATLayer requires agent_task_mapping to identify agent nodes")
        
        _, num_free_agents, _ = agent_task_mapping
        
        # 节点预处理
        x = self.node_mlp(x)
        
        # 多层GAT处理 + agent self-attention
        for layer_idx in range(self.num_layers):
            residual = x
            
            # 1. GAT处理所有边（主要是agent-task边）
            x_gat = self.gat_layers[layer_idx](x, edge_index, edge_attr=edge_attr)
            x_gat = self.layer_norms[layer_idx](x_gat)
            x_gat = F.gelu(x_gat)
            x_gat = self.dropout_layer(x_gat)
            
            # 2. Self-attention处理agent间交互
            if num_free_agents > 0:
                # 提取agent节点特征 (前num_free_agents个节点是free agents)
                agent_features = x[:num_free_agents]  # [num_free_agents, output_dim]
                
                # 对agent特征应用self-attention
                agent_features_updated = agent_features.unsqueeze(0)  # [1, num_free_agents, output_dim]
                
                # 多层self-attention
                for attn_layer, attn_norm in zip(self.agent_self_attention_layers, self.agent_attention_norms):
                    attn_residual = agent_features_updated
                    
                    # Self-attention
                    attn_output, _ = attn_layer(
                        agent_features_updated, 
                        agent_features_updated, 
                        agent_features_updated
                    )
                    
                    # Layer norm + residual
                    agent_features_updated = attn_norm(attn_output + attn_residual)
                
                agent_features_updated = agent_features_updated.squeeze(0)  # [num_free_agents, output_dim]
                
                # 将更新后的agent特征放回x_gat中
                x_gat[:num_free_agents] = agent_features_updated
            
            # 残差连接
            x = residual + x_gat
        
        # 计算边嵌入：将连接的节点特征拼接或相加
        row, col = edge_index
        if self.edge_combine == 'concat':
            edge_features = torch.cat([x[row], x[col]], dim=1)
        else:  # 'add'
            edge_features = x[row] + x[col]
        
        # 通过MLP处理边特征
        edge_embeddings = self.edge_mlp(edge_features)
        
        return edge_embeddings


class GridSPMPNN(nn.Module):
    """
    基于SP-MPNN的Grid GNN，用于处理grid map和agent/task的异构图结构
    
    实现思路：
    1. 对grid图计算k-hop最短路径邻域
    2. 创建agent和task的虚拟节点，通过异构边连接到grid
    3. 异构边包括：agent-grid, pickup-grid, delivery-grid, pickup-delivery
    4. 分别处理不同类型的消息传递
    """
    def __init__(self, grid_feature_dim, hidden_dim, num_layers=3, dropout=0.1, max_distance=3):
        super(GridSPMPNN, self).__init__()
        self.grid_feature_dim = grid_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_distance = max_distance  # SP-MPNN的最大距离k
        
        # Grid节点初始化
        self.grid_node_init = nn.Sequential(
            nn.Linear(grid_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 虚拟节点初始化：使用坐标特征而非共享参数
        self.virtual_node_coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 从2D坐标到hidden_dim
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 异构SP-MPNN层：为每个边类型和距离层创建不同的消息函数
        self.hetero_message_mlps = nn.ModuleList()
        self.hetero_update_mlps = nn.ModuleList()
        self.pre_layer_norms = nn.ModuleList()  # Pre-LayerNorm
        self.dropout_layers = nn.ModuleList()
        
        # 定义所有边类型
        self.edge_types = [
            'agent_grid', 'pickup_grid', 'delivery_grid', 'agent_delivery', 'pickup_delivery'
        ] + [f'dist_{dist}' for dist in range(1, max_distance + 1)]
        
        for layer in range(num_layers):
            # Pre-LayerNorm for each layer
            self.pre_layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
            
            # 每一层包含所有边类型的消息函数
            layer_message_mlps = nn.ModuleDict()
            
            # SP-MPNN距离边的消息函数
            for dist in range(1, max_distance + 1):
                layer_message_mlps[f'dist_{dist}'] = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU()
                )
            
            # 异构边的消息函数
            layer_message_mlps['agent_grid'] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU()
            )
            layer_message_mlps['pickup_grid'] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU()
            )
            layer_message_mlps['delivery_grid'] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU()
            )
            layer_message_mlps['agent_delivery'] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU()
            )
            layer_message_mlps['pickup_delivery'] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU()
            )
            
            self.hetero_message_mlps.append(layer_message_mlps)
            
            # 更新函数：聚合所有边类型的消息 + self (移除内部dropout，在残差后统一处理)
            total_message_types = len(self.edge_types) + 1  # +1 for self
            self.hetero_update_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * total_message_types, hidden_dim),
                nn.GELU()
            ))
        
        # 输出层：用于生成最终的节点表示
        self.free_task_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # pickup + delivery
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.delivering_task_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # only delivery
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def _create_node_mappings(self, obs, batch_size, num_grid_nodes_per_sample, device):
        """
        预先创建节点索引映射，避免Python循环累加
        重新设计虚拟节点结构：
        1. All agents (free + delivering)
        2. All pickup locations (from free tasks)
        3. All delivery locations (from free tasks + delivering tasks)
        """
        mappings = {}
        
        # Grid节点映射 (固定)
        grid_base = torch.arange(batch_size, device=device) * num_grid_nodes_per_sample
        mappings['grid_base'] = grid_base.unsqueeze(1)  # [batch_size, 1]
        
        # 计算每个batch的虚拟节点起始位置
        virtual_base = batch_size * num_grid_nodes_per_sample
        
        # 收集所有实体数量
        free_agents_nums = obs["free_agents_num"]  # [batch_size]
        delivering_agents_nums = obs["delivering_agents_num"]  # [batch_size]
        free_tasks_nums = obs["free_tasks_num"]  # [batch_size]
        delivering_tasks_nums = obs["delivering_tasks_num"]  # [batch_size]
        
        # 计算累积偏移
        agents_offsets = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
        pickups_offsets = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
        deliveries_offsets = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
        
        for b in range(batch_size):
            # All agents = free + delivering
            agents_offsets[b + 1] = agents_offsets[b] + free_agents_nums[b] + delivering_agents_nums[b]
            # Pickup locations = free tasks only
            pickups_offsets[b + 1] = pickups_offsets[b] + free_tasks_nums[b]
            # Delivery locations = free tasks + delivering tasks
            deliveries_offsets[b + 1] = deliveries_offsets[b] + free_tasks_nums[b] + delivering_tasks_nums[b]
        
        # 虚拟节点全局索引基址
        mappings['agents_base'] = virtual_base + agents_offsets[:-1]
        mappings['pickups_base'] = virtual_base + agents_offsets[-1] + pickups_offsets[:-1]
        mappings['deliveries_base'] = virtual_base + agents_offsets[-1] + pickups_offsets[-1] + deliveries_offsets[:-1]
        
        # 为了兼容，保留原有的映射方式
        mappings['free_agents_base'] = virtual_base + agents_offsets[:-1]
        mappings['delivering_agents_base'] = virtual_base + agents_offsets[:-1]  # delivering agents在agents中的偏移由后续计算
        
        return mappings
    
    def _create_hetero_edges_vectorized(self, obs, batch_size, num_grid_nodes_per_sample, width, device, node_mappings):
        """
        向量化创建异构边索引，按照新的虚拟节点结构：
        1. agent-grid: 所有agent与其位置的grid连接
        2. pickup-grid: pickup位置与grid连接
        3. delivery-grid: delivery位置与grid连接  
        4. agent-delivery: delivering agent与其对应delivery位置连接
        5. pickup-delivery: 每个free task的pickup与delivery连接
        """
        from torch_geometric.utils import coalesce
        
        hetero_edges = {
            'agent_grid': [],
            'pickup_grid': [],
            'delivery_grid': [],
            'agent_delivery': [],
            'pickup_delivery': []
        }
        
        # 1. Agent-Grid edges: 所有agent（free + delivering）与grid连接
        all_agents_coords = []
        all_agent_batch_idx = []
        
        for b in range(batch_size):
            # Free agents
            free_agents_num = int(obs["free_agents_num"][b].item())
            if free_agents_num > 0:
                free_coords = obs["free_agents"][b, :free_agents_num, :2].long()
                all_agents_coords.append(free_coords)
                all_agent_batch_idx.extend([b] * free_agents_num)
            
            # Delivering agents  
            delivering_agents_num = int(obs["delivering_agents_num"][b].item())
            if delivering_agents_num > 0:
                delivering_coords = obs["delivering_agents"][b, :delivering_agents_num, :2].long()
                all_agents_coords.append(delivering_coords)
                all_agent_batch_idx.extend([b] * delivering_agents_num)
        
        if all_agents_coords:
            all_agents_coords = torch.cat(all_agents_coords, dim=0)  # [total_agents, 2]
            all_agent_batch_idx = torch.tensor(all_agent_batch_idx, device=device)
            
            # 计算agent虚拟节点索引
            agent_virtual_indices = []
            current_agent_idx = 0
            for b in range(batch_size):
                base_idx = node_mappings['agents_base'][b]
                free_num = int(obs["free_agents_num"][b].item())
                delivering_num = int(obs["delivering_agents_num"][b].item())
                total_agents = free_num + delivering_num
                
                if total_agents > 0:
                    batch_agent_indices = torch.arange(total_agents, device=device) + base_idx
                    agent_virtual_indices.append(batch_agent_indices)
                    current_agent_idx += total_agents
            
            if agent_virtual_indices:
                agent_virtual_indices = torch.cat(agent_virtual_indices, dim=0)
                
                # 计算grid索引
                grid_indices = (node_mappings['grid_base'][all_agent_batch_idx].squeeze(1) + 
                               all_agents_coords[:, 0] * width + all_agents_coords[:, 1])
                
                # 创建双向边
                agent_grid_edges = torch.stack([
                    torch.cat([agent_virtual_indices, grid_indices]),
                    torch.cat([grid_indices, agent_virtual_indices])
                ], dim=0)
                hetero_edges['agent_grid'].append(agent_grid_edges)
        
        # 2. Pickup-Grid edges: 只有free tasks有pickup
        free_tasks_data = obs["free_tasks"]  # [batch_size, max_tasks, 4]
        free_tasks_nums = obs["free_tasks_num"]  # [batch_size]
        
        all_pickup_coords = []
        all_pickup_batch_idx = []
        pickup_virtual_indices = []
        
        for b in range(batch_size):
            free_tasks_num = int(free_tasks_nums[b].item())
            if free_tasks_num > 0:
                pickup_coords = free_tasks_data[b, :free_tasks_num, :2].long()
                all_pickup_coords.append(pickup_coords)
                all_pickup_batch_idx.extend([b] * free_tasks_num)
                
                # Pickup虚拟节点索引
                base_idx = node_mappings['pickups_base'][b]
                batch_pickup_indices = torch.arange(free_tasks_num, device=device) + base_idx
                pickup_virtual_indices.append(batch_pickup_indices)
        
        if all_pickup_coords:
            all_pickup_coords = torch.cat(all_pickup_coords, dim=0)
            all_pickup_batch_idx = torch.tensor(all_pickup_batch_idx, device=device)
            pickup_virtual_indices = torch.cat(pickup_virtual_indices, dim=0)
            
            pickup_grid_indices = (node_mappings['grid_base'][all_pickup_batch_idx].squeeze(1) + 
                                  all_pickup_coords[:, 0] * width + all_pickup_coords[:, 1])
            
            pickup_grid_edges = torch.stack([
                torch.cat([pickup_virtual_indices, pickup_grid_indices]),
                torch.cat([pickup_grid_indices, pickup_virtual_indices])
            ], dim=0)
            hetero_edges['pickup_grid'].append(pickup_grid_edges)
        
        # 3. Delivery-Grid edges: free tasks + delivering tasks的delivery位置
        all_delivery_coords = []
        all_delivery_batch_idx = []
        delivery_virtual_indices = []
        
        for b in range(batch_size):
            current_delivery_idx = 0
            base_idx = node_mappings['deliveries_base'][b]
            
            # Free tasks的delivery位置
            free_tasks_num = int(free_tasks_nums[b].item())
            if free_tasks_num > 0:
                delivery_coords = free_tasks_data[b, :free_tasks_num, 2:4].long()
                all_delivery_coords.append(delivery_coords)
                all_delivery_batch_idx.extend([b] * free_tasks_num)
                
                batch_delivery_indices = torch.arange(free_tasks_num, device=device) + base_idx + current_delivery_idx
                delivery_virtual_indices.append(batch_delivery_indices)
                current_delivery_idx += free_tasks_num
            
            # Delivering tasks的delivery位置
            delivering_tasks_num = int(obs["delivering_tasks_num"][b].item())
            if delivering_tasks_num > 0:
                delivering_delivery_coords = obs["delivering_tasks"][b, :delivering_tasks_num, 3:5].long()
                all_delivery_coords.append(delivering_delivery_coords)
                all_delivery_batch_idx.extend([b] * delivering_tasks_num)
                
                batch_delivery_indices = torch.arange(delivering_tasks_num, device=device) + base_idx + current_delivery_idx
                delivery_virtual_indices.append(batch_delivery_indices)
        
        if all_delivery_coords:
            all_delivery_coords = torch.cat(all_delivery_coords, dim=0)
            all_delivery_batch_idx = torch.tensor(all_delivery_batch_idx, device=device)
            delivery_virtual_indices = torch.cat(delivery_virtual_indices, dim=0)
            
            delivery_grid_indices = (node_mappings['grid_base'][all_delivery_batch_idx].squeeze(1) + 
                                    all_delivery_coords[:, 0] * width + all_delivery_coords[:, 1])
            
            delivery_grid_edges = torch.stack([
                torch.cat([delivery_virtual_indices, delivery_grid_indices]),
                torch.cat([delivery_grid_indices, delivery_virtual_indices])
            ], dim=0)
            hetero_edges['delivery_grid'].append(delivery_grid_edges)
        
        # 4. Agent-Delivery edges: delivering agent与其对应的delivery位置连接
        agent_delivery_sources = []
        agent_delivery_targets = []
        
        for b in range(batch_size):
            free_agents_num = int(obs["free_agents_num"][b].item())
            delivering_agents_num = int(obs["delivering_agents_num"][b].item())
            delivering_tasks_num = int(obs["delivering_tasks_num"][b].item())
            
            if delivering_agents_num > 0 and delivering_tasks_num > 0:
                # Delivering agent的虚拟节点索引（在所有agents中的位置）
                delivering_agent_base = node_mappings['agents_base'][b] + free_agents_num
                delivering_agent_indices = torch.arange(delivering_agents_num, device=device) + delivering_agent_base
                
                # 对应的delivery位置索引
                free_tasks_num = int(free_tasks_nums[b].item())
                delivery_base = node_mappings['deliveries_base'][b] + free_tasks_num  # 跳过free tasks的delivery
                delivery_indices = torch.arange(delivering_tasks_num, device=device) + delivery_base
                
                # 假设agent与task一一对应（需要根据实际逻辑调整）
                min_pairs = min(delivering_agents_num, delivering_tasks_num)
                if min_pairs > 0:
                    agent_delivery_sources.extend([delivering_agent_indices[:min_pairs], delivery_indices[:min_pairs]])
                    agent_delivery_targets.extend([delivery_indices[:min_pairs], delivering_agent_indices[:min_pairs]])
        
        if agent_delivery_sources and len(agent_delivery_sources) > 0:
            agent_delivery_edges = torch.stack([
                torch.cat(agent_delivery_sources),
                torch.cat(agent_delivery_targets)
            ], dim=0)
            hetero_edges['agent_delivery'].append(agent_delivery_edges)
        
        # 5. Pickup-Delivery edges: 每个free task的pickup与delivery连接
        pickup_delivery_sources = []
        pickup_delivery_targets = []
        
        for b in range(batch_size):
            free_tasks_num = int(free_tasks_nums[b].item())
            if free_tasks_num > 0:
                pickup_base = node_mappings['pickups_base'][b]
                delivery_base = node_mappings['deliveries_base'][b]
                
                pickup_indices = torch.arange(free_tasks_num, device=device) + pickup_base
                delivery_indices = torch.arange(free_tasks_num, device=device) + delivery_base
                
                pickup_delivery_sources.extend([pickup_indices, delivery_indices])
                pickup_delivery_targets.extend([delivery_indices, pickup_indices])
        
        if pickup_delivery_sources and len(pickup_delivery_sources) > 0:
            pickup_delivery_edges = torch.stack([
                torch.cat(pickup_delivery_sources),
                torch.cat(pickup_delivery_targets)
            ], dim=0)
            hetero_edges['pickup_delivery'].append(pickup_delivery_edges)
        
        # 合并所有边并去重
        final_hetero_edges = {}
        for edge_type, edge_list in hetero_edges.items():
            if edge_list:
                combined_edges = torch.cat(edge_list, dim=1)
                # 使用PyG的coalesce去重和排序
                final_hetero_edges[edge_type] = coalesce(combined_edges, num_nodes=None)[0]
            else:
                final_hetero_edges[edge_type] = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        return final_hetero_edges
    
    def _create_virtual_node_features(self, obs, batch_size, node_mappings, device):
        """
        向量化创建虚拟节点特征，按照新的虚拟节点结构：
        1. All agents (free + delivering)
        2. All pickup locations (from free tasks)  
        3. All delivery locations (from free tasks + delivering tasks)
        """
        virtual_coords = []
        height, width = obs["grid"].shape[1], obs["grid"].shape[2]
        normalize_factor = torch.tensor([height, width], device=device, dtype=torch.float)
        
        # 1. All agents coordinates (free + delivering)
        for b in range(batch_size):
            # Free agents
            free_agents_num = int(obs["free_agents_num"][b].item())
            if free_agents_num > 0:
                free_coords = obs["free_agents"][b, :free_agents_num, :2].float() / normalize_factor
                virtual_coords.append(free_coords)
            
            # Delivering agents
            delivering_agents_num = int(obs["delivering_agents_num"][b].item())
            if delivering_agents_num > 0:
                delivering_coords = obs["delivering_agents"][b, :delivering_agents_num, :2].float() / normalize_factor
                virtual_coords.append(delivering_coords)
        
        # 2. Pickup locations (only from free tasks)
        for b in range(batch_size):
            free_tasks_num = int(obs["free_tasks_num"][b].item())
            if free_tasks_num > 0:
                pickup_coords = obs["free_tasks"][b, :free_tasks_num, :2].float() / normalize_factor
                virtual_coords.append(pickup_coords)
        
        # 3. Delivery locations (from free tasks + delivering tasks)
        for b in range(batch_size):
            # Free tasks delivery locations
            free_tasks_num = int(obs["free_tasks_num"][b].item())
            if free_tasks_num > 0:
                free_delivery_coords = obs["free_tasks"][b, :free_tasks_num, 2:4].float() / normalize_factor
                virtual_coords.append(free_delivery_coords)
            
            # Delivering tasks delivery locations
            delivering_tasks_num = int(obs["delivering_tasks_num"][b].item())
            if delivering_tasks_num > 0:
                delivering_delivery_coords = obs["delivering_tasks"][b, :delivering_tasks_num, 3:5].float() / normalize_factor
                virtual_coords.append(delivering_delivery_coords)
        
        # 4. 合并所有虚拟节点坐标并初始化特征
        if virtual_coords:
            all_virtual_coords = torch.cat(virtual_coords, dim=0)  # [total_virtual_nodes, 2]
            virtual_features = self.virtual_node_coord_proj(all_virtual_coords)  # [total_virtual_nodes, hidden_dim]
        else:
            virtual_features = torch.zeros((0, self.hidden_dim), device=device)
        
        return virtual_features
    
    def forward(self, obs, device):
        """
        前向传播 - 使用batch图并行处理
        
        Args:
            obs: 观察数据字典，包含grid和agent/task信息
            device: 设备
            
        Returns:
            dict: 包含各类实体特征的字典
        """
        # 解析观察数据
        grid = obs["grid"]  # [batch_size, height, width]
        batch_size, height, width = grid.shape
        num_grid_nodes_per_sample = height * width
        
        # 1. 创建批处理的grid节点特征
        all_grid_coords = []
        batch_indices = []
        
        for b in range(batch_size):
            # 为每个样本创建grid坐标特征
            grid_coords = torch.zeros((num_grid_nodes_per_sample, 2), device=device)
            for i in range(height):
                for j in range(width):
                    idx = i * width + j
                    grid_coords[idx, 0] = i / height  # 归一化x坐标
                    grid_coords[idx, 1] = j / width   # 归一化y坐标
            
            all_grid_coords.append(grid_coords)
            # 记录每个节点属于哪个batch
            batch_indices.extend([b] * num_grid_nodes_per_sample)
        
        # 合并所有batch的grid特征
        batch_grid_coords = torch.cat(all_grid_coords, dim=0)  # [total_grid_nodes, 2]
        batch_indices = torch.tensor(batch_indices, device=device)
        
        # 2. 创建批处理的距离边索引
        batch_distance_edges = {}
        
        for dist in range(1, self.max_distance + 1):
            obs_key = f'sp_mpnn_dist_{dist}'
            dist_key = f'dist_{dist}'
            
            assert obs_key in obs, f"SP-MPNN需要预计算的距离信息，但未找到 {obs_key}"
            
            batch_edges = []
            for b in range(batch_size):
                # 根据obs数据结构获取边索引
                if isinstance(obs[obs_key], (list, tuple)):
                    # 如果是batch形式的列表
                    edges_data = obs[obs_key][b]
                else:
                    # 如果是单个样本或已经是batch tensor，取第b个样本
                    if isinstance(obs[obs_key], torch.Tensor) and obs[obs_key].dim() == 3:
                        # 形状为 [batch_size, 2, max_edges]
                        edges_data = obs[obs_key][b]
                    else:
                        # 形状为 [2, max_edges]，所有batch使用相同的边结构
                        edges_data = obs[obs_key]
                
                if isinstance(edges_data, torch.Tensor):
                    edges_np = edges_data.cpu().numpy()
                else:
                    edges_np = edges_data
                
                if edges_np.shape[1] > 0:
                    # 去除填充的零
                    non_zero_cols = np.any(edges_np != 0, axis=0)
                    if np.any(non_zero_cols):
                        last_nonzero = np.where(non_zero_cols)[0][-1] + 1
                        valid_edges = edges_np[:, :last_nonzero]
                        
                        # 将边索引偏移到正确的batch位置
                        offset = b * num_grid_nodes_per_sample
                        valid_edges = valid_edges + offset
                        
                        batch_edges.append(torch.from_numpy(valid_edges.copy()).to(device).long())
            
            # 合并所有batch的边
            if batch_edges:
                batch_distance_edges[dist_key] = torch.cat(batch_edges, dim=1)
            else:
                batch_distance_edges[dist_key] = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # 3. 初始化所有grid节点特征
        batch_grid_features = self.grid_node_init(batch_grid_coords)  # [total_grid_nodes, hidden_dim]
        
        # 4. 创建节点索引映射
        node_mappings = self._create_node_mappings(obs, batch_size, num_grid_nodes_per_sample, device)
        
        # 5. 向量化创建虚拟节点特征（使用坐标初始化）
        batch_virtual_features = self._create_virtual_node_features(obs, batch_size, node_mappings, device)
        
        # 6. 合并所有节点特征
        all_features = torch.cat([batch_grid_features, batch_virtual_features], dim=0)
        
        # 7. 创建异构边索引（向量化）
        hetero_edges = self._create_hetero_edges_vectorized(obs, batch_size, num_grid_nodes_per_sample, width, device, node_mappings)
        
        # 8. 异构SP-MPNN消息传递（内存优化版本）
        # 预分配复用的张量，避免每层重复创建
        num_nodes = all_features.size(0)
        num_message_types = 1 + len(self.edge_types)  # +1 for self messages
        
        # 预分配张量：聚合后的消息缓冲区 [num_nodes, hidden_dim * num_message_types]
        aggregated_messages = torch.zeros(num_nodes, self.hidden_dim * num_message_types, device=device)
        
        # 预分配可复用的临时张量
        target_messages_buffer = torch.zeros_like(all_features)
        neighbor_counts_buffer = torch.zeros(num_nodes, device=device)
        
        for layer in range(self.num_layers):
            # Pre-LayerNorm
            normalized_features = self.pre_layer_norms[layer](all_features)
            
            # 重置聚合消息缓冲区
            aggregated_messages = torch.zeros_like(aggregated_messages)
            message_offset = 0
            
            # 1. Self messages (直接复制到缓冲区)
            aggregated_messages[:, message_offset:message_offset + self.hidden_dim] = normalized_features
            message_offset += self.hidden_dim
            
            # 2. Distance-based messages (grid to grid) - 内存优化版本
            for dist in range(1, self.max_distance + 1):
                dist_key = f'dist_{dist}'
                
                # 重置可复用缓冲区（避免in-place操作）
                target_messages_buffer = torch.zeros_like(target_messages_buffer)
                neighbor_counts_buffer = torch.zeros_like(neighbor_counts_buffer)
                
                if (dist_key in batch_distance_edges and 
                    batch_distance_edges[dist_key].numel() > 0 and 
                    batch_distance_edges[dist_key].dim() >= 2 and 
                    batch_distance_edges[dist_key].size(1) > 0):
                    
                    edge_idx = batch_distance_edges[dist_key]
                    row, col = edge_idx
                    
                    # 直接计算消息并聚合，避免创建中间张量
                    source_features = normalized_features[row]
                    target_features = normalized_features[col]
                    combined = torch.cat([target_features, source_features], dim=1)
                    messages = self.hetero_message_mlps[layer][dist_key](combined)
                    
                    # 聚合到缓冲区（避免in-place操作）
                    target_messages_buffer = target_messages_buffer.index_add(0, col, messages)
                    neighbor_counts_buffer = neighbor_counts_buffer.index_add(0, col, torch.ones(col.size(0), device=device))
                    
                    # 计算平均并复制到聚合缓冲区（避免in-place操作）
                    neighbor_counts_buffer = torch.clamp(neighbor_counts_buffer, min=1)
                    target_messages_buffer = target_messages_buffer / neighbor_counts_buffer.unsqueeze(1)
                
                # 复制到聚合缓冲区的对应位置
                aggregated_messages[:, message_offset:message_offset + self.hidden_dim] = target_messages_buffer
                message_offset += self.hidden_dim
            
            # 3. Heterogeneous edge messages - 内存优化版本  
            hetero_edge_types = [et for et in self.edge_types if not et.startswith('dist_')]
            for edge_type in hetero_edge_types:
                # 重置可复用缓冲区（避免in-place操作）
                target_messages_buffer = torch.zeros_like(target_messages_buffer)
                neighbor_counts_buffer = torch.zeros_like(neighbor_counts_buffer)
                
                if (edge_type in hetero_edges and 
                    hetero_edges[edge_type].numel() > 0 and 
                    hetero_edges[edge_type].dim() >= 2 and 
                    hetero_edges[edge_type].size(1) > 0):
                    
                    edge_idx = hetero_edges[edge_type]
                    row, col = edge_idx
                    
                    source_features = normalized_features[row]
                    target_features = normalized_features[col]
                    combined = torch.cat([target_features, source_features], dim=1)
                    messages = self.hetero_message_mlps[layer][edge_type](combined)
                    
                    # 聚合到缓冲区（避免in-place操作）
                    target_messages_buffer = target_messages_buffer.index_add(0, col, messages)
                    neighbor_counts_buffer = neighbor_counts_buffer.index_add(0, col, torch.ones(col.size(0), device=device))
                    
                    # 计算平均（避免in-place操作）
                    neighbor_counts_buffer = torch.clamp(neighbor_counts_buffer, min=1)
                    target_messages_buffer = target_messages_buffer / neighbor_counts_buffer.unsqueeze(1)
                
                # 复制到聚合缓冲区的对应位置
                aggregated_messages[:, message_offset:message_offset + self.hidden_dim] = target_messages_buffer
                message_offset += self.hidden_dim
            
            # 4. FFN处理（使用预聚合的消息）
            new_features = self.hetero_update_mlps[layer](aggregated_messages)
            
            # 5. Dropout + 残差连接
            new_features = self.dropout_layers[layer](new_features)
            all_features = all_features + new_features  # 避免in-place操作，防止梯度计算错误
            
            # 6. 定期清理GPU缓存（每2层清理一次）
            if layer % 2 == 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 清理临时张量
        del aggregated_messages, target_messages_buffer, neighbor_counts_buffer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 9. 从统一特征中提取各类实体特征
        return self._extract_entity_features_vectorized(all_features, obs, batch_size, node_mappings, device)
    
    def _extract_entity_features_vectorized(self, all_features, obs, batch_size, node_mappings, device):
        """
        向量化的实体特征提取，按照新的虚拟节点结构
        """
        max_agents = obs["free_agents"].size(1)
        max_tasks = obs["free_tasks"].size(1)
        
        # 预分配结果张量
        batch_free_agents = torch.zeros((batch_size, max_agents, self.hidden_dim), device=device)
        batch_delivering_agents = torch.zeros((batch_size, max_agents, self.hidden_dim), device=device)
        batch_free_tasks = torch.zeros((batch_size, max_tasks, self.hidden_dim), device=device)
        batch_delivering_tasks = torch.zeros((batch_size, max_tasks, self.hidden_dim), device=device)
        
        # 1. 提取agents特征（从统一的agents区域中分离free和delivering）
        for b in range(batch_size):
            free_agents_num = int(obs["free_agents_num"][b].item())
            delivering_agents_num = int(obs["delivering_agents_num"][b].item())
            total_agents = free_agents_num + delivering_agents_num
            
            if total_agents > 0:
                agents_base = node_mappings['agents_base'][b]
                all_agent_features = all_features[agents_base:agents_base + total_agents]
                
                # Free agents: 前free_agents_num个
                if free_agents_num > 0:
                    batch_free_agents[b, :free_agents_num] = all_agent_features[:free_agents_num]
                
                # Delivering agents: 后delivering_agents_num个
                if delivering_agents_num > 0:
                    batch_delivering_agents[b, :delivering_agents_num] = all_agent_features[free_agents_num:free_agents_num + delivering_agents_num]
        
        # 2. 提取Free tasks特征 (组合pickup和delivery特征)
        for b in range(batch_size):
            free_tasks_num = int(obs["free_tasks_num"][b].item())
            if free_tasks_num > 0:
                # Pickup特征
                pickup_base = node_mappings['pickups_base'][b]
                pickup_features = all_features[pickup_base:pickup_base + free_tasks_num]
                
                # Delivery特征 (在delivery区域的前free_tasks_num个)
                delivery_base = node_mappings['deliveries_base'][b]
                delivery_features = all_features[delivery_base:delivery_base + free_tasks_num]
                
                # 组合pickup和delivery特征生成任务特征
                combined_features = torch.cat([pickup_features, delivery_features], dim=1)  # [free_tasks_num, hidden_dim*2]
                task_features = self.free_task_mlp(combined_features)  # [free_tasks_num, hidden_dim]
                
                batch_free_tasks[b, :free_tasks_num] = task_features
        
        # 3. 提取Delivering tasks特征 (只有delivery特征)
        for b in range(batch_size):
            free_tasks_num = int(obs["free_tasks_num"][b].item())
            delivering_tasks_num = int(obs["delivering_tasks_num"][b].item())
            
            if delivering_tasks_num > 0:
                # Delivering tasks的delivery特征（在delivery区域中，跳过free tasks的delivery部分）
                delivery_base = node_mappings['deliveries_base'][b] + free_tasks_num
                delivery_features = all_features[delivery_base:delivery_base + delivering_tasks_num]
                task_features = self.delivering_task_mlp(delivery_features)
                batch_delivering_tasks[b, :delivering_tasks_num] = task_features
        
        return {
            'free_agents': batch_free_agents,
            'delivering_agents': batch_delivering_agents,
            'free_tasks': batch_free_tasks,
            'delivering_tasks': batch_delivering_tasks
        }
    
    def _extract_entity_features(self, all_features, obs, batch_size, num_grid_nodes_per_sample, width, device):
        """
        从统一的节点特征中提取各类实体的特征
        """
        batch_results = []
        current_virtual_idx = batch_size * num_grid_nodes_per_sample  # 虚拟节点起始索引
        
        for b in range(batch_size):
            # 直接提取虚拟节点特征，不再进行额外的消息传递
            
            # Free agents
            free_agents_num = obs["free_agents_num"][b].item()
            free_agent_features = []
            for i in range(int(free_agents_num)):
                free_agent_features.append(all_features[current_virtual_idx])
                current_virtual_idx += 1
            
            # Delivering agents
            delivering_agents_num = obs["delivering_agents_num"][b].item()
            delivering_agent_features = []
            for i in range(int(delivering_agents_num)):
                delivering_agent_features.append(all_features[current_virtual_idx])
                current_virtual_idx += 1
            
            # Free tasks (需要组合pickup和delivery特征)
            free_tasks_num = obs["free_tasks_num"][b].item()
            free_task_features = []
            for i in range(int(free_tasks_num)):
                pickup_feat = all_features[current_virtual_idx]
                delivery_feat = all_features[current_virtual_idx + 1]
                
                # 组合pickup和delivery特征生成任务特征
                task_feat = self.free_task_mlp(torch.cat([pickup_feat, delivery_feat], dim=0))
                free_task_features.append(task_feat)
                
                current_virtual_idx += 2
            
            # Delivering tasks
            delivering_tasks_num = obs["delivering_tasks_num"][b].item()
            delivering_task_features = []
            for i in range(int(delivering_tasks_num)):
                delivery_feat = all_features[current_virtual_idx]
                task_feat = self.delivering_task_mlp(delivery_feat)
                delivering_task_features.append(task_feat)
                current_virtual_idx += 1
            
            # Padding到最大尺寸
            max_agents = obs["free_agents"].size(1)
            max_tasks = obs["free_tasks"].size(1)
            
            # 创建padded tensors
            def pad_features(features, max_size):
                if features:
                    tensor = torch.stack(features)
                    if tensor.size(0) < max_size:
                        pad_size = max_size - tensor.size(0)
                        padding = torch.zeros((pad_size, self.hidden_dim), device=device)
                        tensor = torch.cat([tensor, padding], dim=0)
                    return tensor
                else:
                    return torch.zeros((max_size, self.hidden_dim), device=device)
            
            batch_results.append((
                pad_features(free_agent_features, max_agents),
                pad_features(delivering_agent_features, max_agents),
                pad_features(free_task_features, max_tasks),
                pad_features(delivering_task_features, max_tasks)
            ))
        
        # 组合最终结果
        batch_free_agents = torch.stack([result[0] for result in batch_results])
        batch_delivering_agents = torch.stack([result[1] for result in batch_results])
        batch_free_tasks = torch.stack([result[2] for result in batch_results])
        batch_delivering_tasks = torch.stack([result[3] for result in batch_results])
        
        return {
            'free_agents': batch_free_agents,
            'delivering_agents': batch_delivering_agents,
            'free_tasks': batch_free_tasks,
            'delivering_tasks': batch_delivering_tasks
        }


class EdgeGCNN(nn.Module):
    """
    边聚焦的GIN实现（替换原EdgeGCNN）
    保持边聚焦设计，使用GIN的消息传递机制
    """
    def __init__(self, node_dim=4, edge_dim=128, num_layers=3, edge_attr_dim=3, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.edge_attr_dim = edge_attr_dim
        # 特征投影
        self.node_proj = nn.Linear(node_dim, edge_dim)
        self.edge_proj = nn.Linear(edge_attr_dim, edge_dim)
        # GIN层
        try:
            from torch_geometric.nn import GINEConv
            self.PYG_AVAILABLE = True
        except ImportError:
            self.PYG_AVAILABLE = False
        if self.PYG_AVAILABLE:
            self.gin_layers = nn.ModuleList([
                GINEConv(nn.Sequential(
                    nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim)
                ), edge_dim=edge_dim, train_eps=True) for _ in range(num_layers)
            ])
        else:
            self.gin_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim)
                ) for _ in range(num_layers)
            ])
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(edge_dim) for _ in range(num_layers)
        ])
        # 边分数预测层
        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_dim * 3, edge_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(edge_dim, edge_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_dim // 2, 1)
        )
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    def forward(self, node_features, edge_index, edge_attr, num_agents, num_tasks):
        # 投影特征
        x = self.node_proj(node_features)  # [num_nodes, edge_dim]
        e = self.edge_proj(edge_attr)      # [num_edges, edge_dim]
        # GIN消息传递（双向处理）
        for layer_idx in range(self.num_layers):
            if self.PYG_AVAILABLE:
                # Pass edge attributes to GINEConv
                x_new = self.gin_layers[layer_idx](x, edge_index, edge_attr=e)
            else:
                row, col = edge_index
                neighbor_features = x[col]
                from torch_scatter import scatter_mean
                aggregated = scatter_mean(neighbor_features, row, dim=0, dim_size=x.size(0))
                combined = x + aggregated
                x_new = self.gin_layers[layer_idx](combined)
            x = x + x_new
            x = self.layer_norms[layer_idx](x)
            x = F.relu(x)
        # 从节点特征计算边分数（边聚焦设计）
        row, col = edge_index
        edge_representations = torch.cat([x[row], x[col], e], dim=-1)
        edge_scores = self.edge_predictor(edge_representations).squeeze(-1)
        # edge_scores = torch.clamp(edge_scores, min=-10.0, max=10.0)
        return edge_scores


class UndirectedEdgeGCNN(EdgeGCNN):
    """
    无向版本的边聚焦GIN（EdgeGCNN）
    直接继承EdgeGCNN，forward不做特殊处理，适配无向边输入即可。
    """
    def __init__(self, node_dim, edge_dim=128, num_layers=3, edge_attr_dim=3, **kwargs):
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, num_layers=num_layers, edge_attr_dim=edge_attr_dim)
    def forward(self, node_features, edge_index, edge_attr, num_agents, num_tasks):
        return super().forward(node_features, edge_index, edge_attr, num_agents, num_tasks)


class EdgeNodeLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, edge_attr_dim):
        super().__init__()
        self.enc_eattr = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim + 2*edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.enorm = nn.LayerNorm(edge_dim)
        self.xnorm = nn.LayerNorm(node_dim)

    def forward(self, x, e, edge_index, edge_attr):
        src, dst = edge_index
        ea = self.enc_eattr(edge_attr)
        e_in = torch.cat([x[src], x[dst], e, ea], dim=-1)
        e_new = self.enorm(e + self.edge_mlp(e_in))
        m = torch.zeros(x.size(0), e_new.size(1), device=x.device, dtype=x.dtype)
        m.index_add_(0, dst, e_new)
        x_new = self.xnorm(x + self.node_mlp(torch.cat([x, m], dim=-1)))
        return x_new, e_new


class EdgeNodeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, edge_attr_dim, num_layers=3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_attr_dim = edge_attr_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            EdgeNodeLayer(node_dim, edge_dim, edge_attr_dim) for _ in range(num_layers)
        ])
        self.edge_init = nn.Linear(edge_attr_dim, edge_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, 1),
        )

    def forward(self, node_features, edge_index, edge_attr, num_agents, num_tasks, forward_mask=None):
        x = node_features
        e = self.edge_init(edge_attr)
        for layer in self.layers:
            x, e = layer(x, e, edge_index, edge_attr)

        # If a forward_mask is given, score only those; else score all edges
        if forward_mask is not None:
            idx = forward_mask
            src, dst = edge_index[:, idx]
            ef = e[idx]
        else:
            src, dst = edge_index
            ef = e
        logits = self.score_mlp(torch.cat([x[src], x[dst], ef], dim=-1)).squeeze(-1)
        return logits
