from stable_baselines3.common.policies import BasePolicy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv, EdgeConv, MessagePassing, GATv2Conv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

# 导入优化后的GNN模块
from .gnn import GridGCN, GridGAT, HigherGATLayer, LineGraphGAT, UndirectedHigherGATLayer, UndirectedLineGraphGAT, GridSPMPNN, SelfAttentionGATLayer, EdgeGCNN, UndirectedEdgeGCNN, EdgeNodeGNN
# 导入CNN模块
from .cnn import GridCNNChannels
# 导入工具函数
from .utils import (Sinkhorn, apply_hungarian_algorithm, adjust_no_task_logits,
                   prepare_cnn_channels, process_grid_features, process_higher_graph, calculate_agent_task_scores,
                   generate_action_probabilities, apply_sinkhorn_to_probabilities, create_valid_mask,
                   compute_pretrain_loss, compute_entropy, compute_sinkhorn_log_probs, compute_sigmoid_bce_loss, compute_simplified_edge_loss,
                   compute_logprob_and_entropy_from_L)
# from gnn_check import quick_visualize, visualize_higher_graph  # 注释掉，因为模块不存在


class GNNPolicy(BasePolicy):
    """
    GNN REINFORCE Policy based on PyTorch Geometric
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        grid_feature_dim=2, 
        hidden_dim=128,      
        max_agents=50,      
        max_tasks=500,      
        lower_gnn_type='gcn',  # 默认使用GCN作为低层级GNN
        higher_gnn_type='line_graph',  # 默认使用LineGraphGAT作为高层级GNN
        pretrain_steps=10000,
        fix_div=False,
        not_div=False,
        use_sinkhorn=True,  # 默认使用Sinkhorn算法
        tau=1.0,            # Sinkhorn算法温度参数
        iterations=5,       # Sinkhorn迭代次数
        unassign_threshold=0.3679,  # 默认设为e^-1，用于Sinkhorn算法中的填充值
        invalid_edge_score=-100.0,  # 无效边的分数，设置为一个较小的负值
        use_hungarian_for_deterministic=True,  # 是否在deterministic模式下使用Hungarian算法而非贪心策略
        use_gumbel_hungarian=False,  # Gumbel+Hungarian模式（pretrain用sigmoid+BCE，RL用gumbel+hungarian）
        use_gumbel_sinkhorn=False,  # Gumbel Sinkhorn模式（pretrain用sinkhorn+BCE，RL用gumbel_sinkhorn+hungarian）
        infer_decode_mode='sequential',  # 推理(deteministic)解码模式: sequential 或 hungarian
        rl_policy='row_softmax',  # use_gumbel_hungarian下，RL阶段P的计算：row_softmax 或 sinkhorn
        use_simplified_pretrain_loss=False,  # 是否使用简化的edge-level BCE损失（类似test_gcnn）
        # 新增的GNN参数
        lower_gnn_num_layers=3,   # Lower level GNN层数
        higher_gnn_num_layers=2,  # Higher level GNN层数
        gnn_dropout=0.1,    # GNN dropout率
        gnn_heads=4,        # GAT注意力头数
        edge_combine='add',  # 边特征组合方式：'add' (推荐用于无向边)
        use_undirected=True,    # 是否使用无向图
        max_distance=3,         # SP-MPNN的最大距离k
        self_attention_layers=2,  # Self-attention层数（仅用于self_attention_gat）
        use_sde=False,
        **kwargs
    ):
        # Extract custom kwargs that BasePolicy does not know
        self.rl_n_samples = kwargs.pop('rl_n_samples', 1)

        super(GNNPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            **kwargs
        )
        
        self.lr_schedule = lr_schedule
        self.max_agents = max_agents
        self.max_tasks = max_tasks
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.grid_feature_dim = grid_feature_dim
        self.lower_gnn_type = lower_gnn_type.lower()  # 低层级GNN类型
        self.higher_gnn_type = higher_gnn_type.lower()  # 高层级GNN类型
        
        # 新增的GNN参数
        self.lower_gnn_num_layers = lower_gnn_num_layers
        self.higher_gnn_num_layers = higher_gnn_num_layers
        self.gnn_dropout = gnn_dropout
        self.gnn_heads = gnn_heads
        self.edge_combine = edge_combine
        self.use_undirected = use_undirected
        self.max_distance = max_distance
        self.self_attention_layers = self_attention_layers
        
        # 添加预训练相关参数
        self.pretrain_steps = pretrain_steps
        self.current_step = 0  # 跟踪总训练步数
        self.pretrain_mode = False  # 是否强制使用预训练模式
        self.fix_div = fix_div  # 是否使用固定除数
        self.not_div = not_div  # 是否不进行除法操作
        
        # 添加Sinkhorn相关参数
        self.use_sinkhorn = use_sinkhorn
        self.tau = tau
        self.iterations = iterations
        self.unassign_threshold = unassign_threshold  # 未分配任务的阈值
        self.invalid_edge_score = invalid_edge_score  # 无效边的分数
        self.use_hungarian_for_deterministic = use_hungarian_for_deterministic  # 控制确定性采样方法
        self.use_gumbel_hungarian = use_gumbel_hungarian  # Gumbel+Hungarian模式（pretrain用sigmoid+BCE，RL用gumbel+hungarian）
        self.use_gumbel_sinkhorn = use_gumbel_sinkhorn  # Gumbel Sinkhorn模式（pretrain用sinkhorn+BCE，RL用gumbel_sinkhorn+hungarian）
        self.infer_decode_mode = str(infer_decode_mode).lower()
        if self.infer_decode_mode not in ("sequential", "hungarian"):
            self.infer_decode_mode = "sequential"
        self.rl_policy = rl_policy
        # 多样本只读评估的样本数（用于中心化），已在super前pop
        self.rl_n_samples = int(self.rl_n_samples)
        
        self._build(lr_schedule)
    
    def _build(self, lr_schedule):
        # 低层级GNN选择
        if self.lower_gnn_type == 'gat':
            self.grid_gnn = GridGAT(
                grid_feature_dim=2, 
                hidden_dim=self.hidden_dim,
                num_layers=self.lower_gnn_num_layers,
                dropout=self.gnn_dropout,
                heads=self.gnn_heads
            )
        elif self.lower_gnn_type == 'cnn_channels':
            self.grid_gnn = GridCNNChannels(
                grid_feature_dim=2,
                hidden_dim=self.hidden_dim,
                num_layers=self.lower_gnn_num_layers,
                dropout=self.gnn_dropout,
                input_channels=8,  # 8个通道特征
                use_pretrained=True  # 使用预训练权重
            )
        elif self.lower_gnn_type == 'sp_mpnn':
            self.grid_gnn = GridSPMPNN(
                grid_feature_dim=2,
                hidden_dim=self.hidden_dim,
                num_layers=self.lower_gnn_num_layers,
                dropout=self.gnn_dropout,
                max_distance=self.max_distance  # SP-MPNN的最大距离k
            )
        else:  # 默认使用GCN
            self.grid_gnn = GridGCN(
                grid_feature_dim=2, 
                hidden_dim=self.hidden_dim,
                num_layers=self.lower_gnn_num_layers,
                dropout=self.gnn_dropout
            )
        
        # 高层级GNN选择
        if self.higher_gnn_type == 'gat':
            # 使用GAT作为高层级GNN
            if self.use_undirected:
                self.higher_gnn = UndirectedHigherGATLayer(
                    node_dim=self.hidden_dim, 
                    output_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    dropout=self.gnn_dropout,
                    heads=self.gnn_heads,
                    edge_combine=self.edge_combine,
                    edge_attr_dim=5  # 更新为5维：3维边类型 + 2维距离信息
                )
            else:
                self.higher_gnn = HigherGATLayer(
                    node_dim=self.hidden_dim, 
                    output_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    dropout=self.gnn_dropout,
                    heads=self.gnn_heads,
                    edge_combine=self.edge_combine,
                    edge_attr_dim=5  # 更新为5维：3维边类型 + 2维距离信息
                )
        elif self.higher_gnn_type == 'line_graph':
            # 使用Line Graph GAT
            if self.use_undirected:
                self.higher_gnn = UndirectedLineGraphGAT(
                    node_dim=self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    dropout=self.gnn_dropout,
                    heads=self.gnn_heads,
                    edge_combine=self.edge_combine,
                    edge_attr_dim=5  # 更新为5维：3维边类型 + 2维距离信息
                )
            else:
                self.higher_gnn = LineGraphGAT(
                    node_dim=self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    dropout=self.gnn_dropout,
                    heads=self.gnn_heads,
                    edge_combine=self.edge_combine,
                    edge_attr_dim=5  # 更新为5维：3维边类型 + 2维距离信息
                )
        elif self.higher_gnn_type == 'self_attention_gat':
            # 使用Self-Attention GAT，agent间不连边，通过self-attention交换信息
            self.higher_gnn = SelfAttentionGATLayer(
                node_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=self.higher_gnn_num_layers,
                dropout=self.gnn_dropout,
                heads=self.gnn_heads,
                edge_combine=self.edge_combine,
                edge_attr_dim=2,  # 简化为2维：只包含距离信息，不包含边类型
                self_attention_layers=self.self_attention_layers
            )
        elif self.higher_gnn_type == 'gcnn_match':
            # 使用EdgeGCNN，基于Assignment Graph的GCN方法
            if self.use_undirected:
                self.higher_gnn = UndirectedEdgeGCNN(
                    node_dim=self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    edge_attr_dim=3  # 3维：agent到pickup、pickup到delivery、总距离
                )
            else:
                self.higher_gnn = EdgeGCNN(
                    node_dim=self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    edge_attr_dim=3
                )
        elif self.higher_gnn_type == 'edge_gcnn':
            # 使用EdgeGCNN，基于Assignment Graph的Edge-focused GCN方法
            if self.use_undirected:
                self.higher_gnn = UndirectedEdgeGCNN(
                    node_dim=self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    edge_attr_dim=3
                )
            else:
                self.higher_gnn = EdgeGCNN(
                    node_dim=self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    num_layers=self.higher_gnn_num_layers,
                    edge_attr_dim=3
                )
        elif self.higher_gnn_type == 'edge_node_gnn':
            # 新增：EdgeNodeGNN（仅使用agent->task forward边，3维dist属性）
            self.higher_gnn = EdgeNodeGNN(
                node_dim=self.hidden_dim,
                edge_dim=self.hidden_dim,
                edge_attr_dim=3,
                num_layers=self.higher_gnn_num_layers
            )
        else:
            raise ValueError(f"Invalid higher_gnn_type: {self.higher_gnn_type}. Supported types: 'gat', 'line_graph', 'self_attention_gat', 'gcnn_match', 'edge_gcnn'")
        
        self.agent_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.gnn_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.pickup_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.gnn_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.delivery_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.gnn_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        
        # 初始化Sinkhorn模块
        if self.use_sinkhorn or self.use_gumbel_sinkhorn or self.use_gumbel_hungarian:
            # Sinkhorn的use_gumbel参数不再使用，由上层逻辑控制
            self.sinkhorn = Sinkhorn(tau=self.tau, iterations=self.iterations, unassign_threshold=self.unassign_threshold, use_gumbel=False)
        
        self.apply(self._init_weights)
        
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def _init_weights(self, module):
        """
        Orthogonal initialization, consistent with stable-baselines3
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _create_grid_graph(self, grid_features, grid_width, grid_height, obstacles=None):
        """
        Create grid graph, excluding obstacles
        grid_features: [batch_size, height, width, feature_dim]
        grid_width, grid_height: Specific width and height of the grid
        obstacles: [batch_size, height, width] Obstacle mask, True indicates obstacle
        """
        batch_size = grid_features.shape[0]
        
        batch_edge_indices = []
        batch_node_features = []
        batch_masks = []
        
        # Process each batch
        for b in range(batch_size):
            # Convert grid features to node features [height*width, feature_dim]
            node_features = grid_features[b].reshape(-1, grid_features.shape[-1])
            
            # Calculate traversable position mask
            if obstacles is not None:
                valid_mask = ~obstacles[b].reshape(-1)
            else:
                # Assume all positions are traversable
                valid_mask = torch.ones(grid_height * grid_width, dtype=torch.bool, device=grid_features.device)
            
            # 为所有节点创建坐标特征 (x, y)
            coordinates = torch.zeros((grid_height * grid_width, 2), device=grid_features.device)
            for idx in range(grid_height * grid_width):
                row, col = idx // grid_width, idx % grid_width
                coordinates[idx, 0] = row / grid_height  # 归一化x坐标
                coordinates[idx, 1] = col / grid_width   # 归一化y坐标
                
            # 将节点特征与坐标特征拼接
            combined_features = coordinates
            
            # Get features of valid nodes
            valid_features = combined_features[valid_mask]
            
            # Create edges
            edge_index = []
            valid_indices = torch.nonzero(valid_mask).squeeze(1)
            valid_map = {idx.item(): i for i, idx in enumerate(valid_indices)}
            
            # Create edges between valid nodes
            for idx in valid_indices:
                node_idx = idx.item()
                row, col = node_idx // grid_width, node_idx % grid_width
                
                # Four directions: up, down, left, right
                neighbors = [
                    (row-1, col), (row+1, col), 
                    (row, col-1), (row, col+1)
                ]
                
                for nrow, ncol in neighbors:
                    if 0 <= nrow < grid_height and 0 <= ncol < grid_width:
                        neighbor_idx = nrow * grid_width + ncol
                        if valid_mask[neighbor_idx]:
                            # Use index in valid nodes
                            src = valid_map[node_idx]
                            dst = valid_map[neighbor_idx]
                            edge_index.append([src, dst])
            
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, device=grid_features.device).t()
            else:
                # If there are no edges, create empty edge indices
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=grid_features.device)
            
            batch_edge_indices.append(edge_index)
            batch_node_features.append(coordinates)
            batch_masks.append(valid_mask)
        
        return batch_node_features, batch_edge_indices, batch_masks
    
    def _create_higher_graph(self, grid_node_features, grid_valid_masks,
                            free_agent_pos, delivering_agent_pos,
                            free_task_pickup_pos, free_task_delivery_pos,
                            delivering_task_pos, free_agents_num, 
                            delivering_agents_num, free_tasks_num, 
                            delivering_tasks_num, delivering_task_agent_indices=None,
                            free_agents_nearest_tasks=None, grid_norm: float = None):
        """
        Create higher-level graph, connecting agents and tasks
        使用free_agents_nearest_tasks来限制free agent和free task之间的连接
        """
        batch_size = len(grid_node_features)
        
        batch_higher_features = []
        batch_higher_edge_indices = []
        batch_higher_edge_attrs = []
        batch_agent_task_mappings = []
        
        for b in range(batch_size):
            # Extract features for each node
            valid_mask = grid_valid_masks[b]
            valid_indices = torch.nonzero(valid_mask).squeeze(1)
            valid_map = {idx.item(): i for i, idx in enumerate(valid_indices)}
            # 距离归一化因子（与脚本一致，使用 H+W）
            norm_val = 1.0
            if grid_norm is not None and grid_norm > 0:
                norm_val = float(grid_norm)
            
            # Free Agent features - 只处理有效的自由智能体
            free_agent_features = []
            for i in range(free_agents_num[b].long().item()):
                pos = free_agent_pos[b, i]
                if pos >= 0 and pos < len(valid_mask) and valid_mask[pos]:
                    idx = valid_map[pos.item()]
                    node_feature = grid_node_features[b][idx]
                    free_agent_features.append(self.agent_mlp(node_feature))
            
            # Delivering Agent features - 只处理有效的运送中的智能体
            delivering_agent_features = []
            for i in range(delivering_agents_num[b].long().item()):
                pos = delivering_agent_pos[b, i]
                if pos >= 0 and pos < len(valid_mask) and valid_mask[pos]:
                    idx = valid_map[pos.item()]
                    node_feature = grid_node_features[b][idx]
                    delivering_agent_features.append(self.agent_mlp(node_feature))
            
            # Free Task features - 只处理有效的自由任务
            free_task_features = []
            for i in range(free_tasks_num[b].long().item()):
                pickup_pos = free_task_pickup_pos[b, i]
                delivery_pos = free_task_delivery_pos[b, i]
                if pickup_pos >= 0 and delivery_pos >= 0 and pickup_pos < len(valid_mask) and delivery_pos < len(valid_mask) and valid_mask[pickup_pos] and valid_mask[delivery_pos]:
                    pickup_idx = valid_map[pickup_pos.item()]
                    delivery_idx = valid_map[delivery_pos.item()]
                    pickup_feature = self.pickup_mlp(grid_node_features[b][pickup_idx])
                    delivery_feature = self.delivery_mlp(grid_node_features[b][delivery_idx])
                    task_feature = pickup_feature + delivery_feature
                    free_task_features.append(task_feature)
            
            # Delivering Task features - 只处理有效的运送中的任务
            delivering_task_features = []
            for i in range(delivering_tasks_num[b].long().item()):
                pos = delivering_task_pos[b, i]
                if pos >= 0 and pos < len(valid_mask) and valid_mask[pos]:
                    idx = valid_map[pos.item()]
                    node_feature = self.delivery_mlp(grid_node_features[b][idx])
                    delivering_task_features.append(node_feature)
            
            # Combine all node features
            all_node_features = []
            
            # Add Free Agents
            num_free_agents = len(free_agent_features)
            if num_free_agents > 0:
                all_node_features.extend(free_agent_features)
            
            # Add Delivering Agents
            num_delivering_agents = len(delivering_agent_features)
            if num_delivering_agents > 0:
                all_node_features.extend(delivering_agent_features)
            
            # Add Free Tasks
            num_free_tasks = len(free_task_features)
            if num_free_tasks > 0:
                all_node_features.extend(free_task_features)
            
            # Add Delivering Tasks
            num_delivering_tasks = len(delivering_task_features)
            if num_delivering_tasks > 0:
                all_node_features.extend(delivering_task_features)
            
            if len(all_node_features) == 0:
                # If there are no nodes, create an empty graph
                all_node_features = torch.zeros((0, self.hidden_dim), device=grid_node_features[b].device)
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=grid_node_features[b].device)
                # 根据GNN类型确定边属性维度
                edge_attr_dim = 3 if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn'] else 5
                edge_attr = torch.zeros((0, edge_attr_dim), device=grid_node_features[b].device)
                agent_task_mapping = {}
            else:
                # Stack all node features
                all_node_features = torch.stack(all_node_features)
                
                # Create edges
                edge_index = []
                edge_attr = []
                agent_task_mapping = {}
                
                # Connect edges between Free Agents and Free Tasks using free_agents_nearest_tasks
                if free_agents_nearest_tasks is not None:
                    # 使用环境提供的最近任务信息
                    for i in range(num_free_agents):
                        agent_idx = i
                        # 获取当前智能体的最近任务列表 (形状: [max_nearest_tasks, 3])
                        nearest_tasks_info = free_agents_nearest_tasks[b, i]
                        
                        # 遍历最近任务，只连接有效的任务
                        for j_idx in range(len(nearest_tasks_info)):
                            task_info = nearest_tasks_info[j_idx]
                            task_idx_in_nearest = int(task_info[0].item())  # 任务ID
                            agent_to_pickup_dist = task_info[1].item()       # agent到pickup的归一化距离
                            pickup_to_delivery_dist = task_info[2].item()    # pickup到delivery的归一化距离
                            
                            # 检查任务索引是否有效（-1表示无效）
                            if task_idx_in_nearest >= 0 and task_idx_in_nearest < num_free_tasks:
                                # 在图中的节点索引
                                task_node_idx = num_free_agents + num_delivering_agents + task_idx_in_nearest
                                edge_index.append([agent_idx, task_node_idx])
                                
                                # 根据GNN类型创建不同的边属性
                                if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn']:
                                    # SelfAttentionGAT和EdgeGCNN: 简化的3维边属性，包含距离信息
                                    edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                    edge_attr_simple[0] = agent_to_pickup_dist / norm_val      # 归一化距离
                                    edge_attr_simple[1] = pickup_to_delivery_dist / norm_val   # 归一化距离
                                    edge_attr_simple[2] = (agent_to_pickup_dist + pickup_to_delivery_dist) / norm_val
                                    edge_attr.append(edge_attr_simple)
                                    # edge_node_gnn: 添加 task->agent 反向边用于消息传递
                                    if self.higher_gnn_type == 'edge_node_gnn':
                                        edge_index.append([task_node_idx, agent_idx])
                                        edge_attr.append(edge_attr_simple)
                                else:
                                    if self.higher_gnn_type in ['edge_node_gnn']:
                                        edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                        edge_attr_simple[0] = agent_to_pickup_dist / norm_val
                                        edge_attr_simple[1] = pickup_to_delivery_dist / norm_val
                                        edge_attr_simple[2] = (agent_to_pickup_dist + pickup_to_delivery_dist) / norm_val
                                        edge_attr.append(edge_attr_simple)
                                        # 反向边（task->agent）
                                        edge_index.append([task_node_idx, agent_idx])
                                        edge_attr.append(edge_attr_simple)
                                    else:
                                        # 其他GNN: 5维边属性，包含边类型和距离信息
                                        edge_attr_with_dist = torch.zeros(5, device=all_node_features.device)
                                        edge_attr_with_dist[1] = 1.0  # free_agent-free_task边 (one-hot)
                                        edge_attr_with_dist[3] = agent_to_pickup_dist      # agent到pickup距离
                                        edge_attr_with_dist[4] = pickup_to_delivery_dist   # pickup到delivery距离
                                        edge_attr.append(edge_attr_with_dist)
                                
                                # Record mapping relationship - 使用任务在free_tasks中的索引
                                if i not in agent_task_mapping:
                                    agent_task_mapping[i] = []
                                agent_task_mapping[i].append(task_idx_in_nearest)
                else:
                    # 如果没有提供最近任务信息，回退到原来的全连接方式
                    for i in range(num_free_agents):
                        for j in range(num_free_tasks):
                            agent_idx = i
                            task_idx = num_free_agents + num_delivering_agents + j
                            edge_index.append([agent_idx, task_idx])
                            
                            # 根据GNN类型创建不同的边属性
                            if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn']:
                                # SelfAttentionGAT和EdgeGCNN: 简化的3维边属性，无距离信息时用0填充
                                edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                # 距离信息用0填充（fallback情况下没有距离信息）
                                edge_attr.append(edge_attr_simple)
                                # edge_node_gnn: 添加反向 task->agent 边
                                if self.higher_gnn_type == 'edge_node_gnn':
                                    edge_index.append([task_idx, agent_idx])
                                    edge_attr.append(edge_attr_simple)
                            else:
                                if self.higher_gnn_type in ['edge_node_gnn']:
                                    edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                    edge_attr.append(edge_attr_simple)
                                    # 反向边（task->agent）
                                    edge_index.append([task_idx, agent_idx])
                                    edge_attr.append(edge_attr_simple)
                                else:
                                    # 其他GNN: 5维边属性，包含边类型和距离信息
                                    edge_type = torch.zeros(5, device=all_node_features.device)
                                    edge_type[1] = 1.0  # free_agent-free_task边
                                    # 距离信息用0填充（fallback情况下没有距离信息）
                                    edge_attr.append(edge_type)
                            
                            # Record mapping relationship
                            if i not in agent_task_mapping:
                                agent_task_mapping[i] = []
                            agent_task_mapping[i].append(j)
                
                # Connect edges between Delivering Agents and their corresponding Delivering Tasks
                if delivering_task_agent_indices is not None:
                    # Use agent indices from delivering_task_agent_indices
                    for i in range(num_delivering_tasks):
                        if i < delivering_tasks_num[b] and i < len(delivering_task_agent_indices[b]):
                            delivering_agent_idx = delivering_task_agent_indices[b][i].item()
                            if delivering_agent_idx < num_delivering_agents:
                                agent_idx = num_free_agents + delivering_agent_idx
                                task_idx = num_free_agents + num_delivering_agents + num_free_tasks + i
                                edge_index.append([agent_idx, task_idx])
                                
                                # 根据GNN类型创建不同的边属性
                                if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn']:
                                    # SelfAttentionGAT和EdgeGCNN: 简化的3维边属性，距离信息暂时用0填充
                                    edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                    # TODO: 后续可以添加delivering agent到delivery位置的距离信息
                                    edge_attr.append(edge_attr_simple)
                                else:
                                    if self.higher_gnn_type in ['edge_node_gnn']:
                                        edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                        edge_attr.append(edge_attr_simple)
                                    else:
                                        # 其他GNN: 5维边属性，包含边类型
                                        edge_type = torch.zeros(5, device=all_node_features.device)
                                        edge_type[2] = 1.0  # delivering_agent-delivering_task边
                                        # 距离信息暂时用0填充
                                        edge_attr.append(edge_type)
                else:
                    # Fallback to the old method if no agent indices are provided
                    for i in range(min(num_delivering_agents, num_delivering_tasks)):
                        agent_idx = num_free_agents + i
                        task_idx = num_free_agents + num_delivering_agents + num_free_tasks + i
                        edge_index.append([agent_idx, task_idx])
                        
                        # 根据GNN类型创建不同的边属性
                        if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn']:
                            # SelfAttentionGAT和EdgeGCNN: 简化的3维边属性，距离信息暂时用0填充
                            edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                            edge_attr.append(edge_attr_simple)
                        else:
                            if self.higher_gnn_type in ['edge_node_gnn']:
                                edge_attr_simple = torch.zeros(3, device=all_node_features.device)
                                edge_attr.append(edge_attr_simple)
                            else:
                                # 其他GNN: 5维边属性，包含边类型
                                edge_type = torch.zeros(5, device=all_node_features.device)
                                edge_type[2] = 1.0  # delivering_agent-delivering_task边
                                # 距离信息暂时用0填充
                                edge_attr.append(edge_type)
                
                # Connect edges between all pairs of Agents（edge_node_gnn 也添加，但不区分边种类，使用3维零向量）
                if self.higher_gnn_type not in ['self_attention_gat', 'gcnn_match']:
                    for i in range(num_free_agents + num_delivering_agents):
                        for j in range(i+1, num_free_agents + num_delivering_agents):
                            edge_index.append([i, j])
                            if self.higher_gnn_type in ['edge_node_gnn']:
                                edge_attr.append(torch.zeros(3, device=all_node_features.device))
                            else:
                                # Agent to agent edges - type 0（5维 one-hot 保持旧逻辑）
                                edge_type = torch.zeros(5, device=all_node_features.device)
                                edge_type[0] = 1.0
                                edge_attr.append(edge_type)
                
                if len(edge_index) > 0:
                    edge_index = torch.tensor(edge_index, device=all_node_features.device).t()
                    edge_attr = torch.stack(edge_attr)
                else:
                    # If there are no edges, create empty edge indices and features
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=all_node_features.device)
                    # 根据GNN类型确定边属性维度
                    edge_attr_dim = 3 if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn'] else 5
                    edge_attr = torch.zeros((0, edge_attr_dim), device=all_node_features.device)
            
            batch_higher_features.append(all_node_features)
            batch_higher_edge_indices.append(edge_index)
            batch_higher_edge_attrs.append(edge_attr)
            batch_agent_task_mappings.append((agent_task_mapping, num_free_agents, num_free_tasks))
        
        return batch_higher_features, batch_higher_edge_indices, batch_higher_edge_attrs, batch_agent_task_mappings
    
    def unpack_obs(self, obs):
        """
        Parse observed environment state
        Using specific grid dimensions obtained from the environment
        """
        # Assume obs is a dictionary containing necessary fields
        if isinstance(obs, dict):
            # Extract components from the dictionary
            grid = obs["grid"]  # [batch_size, height, width] - 2D grid where -1 indicates obstacles
            grid_height, grid_width = grid.shape[1], grid.shape[2]
            
            # Extract grid features and obstacle information
            grid_features = grid.unsqueeze(-1)  # [batch_size, height, width, 1]
            
            obstacles = grid == -1  
            
            # Extract other necessary information from the dictionary
            free_agents = obs["free_agents"]  # [batch_size, num_free_agents, 3]
            delivering_agents = obs["delivering_agents"]  # [batch_size, num_delivering_agents, 3]
            free_tasks = obs["free_tasks"]  # [batch_size, num_free_tasks, 4]
            delivering_tasks = obs["delivering_tasks"]  # [batch_size, num_delivering_tasks, 5]
            
            # Get quantity information
            free_agents_num = obs["free_agents_num"]
            delivering_agents_num = obs["delivering_agents_num"]
            free_tasks_num = obs["free_tasks_num"]
            delivering_tasks_num = obs["delivering_tasks_num"]
            
            # Extract free_agents_nearest_tasks information
            free_agents_nearest_tasks = obs.get("free_agents_nearest_tasks", None)
            
            # Extract position information
            free_agent_pos = torch.zeros_like(free_agents[..., 0], dtype=torch.long)
            delivering_agent_pos = torch.zeros_like(delivering_agents[..., 0], dtype=torch.long)
            
            # Convert coordinates to linear indices
            for b in range(grid.shape[0]):
                for i in range(free_agents_num[b].long().item()):
                    x, y = free_agents[b, i, :2].long()
                    free_agent_pos[b, i] = x * grid_width + y
                
                for i in range(delivering_agents_num[b].long().item()):
                    x, y = delivering_agents[b, i, :2].long()
                    delivering_agent_pos[b, i] = x * grid_width + y
            
            # Extract task information
            free_task_pickup_pos = torch.zeros((grid.shape[0], free_tasks.shape[1]), dtype=torch.long, device=grid.device)
            free_task_delivery_pos = torch.zeros((grid.shape[0], free_tasks.shape[1]), dtype=torch.long, device=grid.device)
            
            for b in range(grid.shape[0]):
                for i in range(free_tasks_num[b].long().item()):
                    pickup_x, pickup_y = free_tasks[b, i, 0:2].long()
                    delivery_x, delivery_y = free_tasks[b, i, 2:4].long()
                    
                    free_task_pickup_pos[b, i] = pickup_x * grid_width + pickup_y
                    free_task_delivery_pos[b, i] = delivery_x * grid_width + delivery_y
            
            # Process delivering tasks - with agent indices
            delivering_task_pos = torch.zeros((grid.shape[0], delivering_tasks.shape[1]), dtype=torch.long, device=grid.device)
            delivering_task_agent_indices = torch.zeros((grid.shape[0], delivering_tasks.shape[1]), dtype=torch.long, device=grid.device)
            
            for b in range(grid.shape[0]):
                for i in range(delivering_tasks_num[b].long().item()):
                    # First dimension is delivering agent index
                    agent_idx = delivering_tasks[b, i, 0].long()
                    # Position information (delivery coordinates)
                    delivery_x, delivery_y = delivering_tasks[b, i, 3:5].long()
                    delivering_task_pos[b, i] = delivery_x * grid_width + delivery_y
                    delivering_task_agent_indices[b, i] = agent_idx
            
            return (grid_features, grid_width, grid_height, obstacles,
                    free_agent_pos, delivering_agent_pos,
                    free_task_pickup_pos, free_task_delivery_pos, delivering_task_pos,
                    free_agents_num.long(), delivering_agents_num.long(), free_tasks_num.long(), delivering_tasks_num.long(), 
                    delivering_task_agent_indices, free_agents_nearest_tasks)
        else:
            # Maintain original processing logic, assuming obs is a tensor
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            
            # Assume grid information is contained in the front part of the observation space
            grid_data = obs[..., :self.observation_space.shape[0]-self.max_agents-4]
            
            # Assume the grid is square, and get specific dimensions from the environment
            grid_height = grid_width = int(np.sqrt(grid_data.shape[-1] // self.grid_feature_dim))
            
            # Reshape grid data to get grid features
            grid_features = grid_data.reshape(batch_size, grid_height, grid_width, self.grid_feature_dim)
            
            # Extract obstacle positions (assuming the first feature indicates whether it's an obstacle)
            obstacles = grid_features[..., 0] < 0  # Adjust according to actual situation
            
            # Extract other information
            free_agents_num = obs[..., -4].long()
            delivering_agents_num = obs[..., -3].long()
            free_tasks_num = obs[..., -2].long()
            delivering_tasks_num = torch.zeros_like(free_agents_num)  # 假设没有delivering_tasks的信息
            
            # Extract position information
            agent_pos_data = obs[..., -1-self.max_agents:-1]
            free_agent_pos = agent_pos_data[..., :free_agents_num.max()].long()
            delivering_agent_pos = agent_pos_data[..., free_agents_num.max():free_agents_num.max()+delivering_agents_num.max()].long()
            
            # This needs to be parsed according to the actual data structure of task positions
            # Simplified processing, actually need to extract from the environment
            free_task_pickup_pos = torch.zeros((batch_size, free_tasks_num.max()), dtype=torch.long, device=obs.device)
            free_task_delivery_pos = torch.zeros((batch_size, free_tasks_num.max()), dtype=torch.long, device=obs.device)
            delivering_task_pos = torch.zeros((batch_size, delivering_agents_num.max()), dtype=torch.long, device=obs.device)
            delivering_task_agent_indices = None  # Not available in tensor format
            free_agents_nearest_tasks = None  # Not available in tensor format
            
            return (grid_features, grid_width, grid_height, obstacles,
                    free_agent_pos, delivering_agent_pos,
                    free_task_pickup_pos, free_task_delivery_pos, delivering_task_pos,
                    free_agents_num.long(), delivering_agents_num.long(), free_tasks_num.long(), delivering_tasks_num.long(), 
                    delivering_task_agent_indices, free_agents_nearest_tasks)

    def _compute_policy_features(self, obs):
        """
        计算策略特征的共同逻辑，被forward和evaluate_actions共同使用
        
        :param obs: 观察数据
        :return: 包含所有必要特征的元组
        """
        # 如果使用SP-MPNN，使用两步框架但第一步直接输出实体特征
        if self.lower_gnn_type == 'sp_mpnn':
            # 第一步：SP-MPNN直接处理观察数据并返回所有实体的特征
            entity_features = self.grid_gnn(obs, obs["grid"].device)
            
            # 从SP-MPNN的输出中提取各类实体特征
            free_agent_features = entity_features['free_agents']
            delivering_agent_features = entity_features['delivering_agents'] 
            free_task_features = entity_features['free_tasks']
            delivering_task_features = entity_features['delivering_tasks']
            
            # 提取数量信息
            free_agents_num = obs["free_agents_num"].long()
            delivering_agents_num = obs["delivering_agents_num"].long()
            free_tasks_num = obs["free_tasks_num"].long()
            delivering_tasks_num = obs["delivering_tasks_num"].long()
            
            # 第二步：创建高层级图，但直接使用SP-MPNN输出的实体特征
            batch_size = free_agent_features.size(0)
            device = obs["grid"].device
            
            # 构建高层级图的节点特征和边索引
            batch_higher_features = []
            batch_higher_edge_indices = []
            batch_higher_edge_attrs = []
            batch_agent_task_mappings = []
            
            # 获取free_agents_nearest_tasks信息
            free_agents_nearest_tasks = obs.get("free_agents_nearest_tasks", None)
            
            for b in range(batch_size):
                # 使用SP-MPNN输出的实体特征构建高层级图
                # Free Agent features - 直接使用SP-MPNN的输出
                sp_free_agent_features = []
                for i in range(free_agents_num[b].item()):
                    sp_free_agent_features.append(free_agent_features[b, i])
                
                # Delivering Agent features - 直接使用SP-MPNN的输出
                sp_delivering_agent_features = []
                for i in range(delivering_agents_num[b].item()):
                    sp_delivering_agent_features.append(delivering_agent_features[b, i])
                
                # Free Task features - 直接使用SP-MPNN的输出
                sp_free_task_features = []
                for i in range(free_tasks_num[b].item()):
                    sp_free_task_features.append(free_task_features[b, i])
                
                # Delivering Task features - 直接使用SP-MPNN的输出
                sp_delivering_task_features = []
                for i in range(delivering_tasks_num[b].item()):
                    sp_delivering_task_features.append(delivering_task_features[b, i])
                
                # Combine all node features
                all_node_features = []
                
                # Add Free Agents
                num_free_agents = len(sp_free_agent_features)
                if num_free_agents > 0:
                    all_node_features.extend(sp_free_agent_features)
                
                # Add Delivering Agents
                num_delivering_agents = len(sp_delivering_agent_features)
                if num_delivering_agents > 0:
                    all_node_features.extend(sp_delivering_agent_features)
                
                # Add Free Tasks
                num_free_tasks = len(sp_free_task_features)
                if num_free_tasks > 0:
                    all_node_features.extend(sp_free_task_features)
                
                # Add Delivering Tasks
                num_delivering_tasks = len(sp_delivering_task_features)
                if num_delivering_tasks > 0:
                    all_node_features.extend(sp_delivering_task_features)
                
                if len(all_node_features) == 0:
                    # If there are no nodes, create an empty graph
                    all_node_features = torch.zeros((0, self.hidden_dim), device=device)
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    # 根据GNN类型确定边属性维度
                    edge_attr_dim = 3 if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn'] else 5
                    edge_attr = torch.zeros((0, edge_attr_dim), device=device)
                    agent_task_mapping = {}
                else:
                    # Stack all node features
                    all_node_features = torch.stack(all_node_features)
                    
                    # Create edges（与原来的逻辑相同）
                    edge_index = []
                    edge_attr = []
                    agent_task_mapping = {}
                    
                    # Connect edges between Free Agents and Free Tasks using free_agents_nearest_tasks
                    if free_agents_nearest_tasks is not None:
                        # 使用环境提供的最近任务信息
                        for i in range(num_free_agents):
                            agent_idx = i
                            # 获取当前智能体的最近任务列表 (形状: [max_nearest_tasks, 3])
                            nearest_tasks_info = free_agents_nearest_tasks[b, i]
                            
                            # 遍历最近任务，只连接有效的任务
                            for j_idx in range(len(nearest_tasks_info)):
                                task_info = nearest_tasks_info[j_idx]
                                task_idx_in_nearest = int(task_info[0].item())  # 任务ID
                                agent_to_pickup_dist = task_info[1].item()       # agent到pickup的归一化距离
                                pickup_to_delivery_dist = task_info[2].item()    # pickup到delivery的归一化距离
                                
                                # 检查任务索引是否有效（-1表示无效）
                                if task_idx_in_nearest >= 0 and task_idx_in_nearest < num_free_tasks:
                                    # 在图中的节点索引
                                    task_node_idx = num_free_agents + num_delivering_agents + task_idx_in_nearest
                                    edge_index.append([agent_idx, task_node_idx])
                                    
                                    # 根据GNN类型创建不同的边属性
                                    if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn']:
                                        # SelfAttentionGAT和EdgeGCNN: 简化的3维边属性，包含距离信息
                                        edge_attr_simple = torch.zeros(3, device=device)
                                        edge_attr_simple[0] = agent_to_pickup_dist      # agent到pickup距离
                                        edge_attr_simple[1] = pickup_to_delivery_dist   # pickup到delivery距离
                                        edge_attr_simple[2] = agent_to_pickup_dist + pickup_to_delivery_dist  # 总距离
                                        edge_attr.append(edge_attr_simple)
                                    else:
                                        # 其他GNN: 5维边属性，包含边类型和距离信息
                                        edge_attr_with_dist = torch.zeros(5, device=device)
                                        edge_attr_with_dist[1] = 1.0  # free_agent-free_task边 (one-hot)
                                        edge_attr_with_dist[3] = agent_to_pickup_dist      # agent到pickup距离
                                        edge_attr_with_dist[4] = pickup_to_delivery_dist   # pickup到delivery距离
                                        edge_attr.append(edge_attr_with_dist)
                                    
                                    # Record mapping relationship - 使用任务在free_tasks中的索引
                                    if i not in agent_task_mapping:
                                        agent_task_mapping[i] = []
                                    agent_task_mapping[i].append(task_idx_in_nearest)
                    
                    # Connect edges between all pairs of Agents (除非使用self_attention_gat/edge_gcnn/edge_node_gnn)
                    if self.higher_gnn_type not in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn']:
                        for i in range(num_free_agents + num_delivering_agents):
                            for j in range(i+1, num_free_agents + num_delivering_agents):
                                edge_index.append([i, j])
                                # Agent to agent edges - type 0
                                edge_type = torch.zeros(5, device=device)
                                edge_type[0] = 1.0  # agent-agent边
                                edge_attr.append(edge_type)
                    
                    if len(edge_index) > 0:
                        edge_index = torch.tensor(edge_index, device=device).t()
                        edge_attr = torch.stack(edge_attr)
                    else:
                        # If there are no edges, create empty edge indices and features
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                        # 根据GNN类型确定边属性维度
                        edge_attr_dim = 3 if self.higher_gnn_type in ['self_attention_gat', 'gcnn_match', 'edge_gcnn', 'edge_node_gnn'] else 5
                        edge_attr = torch.zeros((0, edge_attr_dim), device=device)
                
                batch_higher_features.append(all_node_features)
                batch_higher_edge_indices.append(edge_index)
                batch_higher_edge_attrs.append(edge_attr)
                batch_agent_task_mappings.append((agent_task_mapping, num_free_agents, num_free_tasks))
            
            # 处理高层级图，提取边特征
            batch_edge_features = process_higher_graph(
                self.higher_gnn, batch_higher_features, batch_higher_edge_indices, batch_higher_edge_attrs,
                self.higher_gnn_type, self.output_dim, device, batch_agent_task_mappings
            )
            
            # 计算智能体任务分数
            batch_agent_task_scores = calculate_agent_task_scores(
                batch_edge_features, batch_agent_task_mappings, self.action_net, 
                self.invalid_edge_score, device, self.higher_gnn_type
            )
            
        # else:
        #     # 传统的两步图构建流程
        #     # 解包观察数据
        #     unpacked_obs = self.unpack_obs(obs)
        #     (grid_features, grid_width, grid_height, obstacles,
        #         free_agent_pos, delivering_agent_pos,
        #         free_task_pickup_pos, free_task_delivery_pos, delivering_task_pos,
        #         free_agents_num, delivering_agents_num, free_tasks_num, delivering_tasks_num, 
        #         delivering_task_agent_indices, free_agents_nearest_tasks) = unpacked_obs
            
        #     # 创建网格图
        #     grid_node_features, grid_edge_indices, grid_valid_masks = self._create_grid_graph(
        #         grid_features, grid_width, grid_height, obstacles
        #     )
            
        #     # 准备CNN通道数据
        #     cnn_channels = prepare_cnn_channels(obs, self.lower_gnn_type)
            
        #     # 处理网格特征
        #     processed_grid_features = process_grid_features(
        #         self.grid_gnn, grid_node_features, grid_edge_indices, cnn_channels, self.lower_gnn_type
        #     )
            
        #     # 创建高层级图
        #     higher_node_features, higher_edge_indices, higher_edge_attrs, agent_task_mappings = self._create_higher_graph(
        #         processed_grid_features, grid_valid_masks,
        #         free_agent_pos, delivering_agent_pos,
        #         free_task_pickup_pos, free_task_delivery_pos, delivering_task_pos,
        #         free_agents_num, delivering_agents_num, free_tasks_num, delivering_tasks_num,
        #         delivering_task_agent_indices, free_agents_nearest_tasks,
        #         grid_norm=(grid_height + grid_width)
        #     )
            
        #     # 处理高层级图，提取边特征
        #     batch_edge_features = process_higher_graph(
        #         self.higher_gnn, higher_node_features, higher_edge_indices, higher_edge_attrs,
        #         self.higher_gnn_type, self.output_dim, grid_features.device, agent_task_mappings
        #     )
            
        #     # 计算智能体任务分数
        #     batch_agent_task_scores = calculate_agent_task_scores(
        #         batch_edge_features, agent_task_mappings, self.action_net, 
        #         self.invalid_edge_score, grid_features.device, self.higher_gnn_type
        #     )
                
        #     device = grid_features.device
        
        # 生成动作概率分布
        # 注意：use_sinkhorn时，action_probs包含的是log-cost矩阵，original_scores包含原始分数
        action_probs, original_scores = generate_action_probabilities(
            batch_agent_task_scores, free_agents_num, free_tasks_num, 
            self.use_sinkhorn, device
        )
        
        valid_mask = create_valid_mask(
            batch_size, self.max_agents, self.max_tasks, 
            free_agents_num, free_tasks_num, free_agents_nearest_tasks, device
        )


        if self.use_sinkhorn or self.use_gumbel_sinkhorn or self.use_gumbel_hungarian:
            # 预训练阶段 & Gumbel Hungarian模式使用real domain sinkhorn
            use_real_domain = (self.current_step < self.pretrain_steps or self.pretrain_mode or self.use_gumbel_hungarian)
            
            # Gumbel Sinkhorn模式：根据阶段决定是否添加Gumbel噪声
            # if self.use_gumbel_sinkhorn:
            #     # 预训练阶段：不加Gumbel噪声
            #     # RL阶段：加Gumbel噪声
            #     add_gumbel_noise = not (self.current_step < self.pretrain_steps or self.pretrain_mode)
            # else:
            #     # 传统模式：使用Sinkhorn模块的默认设置
            #     add_gumbel_noise = None
            
            if self.use_gumbel_hungarian:
                use_real_domain = True
                add_gumbel_noise = False
                
            action_probs = apply_sinkhorn_to_probabilities(
                action_probs, self.sinkhorn, free_agents_num, free_tasks_num, valid_mask,
                training=self.training, use_real_domain=use_real_domain, add_gumbel_noise=add_gumbel_noise
            )
        
        # 创建有效掩码
        batch_size = len(action_probs)
        
        
        return (action_probs, original_scores, valid_mask, free_agents_num, free_tasks_num, 
                free_agents_nearest_tasks, device, batch_size)

    def _build_sequential_row_masks(self, valid_mask_b, selected_tasks, num_tasks, no_task_quota, device):
        """
        Build per-agent action mask for no-replacement sequential sampling.
        - Real tasks cannot be repeated.
        - no-task column is controlled by quota (max(0, M-N)).
        """
        mask = torch.ones(num_tasks + 1, dtype=torch.bool, device=device)
        if num_tasks > 0:
            mask[:num_tasks] = ~selected_tasks
        if valid_mask_b is not None:
            row_valid = valid_mask_b[:num_tasks + 1].bool()
            mask = mask & row_valid
        if no_task_quota <= 0:
            mask[num_tasks] = False
        return mask

    def _sequential_no_replacement_policy(
        self,
        original_scores,
        free_agents_num,
        free_tasks_num,
        valid_mask,
        deterministic=False,
        provided_actions=None,
    ):
        """
        Sequential no-replacement policy over rows.
        Returns:
          - actions [B, max_agents]
          - log_probs [B]
          - entropies [B] (mean over active agents)
        """
        batch_size = len(original_scores)
        device = original_scores[0].device if batch_size > 0 else self.device
        actions_out = torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
        log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)
        entropies = torch.zeros(batch_size, device=device, dtype=torch.float32)

        tau = max(float(self.tau), 1e-6)

        for b in range(batch_size):
            M = int(free_agents_num[b].item())
            N = int(free_tasks_num[b].item())
            if M <= 0:
                continue

            actions_out[b].fill_(N)
            selected_tasks = torch.zeros(N, dtype=torch.bool, device=device)
            no_task_quota = max(0, M - N)
            agent_entropy = []

            for i in range(M):
                logits = original_scores[b][i, :N + 1] / tau
                vm = None if valid_mask is None else valid_mask[b, i, :N + 1]
                mask = self._build_sequential_row_masks(vm, selected_tasks, N, no_task_quota, device)

                # Fallback to avoid empty support
                if not mask.any():
                    if no_task_quota > 0:
                        mask[N] = True
                    elif N > 0 and (~selected_tasks).any():
                        mask[:N] = ~selected_tasks
                    else:
                        mask[N] = True

                masked_logits = logits.clone()
                masked_logits[~mask] = -1e9
                dist = Categorical(logits=masked_logits)

                if provided_actions is None:
                    act = torch.argmax(masked_logits) if deterministic else dist.sample()
                else:
                    act_raw = int(provided_actions[b, i].item()) if i < provided_actions.shape[1] else N
                    if 0 <= act_raw < (N + 1) and bool(mask[act_raw]):
                        act = torch.tensor(act_raw, device=device, dtype=torch.long)
                    elif bool(mask[N]):
                        act = torch.tensor(N, device=device, dtype=torch.long)
                    else:
                        act = torch.where(mask)[0][0].long()

                actions_out[b, i] = act
                log_probs[b] = log_probs[b] + dist.log_prob(act)
                agent_entropy.append(dist.entropy())

                a_int = int(act.item())
                if a_int < N:
                    selected_tasks[a_int] = True
                else:
                    no_task_quota = max(0, no_task_quota - 1)

            if agent_entropy:
                entropies[b] = torch.stack(agent_entropy).mean()

        return actions_out, log_probs, entropies

    def forward(self, obs, deterministic=False):
        """
        Forward pass, generating actions
        """
        # 使用共同的特征计算逻辑
        (action_probs, original_scores, valid_mask, free_agents_num, free_tasks_num, 
         free_agents_nearest_tasks, device, batch_size) = self._compute_policy_features(obs)

        obs_id = obs["env_id"]
        
        # 获取专家动作
        if isinstance(obs, dict) and "expert_actions" in obs:
            expert_actions = obs["expert_actions"].long()
        else:
            # 如果没有专家动作，使用空动作
            expert_actions = torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
        
        # 采样动作
        sampled_log_probs = None
        sampled_entropy = None

        if self.use_gumbel_sinkhorn:
            # Gumbel Sinkhorn模式：预训练用Sinkhorn+BCE，RL用Gumbel Sinkhorn+Hungarian
            if self.current_step < self.pretrain_steps or self.pretrain_mode:
                # 预训练阶段：使用专家动作
                actions = expert_actions.clone() if expert_actions is not None else torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
            else:
                # RL阶段：Sinkhorn已经包含Gumbel噪声，使用Hungarian算法硬采样
                actions = self.sample_action(
                    distribution=action_probs,  # 传递Gumbel Sinkhorn处理后的概率分布
                    valid_mask=valid_mask,
                    free_agents_num=free_agents_num,
                    tasks_num=free_tasks_num,
                    deterministic=deterministic,
                    expert_actions=expert_actions,
                    free_agents_nearest_tasks=free_agents_nearest_tasks,
                    use_gumbel_hungarian_mode=True  # 使用Hungarian算法
                )
        elif self.use_sinkhorn and not self.use_gumbel_hungarian:
            if deterministic:
                # 确定性模式（推理时）：不管是否use_gumbel，都不加噪声，将log-cost转换为实数域cost再传递给Hungarian算法
                # exp_cost_scores = []
                # for b in range(len(original_scores)):
                #     # 将log-cost转换为实数域的cost: exp(log-cost)
                #     # 注意：原始分数是log-cost，需要exp变换到实数域再取负值作为代价
                #     exp_cost_matrix = torch.exp(original_scores[b])
                #     exp_cost_scores.append(exp_cost_matrix)
                
                # actions = self.sample_action(
                #     distribution=exp_cost_scores,  # 传递exp变换后的cost矩阵
                #     valid_mask=valid_mask,
                #     free_agents_num=free_agents_num,
                #     tasks_num=free_tasks_num,
                #     deterministic=True,
                #     expert_actions=expert_actions,
                #     free_agents_nearest_tasks=free_agents_nearest_tasks
                # )
                actions = self.sample_action(
                    distribution=action_probs,  # 传递exp变换后的cost矩阵
                    valid_mask=valid_mask,
                    free_agents_num=free_agents_num,
                    tasks_num=free_tasks_num,
                    deterministic=True,
                    expert_actions=expert_actions,
                    free_agents_nearest_tasks=free_agents_nearest_tasks
                )
            else:
                # 随机采样：使用Sinkhorn处理后的概率分布
                actions = self.sample_action(
                    distribution=action_probs,  # 传递Sinkhorn处理后的概率分布
                    valid_mask=valid_mask,
                    free_agents_num=free_agents_num,
                    tasks_num=free_tasks_num,
                    deterministic=False,
                    expert_actions=expert_actions,
                    free_agents_nearest_tasks=free_agents_nearest_tasks
                )
        elif self.use_gumbel_hungarian:
            # 新的Gumbel+Hungarian模式
            if (self.current_step < self.pretrain_steps or self.pretrain_mode) and not deterministic:
                # 预训练阶段：使用专家动作
                actions = expert_actions.clone() if expert_actions is not None else torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
            elif deterministic:
                use_expert = os.environ.get("USE_EXPERT", "False")
                if use_expert == "True":
                    actions = expert_actions.clone() if expert_actions is not None else torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
                elif self.infer_decode_mode == "hungarian":
                    # 仅推理路径使用Hungarian全局解码，不影响训练采样与log_prob定义
                    hungarian_mats = apply_hungarian_algorithm(
                        original_scores, free_agents_num, free_tasks_num, use_probabilities=False
                    )
                    actions = torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
                    for b in range(batch_size):
                        n = int(free_tasks_num[b].item())
                        m = int(free_agents_num[b].item())
                        actions[b].fill_(n)
                        if m <= 0:
                            continue
                        mat = hungarian_mats[b]
                        if mat.numel() == 0:
                            continue
                        rows, cols = mat.nonzero(as_tuple=True)
                        for ri, cj in zip(rows.tolist(), cols.tolist()):
                            if 0 <= ri < m and 0 <= cj < (n + 1):
                                actions[b, ri] = int(cj)
                else:
                    actions, sampled_log_probs, sampled_entropy = self._sequential_no_replacement_policy(
                        original_scores=original_scores,
                        free_agents_num=free_agents_num,
                        free_tasks_num=free_tasks_num,
                        valid_mask=valid_mask,
                        deterministic=True,
                        provided_actions=None,
                    )
            else:
                # 训练模式：使用无放回顺序采样，确保动作可执行且与log_prob完全一致
                actions, sampled_log_probs, sampled_entropy = self._sequential_no_replacement_policy(
                    original_scores=original_scores,
                    free_agents_num=free_agents_num,
                    free_tasks_num=free_tasks_num,
                    valid_mask=valid_mask,
                    deterministic=False,
                    provided_actions=None,
                )
        else:
            actions = self.sample_action(
                distribution=original_scores,  # 直接传递列表
                valid_mask=valid_mask,
                free_agents_num=free_agents_num,
                tasks_num=free_tasks_num,
                deterministic=deterministic,
                expert_actions=expert_actions,
                free_agents_nearest_tasks=free_agents_nearest_tasks
            )
        
        # 计算log概率
        if self.use_gumbel_hungarian and sampled_log_probs is not None:
            log_probs = sampled_log_probs
        else:
            log_probs = torch.zeros(batch_size, device=device)
            for b in range(batch_size):
                agent_log_probs = []
                for i in range(free_agents_num[b].long().item()):
                    if i < action_probs[b].size(0) and i < actions.shape[1]:
                        action = actions[b, i]
                        if 0 <= action < action_probs[b][i].size(0):
                            agent_log_probs.append(torch.log(action_probs[b][i][action] + 1e-10))
                        else:
                            agent_log_probs.append(torch.tensor(-10.0, device=device))
                if len(agent_log_probs) > 0:
                    log_probs[b] = torch.stack(agent_log_probs).sum()
        
        # 生成一个零张量作为值函数返回值
        values = torch.zeros(batch_size, device=device)
        
        # 检查如果原先有[:,:-1]的操作，保留该操作以确保一致性
        if actions.shape[1] > self.max_agents:
            actions = actions[:,:-1]
        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.
        
        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # 使用共同的特征计算逻辑
        (action_probs, original_scores, valid_mask, free_agents_num, free_tasks_num, 
         free_agents_nearest_tasks, device, batch_size) = self._compute_policy_features(obs)
        
        # 使用零张量替代None作为值函数返回值
        values = torch.zeros(batch_size, device=device)
        
        # 检查是否处于预训练阶段
        if self.current_step < self.pretrain_steps or self.pretrain_mode:
            # 预训练阶段：使用与RL一致的无放回顺序策略监督目标（expert NLL）
            if isinstance(obs, dict) and "expert_actions" in obs:
                expert_actions = obs["expert_actions"].long()
            else:
                expert_actions = actions.long()
            _, expert_log_prob, expert_entropy = self._sequential_no_replacement_policy(
                original_scores=original_scores,
                free_agents_num=free_agents_num,
                free_tasks_num=free_tasks_num,
                valid_mask=valid_mask,
                deterministic=False,
                provided_actions=expert_actions,
            )
            # 在REINFORCE.train的pretrain分支里，loss直接取log_prob并最小化
            # 这里返回正的NLL作为"log_prob"占位，保持外部训练代码不改动
            # 为了消除不同样本agent数量差异导致的loss幅度波动，按有效agent数归一化
            agent_denom = torch.clamp(free_agents_num.float(), min=1.0)
            log_prob = (-expert_log_prob) / agent_denom
            entropy = expert_entropy
        else:
            # 标准RL训练阶段：计算动作的对数概率（与前向采样使用同一份 L）
            if self.use_gumbel_hungarian:
                actions = actions.long()
                _, log_prob, entropy = self._sequential_no_replacement_policy(
                    original_scores=original_scores,
                    free_agents_num=free_agents_num,
                    free_tasks_num=free_tasks_num,
                    valid_mask=valid_mask,
                    deterministic=False,
                    provided_actions=actions,
                )
        
        return values, log_prob, entropy
    
    def _predict(self, observation, deterministic=False):
        """
        Generate actions
        """
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions
    
    def predict_values(self, obs):
        """
        Predict state values, since we're using REINFORCE algorithm and baseline is calculated in the environment, return zeros
        """
        # Return a zero tensor instead of None
        if isinstance(obs, dict):
            batch_size = obs["grid"].shape[0]
            device = obs["grid"].device
        else:
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            device = obs.device
            
        return torch.zeros(batch_size, device=device)
        
    def _get_constructor_parameters(self) -> dict:
        """
        Get constructor parameters for saving and loading the model
        Following the approach of BasePolicy and ActorCriticPolicy
        """
        data = super()._get_constructor_parameters()
        
        data.update(dict(
            grid_feature_dim=self.grid_feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            max_agents=self.max_agents,
            max_tasks=self.max_tasks,
            lower_gnn_type=self.lower_gnn_type,
            higher_gnn_type=self.higher_gnn_type,
            pretrain_steps=self.pretrain_steps,
            fix_div=self.fix_div,
            not_div=self.not_div,
            use_sinkhorn=self.use_sinkhorn,
            tau=self.tau,
            iterations=self.iterations,
            unassign_threshold=self.unassign_threshold,
            invalid_edge_score=self.invalid_edge_score,
            use_hungarian_for_deterministic=self.use_hungarian_for_deterministic,  # 添加新参数
            use_gumbel_hungarian=self.use_gumbel_hungarian,  # 添加新参数
            use_gumbel_sinkhorn=self.use_gumbel_sinkhorn,  # 添加新参数
            infer_decode_mode=self.infer_decode_mode,
            # Use dummy scheduler as placeholder
            lr_schedule=self._dummy_schedule,
            lower_gnn_num_layers=self.lower_gnn_num_layers,
            higher_gnn_num_layers=self.higher_gnn_num_layers,
            gnn_dropout=self.gnn_dropout,
            gnn_heads=self.gnn_heads,
            edge_combine=self.edge_combine,
            use_undirected=self.use_undirected,
            max_distance=self.max_distance,
            self_attention_layers=self.self_attention_layers,
            rl_n_samples=self.rl_n_samples,
        ))
        
        return data
    
    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """
        Dummy learning rate scheduler for serialization
        """
        return 0.0

    def sample_action(self, distribution, valid_mask, free_agents_num, tasks_num, deterministic=False, expert_actions=None, free_agents_nearest_tasks=None, use_hungarian_for_gumbel=False, use_gumbel_hungarian_mode=False):
        """
        根据分布和策略采样动作，采用两阶段采样：
        1. 第一阶段：只考虑真实任务，尽可能多地进行agent-task匹配
        2. 第二阶段：将剩余的agent分配到"no task"
        
        :param distribution: 概率分布列表 [batch_size个 tensor, 每个形状为(agents, tasks+1)]
        :param valid_mask: 有效的mask矩阵 [batch_size, max_agents, max_tasks+1]
        :param free_agents_num: 每个批次自由智能体的数量 [batch_size]
        :param tasks_num: 每个批次任务的数量 [batch_size]
        :param deterministic: 是否使用确定性策略
        :param expert_actions: 专家动作，用于预训练
        :param free_agents_nearest_tasks: 每个智能体的最近任务列表 [batch_size, max_agents, max_nearest_tasks]
        :param use_hungarian_for_gumbel: 是否使用Hungarian算法进行Gumbel采样
        :param use_gumbel_hungarian_mode: 是否使用Gumbel+Hungarian模式进行采样
        :return: 采样的动作 [batch_size, max_agents]
        """
        batch_size = len(distribution)
        device = valid_mask.device
        
        # 预训练阶段：使用专家动作
        if (not deterministic) and (self.current_step < self.pretrain_steps or self.pretrain_mode) and expert_actions is not None:
            return expert_actions.clone()
        
        # 准备结果容器
        actions = torch.zeros((batch_size, self.max_agents), device=device, dtype=torch.long)
        
        # 确定性策略
        if deterministic:
            if self.use_hungarian_for_deterministic or use_gumbel_hungarian_mode:
                # 纯Sinkhorn确定性采样：对实数域cost矩阵使用Hungarian算法
                # 注意：这里的distribution已经是exp(log-cost)，是实数域的cost，不是概率分布
                hungarian_matrices = apply_hungarian_algorithm(distribution, free_agents_num, tasks_num, use_probabilities=True)
                
                # 从Hungarian矩阵提取动作
                for b in range(batch_size):
                    if free_agents_num[b] > 0:
                        hungarian_matrix = hungarian_matrices[b]
                        for i in range(free_agents_num[b]):
                            if i < hungarian_matrix.shape[0]:
                                task_idx = torch.argmax(hungarian_matrix[i])
                                if task_idx < tasks_num[b] + 1:
                                    actions[b, i] = task_idx
                                else:
                                    actions[b, i] = tasks_num[b]  # 选择"no task"
            else:
                # 使用两阶段贪心策略
                for b in range(batch_size):
                    num_agents = free_agents_num[b].long().item()
                    num_tasks = tasks_num[b].long().item()
                    
                    # 初始化所有agent为"no task"
                    batch_actions = torch.full((num_agents,), num_tasks, device=device, dtype=torch.long)
                    
                    # 第一阶段：只考虑真实任务进行匹配
                    if num_tasks > 0:
                        # 创建任务可用性掩码
                        available_tasks = torch.ones(num_tasks, device=device, dtype=torch.bool)
                        
                        # 按概率排序智能体，优先处理概率高的智能体
                        agent_priorities = []
                        for i in range(num_agents):
                            if i < distribution[b].shape[0]:
                                # 只考虑真实任务的最大概率（排除no task）
                                max_prob = torch.max(distribution[b][i, :num_tasks]) if num_tasks > 0 else 0.0
                                agent_priorities.append((max_prob.item(), i))
                            else:
                                agent_priorities.append((0.0, i))
                        
                        # 按优先级排序（概率高的优先）
                        agent_priorities.sort(key=lambda x: x[0], reverse=True)
                        
                        for _, i in agent_priorities:
                            if i < distribution[b].shape[0] and available_tasks.any():
                                # 获取当前智能体对真实任务的概率分布
                                task_probs = distribution[b][i, :num_tasks]
                                
                                # 创建只包含最近任务的掩码
                                nearest_task_mask = torch.zeros(num_tasks, device=device, dtype=torch.bool)
                                
                                if free_agents_nearest_tasks is not None:
                                    # 添加最近任务到掩码
                                    nearest_tasks_info = free_agents_nearest_tasks[b, i]
                                    for task_info in nearest_tasks_info:
                                        task_idx_int = int(task_info[0].item())  # 任务ID在第0个位置
                                        if task_idx_int >= 0 and task_idx_int < num_tasks:
                                            nearest_task_mask[task_idx_int] = True
                                else:
                                    # 如果没有最近任务信息，所有任务都可用
                                    nearest_task_mask[:num_tasks] = True
                                
                                # 结合可用性掩码和最近任务掩码
                                combined_mask = available_tasks & nearest_task_mask
                                masked_indices = torch.arange(num_tasks, device=device)[combined_mask]
                                
                                if len(masked_indices) > 0:
                                    # 从可用的最近任务中找出最大概率的任务
                                    available_probs = task_probs[masked_indices]
                                    local_max_idx = torch.argmax(available_probs)
                                    task_idx = masked_indices[local_max_idx]
                                    
                                    # 分配任务并标记为不可用
                                    batch_actions[i] = task_idx
                                    available_tasks[task_idx] = False
                    
                    # 将结果复制到actions中
                    actions[b, :num_agents] = batch_actions
        else:
            # 随机采样策略
            if use_hungarian_for_gumbel and not (self.current_step < self.pretrain_steps or self.pretrain_mode):
                # Gumbel + Sinkhorn + Hungarian采样（非预训练阶段）：对概率矩阵使用Hungarian算法
                hungarian_matrices = apply_hungarian_algorithm(distribution, free_agents_num, tasks_num, use_probabilities=True)
                
                # 从Hungarian矩阵提取动作
                for b in range(batch_size):
                    if free_agents_num[b] > 0:
                        hungarian_matrix = hungarian_matrices[b]
                        for i in range(free_agents_num[b]):
                            if i < hungarian_matrix.shape[0]:
                                task_idx = torch.argmax(hungarian_matrix[i])
                                if task_idx < tasks_num[b] + 1:
                                    actions[b, i] = task_idx
                                else:
                                    actions[b, i] = tasks_num[b]  # 选择"no task"
            elif use_gumbel_hungarian_mode and not (self.current_step < self.pretrain_steps or self.pretrain_mode):
                # Gumbel+Hungarian模式：RL阶段使用Hungarian算法
                hungarian_matrices = apply_hungarian_algorithm(distribution, free_agents_num, tasks_num, use_probabilities=True)
                
                # 从Hungarian矩阵提取动作
                for b in range(batch_size):
                    if free_agents_num[b] > 0:
                        hungarian_matrix = hungarian_matrices[b]
                        for i in range(free_agents_num[b]):
                            if i < hungarian_matrix.shape[0]:
                                task_idx = torch.argmax(hungarian_matrix[i])
                                if task_idx < tasks_num[b] + 1:
                                    actions[b, i] = task_idx
                                else:
                                    actions[b, i] = tasks_num[b]  # 选择"no task"
            elif self.use_sinkhorn:
                # Sinkhorn情况：使用全局贪心采样
                for b in range(batch_size):
                    num_agents = free_agents_num[b].long().item()
                    num_tasks = tasks_num[b].long().item()
                    
                    # 初始化所有agent为"no task"
                    batch_actions = torch.full((num_agents,), num_tasks, device=device, dtype=torch.long)
                
                    if num_agents > 0 and num_tasks >= 0:
                        # 创建概率矩阵的副本用于采样
                        prob_matrix = distribution[b][:num_agents, :num_tasks+1].clone()  # 包括"no task"列
                        
                        # 创建行列可用性掩码
                        available_agents = torch.ones(num_agents, device=device, dtype=torch.bool)
                        available_tasks = torch.ones(num_tasks + 1, device=device, dtype=torch.bool)  # +1 for "no task"
                        
                        # 计算"no task"列可以被选择的次数
                        if num_agents > num_tasks:
                            # agent数量大于task数量：空闲列可以被选择 |agent_num - task_num| 次
                            no_task_quota = num_agents - num_tasks
                        else:
                            # agent数量小于等于task数量：空闲列可以被选择 |task_num - agent_num| 次
                            # 但实际上在这种情况下，我们希望尽量分配真实任务
                            no_task_quota = max(0, num_agents - num_tasks)
                        
                        # 全局贪心采样过程
                        assignments_made = 0
                        while assignments_made < num_agents and available_agents.any():
                            # 在可用的行列范围内找到最大概率
                            masked_prob_matrix = prob_matrix.clone()
                            
                            # 屏蔽不可用的行
                            for i in range(num_agents):
                                if not available_agents[i]:
                                    masked_prob_matrix[i, :] = -float('inf')
                            
                            # 屏蔽不可用的列
                            for j in range(num_tasks + 1):
                                if not available_tasks[j]:
                                    masked_prob_matrix[:, j] = -float('inf')
                            
                            # 特殊处理"no task"列的配额
                            if no_task_quota <= 0:
                                masked_prob_matrix[:, num_tasks] = -float('inf')  # 屏蔽"no task"列
                            
                            # 找到全局最大概率位置
                            flat_idx = torch.argmax(masked_prob_matrix.flatten())
                            agent_idx = flat_idx // (num_tasks + 1)
                            task_idx = flat_idx % (num_tasks + 1)
                            
                            # 检查是否找到有效位置
                            if masked_prob_matrix[agent_idx, task_idx] == -float('inf'):
                                # 没有找到有效位置，随机分配剩余agent到"no task"
                                remaining_agents = torch.nonzero(available_agents).flatten()
                                for remaining_agent in remaining_agents:
                                    batch_actions[remaining_agent] = num_tasks
                                break
                            
                            # 进行分配
                            batch_actions[agent_idx] = task_idx
                            
                            # 标记该agent为不可用
                            available_agents[agent_idx] = False
                            
                            # 处理任务可用性
                            if task_idx == num_tasks:
                                # 选择了"no task"，减少配额
                                no_task_quota -= 1
                                if no_task_quota <= 0:
                                    available_tasks[num_tasks] = False
                            else:
                                # 选择了真实任务，标记该任务为不可用
                                available_tasks[task_idx] = False
                
                            assignments_made += 1
                
                    # 将结果复制到actions中
                    actions[b, :num_agents] = batch_actions
        return actions

    def compute_centered_returns(self, vec_env, obs, actions_current, r0, deterministic=False, cached_log_prob=None, cached_entropy=None):
        """
        Multi-sample update for use_gumbel_hungarian (row_softmax):
        - Compute distribution once from obs.
        - Build K actions (first is actions_current, others sampled with Gumbel+Hungarian).
        - Rewards: use r0 for the first action, evaluate K-1 via env.evaluate_actions_storage.
        - Return (log_prob_all[B,K], centered_all[B,K]).
        """
        if self.rl_n_samples is None or self.rl_n_samples <= 1:
            return None

        (action_probs, original_scores, valid_mask, free_agents_num, free_tasks_num,
         free_agents_nearest_tasks, device, batch_size) = self._compute_policy_features(obs)

        obs_id = obs['env_id']
        K = int(self.rl_n_samples)
        B = batch_size
        Amax = actions_current.shape[1]

        # Sample K-1 extra actions using the same no-replacement policy (aligned with log_prob)
        extra_samples = []
        num_extra = max(0, K - 1)
        # log_prob_all = torch.zeros((B, K), device=device)
        log_prob_all = [cached_log_prob]
        # ent_all = torch.zeros((B, K), device=device)
        ent_all = [cached_entropy]

        invalid_mask = torch.abs(r0+1) < 1e-6

        if num_extra > 0:
            for _ in range(num_extra):
                sampled_actions, log_prob, entropy = self._sequential_no_replacement_policy(
                    original_scores=original_scores,
                    free_agents_num=free_agents_num,
                    free_tasks_num=free_tasks_num,
                    valid_mask=valid_mask,
                    deterministic=deterministic,
                    provided_actions=None,
                )
                extra_samples.append(sampled_actions.detach().clone())
                log_prob_all.append(log_prob)
                ent_all.append(entropy)
        # Assemble actions_all: [B, K, A]
        actions_all = torch.zeros((B, K, Amax), device=device, dtype=torch.long)
        actions_all[:, 0, :] = actions_current
        for idx, s in enumerate(extra_samples):
            actions_all[:, idx + 1, :] = s

        log_prob_all = torch.stack(log_prob_all, dim=1)
        ent_all = torch.stack(ent_all, dim=1)

        # Rewards for K actions: first from r0, others from evaluate_actions_storage
        returns_all = torch.zeros((B, K), device=device, dtype=torch.float32)
        returns_all[:, 0] = r0.detach().to(device=device, dtype=torch.float32)

        actions_dict = {}
        for env_idx in range(B):
            if num_extra <= 0:
                continue
            actions_list = []
            for k in range(1, K):
                ai = [int(x) for x in actions_all[env_idx, k].detach().cpu().tolist()]
                actions_list.append(ai)
            env_id = obs_id[env_idx].item()
            if invalid_mask[env_idx]:
                actions_dict[env_id] = -1
            else:
                actions_dict[env_id] = actions_list
        res = vec_env.env_method('evaluate_actions_storage', [actions_dict], indices=None)
        id2row = {int(obs_id[b].item()): b for b in range(B)}
        for item in res:
            env_idx, results = item
            if results is None:
                continue
            b = id2row[int(env_idx)]
            for k in range(1, K):
                returns_all[b, k] = float(results[k-1][0])

        # Leave-one-out centering across K samples per env
        # For each env b and sample k: A_k = r_k - mean_{j!=k}(r_j)
        # Vectorized: baseline = (sum_r - r_k) / (K-1)
        sum_r = returns_all.sum(dim=1, keepdim=True)
        denom = max(K - 1, 1)
        loo_baseline = (sum_r - returns_all) / denom
        centered_all = (returns_all - loo_baseline).detach()

        log_prob_all[invalid_mask, 1:] = 0
        centered_all[invalid_mask, 1:] = 0
        returns_all[invalid_mask, 1:] = 0

        return log_prob_all, centered_all, returns_all


