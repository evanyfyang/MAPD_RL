import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import math


class Sinkhorn(nn.Module):
    """
    Sinkhorn算法实现，用于双边约束下的最优分配
    
    注意：该实现接受log-cost矩阵作为输入，温度参数tau在对数域中正确应用：
    - tau < 1.0: 锐化分布（减少随机性）
    - tau = 1.0: 原始分布
    - tau > 1.0: 平滑分布（增加随机性）
    """
    def __init__(self, tau=1.0, iterations=5, unassign_threshold=np.exp(-1), use_gumbel=False):
        super(Sinkhorn, self).__init__()
        self.tau = tau
        self.iterations = iterations
        self.unassign_threshold = unassign_threshold
        self.use_gumbel = use_gumbel
    

    def sinkhorn_log(self, score_matrix, num_agents, num_tasks, valid_mask, eps=1e-8):
        """
        Log-space Sinkhorn，与 sinkhorn_log 行为保持一致：
        - 相同的增广规则（M<=N 加虚拟行；M>N 加 no-task 列）
        - 相同的 clamp 到 [-50, 0]，再除以 tau（对应 real 版的 exp(a_aug/tau)）
        - 相同的边缘约束向量 lambda_vec / mu_vec
        - 相同的迭代与最终切片规则
        """
        import torch

        M, N = num_agents, num_tasks
        device = score_matrix.device
        dtype = score_matrix.dtype

        # print(f"[dbg] pre score finite={torch.isfinite(score_matrix).all().item()} "
        #     f"max={float(score_matrix.abs().max().item())}")
        # if score_matrix.requires_grad:
        #     score_matrix.retain_grad()
        #     score_matrix.register_hook(lambda g: print(f"[dbg] backword score finite="
        #                                             f"{torch.isfinite(g).all().item()} "
        #                                             f"max={float(g.abs().max().item())}"))
        if M <= N:
            a_aug = torch.cat(
                [score_matrix,
                torch.full((1, N), self.unassign_threshold, device=device, dtype=dtype)],
                dim=0
            )
            lambda_vec = torch.cat(
                [torch.ones(M, device=device, dtype=dtype),
                torch.tensor([N - M], device=device, dtype=dtype)]
            )
            mu_vec = torch.ones(N, device=device, dtype=dtype)
            valid_mask_local = torch.zeros_like(a_aug, dtype=torch.bool)
            valid_mask_local[:M, :N] = valid_mask[:M, :N].bool()
        else:
            K = M - N
            no_task_cols = torch.full((M, K), self.unassign_threshold, device=device, dtype=dtype)
            a_aug = torch.cat([score_matrix, no_task_cols], dim=1)

            lambda_vec = torch.ones(M, device=device, dtype=dtype)
            mu_vec = torch.ones(N + K, device=device, dtype=dtype)
            # mu_vec = torch.cat(
            #     [torch.ones(N, device=device, dtype=dtype),
            #     torch.tensor([M - N], device=device, dtype=dtype)]
            # )
            valid_mask_local = torch.zeros_like(a_aug, dtype=torch.bool)
            valid_mask_local[:M, :N] = valid_mask[:M, :N].bool()
            valid_mask_local[:M, N:N + K] = True

        v = a_aug[valid_mask_local]
        mu = v.mean()
        sd = v.std(unbiased=False)
        log_M = (a_aug - mu) / (sd + eps)
        log_M = log_M / self.tau                        

        log_lambda = torch.log(lambda_vec.clamp_min(eps)) 
        log_mu = torch.log(mu_vec.clamp_min(eps))

        # ---------- Sinkhorn 迭代（log-space 归一化） ----------
        # 先按列、再按行，与 real 版相同的顺序（列归一 -> 行归一）
        for _ in range(self.iterations):
            # 列归一：使每列和为 mu_j
            col_log_sum = torch.logsumexp(log_M, dim=0, keepdim=True)  # [1, C]
            log_M = log_M + (log_mu.view(1, -1) - col_log_sum)

            # 行归一：使每行和为 lambda_i
            row_log_sum = torch.logsumexp(log_M, dim=1, keepdim=True)  # [R, 1]
            log_M = log_M + (log_lambda.view(-1, 1) - row_log_sum)

        # 回到正域
        s = torch.exp(log_M)

        if M <= N:
            result = s[:M, :N]         # 无显式 no-task 列（与 real 版一致）
        else:
            result = s[:M, :N + K]     # 包含 no-task 列

        # print(f"[dbg] pre result finite={torch.isfinite(result).all().item()} "
        #     f"max={float(result.abs().max().item())}")
        # if result.requires_grad:
        #     result.retain_grad()
        #     result.register_hook(lambda g: print(f"[dbg] backword result finite="
        #                                         f"{torch.isfinite(g).all().item()} "
        #                                         f"max={float(g.abs().max().item())}"))
        return result

    def sinkhorn_real(self, score_matrix, num_agents, num_tasks, eps=1e-5):
        """
        Real domain Sinkhorn-Knopp algorithm
        参考test_gcnn.py的实现，但保持与原本逻辑一致
        
        Args:
            score_matrix: [num_agents, num_tasks] 实数域的分数矩阵，值越大表示匹配越好
            num_agents: agent数量
            num_tasks: task数量
            eps: 数值稳定性参数
            
        Returns:
            doubly_stochastic_matrix: [num_agents, num_tasks+1] 双随机矩阵（包含no-task列）
        """
        M, N = num_agents, num_tasks
        device = score_matrix.device

        print(f"[dbg] pre score finite={torch.isfinite(score_matrix).all().item()} "
                f"max={float(score_matrix.abs().max().item())}")
        if score_matrix.requires_grad:
            score_matrix.retain_grad()
            score_matrix.register_hook(lambda g: print(f"[dbg] backword score finite="
                                                        f"{torch.isfinite(g).all().item()} "
                                                        f"max={float(g.abs().max().item())}"))
        
        # 根据较少的一侧决定增广方式
        if M <= N:
            # agent较少: 添加一个虚拟agent行
            # 将原始分数矩阵与unassign_threshold填充行拼接
            a_aug = torch.cat([
                score_matrix, 
                torch.full((1, N), self.unassign_threshold, device=device)
            ], dim=0)
            
            # 设置约束向量
            lambda_vec = torch.cat([
                torch.ones(M, device=device), 
                torch.tensor([N-M], device=device)
            ], dim=0)
            mu_vec = torch.ones(N, device=device)
        else:
            # task较少: 添加一个虚拟task列（no-task列）
            # 将原始分数矩阵与unassign_threshold填充列拼接
            if N == score_matrix.shape[1]:
                a_aug = torch.cat([
                    score_matrix, 
                    torch.full((M, 1), self.unassign_threshold, device=device)
                ], dim=1)
            else:
                a_aug = score_matrix
                
            # 设置约束向量
            lambda_vec = torch.ones(M, device=device)
            mu_vec = torch.cat([
                torch.ones(N, device=device), 
                torch.tensor([M-N], device=device)
            ], dim=0)
        
        a_aug = a_aug.clamp(min=-50.0, max=0.0)

        # 转换为指数域开始迭代
        s = torch.exp(a_aug / self.tau)  # 应用温度参数
        
        # Sinkhorn迭代
        for _ in range(self.iterations):
            s = (s / (s.sum(dim=0, keepdim=True) + eps)) * mu_vec.view(1, -1)
            s = (s / (s.sum(dim=1, keepdim=True) + eps)) * lambda_vec.view(-1, 1)
            
        
        if M <= N:
            result = s[:M, :N]
        else:
            result = s[:M, :N+1]
        
        print(f"[dbg] pre result finite={torch.isfinite(result).all().item()} "
            f"max={float(result.abs().max().item())}")
        if result.requires_grad:
            result.retain_grad()
            result.register_hook(lambda g: print(f"[dbg] backword result finite="
                                                        f"{torch.isfinite(g).all().item()} "
                                                        f"max={float(g.abs().max().item())}"))
        return result
        # # 根据增广方式返回正确的结果
        # if M <= N:
        #     # 返回原始大小的部分，同时添加no-task列
        #     original_part = s[:M, :N]  # [M, N]
        #     # 创建no-task列：如果有剩余任务分配给虚拟行，则对应的agent应该选择no-task
        #     no_task_col = torch.zeros((M, 1), device=device)
        #     # 虚拟行的分配表示某些任务未被分配，对应的agent选择no-task
        #     if N > M:
        #         # 计算每个agent选择no-task的概率
        #         # 这是基于虚拟行分配的逆向推理
        #         virtual_row_allocation = s[M, :N]  # [N]
        #         # 将虚拟行的分配均匀分布到前M个agent的no-task选择上
        #         no_task_prob = virtual_row_allocation.sum() / M if M > 0 else 0.0
        #         no_task_col.fill_(no_task_prob)
            
        #     result = torch.cat([original_part, no_task_col], dim=1)  # [M, N+1]
        # else:
        #     # task较少的情况：agent > task，直接返回包含no-task列的结果
        #     result = s[:M, :N+1]  # [M, N+1]
        
        # return result
    
    def sinkhorn_log_domain(self, log_matrix, desired_row_sums=None, desired_col_sums=None, iterations=20, eps=1e-8):
        """
        直接在对数域进行Sinkhorn迭代算法计算双边约束下的最优分配
        log_matrix: 输入的对数域分数矩阵，值越大表示匹配越好
        desired_row_sums: 期望的行和，如果为None则默认为全1
        desired_col_sums: 期望的列和，如果为None则默认为全1
        
        注意：矩阵形状可能是非方阵，其中较小维度的元素必须全部分配，
        较大维度的多余元素会被分配到虚拟行/列
        """
        row_len = log_matrix.shape[0] 
        col_len = log_matrix.shape[1] 
        
        # 设置行列和的约束
        if desired_row_sums is None:
            log_desired_row_sums = torch.zeros(row_len, requires_grad=False, device=log_matrix.device)  # log(1) = 0
        else:
            log_desired_row_sums = torch.log(desired_row_sums.float() + eps).to(log_matrix.device)
            
        if desired_col_sums is None:
            log_desired_col_sums = torch.zeros(col_len, requires_grad=False, device=log_matrix.device)  # log(1) = 0
        else:
            log_desired_col_sums = torch.log(desired_col_sums.float() + eps).to(log_matrix.device)
        
        # 直接使用输入的对数域矩阵
        log_M = log_matrix.clone()
        
        for _ in range(iterations):
            # 行归一化：log_M[i,j] = log_M[i,j] + log(desired_row_sum[i]) - log(sum_j exp(log_M[i,j]))
            row_log_sum = torch.logsumexp(log_M, dim=1, keepdim=True)
            log_M = log_M + (log_desired_row_sums.unsqueeze(1) - row_log_sum)
            
            # 列归一化：log_M[i,j] = log_M[i,j] + log(desired_col_sum[j]) - log(sum_i exp(log_M[i,j]))
            col_log_sum = torch.logsumexp(log_M, dim=0, keepdim=True)
            log_M = log_M + (log_desired_col_sums.unsqueeze(0) - col_log_sum)
        
        # 返回概率域的结果
        return torch.exp(log_M)

    def forward(self, logits, free_agents_num, tasks_num, valid_mask,training=True, use_real_domain=False, add_gumbel_noise=None):
        """
        对每个批次应用Sinkhorn算法
        logits: log-cost矩阵 [batch_size, max_agents (or +1), max_tasks (or +1)]
        free_agents_num: 每个批次中自由智能体的数量
        tasks_num: 每个批次中任务的数量
        training: 是否在训练阶段（用于控制是否添加Gumbel噪声）
        use_real_domain: 是否使用real domain的sinkhorn（预训练阶段推荐）
        add_gumbel_noise: 是否添加Gumbel噪声，如果为None则使用self.use_gumbel
        """
        batch_size = logits.size(0)
        distribution = []
        H_t, W_t = logits.size(1), logits.size(2)
        for b in range(batch_size):
            num_agents = free_agents_num[b].long().item()
            num_tasks = tasks_num[b].long().item()
            
            if num_agents > 0 and num_tasks > 0:
                current_logits = logits[b, :num_agents, :num_tasks]
                current_logits = current_logits - current_logits.max()   
                                
                result = self.sinkhorn_log(current_logits, num_agents, num_tasks, valid_mask[b])
                pad_h = H_t - result.size(0)
                pad_w = W_t - result.size(1)
                result = torch.nn.functional.pad(result, (0, pad_w, 0, pad_h))
                distribution.append(result)
        distribution = torch.stack(distribution, dim=0) 
        return distribution


def apply_hungarian_algorithm(input_matrix, free_agents_num, tasks_num, use_probabilities=False):
    """
    应用Hungarian算法进行最优分配
    
    :param input_matrix: 输入矩阵列表（可以是logits或概率分布）
    :param free_agents_num: 每个批次自由智能体的数量 [batch_size]
    :param tasks_num: 每个批次任务的数量 [batch_size]
    :param use_probabilities: 是否输入的是概率分布（True）还是logits（False）
    :return: Hungarian分配结果列表
    """
    batch_size = len(input_matrix)
    device = free_agents_num.device
    
    # 初始化结果矩阵
    hungarian_matrices = []
    
    for b in range(batch_size):
        num_agents = free_agents_num[b].long().item()
        num_tasks = tasks_num[b].long().item()
        
        if num_agents == 0 or num_tasks == 0:
            # 创建空的Hungarian矩阵
            hungarian_matrices.append(torch.zeros((0, 0), device=device))
            continue
        
        # 提取当前批次的输入矩阵
        batch_input = input_matrix[b][:num_agents, :num_tasks].clone()
        
        if use_probabilities:
            # 如果输入是概率分布，转换为代价矩阵（Hungarian算法最小化代价）
            # 使用 -log(probability) 作为代价，概率越高代价越低
            eps = 1e-10
            batch_input = torch.clamp(batch_input, min=eps, max=1.0)  # 避免log(0)
            cost_matrix = -torch.log(batch_input).detach().cpu().numpy()
        else:
            # 如果输入是logits，直接使用负分数作为代价
            # 使用一个足够大的正代价来屏蔽无效位置，避免被误选
            cost_matrix = -batch_input.detach().cpu().numpy()
        
        # 应用Hungarian算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 创建结果矩阵
        hungarian_matrix = torch.zeros((num_agents, num_tasks), device=device)
        for i, j in zip(row_ind, col_ind):
            hungarian_matrix[i, j] = 1.0
        
        hungarian_matrices.append(hungarian_matrix)
    
    return hungarian_matrices


def adjust_no_task_logits(scores, num_agents, num_tasks, device):
    """
    在num_agents > num_tasks的情况下，调整"no task"的logit以反映其在Plackett-Luce采样中被复制的次数
    
    :param scores: 原始分数 [tasks+1]
    :param num_agents: 智能体数量
    :param num_tasks: 任务数量
    :param device: 设备
    :return: 调整后的分数
    """
    if num_agents > num_tasks:
        # 计算"no task"被复制的次数
        num_extra_no_tasks = max(0, num_agents - num_tasks - 1)
        total_no_task_copies = 1 + num_extra_no_tasks  # 原本的1个 + 额外复制的
        
        # 调整后的scores，增加"no task"的logit以反映其被复制的效果
        adjusted_scores = scores.clone()
        if len(adjusted_scores) > num_tasks:  # 确保有"no task"选项
            # 使用log(复制次数)来调整logit，使得softmax后的概率能反映实际采样概率
            adjusted_scores[num_tasks] += torch.log(torch.tensor(total_no_task_copies, device=device, dtype=torch.float))
        
        return adjusted_scores
    else:
        return scores 


def prepare_cnn_channels(obs, lower_gnn_type):
    """
    准备CNN通道数据（如果使用cnn_channels模型）
    
    :param obs: 观察数据
    :param lower_gnn_type: 低层级GNN类型
    :return: CNN通道数据或None
    """
    cnn_channels = None
    if lower_gnn_type == 'cnn_channels' and isinstance(obs, dict):
        # 从观察中提取8个通道特征
        channel_names = [
            'obstacle_map', 'free_agent_map', 'delivering_agent_map', 
            'delivering_task_id_map', 'pickup_location_map', 'delivery_location_map',
            'pickup_distances', 'delivery_distances'
        ]
        
        # 检查所有通道是否都存在
        if all(name in obs for name in channel_names):
            # 将8个通道堆叠成一个tensor [batch_size, 8, height, width]
            channels = []
            for name in channel_names:
                channels.append(obs[name])
            cnn_channels = torch.stack(channels, dim=1)  # [batch_size, 8, height, width]
    
    return cnn_channels


def process_grid_features(grid_gnn, grid_node_features, grid_edge_indices, cnn_channels, lower_gnn_type):
    """
    处理网格特征
    
    :param grid_gnn: 网格GNN模型
    :param grid_node_features: 网格节点特征
    :param grid_edge_indices: 网格边索引
    :param cnn_channels: CNN通道数据
    :param lower_gnn_type: 低层级GNN类型
    :return: 处理后的网格特征
    """
    batch_size = len(grid_node_features)
    processed_grid_features = []
    
    if lower_gnn_type == 'cnn_channels' and cnn_channels is not None:
        node_feats = grid_gnn(grid_node_features, grid_edge_indices, cnn_channels)
        processed_grid_features = node_feats
    elif lower_gnn_type == 'sp_mpnn':
        # SP-MPNN需要特殊处理，因为它直接处理观察数据并返回entity features
        # 这里我们不需要进一步处理，因为SP-MPNN会直接返回实体特征
        # 但为了保持接口兼容性，我们还是需要返回网格特征格式
        for b in range(batch_size):
            node_feats = grid_node_features[b]  # SP-MPNN的前向传播会在策略中单独调用
            processed_grid_features.append(node_feats)
    else:
        for b in range(batch_size):
            node_feats = grid_gnn(grid_node_features[b], grid_edge_indices[b])
            processed_grid_features.append(node_feats)
    
    return processed_grid_features


def process_higher_graph(
    higher_gnn,
    higher_node_features,
    higher_edge_indices,
    higher_edge_attrs,
    higher_gnn_type,
    output_dim,
    device,
    agent_task_mappings=None,
    higher_edge_types=None,
):
    """
    处理高层级图，提取边特征
    
    :param higher_gnn: 高层级GNN模型
    :param higher_node_features: 高层级节点特征
    :param higher_edge_indices: 高层级边索引
    :param higher_edge_attrs: 高层级边属性
    :param higher_gnn_type: 高层级GNN类型
    :param output_dim: 输出维度
    :param device: 设备
    :param agent_task_mappings: 智能体任务映射（用于self_attention_gat）
    :return: 批次边特征列表
    """
    batch_size = len(higher_node_features)
    batch_edge_features = []
    
    for b in range(batch_size):
        # Higher-level GNN processing
        if higher_node_features[b].size(0) > 0 and higher_edge_indices[b].size(1) > 0:
            # 根据higher_gnn类型调用不同的方法
            if higher_gnn_type == 'line_graph':
                # LineGraphGAT现在也需要edge_attr参数
                edge_feats = higher_gnn(
                    higher_node_features[b],
                    higher_edge_indices[b],
                    higher_edge_attrs[b]
                )
            elif higher_gnn_type == 'self_attention_gat':
                # SelfAttentionGATLayer需要agent_task_mapping参数
                agent_task_mapping = agent_task_mappings[b] if agent_task_mappings else None
                edge_feats = higher_gnn(
                    higher_node_features[b],
                    higher_edge_indices[b],
                    higher_edge_attrs[b],
                    agent_task_mapping=agent_task_mapping
                )
            elif higher_gnn_type in ['gcnn_match', 'edge_gcnn']:
                # EdgeGCNN/UndirectedEdgeGCNN需要额外的num_agents和num_tasks参数
                agent_task_mapping, num_free_agents, num_free_tasks = agent_task_mappings[b]
                edge_feats = higher_gnn(
                    higher_node_features[b],
                    higher_edge_indices[b],
                    higher_edge_attrs[b],
                    num_free_agents,
                    num_free_tasks
                )
            elif higher_gnn_type == 'edge_node_gnn':
                # EdgeNodeGNN: 仅对 free-agent -> free-task 的forward边打分
                agent_task_mapping, num_free_agents, num_free_tasks = agent_task_mappings[b]
                edge_index_b = higher_edge_indices[b]
                if edge_index_b.numel() == 0 or num_free_agents == 0 or num_free_tasks == 0:
                    edge_feats = torch.zeros((0,), device=device)
                else:
                    src, dst = edge_index_b
                    # 估计free task在节点序列中的起始索引：
                    # 在所有 src < num_free_agents 的边中，dst 的最小值即为 task_start（= num_free_agents + num_delivering_agents）
                    mask_from_free_agents = (src < num_free_agents)
                    if mask_from_free_agents.any():
                        min_dst = torch.min(dst[mask_from_free_agents])
                        task_start = int(min_dst.item())
                        task_end = task_start + int(num_free_tasks)
                        forward_mask = (src < num_free_agents) & (dst >= task_start) & (dst < task_end)
                    else:
                        forward_mask = torch.zeros_like(src, dtype=torch.bool)
                    # 前向只对forward边打分
                    edge_feats = higher_gnn(
                        higher_node_features[b],
                        higher_edge_indices[b],
                        higher_edge_attrs[b],
                        num_free_agents,
                        num_free_tasks,
                        forward_mask=forward_mask
                    )
            elif higher_gnn_type == 'edge_node_gnn_complex':
                mapping_meta, _, _ = agent_task_mappings[b]
                edge_index_b = higher_edge_indices[b]
                edge_type_b = higher_edge_types[b] if higher_edge_types is not None else None
                score_mask = mapping_meta.get("score_mask", None) if isinstance(mapping_meta, dict) else None
                if edge_index_b.numel() == 0 or edge_type_b is None or score_mask is None or int(score_mask.sum().item()) == 0:
                    edge_feats = torch.zeros((0,), device=device)
                else:
                    edge_feats = higher_gnn(
                        higher_node_features[b],
                        higher_edge_indices[b],
                        higher_edge_attrs[b],
                        edge_type_b,
                        score_mask=score_mask,
                    )
            else:
                # HigherGATLayer需要三个参数
                edge_feats = higher_gnn(
                    higher_node_features[b],
                    higher_edge_indices[b],
                    higher_edge_attrs[b]
                )
            batch_edge_features.append(edge_feats)
        else:
            # If there are no nodes or edges, return empty features
            batch_edge_features.append(torch.zeros((0, output_dim), device=device))
    
    return batch_edge_features


def calculate_agent_task_scores(
    batch_edge_features,
    agent_task_mappings,
    action_net,
    invalid_edge_score,
    device,
    higher_gnn_type=None,
    use_explicit_path_feature=True,
):
    """
    计算每个自由智能体对每个任务的分数
    
    :param batch_edge_features: 批次边特征或边分数
    :param agent_task_mappings: 智能体任务映射
    :param action_net: 动作网络
    :param invalid_edge_score: 无效边分数
    :param device: 设备
    :param higher_gnn_type: 高层级GNN类型，用于判断是否为EdgeGCNN
    :return: 批次智能体任务分数列表
    """
    batch_size = len(batch_edge_features)
    batch_agent_task_scores = []
    
    use_split_heads = isinstance(action_net, dict)
    fa_head = action_net.get("fa", None) if use_split_heads else None
    da_head = action_net.get("da", None) if use_split_heads else None

    for b in range(batch_size):
        agent_task_mapping, num_free_agents, num_free_tasks = agent_task_mappings[b]

        if higher_gnn_type == "edge_node_gnn_complex" and isinstance(agent_task_mapping, dict):
            scores = torch.full((num_free_agents, num_free_tasks + 1), invalid_edge_score, device=device).float()
            scores[:, -1] = np.exp(-1)

            score_agent_indices = agent_task_mapping.get("score_agent_indices", [])
            score_task_indices = agent_task_mapping.get("score_task_indices", [])
            if len(score_agent_indices) == 0 or batch_edge_features[b].numel() == 0:
                batch_agent_task_scores.append(scores)
                continue

            row_idx = torch.tensor(score_agent_indices, device=device, dtype=torch.long)
            col_idx = torch.tensor(score_task_indices, device=device, dtype=torch.long)
            edge_vals = batch_edge_features[b].to(device).float()
            valid_len = min(len(row_idx), len(col_idx), edge_vals.numel())
            if valid_len <= 0:
                batch_agent_task_scores.append(scores)
                continue

            row_idx = row_idx[:valid_len]
            col_idx = col_idx[:valid_len]
            edge_vals = edge_vals[:valid_len]

            delta = torch.zeros((num_free_agents, num_free_tasks + 1), device=device, dtype=edge_vals.dtype)
            delta = delta.index_put((row_idx, col_idx), edge_vals, accumulate=True)

            mask = torch.zeros_like(delta)
            ones = torch.ones_like(edge_vals, dtype=delta.dtype)
            mask = mask.index_put((row_idx, col_idx), ones, accumulate=True)
            mask = (mask > 0).to(delta.dtype)

            base = torch.full_like(delta, invalid_edge_score)
            base[:, -1] = math.exp(-1)
            scores = torch.where(mask.bool(), delta, base)
            batch_agent_task_scores.append(scores)
            continue
        
        # Create score matrix - 分数越大表示匹配越好
        # 初始化为较小的负值，表示无效边，GNN会学习到更好的分数
        scores = torch.full((num_free_agents, num_free_tasks + 1), invalid_edge_score, device=device).float()  # +1 for "no task" option
        
        # 最后一列（不分配任务选项）设为0，作为baseline
        scores[:, -1] = np.exp(-1)

        row_idx = []
        col_idx = []
        edge_vals = []
        edge_idx = 0

        for agent_idx, task_indices in agent_task_mapping.items():
            for mapping_item in task_indices:
                if edge_idx < batch_edge_features[b].size(0):
                    if isinstance(mapping_item, dict):
                        task_idx = int(mapping_item.get("task_idx", -1))
                        edge_kind = str(mapping_item.get("edge_kind", "fa_ft"))
                        path_total = float(mapping_item.get("path_total", 0.0))
                        da_path_total = float(mapping_item.get("da_path_total", 0.0))
                    else:
                        # backward compatibility with old mapping format: int task_idx
                        task_idx = int(mapping_item)
                        edge_kind = "fa_ft"
                        path_total = 0.0
                        da_path_total = 0.0
                    if task_idx < 0 or task_idx >= num_free_tasks:
                        edge_idx += 1
                        continue
                    row_idx.append(agent_idx)
                    col_idx.append(task_idx)
                    base_edge_val = batch_edge_features[b][edge_idx]
                    if base_edge_val.dim() > 0:
                        base_edge_val = base_edge_val.reshape(-1)[0]
                    base_edge_val = base_edge_val.float()
                    if use_split_heads:
                        if edge_kind == "da_ft" and da_head is not None:
                            # 显式DA总路径信息，压缩到稳定尺度
                            if use_explicit_path_feature:
                                path_feat = torch.log1p(
                                    torch.tensor(max(0.0, da_path_total), device=device, dtype=torch.float32)
                                )
                            else:
                                path_feat = torch.zeros((), device=device, dtype=torch.float32)
                            head_in = torch.stack([base_edge_val, path_feat], dim=0).unsqueeze(0)
                            edge_score = da_head(head_in).reshape(-1)[0]
                        elif edge_kind == "fa_ft" and fa_head is not None:
                            fa_total = path_total if path_total > 0.0 else da_path_total
                            if use_explicit_path_feature:
                                path_feat = torch.log1p(
                                    torch.tensor(max(0.0, fa_total), device=device, dtype=torch.float32)
                                )
                            else:
                                path_feat = torch.zeros((), device=device, dtype=torch.float32)
                            head_in = torch.stack(
                                [base_edge_val, path_feat],
                                dim=0
                            ).unsqueeze(0)
                            edge_score = fa_head(head_in).reshape(-1)[0]
                        else:
                            edge_score = base_edge_val
                    else:
                        edge_score = base_edge_val
                    edge_vals.append(edge_score)
                    edge_idx += 1

        if len(row_idx) == 0:
            batch_agent_task_scores.append(scores)
            continue

        row_idx = torch.tensor(row_idx, device=device, dtype=torch.long)
        col_idx = torch.tensor(col_idx, device=device, dtype=torch.long)
        edge_vals = torch.stack(edge_vals).to(device).float()  

        delta = torch.zeros((num_free_agents, num_free_tasks + 1), device=device, dtype=edge_vals.dtype)
        delta = delta.index_put((row_idx, col_idx), edge_vals, accumulate=True)

        mask = torch.zeros_like(delta) 
        ones = torch.ones_like(edge_vals, dtype=delta.dtype)
        mask = mask.index_put((row_idx, col_idx), ones, accumulate=True)
        mask = (mask > 0).to(delta.dtype)

        base = torch.full_like(delta, invalid_edge_score)
        base[:, -1] = math.exp(-1)
        scores = torch.where(mask.bool(), delta, base)
        
        batch_agent_task_scores.append(scores)
    
    return batch_agent_task_scores


def generate_action_probabilities(batch_agent_task_scores, free_agents_num, free_tasks_num, use_sinkhorn, device):
    """
    生成动作概率分布
    
    :param batch_agent_task_scores: 批次智能体任务分数
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量
    :param use_sinkhorn: 是否使用Sinkhorn算法
    :param device: 设备
    :return: 动作概率列表和原始分数列表
    """
    batch_size = len(batch_agent_task_scores)
    action_probs = []
    original_scores = []  # 存储原始分数用于Hungarian算法
    
    for b in range(batch_size):
        agent_probs = []
        num_agents = free_agents_num[b].long().item()
        num_tasks = free_tasks_num[b].long().item()
        
        for i in range(num_agents):
            scores = batch_agent_task_scores[b][i]
            if not use_sinkhorn:
                # 非Sinkhorn：调整"no task"的logit并计算softmax
                adjusted_scores = adjust_no_task_logits(scores, num_agents, num_tasks, device)
                probs = F.softmax(adjusted_scores, dim=0)
            else:
                # Sinkhorn：原始分数被当作log-cost，直接传递不做变换
                # 这里的scores将被当作log-cost矩阵传递给Sinkhorn算法
                probs = scores.clone()
            agent_probs.append(probs)
        
        agent_probs = torch.stack(agent_probs)
        
        action_probs.append(agent_probs)
        original_scores.append(batch_agent_task_scores[b])  
    
    return action_probs, original_scores


def apply_sinkhorn_to_probabilities(action_probs, sinkhorn, free_agents_num, free_tasks_num, valid_mask, training=True, use_real_domain=True, add_gumbel_noise=None):
    """
    对log-cost矩阵应用Sinkhorn处理，得到概率分布
    
    :param action_probs: log-cost矩阵列表（当use_sinkhorn时）
    :param sinkhorn: Sinkhorn模块
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量
    :param training: 是否在训练阶段（用于控制是否添加Gumbel噪声）
    :param use_real_domain: 是否使用real domain的sinkhorn（预训练阶段推荐）
    :param add_gumbel_noise: 是否添加Gumbel噪声，如果为None则使用sinkhorn模块的设置
    :return: 处理后的概率分布列表
    """
    batch_size = len(action_probs)
    
    # 对每个批次单独应用Sinkhorn算法
    for b in range(batch_size):
        # 只处理有智能体和任务的批次
        if free_agents_num[b] > 0 and free_tasks_num[b] > 0:
            # 创建一个单批次版本的输入
            single_batch_probs = action_probs[b].unsqueeze(0)
            single_batch_free_agents = free_agents_num[b].unsqueeze(0)
            single_batch_free_tasks = free_tasks_num[b].unsqueeze(0)
            
            # 应用Sinkhorn算法
            sinkhorn_distribution = sinkhorn(
                single_batch_probs,
                single_batch_free_agents,
                single_batch_free_tasks,
                valid_mask[b].unsqueeze(0),
                training=training,
                use_real_domain=use_real_domain,
                add_gumbel_noise=add_gumbel_noise
            )
            
            # 更新概率分布
            num_agents = action_probs[b].shape[0]
            num_tasks = action_probs[b].shape[1]
            action_probs[b] = sinkhorn_distribution[0, :num_agents, :num_tasks]
    
    return action_probs


def create_valid_mask(
    batch_size, max_agents, max_tasks,
    free_agents_num, free_tasks_num,
    free_agents_nearest_tasks,
    assignable_agent_is_delivering,
    assignable_agent_delivering_task_idx,
    delivering_tasks_nearest_tasks,
    device
):
    """
    创建有效掩码
    
    :param batch_size: 批次大小
    :param max_agents: 最大智能体数量
    :param max_tasks: 最大任务数量
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量
    :param device: 设备
    :return: 有效掩码
    """
    free_agents_mask = torch.ones((batch_size, max_agents, 1), device=device)
    tasks_mask = torch.zeros((batch_size, 1, max_tasks+1), device=device)
    
    for b in range(batch_size):
        if free_agents_num[b] > free_tasks_num[b]:
            tasks_mask[b, 0, free_tasks_num[b]] = 1
    
    valid_mask = torch.bmm(free_agents_mask, tasks_mask)
    for b in range(batch_size):
        n_agents = int(free_agents_num[b].item())
        for agent_idx in range(n_agents):
            # 兼容旧链路：缺字段则默认free agent语义
            is_delivering = 0.0
            if assignable_agent_is_delivering is not None:
                try:
                    is_delivering = float(assignable_agent_is_delivering[b, agent_idx, 0].item())
                except Exception:
                    is_delivering = 0.0

            if is_delivering < 0.5:
                # free agent: 使用free_agents_nearest_tasks
                if free_agents_nearest_tasks is None:
                    continue
                nearest_tasks_info = free_agents_nearest_tasks[b, agent_idx]
            else:
                # delivering agent: 使用delivering_tasks_nearest_tasks
                if delivering_tasks_nearest_tasks is None or assignable_agent_delivering_task_idx is None:
                    continue
                dt_idx = int(assignable_agent_delivering_task_idx[b, agent_idx, 0].item())
                if dt_idx < 0 or dt_idx >= delivering_tasks_nearest_tasks.shape[1]:
                    continue
                nearest_tasks_info = delivering_tasks_nearest_tasks[b, dt_idx]

            for task_info in nearest_tasks_info:
                task_idx_int = int(task_info[0].item())
                if 0 <= task_idx_int < max_tasks:
                    valid_mask[b, agent_idx, task_idx_int] = 1
    return valid_mask


def compute_pretrain_loss(action_probs, obs, free_agents_num, free_tasks_num, free_agents_nearest_tasks, 
                         use_sinkhorn, not_div, fix_div, max_agents, device, sinkhorn_module=None):
    """
    计算预训练阶段的BCE损失
    
    :param action_probs: 动作概率列表
    :param obs: 观察数据
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量
    :param free_agents_nearest_tasks: 智能体最近任务
    :param use_sinkhorn: 是否使用Sinkhorn
    :param not_div: 是否不进行除法
    :param fix_div: 是否使用固定除数
    :param max_agents: 最大智能体数量
    :param device: 设备
    :param sinkhorn_module: Sinkhorn模块实例（预训练时用于real domain计算）
    :return: 预训练损失
    """
    import torch.nn.functional as F
    
    batch_size = len(action_probs)
    policy_losses = []
    
    for b in range(batch_size):
        # 从观察中提取专家动作
        if isinstance(obs, dict) and "expert_actions" in obs:
            expert_actions_b = obs["expert_actions"][b].long()
        else:
            expert_actions_b = torch.zeros(max_agents, device=device, dtype=torch.long)
        
        # 为当前批次计算预训练损失
        num_free_agents = free_agents_num[b].item()
        num_tasks = free_tasks_num[b].item() + 1  # +1 for "no task" option
        
        if num_free_agents > 0:
            # 将专家动作转换为one-hot编码
            expert_actions_one_hot = F.one_hot(
                expert_actions_b[:num_free_agents].clamp(min=0), 
                num_classes=num_tasks
            ).float()
            
            pred_probs = action_probs[b][:num_free_agents]
            
            # 确保概率有效（防止数值误差）
            pred_probs = torch.clamp(pred_probs, min=0, max=1)
            
            # 计算BCELoss
            bce_loss = F.binary_cross_entropy(
                pred_probs, 
                expert_actions_one_hot, 
                reduction='none'
            )
            
            # 创建权重矩阵，只对free_agents_nearest_tasks中的有效任务设置权重
            weight = torch.zeros_like(expert_actions_one_hot)
            
            if free_agents_nearest_tasks is not None:
                # 使用free_agents_nearest_tasks信息设置权重
                for agent_idx in range(num_free_agents):
                    nearest_tasks_info = free_agents_nearest_tasks[b, agent_idx]
                    
                    # 对最近任务设置权重
                    for task_info in nearest_tasks_info:
                        task_idx_int = int(task_info[0].item())  # 任务ID在第0个位置
                        if task_idx_int >= 0 and task_idx_int < num_tasks - 1:  # 排除"no task"选项
                            # 如果是专家选择的任务，设置高权重
                            if expert_actions_one_hot[agent_idx, task_idx_int] > 0:
                                weight[agent_idx, task_idx_int] = 10.0
                            else:
                                # 如果不是专家选择的任务，设置低权重
                                weight[agent_idx, task_idx_int] = 1
                    
                    # "no task"选项的权重处理
                    if expert_actions_one_hot[agent_idx, -1] > 0:
                        weight[agent_idx, -1] = 0.0
                    else:
                        weight[agent_idx, -1] = 0.0
            else:
                # 如果没有nearest_tasks信息，使用原来的权重设置
                weight = expert_actions_one_hot * 1.0 + (1 - expert_actions_one_hot) * 0.1
            
            policy_loss = (bce_loss * weight).sum(dim=-1)
            
            weight_sum = weight.sum()
            if weight_sum > 0:
                policy_loss = policy_loss / weight_sum
                    
            policy_losses.append(policy_loss.sum())
        else:
            # 如果没有自由智能体，使用零损失
            policy_losses.append(torch.tensor(0.0, device=device))
    
    # 合并所有批次的损失
    if policy_losses:
        log_prob = torch.stack(policy_losses)
    else:
        log_prob = torch.zeros(batch_size, device=device)
    
    return log_prob


def compute_simplified_pretrain_loss(original_scores, obs, free_agents_num, free_tasks_num, device):
    """
    计算简化的预训练损失，类似于test_gcnn的方法
    使用Hungarian algorithm生成edge labels，然后用weighted BCE
    
    :param original_scores: 原始分数列表 [batch_size个 tensor, 每个形状为(agents, tasks+1)]
    :param obs: 观察数据
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量
    :param device: 设备
    :return: 简化的预训练损失
    """
    import torch.nn.functional as F
    
    batch_size = len(original_scores)
    policy_losses = []
    
    for b in range(batch_size):
        num_agents = free_agents_num[b].long().item()
        num_tasks = free_tasks_num[b].long().item()
        
        if num_agents == 0 or num_tasks == 0:
            policy_losses.append(torch.tensor(0.0, device=device))
            continue
            
        # 使用原始分数（不包括no-task列）构建cost矩阵
        cost_matrix = original_scores[b][:num_agents, :num_tasks]
        
        # 应用Hungarian algorithm获得最优分配
        hungarian_matrices = apply_hungarian_algorithm(
            [cost_matrix], 
            free_agents_num[b:b+1], 
            torch.tensor([num_tasks], device=device),
            use_probabilities=False
        )
        optimal_assignment = hungarian_matrices[0]  # [num_agents, num_tasks]
        
        # 计算edge-level的预测概率
        edge_scores = cost_matrix.flatten()  # [num_agents * num_tasks]
        edge_probs = torch.sigmoid(edge_scores)
        
        # 构建edge labels
        edge_labels = optimal_assignment.flatten().float()  # [num_agents * num_tasks]
        
        # 计算权重 (类似test_gcnn的处理)
        pos = edge_labels.sum().item()
        neg = edge_labels.numel() - pos
        pos_weight = (pos + neg) / (pos + 1e-9) if pos > 0 else 1.0
        neg_weight = (pos + neg) / (neg + 1e-9) if neg > 0 else 1.0
        
        weights = torch.where(edge_labels == 1, pos_weight, neg_weight)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy(edge_probs, edge_labels, weight=weights)
        policy_losses.append(bce_loss)
    
    return torch.stack(policy_losses)


def compute_sinkhorn_log_probs(action_probs, actions, free_agents_num):
    """
    计算Sinkhorn情况下的log概率
    直接使用处理过的action_probs计算log概率
    
    :param action_probs: 处理过的动作概率列表 [batch_size个 tensor, 每个形状为(agents, tasks+1)]
    :param actions: 实际选择的动作 [batch_size, max_agents]
    :param free_agents_num: 每个批次自由智能体的数量 [batch_size]
    :return: log概率 [batch_size]
    """
    batch_size = len(action_probs)
    device = actions.device
    log_probs = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        num_agents = free_agents_num[b].long().item()
        
        if num_agents == 0:
            continue
        
        agent_log_probs = []
        
        for i in range(num_agents):
            if i < len(action_probs[b]) and i < actions.shape[1]:
                action = actions[b, i].long().item()
                
                if i < action_probs[b].shape[0] and action < action_probs[b][i].shape[0]:
                    action_prob = action_probs[b][i][action]
                    agent_log_probs.append(torch.log(action_prob + 1e-10))
                else:
                    agent_log_probs.append(torch.tensor(-10.0, device=device))
            else:
                agent_log_probs.append(torch.tensor(-10.0, device=device))
        
        if agent_log_probs:
            log_probs[b] = torch.stack(agent_log_probs).sum()
    
    return log_probs


def compute_entropy(action_probs, free_agents_num, free_agents_nearest_tasks, free_tasks_num, 
                   use_sinkhorn, not_div, fix_div, max_agents, device):
    """
    计算熵
    
    :param action_probs: 动作概率列表
    :param free_agents_num: 自由智能体数量
    :param free_agents_nearest_tasks: 智能体最近任务
    :param free_tasks_num: 自由任务数量
    :param use_sinkhorn: 是否使用Sinkhorn
    :param not_div: 是否不进行除法
    :param fix_div: 是否使用固定除数
    :param max_agents: 最大智能体数量
    :param device: 设备
    :return: 熵
    """
    batch_size = len(action_probs)
    entropy = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        agent_entropy = []
        num_tasks = free_tasks_num[b].long().item()
        
        for i in range(free_agents_num[b].long().item()):
            if i < len(action_probs[b]):
                probs = action_probs[b][i]
                log_probs = torch.log(probs + 1e-10)
                
                # 计算熵: -sum(p * log(p))
                ent = -torch.sum(probs * log_probs)
                
                # 归一化熵值到[0,1]范围
                # 最大熵是log(选择数量)，这里选择数量是num_tasks+1（包括"no task"）
                max_entropy = torch.log(torch.tensor(num_tasks + 1, device=device, dtype=torch.float))
                normalized_ent = ent / max_entropy if max_entropy > 0 else ent
                
                agent_entropy.append(normalized_ent)
        
        if agent_entropy:
            if not not_div:
                if not fix_div:
                    # 使用智能体数量进行归一化
                    if use_sinkhorn:
                        # Sinkhorn情况：使用最近任务数量进行归一化
                        if free_agents_nearest_tasks is not None:
                            # 计算平均归一化熵
                            entropy[b] = torch.stack(agent_entropy).mean()
                        else:
                            entropy[b] = torch.stack(agent_entropy).mean()
                    else:
                        # 非Sinkhorn情况：使用智能体数量进行归一化
                        entropy[b] = torch.stack(agent_entropy).mean()
                else:
                    # 使用固定值进行归一化 - 这里也改为平均值
                    entropy[b] = torch.stack(agent_entropy).mean()
            else:
                # 不进行除法操作时，仍然使用平均值而不是累加
                entropy[b] = torch.stack(agent_entropy).mean()
    
    return entropy 


def compute_sigmoid_bce_loss(original_scores, obs, free_agents_num, free_tasks_num, free_agents_nearest_tasks,
                           not_div, fix_div, max_agents, device):
    """
    计算Gumbel+Hungarian模式下预训练阶段的sigmoid+BCE损失
    
    :param original_scores: 原始分数列表 [batch_size个 tensor, 每个形状为(agents, tasks+1)]
    :param obs: 观察数据
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量  
    :param free_agents_nearest_tasks: 智能体最近任务列表
    :param not_div: 是否不进行除法
    :param fix_div: 是否使用固定除数
    :param max_agents: 最大智能体数量
    :param device: 设备
    :return: BCE损失
    """
    import torch.nn.functional as F
    
    batch_size = len(original_scores)
    policy_losses = []
    
    for b in range(batch_size):
        # 从观察中提取专家动作
        if isinstance(obs, dict) and "expert_actions" in obs:
            expert_actions_b = obs["expert_actions"][b].long()
        else:
            expert_actions_b = torch.zeros(max_agents, device=device, dtype=torch.long)

        num_agents = free_agents_num[b].long().item()
        num_tasks = free_tasks_num[b].long().item()

        if num_agents == 0:
            policy_losses.append(torch.tensor(0.0, device=device))
            continue

        # logits 和标签/权重矩阵（包含 no-task 列）
        logits_b = original_scores[b][:num_agents, :num_tasks + 1]
        labels_b = torch.zeros_like(logits_b)
        weights_b = torch.ones_like(logits_b)
        for i in range(num_agents):
            if i < len(expert_actions_b):
                a = int(expert_actions_b[i].item())
                if 0 <= a < labels_b.shape[1]:
                    labels_b[i, a] = 1.0
                    weights_b[i, a] = 10.0  # 专家选中位置权重=10，其余=1

        # 直接用 BCEWithLogitsLoss（更数值稳定，与脚本 raw/bce 路径一致风格）
        batch_loss = F.binary_cross_entropy_with_logits(
            logits_b, labels_b, weight=weights_b, reduction='mean'
        )
        policy_losses.append(batch_loss)
    
    return torch.stack(policy_losses)

def compute_simplified_edge_loss(original_scores, obs, free_agents_num, free_tasks_num, device):
    """
    计算简化的EdgeGCNN风格损失，直接模仿test_gcnn的实现
    将expert actions转换为edge labels，使用weighted BCE
    
    :param original_scores: 原始分数列表 [batch_size个 tensor, 每个形状为(agents, tasks+1)]
    :param obs: 观察数据
    :param free_agents_num: 自由智能体数量
    :param free_tasks_num: 自由任务数量
    :param device: 设备
    :return: 简化的预训练损失
    """
    import torch.nn.functional as F
    
    batch_size = len(original_scores)
    losses = []
    
    for b in range(batch_size):
        # 从观察中提取专家动作
        if isinstance(obs, dict) and "expert_actions" in obs:
            expert_actions_b = obs["expert_actions"][b].long()
        else:
            expert_actions_b = torch.zeros(free_agents_num[b].item(), device=device, dtype=torch.long)
        
        num_agents = free_agents_num[b].long().item()
        num_tasks = free_tasks_num[b].long().item()
        
        if num_agents == 0 or num_tasks == 0:
            losses.append(torch.tensor(0.0, device=device))
            continue
        
        # 只考虑agent-task连接，不包括"no task"选项
        score_matrix = original_scores[b][:num_agents, :num_tasks]  # [num_agents, num_tasks]
        
        # 将expert actions转换为二分配标签
        assignment_matrix = torch.zeros((num_agents, num_tasks), device=device)
        for i in range(num_agents):
            if i < len(expert_actions_b):
                task_idx = expert_actions_b[i].item()
                if 0 <= task_idx < num_tasks:  # 只处理真实任务分配
                    assignment_matrix[i, task_idx] = 1.0
        
        # 展平为edge-level
        edge_scores = score_matrix.flatten()  # [num_agents * num_tasks]
        edge_labels = assignment_matrix.flatten()  # [num_agents * num_tasks]
        
        # 计算正负样本权重（完全模仿test_gcnn）
        pos = edge_labels.sum().item()
        neg = edge_labels.numel() - pos
        pos_weight = (pos + neg) / (pos + 1e-9) if pos > 0 else 1.0
        neg_weight = (pos + neg) / (neg + 1e-9) if neg > 0 else 1.0
        
        weights = torch.where(edge_labels == 1, pos_weight, neg_weight)
        
        # 使用BCEWithLogitsLoss（test_gcnn风格）
        loss = F.binary_cross_entropy_with_logits(
            edge_scores, edge_labels, weight=weights, reduction='mean'
        )
        
        losses.append(loss)
    
    return torch.stack(losses)


# ---------------- RL (use_gumbel_hungarian) helper functions: row_softmax / sinkhorn ---------------- #

def _add_gumbel_then_temp(logits, tau, device):
    """
    Add standard Gumbel noise then divide by temperature tau.
    Order: logits + Gumbel -> divide by tau. Return scaled noisy logits.
    """
    gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(device)
    scaled = (logits + gumbel_noise) / max(float(tau), 1e-6)
    return scaled


def compute_row_softmax_log_probs_gumbel(original_scores, actions, free_agents_num, free_tasks_num, tau):
    """
    RL log-prob under row-softmax policy on Gumbel-perturbed logits.
    For each agent i: softmax over tasks+1 (including no-task).
    Order: add Gumbel -> divide by tau -> row-wise log_softmax.
    """
    batch_size = len(original_scores)
    device = original_scores[0].device if batch_size > 0 else actions.device
    log_probs = torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        num_agents = int(free_agents_num[b].item())
        num_tasks = int(free_tasks_num[b].item())
        if num_agents == 0:
            continue
        logits = original_scores[b][:num_agents, :num_tasks + 1]
        noisy = _add_gumbel_then_temp(logits, tau, device)
        # Max-shift per row for stability
        noisy = noisy - noisy.max(dim=-1, keepdim=True)[0]
        row_log_probs = torch.log_softmax(noisy, dim=-1)
        agent_logs = []
        for i in range(num_agents):
            a = int(actions[b, i].item()) if i < actions.shape[1] else num_tasks
            if 0 <= a < row_log_probs.shape[1]:
                agent_logs.append(row_log_probs[i, a])
        if agent_logs:
            log_probs[b] = torch.stack(agent_logs).sum()

    return log_probs


def compute_row_softmax_entropy_gumbel(original_scores, free_agents_num, free_tasks_num, tau):
    """
    Entropy under row-softmax policy on Gumbel-perturbed logits.
    Returns per-batch averaged entropy across agents.
    """
    batch_size = len(original_scores)
    device = original_scores[0].device if batch_size > 0 else 'cpu'
    entropies = torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        num_agents = int(free_agents_num[b].item())
        num_tasks = int(free_tasks_num[b].item())
        if num_agents == 0:
            continue
        logits = original_scores[b][:num_agents, :num_tasks + 1]
        noisy = _add_gumbel_then_temp(logits, tau, device)
        noisy = noisy - noisy.max(dim=-1, keepdim=True)[0]
        probs = torch.softmax(noisy, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        ent = -(probs * torch.log(probs)).sum(dim=-1)  # [num_agents]
        entropies[b] = ent.mean()

    return entropies

def expand_no_task_columns(mat, num_agents, num_tasks):
    if num_agents <= num_tasks:
        return mat, 0
    k = num_agents - num_tasks
    if mat.size(1) >= num_tasks + 1:
        no_task_col = mat[:, num_tasks:num_tasks + 1]
    else:
        no_task_col = torch.zeros((num_agents, 1), device=mat.device, dtype=mat.dtype)
    mat_ext = torch.cat([mat[:, :num_tasks], no_task_col.repeat(1, k)], dim=1)
    return mat_ext, k

def masked_entropy_from_probs(P, row_mask):
    if row_mask is None:
        p = P
    else:
        p = P * row_mask
    s = p.sum()
    if s <= 0:
        return None
    p = p / s
    ent = -(p * torch.log(p + 1e-10)).sum()
    if row_mask is not None:
        cnt = row_mask.sum()
        if cnt > 1:
            ent = ent / torch.log(cnt)
    return ent


def compute_logprob_and_entropy_from_L(gumbel_noise_dict, original_scores,obs_id, actions, valid_mask, free_agents_num, free_tasks_num, sinkhorn_module, tau, normalize_by_M=True):
    """
    Compute (log_prob, entropy) from pre-built noisy L = (logits + gumbel)/tau without re-sampling.
    Inputs:
      - noisy_L_list: list[Tensor], each Tensor shape [M, N]
      - actions: Tensor [B, max_agents]
      - valid_mask: Tensor [B, max_agents, max_tasks]
      - free_agents_num: Tensor [B]
      - free_tasks_num: Tensor [B]
      - sinkhorn_module: Sinkhorn instance
      - normalize_by_M: if True, divide log_prob by number of agents
    Returns:
      - log_probs: Tensor [B]
      - entropies: Tensor [B]
    """
    batch_size = len(original_scores)
    device = actions.device
    lp_list = []
    ent_list = []

    for b in range(batch_size):
        M = int(free_agents_num[b].item())
        N = int(free_tasks_num[b].item())
        bi = obs_id[b].item()
        if M == 0:
            continue

        logits = original_scores[b][:M, :N]
        noise = gumbel_noise_dict[bi][:M, :N] * valid_mask[b, :M, :N]
        Lb = logits + noise
        
        P = sinkhorn_module.sinkhorn_log(Lb, M, N, valid_mask[b])
        # P = F.pad(sinkhorn_result, (0, (N+1)-sinkhorn_result.size(1), 0, 0))

        P = torch.clamp(P, min=1e-8, max=1.0)

        # log_prob
        agent_logs = []
        for i in range(M):
            a = int(actions[b, i].item()) if i < actions.shape[1] else N
            if 0 <= a < P.shape[1]:
                agent_logs.append(torch.log(P[i, a]))
        if agent_logs:
            lp = torch.stack(agent_logs).sum()
            # if normalize_by_M:
            #     lp = lp / max(M, 1)

        # entropy (mean over agents)
        ent_agents = []
        for i in range(M):
            if valid_mask is not None:
                row_mask = valid_mask[b, i, :P.shape[1]].float()
            else:
                row_mask = None
            if row_mask is not None:
                p = P[i, :P.shape[1]] * row_mask
            else:
                p = P[i, :P.shape[1]]
            s = p.sum()
            if s > 0:
                p = p / s
                ent_i = -(p * torch.log(p + 1e-10)).sum()
                if row_mask is not None:
                    cnt = row_mask.sum()
                    if cnt > 1:
                        ent_i = ent_i / torch.log(cnt)
                ent_agents.append(ent_i)
        ent = torch.stack(ent_agents).mean() if ent_agents else torch.tensor(0.0, device=device)

        lp_list.append(lp)
        ent_list.append(ent)
    log_probs = torch.stack(lp_list, dim=0)
    entropies = torch.stack(ent_list, dim=0)

    return log_probs, entropies