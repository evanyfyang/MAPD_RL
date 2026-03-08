#!/usr/bin/env python3
"""
GNN调试可视化工具
用于可视化obs中的CNN通道信息，便于调试
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
from datetime import datetime
import networkx as nx


def visualize_cnn_channels(obs, batch_idx=0, save_dir="./debug_visualizations", prefix="debug"):
    """
    可视化CNN通道信息
    
    Args:
        obs: 观察字典，包含所有CNN通道信息
        batch_idx: 批次索引，默认为0
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取数据
    if isinstance(obs, dict):
        # 提取基本信息
        grid = obs["grid"][batch_idx].cpu().numpy() if torch.is_tensor(obs["grid"]) else obs["grid"][batch_idx]
        free_agents = obs["free_agents"][batch_idx].cpu().numpy() if torch.is_tensor(obs["free_agents"]) else obs["free_agents"][batch_idx]
        delivering_agents = obs["delivering_agents"][batch_idx].cpu().numpy() if torch.is_tensor(obs["delivering_agents"]) else obs["delivering_agents"][batch_idx]
        free_tasks = obs["free_tasks"][batch_idx].cpu().numpy() if torch.is_tensor(obs["free_tasks"]) else obs["free_tasks"][batch_idx]
        delivering_tasks = obs["delivering_tasks"][batch_idx].cpu().numpy() if torch.is_tensor(obs["delivering_tasks"]) else obs["delivering_tasks"][batch_idx]
        
        # 提取数量信息，确保转换为整数
        free_agents_num = int(obs["free_agents_num"][batch_idx].item() if torch.is_tensor(obs["free_agents_num"]) else obs["free_agents_num"][batch_idx])
        delivering_agents_num = int(obs["delivering_agents_num"][batch_idx].item() if torch.is_tensor(obs["delivering_agents_num"]) else obs["delivering_agents_num"][batch_idx])
        free_tasks_num = int(obs["free_tasks_num"][batch_idx].item() if torch.is_tensor(obs["free_tasks_num"]) else obs["free_tasks_num"][batch_idx])
        delivering_tasks_num = int(obs["delivering_tasks_num"][batch_idx].item() if torch.is_tensor(obs["delivering_tasks_num"]) else obs["delivering_tasks_num"][batch_idx])
        
        # 提取CNN通道
        pickup_distances = obs["pickup_distances"][batch_idx].cpu().numpy() if torch.is_tensor(obs["pickup_distances"]) else obs["pickup_distances"][batch_idx]
        delivery_distances = obs["delivery_distances"][batch_idx].cpu().numpy() if torch.is_tensor(obs["delivery_distances"]) else obs["delivery_distances"][batch_idx]
        obstacle_map = obs["obstacle_map"][batch_idx].cpu().numpy() if torch.is_tensor(obs["obstacle_map"]) else obs["obstacle_map"][batch_idx]
        
    else:
        raise ValueError("obs必须是字典格式")
    
    height, width = grid.shape
    grid_max_distance = height + width
    
    print(f"开始可视化CNN通道信息...")
    print(f"网格大小: {height}x{width}")
    print(f"自由智能体数量: {free_agents_num}")
    print(f"运送中智能体数量: {delivering_agents_num}")
    print(f"自由任务数量: {free_tasks_num}")
    print(f"运送中任务数量: {delivering_tasks_num}")
    
    # 图1: 综合地图
    create_comprehensive_map(
        grid, obstacle_map, free_agents, delivering_agents, free_tasks, delivering_tasks,
        free_agents_num, delivering_agents_num, free_tasks_num, delivering_tasks_num,
        save_dir, f"{prefix}_comprehensive_map_{timestamp}.png"
    )
    
    # 图2: Pickup距离图
    create_distance_map(
        pickup_distances, obstacle_map, grid_max_distance, "Pickup Distances",
        save_dir, f"{prefix}_pickup_distances_{timestamp}.png"
    )
    
    # 图3: Delivery距离图
    create_distance_map(
        delivery_distances, obstacle_map, grid_max_distance, "Delivery Distances",
        save_dir, f"{prefix}_delivery_distances_{timestamp}.png"
    )
    
    print(f"可视化完成！图片保存在: {save_dir}")
    print(f"文件前缀: {prefix}_{timestamp}")


def create_comprehensive_map(grid, obstacle_map, free_agents, delivering_agents, free_tasks, delivering_tasks,
                           free_agents_num, delivering_agents_num, free_tasks_num, delivering_tasks_num,
                           save_dir, filename):
    """创建综合地图，显示所有智能体和任务"""
    
    height, width = grid.shape
    
    # 根据网格大小动态调整图形尺寸和字体大小
    fig_width = max(10, width * 0.6)
    fig_height = max(8, height * 0.6)
    
    # 动态字体大小：基于网格大小调整，增大基础字体
    base_fontsize = max(6, min(16, 200 / max(height, width)))
    agent_fontsize = base_fontsize
    task_fontsize = max(5, base_fontsize - 1)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 创建基础地图（白色为自由空间，黑色为障碍物）
    display_map = np.ones((height, width, 3))  # RGB格式
    
    # 标记障碍物为黑色
    obstacle_positions = obstacle_map > 0.5
    display_map[obstacle_positions] = [0, 0, 0]  # 黑色
    
    # 显示地图
    ax.imshow(display_map, origin='upper')
    
    # 添加网格
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 动态调整标记大小
    marker_size = max(0.2, min(0.45, 12.0 / max(height, width)))
    
    # 绘制自由智能体
    for i in range(int(free_agents_num)):  # 确保是整数
        x, y = int(free_agents[i, 0]), int(free_agents[i, 1])  # 确保坐标是整数
        if 0 <= x < height and 0 <= y < width:
            # 绘制蓝色圆圈表示自由智能体
            circle = plt.Circle((y, x), marker_size, color='blue', alpha=0.8)
            ax.add_patch(circle)
            ax.text(y, x, f'A{i}', ha='center', va='center', fontsize=agent_fontsize, color='white', weight='bold')
    
    # 绘制运送中智能体
    for i in range(int(delivering_agents_num)):  # 确保是整数
        x, y = int(delivering_agents[i, 0]), int(delivering_agents[i, 1])  # 确保坐标是整数
        if 0 <= x < height and 0 <= y < width:
            # 绘制深蓝色方块表示运送中智能体
            rect = plt.Rectangle((y-marker_size, x-marker_size), marker_size*2, marker_size*2, color='darkblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(y, x, f'D{i}', ha='center', va='center', fontsize=agent_fontsize, color='white', weight='bold')
    
    # 绘制自由任务
    for i in range(int(free_tasks_num)):  # 确保是整数
        # 取货位置
        pickup_x, pickup_y = int(free_tasks[i, 0]), int(free_tasks[i, 1])  # 确保坐标是整数
        # 送货位置
        delivery_x, delivery_y = int(free_tasks[i, 2]), int(free_tasks[i, 3])  # 确保坐标是整数
        
        if (0 <= pickup_x < height and 0 <= pickup_y < width and 
            0 <= delivery_x < height and 0 <= delivery_y < width):
            
            # 绘制取货位置（绿色三角形）
            triangle_pickup = plt.Polygon([(pickup_y, pickup_x-marker_size), 
                                         (pickup_y-marker_size, pickup_x+marker_size), 
                                         (pickup_y+marker_size, pickup_x+marker_size)], 
                                        color='green', alpha=0.8)
            ax.add_patch(triangle_pickup)
            ax.text(pickup_y, pickup_x+marker_size*0.3, f'P{i}', ha='center', va='center', 
                   fontsize=task_fontsize, color='white', weight='bold')
            
            # 绘制送货位置（红色倒三角形）
            triangle_delivery = plt.Polygon([(delivery_y, delivery_x+marker_size), 
                                           (delivery_y-marker_size, delivery_x-marker_size), 
                                           (delivery_y+marker_size, delivery_x-marker_size)], 
                                          color='red', alpha=0.8)
            ax.add_patch(triangle_delivery)
            ax.text(delivery_y, delivery_x-marker_size*0.3, f'D{i}', ha='center', va='center', 
                   fontsize=task_fontsize, color='white', weight='bold')
            
            # 绘制连接线
            ax.plot([pickup_y, delivery_y], [pickup_x, delivery_x], 
                   color='orange', linewidth=max(1.5, marker_size*6), alpha=0.6, linestyle='--')
    
    # 绘制运送中任务
    for i in range(int(delivering_tasks_num)):  # 确保是整数
        # 运送中任务的送货位置
        agent_id = int(delivering_tasks[i, 0])  # 确保是整数
        delivery_x, delivery_y = int(delivering_tasks[i, 3]), int(delivering_tasks[i, 4])  # 确保坐标是整数
        
        if 0 <= delivery_x < height and 0 <= delivery_y < width:
            # 绘制紫色菱形表示运送中任务的目标
            diamond = plt.Polygon([(delivery_y, delivery_x-marker_size), 
                                 (delivery_y+marker_size, delivery_x), 
                                 (delivery_y, delivery_x+marker_size), 
                                 (delivery_y-marker_size, delivery_x)], 
                                color='purple', alpha=0.8)
            ax.add_patch(diamond)
            ax.text(delivery_y, delivery_x, f'T{i}', ha='center', va='center', 
                   fontsize=task_fontsize, color='white', weight='bold')
    
    # 设置标题和标签
    title_fontsize = max(10, min(20, base_fontsize + 6))
    ax.set_title('Comprehensive Map View\n(Blue=Free Agents, DarkBlue=Delivering Agents, Green=Pickup, Red=Delivery, Purple=Delivering Target)', 
                fontsize=title_fontsize, pad=20)
    ax.set_xlabel('Y Coordinate', fontsize=max(8, base_fontsize + 2))
    ax.set_ylabel('X Coordinate', fontsize=max(8, base_fontsize + 2))
    
    # 设置坐标轴
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.invert_yaxis()  # 让(0,0)在左上角
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=max(6, base_fontsize))
    
    # 添加图例
    legend_fontsize = max(8, base_fontsize)
    legend_markersize = max(8, base_fontsize + 2)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=legend_markersize, label='Free Agents'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue', markersize=legend_markersize, label='Delivering Agents'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=legend_markersize, label='Pickup Locations'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=legend_markersize, label='Delivery Locations'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=legend_markersize-2, label='Delivering Targets'),
        plt.Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Task Connections'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=legend_markersize, label='Obstacles')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合地图已保存: {filename}")


def create_distance_map(distance_data, obstacle_map, grid_max_distance, title, save_dir, filename):
    """创建距离地图"""
    
    height, width = distance_data.shape
    
    # 根据网格大小动态调整图形尺寸和字体大小
    fig_width = max(10, width * 0.5)
    fig_height = max(8, height * 0.5)
    
    # 动态字体大小，增大基础字体
    base_fontsize = max(5, min(14, 150 / max(height, width)))
    text_fontsize = max(4, base_fontsize)
    
    # 还原距离值（从归一化恢复到实际距离）
    restored_distances = distance_data.copy()
    
    # 处理归一化的距离值
    valid_mask = distance_data >= 0  # 非-1的位置
    restored_distances[valid_mask] = distance_data[valid_mask] * grid_max_distance
    
    # 创建显示用的数组
    display_data = restored_distances.copy()
    
    # 将-1（不可达）和障碍物位置设为特殊值用于显示
    obstacle_positions = obstacle_map > 0.5
    unreachable_positions = distance_data < 0
    
    # 设置显示范围
    max_distance = np.max(restored_distances[valid_mask]) if np.any(valid_mask) else grid_max_distance
    
    # 创建自定义颜色映射
    # 使用viridis颜色映射，但为障碍物和不可达区域设置特殊颜色
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # 创建掩码数组用于不同的显示
    masked_data = np.ma.masked_where(obstacle_positions | unreachable_positions, display_data)
    
    # 绘制距离热图
    im = ax.imshow(masked_data, cmap='viridis', origin='upper', vmin=0, vmax=max_distance)
    
    # 绘制障碍物（黑色）
    obstacle_display = np.zeros((height, width, 4))  # RGBA
    obstacle_display[obstacle_positions] = [0, 0, 0, 1]  # 黑色，完全不透明
    ax.imshow(obstacle_display, origin='upper')
    
    # 绘制不可达区域（深灰色）
    unreachable_display = np.zeros((height, width, 4))  # RGBA
    unreachable_display[unreachable_positions & ~obstacle_positions] = [0.3, 0.3, 0.3, 1]  # 深灰色
    ax.imshow(unreachable_display, origin='upper')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Distance (grid units)', rotation=270, labelpad=15, fontsize=max(8, base_fontsize + 2))
    cbar.ax.tick_params(labelsize=max(6, base_fontsize))
    
    # 添加网格
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 在每个格子中显示数值（调整显示策略）
    if max(height, width) <= 15:  # 小网格显示所有数值
        for i in range(int(height)):  # 确保是整数
            for j in range(int(width)):  # 确保是整数
                if obstacle_positions[i, j]:
                    text = "OBS"
                    color = "white"
                elif unreachable_positions[i, j]:
                    text = "N/A"
                    color = "white"
                else:
                    text = f"{restored_distances[i, j]:.1f}"
                    color = "white" if restored_distances[i, j] < max_distance * 0.5 else "black"
                
                ax.text(j, i, text, ha='center', va='center', fontsize=text_fontsize, color=color, weight='bold')
    elif max(height, width) <= 25:  # 中等网格显示简化数值
        step = max(1, max(height, width) // 12)  # 每隔几个格子显示一个数值
        for i in range(0, int(height), step):
            for j in range(0, int(width), step):
                if obstacle_positions[i, j]:
                    text = "X"
                    color = "white"
                elif unreachable_positions[i, j]:
                    text = "-"
                    color = "white"
                else:
                    text = f"{restored_distances[i, j]:.0f}"
                    color = "white" if restored_distances[i, j] < max_distance * 0.5 else "black"
                
                ax.text(j, i, text, ha='center', va='center', fontsize=text_fontsize, color=color, weight='bold')
    elif max(height, width) <= 40:  # 大网格显示更少数值
        step = max(1, max(height, width) // 8)  # 每隔更多格子显示一个数值
        for i in range(0, int(height), step):
            for j in range(0, int(width), step):
                if obstacle_positions[i, j]:
                    text = "X"
                    color = "white"
                elif unreachable_positions[i, j]:
                    text = "-"
                    color = "white"
                else:
                    text = f"{restored_distances[i, j]:.0f}"
                    color = "white" if restored_distances[i, j] < max_distance * 0.5 else "black"
                
                ax.text(j, i, text, ha='center', va='center', fontsize=text_fontsize, color=color, weight='bold')
    
    # 设置标题和标签
    title_fontsize = max(10, min(18, base_fontsize + 5))
    ax.set_title(f'{title}\n(Black=Obstacles, Gray=Unreachable, Colors=Distance)', 
                fontsize=title_fontsize, pad=15)
    ax.set_xlabel('Y Coordinate', fontsize=max(8, base_fontsize + 2))
    ax.set_ylabel('X Coordinate', fontsize=max(8, base_fontsize + 2))
    
    # 设置坐标轴
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    
    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=max(6, base_fontsize))
    
    # 添加统计信息
    if np.any(valid_mask):
        min_dist = np.min(restored_distances[valid_mask])
        max_dist = np.max(restored_distances[valid_mask])
        mean_dist = np.mean(restored_distances[valid_mask])
        
        stats_text = f"Min: {min_dist:.1f}, Max: {max_dist:.1f}, Mean: {mean_dist:.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=max(8, base_fontsize + 1), 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{title}已保存: {filename}")


def quick_visualize(obs, save_dir="./debug_visualizations"):
    """
    快速可视化函数，用于在pdb调试时调用
    
    使用方法:
    在pdb中：
    >>> from gnn_check import quick_visualize
    >>> quick_visualize(obs)
    """
    try:
        visualize_cnn_channels(obs, batch_idx=0, save_dir=save_dir, prefix="quick_debug")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 打印调试信息
        if isinstance(obs, dict):
            print("观察数据键:", list(obs.keys()))
            for key in ['grid', 'free_agents_num', 'delivering_agents_num', 'free_tasks_num', 'delivering_tasks_num']:
                if key in obs:
                    data = obs[key]
                    if torch.is_tensor(data):
                        print(f"{key}: shape={data.shape}, dtype={data.dtype}")
                    else:
                        print(f"{key}: type={type(data)}, value={data}")
        raise e


def visualize_batch(obs, batch_idx, save_dir="./debug_visualizations"):
    """
    可视化指定批次的数据
    
    使用方法:
    在pdb中：
    >>> from gnn_check import visualize_batch
    >>> visualize_batch(obs, batch_idx=1)
    """
    try:
        visualize_cnn_channels(obs, batch_idx=batch_idx, save_dir=save_dir, prefix=f"batch_{batch_idx}")
    except Exception as e:
        print(f"可视化批次 {batch_idx} 时出现错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 打印调试信息
        if isinstance(obs, dict):
            print("观察数据键:", list(obs.keys()))
            for key in ['grid', 'free_agents_num', 'delivering_agents_num', 'free_tasks_num', 'delivering_tasks_num']:
                if key in obs:
                    data = obs[key]
                    if torch.is_tensor(data):
                        print(f"{key}: shape={data.shape}, dtype={data.dtype}")
                        if batch_idx < data.shape[0]:
                            print(f"{key}[{batch_idx}]: {data[batch_idx]}")
                    else:
                        print(f"{key}: type={type(data)}, value={data}")
        raise e


def visualize_higher_graph(higher_node_features, higher_edge_indices, agent_task_mappings, 
                          free_agents_num, delivering_agents_num, free_tasks_num, delivering_tasks_num,
                          batch_idx=0, save_dir="./debug_visualizations", prefix="higher_graph"):
    """
    可视化高层级图的结构
    
    Args:
        higher_node_features: 高层级节点特征列表 [batch_size个tensor]
        higher_edge_indices: 高层级边索引列表 [batch_size个tensor]
        agent_task_mappings: 智能体-任务映射关系列表
        free_agents_num: 自由智能体数量
        delivering_agents_num: 运送中智能体数量
        free_tasks_num: 自由任务数量
        delivering_tasks_num: 运送中任务数量
        batch_idx: 批次索引
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取指定批次的数据
    if batch_idx >= len(higher_node_features):
        print(f"批次索引 {batch_idx} 超出范围，最大批次数: {len(higher_node_features)}")
        return
    
    node_features = higher_node_features[batch_idx]
    edge_indices = higher_edge_indices[batch_idx]
    agent_task_mapping, num_free_agents, num_free_tasks = agent_task_mappings[batch_idx]
    
    # 转换为CPU numpy数组
    if torch.is_tensor(node_features):
        node_features = node_features.cpu().detach().numpy()
    if torch.is_tensor(edge_indices):
        edge_indices = edge_indices.cpu().numpy()
    
    # 获取数量信息
    num_free_agents = int(free_agents_num[batch_idx].item() if torch.is_tensor(free_agents_num) else free_agents_num[batch_idx])
    num_delivering_agents = int(delivering_agents_num[batch_idx].item() if torch.is_tensor(delivering_agents_num) else delivering_agents_num[batch_idx])
    num_free_tasks = int(free_tasks_num[batch_idx].item() if torch.is_tensor(free_tasks_num) else free_tasks_num[batch_idx])
    num_delivering_tasks = int(delivering_tasks_num[batch_idx].item() if torch.is_tensor(delivering_tasks_num) else delivering_tasks_num[batch_idx])
    
    print(f"可视化高层级图 - 批次 {batch_idx}")
    print(f"节点数量: {node_features.shape[0] if len(node_features.shape) > 0 else 0}")
    print(f"边数量: {edge_indices.shape[1] if len(edge_indices.shape) > 1 else 0}")
    print(f"自由智能体: {num_free_agents}, 运送中智能体: {num_delivering_agents}")
    print(f"自由任务: {num_free_tasks}, 运送中任务: {num_delivering_tasks}")
    
    # 创建NetworkX图
    G = nx.Graph()
    
    # 节点类型和标签
    node_types = []
    node_labels = {}
    node_colors = []
    
    # 添加自由智能体节点
    for i in range(num_free_agents):
        G.add_node(i)
        node_types.append('free_agent')
        node_labels[i] = f'FA{i}'
        node_colors.append('lightblue')
    
    # 添加运送中智能体节点
    for i in range(num_delivering_agents):
        node_id = num_free_agents + i
        G.add_node(node_id)
        node_types.append('delivering_agent')
        node_labels[node_id] = f'DA{i}'
        node_colors.append('darkblue')
    
    # 添加自由任务节点
    for i in range(num_free_tasks):
        node_id = num_free_agents + num_delivering_agents + i
        G.add_node(node_id)
        node_types.append('free_task')
        node_labels[node_id] = f'FT{i}'
        node_colors.append('lightgreen')
    
    # 添加运送中任务节点
    for i in range(num_delivering_tasks):
        node_id = num_free_agents + num_delivering_agents + num_free_tasks + i
        G.add_node(node_id)
        node_types.append('delivering_task')
        node_labels[node_id] = f'DT{i}'
        node_colors.append('purple')
    
    # 添加边 - 只添加agent-task和delivering连接
    edge_types = []
    agent_task_edges = 0
    delivering_edges = 0
    
    if edge_indices.shape[1] > 0:
        for i in range(edge_indices.shape[1]):
            src, dst = edge_indices[0, i], edge_indices[1, i]
            if src < len(node_types) and dst < len(node_types):
                # 确定边的类型
                src_type = node_types[src]
                dst_type = node_types[dst]
                
                # 只添加agent-task连接和delivering连接
                if (src_type == 'free_agent' and dst_type == 'free_task') or \
                   (src_type == 'free_task' and dst_type == 'free_agent'):
                    G.add_edge(src, dst)
                    edge_types.append('agent_task')
                    agent_task_edges += 1
                elif (src_type == 'delivering_agent' and dst_type == 'delivering_task') or \
                     (src_type == 'delivering_task' and dst_type == 'delivering_agent'):
                    G.add_edge(src, dst)
                    edge_types.append('delivering')
                    delivering_edges += 1
                # 跳过agent-agent连接
    
    print(f"Agent-Task连接数: {agent_task_edges}")
    print(f"Delivering连接数: {delivering_edges}")
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：网络图布局
    if len(G.nodes()) > 0:
        # 使用spring布局，但为不同类型的节点设置初始位置
        pos = {}
        
        # 自由智能体在左上
        for i in range(num_free_agents):
            pos[i] = (0, 1 - i * 0.3)
        
        # 运送中智能体在左下
        for i in range(num_delivering_agents):
            node_id = num_free_agents + i
            pos[node_id] = (0, -1 - i * 0.3)
        
        # 自由任务在右上
        for i in range(num_free_tasks):
            node_id = num_free_agents + num_delivering_agents + i
            pos[node_id] = (2, 1 - i * 0.2)
        
        # 运送中任务在右下
        for i in range(num_delivering_tasks):
            node_id = num_free_agents + num_delivering_agents + num_free_tasks + i
            pos[node_id] = (2, -1 - i * 0.3)
        
        # 使用spring布局优化位置
        if len(G.edges()) > 0:
            pos = nx.spring_layout(G, pos=pos, k=1.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.8, ax=ax1)
        
        # 绘制边，根据类型使用不同颜色
        edge_colors = []
        edge_styles = []
        for i, edge in enumerate(G.edges()):
            if i < len(edge_types):
                if edge_types[i] == 'agent_task':
                    edge_colors.append('red')
                    edge_styles.append('-')
                elif edge_types[i] == 'delivering':
                    edge_colors.append('purple')
                    edge_styles.append('-')
                else:
                    edge_colors.append('black')
                    edge_styles.append(':')
            else:
                edge_colors.append('black')
                edge_styles.append('-')
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              style=edge_styles, alpha=0.7, width=2, ax=ax1)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, node_labels, font_size=12, 
                               font_weight='bold', ax=ax1)
    
    ax1.set_title('Higher-Level Graph Structure\n(Agent-Task & Delivering Connections)', 
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=12, label='Free Agents'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                  markersize=12, label='Delivering Agents'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=12, label='Free Tasks'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=12, label='Delivering Tasks'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Agent-Task Connections'),
        plt.Line2D([0], [0], color='purple', linewidth=3, label='Delivering Connections')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=11)
    
    # 右图：连接矩阵（只显示相关连接）
    total_nodes = num_free_agents + num_delivering_agents + num_free_tasks + num_delivering_tasks
    if total_nodes > 0:
        adjacency_matrix = np.zeros((total_nodes, total_nodes))
        
        # 只填充agent-task和delivering连接
        if edge_indices.shape[1] > 0:
            for i in range(edge_indices.shape[1]):
                src, dst = edge_indices[0, i], edge_indices[1, i]
                if src < total_nodes and dst < total_nodes:
                    src_type = node_types[src] if src < len(node_types) else 'unknown'
                    dst_type = node_types[dst] if dst < len(node_types) else 'unknown'
                    
                    # 只记录agent-task和delivering连接
                    if ((src_type == 'free_agent' and dst_type == 'free_task') or 
                        (src_type == 'free_task' and dst_type == 'free_agent') or
                        (src_type == 'delivering_agent' and dst_type == 'delivering_task') or 
                        (src_type == 'delivering_task' and dst_type == 'delivering_agent')):
                        adjacency_matrix[src, dst] = 1
                        adjacency_matrix[dst, src] = 1  # 无向图
        
        im = ax2.imshow(adjacency_matrix, cmap='Blues', alpha=0.8)
        
        # 添加网格线分隔不同类型的节点
        ax2.axhline(y=num_free_agents-0.5, color='red', linewidth=2)
        ax2.axhline(y=num_free_agents+num_delivering_agents-0.5, color='red', linewidth=2)
        ax2.axhline(y=num_free_agents+num_delivering_agents+num_free_tasks-0.5, color='red', linewidth=2)
        
        ax2.axvline(x=num_free_agents-0.5, color='red', linewidth=2)
        ax2.axvline(x=num_free_agents+num_delivering_agents-0.5, color='red', linewidth=2)
        ax2.axvline(x=num_free_agents+num_delivering_agents+num_free_tasks-0.5, color='red', linewidth=2)
        
        # 设置刻度标签
        tick_labels = []
        for i in range(total_nodes):
            if i in node_labels:
                tick_labels.append(node_labels[i])
            else:
                tick_labels.append(f'N{i}')
        
        ax2.set_xticks(range(total_nodes))
        ax2.set_yticks(range(total_nodes))
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
        ax2.set_yticklabels(tick_labels, fontsize=10)
        
        # 添加区域标签
        if num_free_agents > 0:
            ax2.text(num_free_agents/2-0.5, -1, 'Free\nAgents', ha='center', va='top', fontweight='bold')
        if num_delivering_agents > 0:
            ax2.text(num_free_agents+num_delivering_agents/2-0.5, -1, 'Delivering\nAgents', ha='center', va='top', fontweight='bold')
        if num_free_tasks > 0:
            ax2.text(num_free_agents+num_delivering_agents+num_free_tasks/2-0.5, -1, 'Free\nTasks', ha='center', va='top', fontweight='bold')
        if num_delivering_tasks > 0:
            ax2.text(num_free_agents+num_delivering_agents+num_free_tasks+num_delivering_tasks/2-0.5, -1, 'Delivering\nTasks', ha='center', va='top', fontweight='bold')
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
    
    ax2.set_title('Adjacency Matrix\n(Agent-Task & Delivering Connections Only)', 
                 fontsize=14, fontweight='bold')
    
    # 添加统计信息
    stats_text = f"""Graph Statistics:
Nodes: {total_nodes}
  - Free Agents: {num_free_agents}
  - Delivering Agents: {num_delivering_agents}
  - Free Tasks: {num_free_tasks}
  - Delivering Tasks: {num_delivering_tasks}

Connections:
  - Agent-Task: {agent_task_edges}
  - Delivering: {delivering_edges}
  - Total: {agent_task_edges + delivering_edges}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=11, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 保存图片
    filename = f"{prefix}_batch_{batch_idx}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"高层级图可视化已保存: {filename}")


def debug_higher_graph_from_obs(obs, batch_idx=0, save_dir="./debug_visualizations"):
    """
    从观察数据中提取并可视化高层级图
    这个函数模拟GNNPolicy中的_create_higher_graph过程
    
    Args:
        obs: 观察字典
        batch_idx: 批次索引
        save_dir: 保存目录
    """
    try:
        # 这里需要导入GNNPolicy来使用其方法
        # 由于循环导入问题，我们提供一个简化版本的提示
        print("要完整可视化高层级图，请在调试时使用以下代码:")
        print("""
# 在pdb调试中，假设你有一个policy实例:
from gnn_check import visualize_higher_graph

# 在GNNPolicy.forward()或evaluate_actions()中，在创建高层级图后添加:
visualize_higher_graph(
    higher_node_features, 
    higher_edge_indices, 
    agent_task_mappings,
    free_agents_num, 
    delivering_agents_num, 
    free_tasks_num, 
    delivering_tasks_num,
    batch_idx=0
)
""")
        
        # 提取基本信息用于简单可视化
        if isinstance(obs, dict):
            free_agents_num = obs.get("free_agents_num", torch.tensor([0]))[batch_idx]
            delivering_agents_num = obs.get("delivering_agents_num", torch.tensor([0]))[batch_idx]
            free_tasks_num = obs.get("free_tasks_num", torch.tensor([0]))[batch_idx]
            delivering_tasks_num = obs.get("delivering_tasks_num", torch.tensor([0]))[batch_idx]
            
            print(f"观察数据概览 - 批次 {batch_idx}:")
            print(f"自由智能体数量: {free_agents_num}")
            print(f"运送中智能体数量: {delivering_agents_num}")
            print(f"自由任务数量: {free_tasks_num}")
            print(f"运送中任务数量: {delivering_tasks_num}")
            
            if "free_agents_nearest_tasks" in obs:
                nearest_tasks = obs["free_agents_nearest_tasks"][batch_idx]
                print(f"最近任务映射形状: {nearest_tasks.shape}")
                print(f"最近任务映射内容:\n{nearest_tasks}")
        
    except Exception as e:
        print(f"可视化高层级图时出现错误: {e}")


if __name__ == "__main__":
    print("GNN调试可视化工具")
    print("使用方法:")
    print("1. 在pdb调试时导入: from gnn_check import quick_visualize, visualize_higher_graph")
    print("2. 调用函数: quick_visualize(obs)")
    print("3. 可视化高层级图: visualize_higher_graph(higher_node_features, higher_edge_indices, agent_task_mappings, ...)")
    print("4. 图片将保存在 ./debug_visualizations/ 目录下") 