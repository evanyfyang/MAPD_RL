import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, hidden_size: int = 128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=hidden_size)
        
        # Observation space properties
        self.free_agents_grid_space = observation_space.spaces["free_agents_grid"]
        self.delivering_agents_grid_space = observation_space.spaces["delivering_agents_grid"]
        self.tasks_grid_space = observation_space.spaces["tasks_grid"]

        # Shared CNN
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # MLP layers for free agents, delivering agents, and tasks
        self.free_agent_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.delivering_agent_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.task_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Shared Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Final fully connected layer
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + 3, hidden_size),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        
        # Extract valid counts
        free_agents_num = observations["free_agents_num"].long().to(device)
        delivering_agents_num = observations["delivering_agents_num"].long().to(device)
        tasks_num = observations["tasks_num"].long().to(device)
        
        free_agents_grid = observations["free_agents_grid"].to(device)
        delivering_agents_grid = observations["delivering_agents_grid"].to(device)
        tasks_grid = observations["tasks_grid"].to(device)
        
        batch_size = free_agents_grid.size(0)

        def extract_valid(grids, num):
            valid_grids = [grids[i, :num[i].item(), :, :] for i in range(batch_size)]
            max_valid = max([g.size(0) for g in valid_grids])
            padded_grids = torch.stack([
                F.pad(g, (0, 0, 0, 0, 0, max_valid - g.size(0)), value=0)
                if g.size(0) > 0 else torch.zeros((1, grids.size(2), grids.size(3)), device=device)
                for g in valid_grids
            ], dim=0)
            mask = torch.tensor([[1] * g.size(0) + [0] * (max_valid - g.size(0)) for g in valid_grids], device=device)
            return padded_grids, mask  # padded grids and attention mask

        # Extract valid grids and masks
        free_agents_valid, free_agents_mask = extract_valid(free_agents_grid, free_agents_num)
        delivering_agents_valid, delivering_agents_mask = extract_valid(delivering_agents_grid, delivering_agents_num)
        tasks_valid, tasks_mask = extract_valid(tasks_grid, tasks_num)

        # Process grids through shared CNN
        def process_grids(valid_grids, mlp):
            b, n, x, y = valid_grids.size()
            valid_grids = valid_grids.view(b * n, 1, x, y)
            cnn_features = self.shared_conv(valid_grids)  # (b*n, hidden_size)
            mlp_features = mlp(cnn_features)  # (b*n, hidden_size)
            mlp_features = mlp_features.view(b, n, -1)  # (batch_size, max_valid, hidden_size)
            return mlp_features

        free_agents_features = process_grids(free_agents_valid, self.free_agent_mlp)
        delivering_agents_features = process_grids(delivering_agents_valid, self.delivering_agent_mlp)
        tasks_features = process_grids(tasks_valid, self.task_mlp)

        # Concatenate features for agents
        agents_features = torch.cat([free_agents_features, delivering_agents_features], dim=1)
        agents_mask = torch.cat([free_agents_mask, delivering_agents_mask], dim=1)

        # 拼接 Agents 和 Tasks 的特征
        combined_features = torch.cat([agents_features, tasks_features], dim=1)  # (batch_size, agents_num + tasks_num, hidden_size)
        combined_mask = torch.cat([agents_mask, tasks_mask], dim=1)  # (batch_size, agents_num + tasks_num)

        # 调整形状以匹配 Transformer 输入
        combined_features = combined_features.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        combined_mask = combined_mask == 0  # 转换为 mask 格式 (0: keep, 1: mask)

        # 通过共享 Transformer
        combined_embedding = self.shared_transformer(combined_features, src_key_padding_mask=combined_mask)  # (seq_len, batch_size, hidden_size)

        return combined_embedding, free_agents_num, delivering_agents_num, tasks_num
