import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
from torchvision.models import resnet18

class MAPDFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, grid_size: tuple, hidden_size: int = 128):
        super(MAPDFeatureExtractor, self).__init__(observation_space, features_dim=hidden_size)
        
        self.hidden_size = hidden_size
        
        # 获取各网格空间的预设最大数量
        self.max_free_agents = observation_space.spaces["free_agents_grid"].shape[1]  # 预设最大空闲agent数
        self.max_delivering_agents = observation_space.spaces["delivering_agents_grid"].shape[1]  # 预设最大运输中agent数
        self.max_tasks = observation_space.spaces["tasks_grid"].shape[1]  # 预设最大任务数

        # Shared CNN
        self.agent_conv = resnet18(num_classes=hidden_size)
        
        self.task_pos_emb = nn.Embedding(2 * grid_size[0] * grid_size[1], hidden_size)

        self.length_emb = nn.Embedding(10 * (grid_size[0] + grid_size[1]), hidden_size)

        self.grid_size = grid_size
        
        self.task_conv = resnet18(num_classes=hidden_size)

        # self.task_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten()
        # )
        
        # MLP layers for free agents, delivering agents, and tasks
        self.free_agent_mlp = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        )
        self.delivering_agent_mlp = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        )
        self.task_mlp = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Shared Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.task_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.init_net()

    def init_net(self):
        def init_weights_orthogonal2(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def init_cnn_kaiming(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.agent_conv.apply(init_cnn_kaiming)
        self.task_conv.apply(init_cnn_kaiming)
        self.free_agent_mlp.apply(init_weights_orthogonal2)
        self.delivering_agent_mlp.apply(init_weights_orthogonal2)
        self.task_mlp.apply(init_weights_orthogonal2)
        self.delivering_agent_mlp.apply(init_weights_orthogonal2)
        self.shared_transformer.apply(init_cnn_kaiming)

    def pack_obs(self, combined_feature, free_agents_num, delivering_agents_num, tasks_num, expert_actions):
        batch_size = combined_feature.shape[0]
        seq_len = combined_feature.shape[1]
        hidden_size = combined_feature.shape[2]
        combined_feature = combined_feature.reshape(batch_size, seq_len * hidden_size)

        obs = torch.cat(
            [
                combined_feature, 
                free_agents_num.unsqueeze(1), 
                delivering_agents_num.unsqueeze(1), 
                tasks_num.unsqueeze(1),
                expert_actions
            ],
            dim=1 
        )
        return obs


    def extract_valid_2d(self, grids, num):
        batch_size = grids.size(0)
        valid_grids = [grids[i, :num[i].item(), :, :, :] for i in range(batch_size)]
        # max_num = grids.shape[1]
        max_num = max([g.size(0) for g in valid_grids])
        padded_grids = torch.stack([
            F.pad(g, (0, 0, 0, 0, 0, 0, 0, max_num - g.size(0)), value=0)
            if g.size(0) > 0 else torch.zeros((max_num, grids.size(2), grids.size(3), grids.size(4)), device=grids.device)
            for g in valid_grids
        ], dim=0)
        mask = torch.tensor([[1] * g.size(0) + [0] * (max_num - g.size(0)) for g in valid_grids], device=grids.device)
        return padded_grids, mask  
        
    def extract_valid_1d(self, locs, num):
        batch_size = locs.size(0)
        valid_locs = [locs[i, :num[i].item(), :] for i in range(batch_size)]
        max_num = max([l.size(0) for l in valid_locs])
        padded_locs = torch.stack([
            F.pad(l, (0, 0, 0, max_num - l.size(0)), value=0)
            if l.size(0) > 0 else torch.zeros((max_num, locs.size(2)), device=locs.device)
            for l in valid_locs
        ], dim=0)
        mask = torch.tensor([[1] * l.size(0) + [0] * (max_num - l.size(0)) for l in valid_locs], device=locs.device)
        return padded_locs, mask  
    
    def process_agent_grids(self, valid_grids, valid_path_length, conv, mlp):
        b, n, t, x, y = valid_grids.size()
        valid_grids = valid_grids.view(b * n, t, x, y)
        cnn_features = conv(valid_grids).view(b, n, self.hidden_size)  
        path_length_features = self.length_emb(valid_path_length.long()).squeeze(2)  
        mlp_features = mlp(torch.cat([cnn_features, path_length_features], dim=-1)) 
        return mlp_features
    
    def process_task_grids(self, valid_grids, conv):
        b, n, t, x, y = valid_grids.size()
        valid_grids = valid_grids.view(b * n, t, x, y)
        cnn_features = conv(valid_grids).view(b, n, self.hidden_size)  
        return cnn_features

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = observations["free_agents_num"].shape[0]
        # Extract valid counts
        free_agents_num = torch.argmax(observations["free_agents_num"].long().to(device), dim=-1).reshape(batch_size)
        delivering_agents_num = torch.argmax(observations["delivering_agents_num"].long().to(device), dim=-1).reshape(batch_size)
        tasks_num = torch.argmax(observations["tasks_num"].long().to(device), dim=-1).reshape(batch_size)
        task_loc = observations["tasks_loc"].to(device)
        free_agents_path_length = observations["free_agents_path_length"].to(device)
        delivering_agents_path_length = observations["delivering_agents_path_length"].to(device)
        free_agents_grid = observations["free_agents_grid"].to(device)
        delivering_agents_grid = observations["delivering_agents_grid"].to(device)
        tasks_grid = observations["tasks_grid"].to(device)
        
        assert torch.all((tasks_num > 0) )
        batch_size = free_agents_grid.size(0) 

        # Extract valid grids and masks
        free_agents_valid, free_agents_mask = self.extract_valid_2d(free_agents_grid, free_agents_num)
        delivering_agents_valid, delivering_agents_mask = self.extract_valid_2d(delivering_agents_grid, delivering_agents_num)
        task_locs_valid, tasks_mask = self.extract_valid_1d(task_loc, tasks_num)
        tasks_grid_valid, tasks_grid_mask = self.extract_valid_2d(tasks_grid, tasks_num)
        free_agents_path_length_valid, free_agents_path_length_mask = self.extract_valid_1d(free_agents_path_length, free_agents_num)
        delivering_agents_path_length_valid, delivering_agents_path_length_mask = self.extract_valid_1d(delivering_agents_path_length, delivering_agents_num)
        
        free_agents_features = self.process_agent_grids(free_agents_valid, free_agents_path_length_valid, self.agent_conv, self.free_agent_mlp)
        delivering_agents_features = self.process_agent_grids(delivering_agents_valid, delivering_agents_path_length_valid, self.agent_conv, self.delivering_agent_mlp)
        
        tasks_features = self.process_task_grids(tasks_grid_valid, self.task_conv)
        # tasks_features_init = self.task_pos_emb(task_locs_valid.long())
        # tasks_embedding = torch.cat([tasks_features_init[:,:,0,:], tasks_features_init[:,:,1,:]], dim=-1)
        # tasks_embedding = self.task_mlp(tasks_embedding)

        agents_features = torch.cat([free_agents_features, delivering_agents_features], dim=1)
        agents_mask = torch.cat([free_agents_mask, delivering_agents_mask], dim=1)

        # # 拼接 Agents 和 Tasks 的特征
        # combined_features = torch.cat([agents_features, tasks_features], dim=1)  # (batch_size, agents_num + tasks_num, hidden_size)
        combined_mask = torch.cat([agents_mask, tasks_mask], dim=1)  # (batch_size, agents_num + tasks_num)

        # 调整形状以匹配 Transformer 输入
        agents_features = agents_features.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        agents_mask_trans = agents_mask == 0  # 转换为 mask 格式 (0: keep, 1: mask)
        # tasks_features = tasks_embedding.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        # tasks_mask_trans = tasks_mask == 0  # 转换为 mask 格式 (0: keep, 1: mask)

        # 通过共享 Transformer
        agents_embedding = self.shared_transformer(agents_features, src_key_padding_mask=agents_mask_trans).permute(1,0,2)  # (seq_len, batch_size, hidden_size)
        # tasks_embedding = self.task_transformer(tasks_features, src_key_padding_mask=tasks_mask_trans).permute(1,0,2)  # (seq_len, batch_size, hidden_size)
        combined_embedding = torch.cat([agents_embedding, tasks_features], dim=1)
        target_length = self.max_free_agents + self.max_delivering_agents + self.max_tasks
        current_length = combined_embedding.size(1)
        if current_length < target_length:
            pad_size = target_length - current_length
            combined_embedding = F.pad(combined_embedding, (0, 0, 0, pad_size, 0, 0))  # (batch, seq, hidden)
            combined_mask = F.pad(combined_mask, (0, pad_size), value=1)  # 新padding部分用1（需要mask）

        # 获取专家动作
        expert_actions = observations["expert_actions"].to(device).float()
        
        return self.pack_obs(
            combined_embedding, 
            free_agents_num, 
            delivering_agents_num, 
            tasks_num,
            expert_actions
        )
