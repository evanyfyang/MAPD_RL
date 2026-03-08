import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)  # 残差跳连


class PathCNN(nn.Module):
    def __init__(self, in_channels=8, hidden_size=64):
        super().__init__()
        # Stem: 修改 conv1 接受 8 通道，保持 kernel=7、stride=1、pad=3
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )
        # 阶段1: 2 块 dilation=1
        self.l1 = nn.Sequential(
            DilatedBlock(hidden_size, dilation=1),
            DilatedBlock(hidden_size, dilation=1),
        )
        # 阶段2: 2 块 dilation=2
        self.l2 = nn.Sequential(
            DilatedBlock(hidden_size, dilation=2),
            DilatedBlock(hidden_size, dilation=2),
        )
        # 阶段3: 2 块 dilation=2（同上）
        self.l3 = nn.Sequential(
            DilatedBlock(hidden_size, dilation=2),
            DilatedBlock(hidden_size, dilation=2),
        )

    

    def forward(self, x):
        x = self.stem(x)  # (N,64,H,W)
        x = self.l1(x)    # 保留局部细节
        x = self.l2(x)    # 扩展至 ≈ 27×27 感受野
        x = self.l3(x)    # 扩展至 ≈ 55×55+ 感受野
        return x  # 返回 (N,64,H,W) 的位置表示


class GridCNNChannels(nn.Module):
    """
    基于PathCNN的网格模型，直接处理环境提供的8个通道特征
    输入: 8通道的2D特征图 (obstacle_map, free_agent_map, delivering_agent_map, 
          delivering_task_id_map, pickup_location_map, delivery_location_map, 
          pickup_distances, delivery_distances)
    输出: 节点特征，与GridGCN接口兼容
    """
    def __init__(self, grid_feature_dim=2, hidden_dim=128, num_layers=3, dropout=0.1, 
                 input_channels=8, use_pretrained=True):
        super(GridCNNChannels, self).__init__()
        self.grid_feature_dim = grid_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.use_pretrained = use_pretrained
        
        # PathCNN backbone处理多通道输入
        # 为了更好地利用ResNet预训练权重，PathCNN使用64通道
        self.path_cnn = PathCNN(in_channels=input_channels, hidden_size=hidden_dim)
        
        # 如果hidden_dim != 64，添加投影层
        if hidden_dim != 64:
            self.cnn_projection = nn.Conv2d(64, hidden_dim, kernel_size=1, bias=False)
        else:
            self.cnn_projection = nn.Identity()
        
        # 如果使用预训练权重，尝试加载ResNet权重
        if use_pretrained and self.hidden_dim==64:
            self._load_pretrained_weights()
        
    def _load_pretrained_weights(self):
        """尝试加载ResNet-18的预训练权重"""
        try:
            from torchvision.models import resnet18
            
            # 加载预训练的ResNet-18
            pre = resnet18(weights='IMAGENET1K_V1')  # 使用新的weights参数
            
            # 复制 conv1 权重：先平均再扩展到 8 通道
            w = pre.conv1.weight.data            # (64,3,7,7)
            w_avg = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
            
            # 确保我们的PathCNN使用相同的hidden_size
            if self.path_cnn.stem[0].weight.data.shape[0] == w.shape[0]:  # 都是64通道
                self.path_cnn.stem[0].weight.data = w_avg.repeat(1, self.input_channels, 1, 1)  # (64,8,7,7)
                
                # 复制BatchNorm权重
                self.path_cnn.stem[1].weight.data = pre.bn1.weight.data.clone()
                self.path_cnn.stem[1].bias.data = pre.bn1.bias.data.clone()
                self.path_cnn.stem[1].running_mean.data = pre.bn1.running_mean.data.clone()
                self.path_cnn.stem[1].running_var.data = pre.bn1.running_var.data.clone()
                
                print(f"✓ 成功加载ResNet-18预训练权重到stem层")
                
                # 尝试加载layer1的权重到l1
                try:
                    # ResNet的layer1对应我们的l1
                    # layer1.0.conv1 -> l1.0.conv1
                    if hasattr(pre.layer1[0], 'conv1') and hasattr(self.path_cnn.l1[0], 'conv1'):
                        self.path_cnn.l1[0].conv1.weight.data = pre.layer1[0].conv1.weight.data.clone()
                        self.path_cnn.l1[0].bn1.weight.data = pre.layer1[0].bn1.weight.data.clone()
                        self.path_cnn.l1[0].bn1.bias.data = pre.layer1[0].bn1.bias.data.clone()
                        self.path_cnn.l1[0].bn1.running_mean.data = pre.layer1[0].bn1.running_mean.data.clone()
                        self.path_cnn.l1[0].bn1.running_var.data = pre.layer1[0].bn1.running_var.data.clone()
                        
                        self.path_cnn.l1[0].conv2.weight.data = pre.layer1[0].conv2.weight.data.clone()
                        self.path_cnn.l1[0].bn2.weight.data = pre.layer1[0].bn2.weight.data.clone()
                        self.path_cnn.l1[0].bn2.bias.data = pre.layer1[0].bn2.bias.data.clone()
                        self.path_cnn.l1[0].bn2.running_mean.data = pre.layer1[0].bn2.running_mean.data.clone()
                        self.path_cnn.l1[0].bn2.running_var.data = pre.layer1[0].bn2.running_var.data.clone()
                        
                        # 第二个block
                        if len(pre.layer1) > 1 and len(self.path_cnn.l1) > 1:
                            self.path_cnn.l1[1].conv1.weight.data = pre.layer1[1].conv1.weight.data.clone()
                            self.path_cnn.l1[1].bn1.weight.data = pre.layer1[1].bn1.weight.data.clone()
                            self.path_cnn.l1[1].bn1.bias.data = pre.layer1[1].bn1.bias.data.clone()
                            self.path_cnn.l1[1].bn1.running_mean.data = pre.layer1[1].bn1.running_mean.data.clone()
                            self.path_cnn.l1[1].bn1.running_var.data = pre.layer1[1].bn1.running_var.data.clone()
                            
                            self.path_cnn.l1[1].conv2.weight.data = pre.layer1[1].conv2.weight.data.clone()
                            self.path_cnn.l1[1].bn2.weight.data = pre.layer1[1].bn2.weight.data.clone()
                            self.path_cnn.l1[1].bn2.bias.data = pre.layer1[1].bn2.bias.data.clone()
                            self.path_cnn.l1[1].bn2.running_mean.data = pre.layer1[1].bn2.running_mean.data.clone()
                            self.path_cnn.l1[1].bn2.running_var.data = pre.layer1[1].bn2.running_var.data.clone()
                        
                        print(f"✓ 成功加载ResNet-18的layer1权重到l1")
                        
                except Exception as e:
                    print(f"⚠ 无法加载layer1权重: {e}")
                    
            else:
                print(f"⚠ 通道数不匹配: PathCNN={self.path_cnn.stem[0].weight.data.shape[0]}, ResNet={w.shape[0]}")
                
        except Exception as e:
            print(f"⚠ 无法加载ResNet-18预训练权重: {e}")
            print("  使用默认初始化")
        
    def forward(self, grid_features, edge_index, cnn_channels=None):
        """
        grid_features: [num_nodes, grid_feature_dim] - 节点坐标特征
        edge_index: [2, num_edges] - 边索引（用于兼容性，实际不使用）
        cnn_channels: [batch_size, 8, height, width] - 8通道的CNN输入特征
        
        Returns: [num_nodes, hidden_dim] - 处理后的节点特征
        """
        device = grid_features[0].device
        num_nodes = grid_features[0].shape[0]
        
        if num_nodes == 0:
            return torch.zeros((0, self.hidden_dim), device=device)
        
        cnn_input = cnn_channels  # [batch_size, 8, height, width]
        
        cnn_output = self.path_cnn(cnn_input)  
        
        batch_size, _, height, width = cnn_output.shape

        output_features = cnn_output.reshape(batch_size, -1, height * width).permute(0,2,1)
        
        return output_features 