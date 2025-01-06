import torch
from torch.nn import functional as F
from torch import nn
import math
from transformer import Transformer
from ..mapf_solver import mapf_solver

class RLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Transformer()
        self.baseline_model = Transformer()

        self.read_grids()
        # 每个step随机选择一种grid生成任务，使用模型计算model的reward并且更新
        # grid都存储在一个文件夹下
        # init的时候进行全部的solver的初始化，即获取其heuristic
            # 也许c++代码需要更新
        self.init_mapf_solvers()

    def read_grids(self):
        return None

    def init_mapf_solvers(self):
        # add args for mapf_solver for each grid type
        self.solver = mapf_solver.PBSSolver()

    # 每个step中根据给定的grid生成，并且调用solver的update_task
    def generate_tasks(self):

        
        return None
    
    # Input: 
    def solve_grid(self):
        return None

    def get_reward(self):
        return None

    def forward(self,x):
        tasks = self.generate_tasks()

        self.solve_grid()
        rewards = self.get_rewards()
        return None