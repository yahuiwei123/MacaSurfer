import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MGDASolver:
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.num_tasks = len(tasks)
    
    def solve_mgda_weights(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        求解MGDA权重
        """
        if self.num_tasks == 1:
            return torch.tensor([1.0])
        
        # 构建Gram矩阵
        gram_matrix = torch.zeros(self.num_tasks, self.num_tasks)
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                gram_matrix[i, j] = torch.dot(gradients[i], gradients[j])
        
        # 求解二次规划问题
        try:
            # 方法1: 使用最小二乘求解
            ones = torch.ones(self.num_tasks, 1)
            weights = torch.linalg.solve(gram_matrix, ones).squeeze()
            
            # 确保权重非负
            weights = torch.clamp(weights, min=0)
            
            # 归一化
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = torch.ones(self.num_tasks) / self.num_tasks
                
        except:
            # 如果求解失败，使用均匀权重
            weights = torch.ones(self.num_tasks) / self.num_tasks
        
        return weights