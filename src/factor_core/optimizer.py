import torch
import numpy as np
from typing import List, Tuple, Optional
from alphagen.data.expression import Expression

class FactorEnsembleOptimizer:
    """
    因子组合优化器：使用 L1 正则化优化因子权重，
    能够自动识别并降低冗余因子的权重。
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def optimize_weights(self, 
                         single_ics: np.ndarray, 
                         mutual_ics: np.ndarray, 
                         alpha: float = 0.005, 
                         lr: float = 0.0005, 
                         n_iter: int = 500) -> np.ndarray:
        """
        核心优化逻辑：最小化 (mut_ic_sum - 2 * ret_ic_sum + 1) + alpha * L1_norm
        """
        n = len(single_ics)
        if n == 0: return np.array([])
        if n == 1: return np.array([1.0])

        ics_ret = torch.from_numpy(single_ics).to(self.device).float()
        ics_mut = torch.from_numpy(mutual_ics).to(self.device).float()
        weights = torch.zeros(n, device=self.device, requires_grad=True)
        
        # 初始权重设为单因子 IC
        with torch.no_grad():
            weights.copy_(ics_ret)
            
        optimizer = torch.optim.Adam([weights], lr=lr)

        best_w = weights.detach().cpu().numpy()
        min_loss = float('inf')

        for _ in range(n_iter):
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_l1 = torch.norm(weights, p=1)
            loss = loss_ic + alpha * loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_w = weights.detach().cpu().numpy()

        return best_w
