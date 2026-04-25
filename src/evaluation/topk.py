import torch
import numpy as np
from typing import List, Dict, Any, Optional

def calculate_topk_returns(
    factor: torch.Tensor, 
    target: torch.Tensor, 
    k: int = 50, 
    long_short: bool = True
) -> Dict[str, np.ndarray]:
    """
    计算每日 Top-K 收益。
    factor, target: [n_dates, n_assets]
    """
    n_dates = factor.shape[0]
    long_rets = np.zeros(n_dates)
    short_rets = np.zeros(n_dates)
    ls_rets = np.zeros(n_dates)
    
    for i in range(n_dates):
        f_row = factor[i]
        t_row = target[i]
        
        # 排除无效值
        mask = ~(torch.isnan(f_row) | torch.isnan(t_row) | torch.isinf(f_row) | torch.isinf(t_row))
        if mask.sum() < k * 2: # 至少要有 2k 个有效资产才能做多空
            # 如果资产不足，退而求其次
            valid_k = max(1, mask.sum().item() // 4)
        else:
            valid_k = k
            
        if mask.sum() < 2:
            continue
            
        f_valid = f_row[mask]
        t_valid = t_row[mask]
        
        # 排序
        indices = f_valid.argsort()
        
        # Long (Top K)
        long_idx = indices[-valid_k:]
        long_ret = t_valid[long_idx].mean().item()
        long_rets[i] = long_ret
        
        # Short (Bottom K)
        short_idx = indices[:valid_k]
        short_ret = t_valid[short_idx].mean().item()
        short_rets[i] = short_ret
        
        ls_rets[i] = long_ret - short_ret
        
    return {
        "long_returns": long_rets,
        "short_returns": short_rets,
        "ls_returns": ls_rets
    }

def calculate_topk_turnover(factor: torch.Tensor, k: int = 50) -> float:
    """
    计算 Top-K 换手率。
    """
    n_dates = factor.shape[0]
    if n_dates < 2:
        return 0.0
        
    turnovers = []
    prev_top_indices = None
    
    for i in range(n_dates):
        f_row = factor[i]
        mask = ~(torch.isnan(f_row) | torch.isinf(f_row))
        if mask.sum() < k:
            continue
            
        # 获取 Top K 的资产索引
        # 注意：这里需要原始索引
        valid_indices = torch.where(mask)[0]
        f_valid = f_row[mask]
        
        top_k_sub_idx = f_valid.argsort()[-k:]
        top_k_orig_idx = set(valid_indices[top_k_sub_idx].tolist())
        
        if prev_top_indices is not None:
            intersection = top_k_orig_idx.intersection(prev_top_indices)
            turnover = 1.0 - len(intersection) / k
            turnovers.append(turnover)
            
        prev_top_indices = top_k_orig_idx
        
    return float(np.mean(turnovers)) if turnovers else 0.0
