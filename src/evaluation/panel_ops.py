import torch
import numpy as np
from typing import Optional, Tuple

def valid_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """生成非 NaN 且非 Inf 的掩码"""
    return ~(torch.isnan(x) | torch.isnan(y) | torch.isinf(x) | torch.isinf(y))

def pearson_corr_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算两个 1D tensor 的 Pearson 相关系数"""
    mask = valid_mask(x, y)
    if mask.sum() < 2:
        return 0.0
    x_masked = x[mask]
    y_masked = y[mask]
    
    # 检查标准差是否为 0
    if x_masked.std() == 0 or y_masked.std() == 0:
        return 0.0
        
    return torch.corrcoef(torch.stack([x_masked, y_masked]))[0, 1].item()

def spearman_corr_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算两个 1D tensor 的 Spearman Rank 相关系数"""
    mask = valid_mask(x, y)
    if mask.sum() < 2:
        return 0.0
    x_masked = x[mask]
    y_masked = y[mask]
    
    # 转换为 rank
    x_rank = x_masked.argsort().argsort().float()
    y_rank = y_masked.argsort().argsort().float()
    
    return torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1].item()

def cross_sectional_ic(factor: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    计算每日截面 IC。
    factor, target: [n_dates, n_assets]
    """
    n_dates = factor.shape[0]
    ics = np.zeros(n_dates)
    for i in range(n_dates):
        ics[i] = pearson_corr_1d(factor[i], target[i])
    return ics

def cross_sectional_rank_ic(factor: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    计算每日截面 Rank IC。
    factor, target: [n_dates, n_assets]
    """
    n_dates = factor.shape[0]
    ics = np.zeros(n_dates)
    for i in range(n_dates):
        ics[i] = spearman_corr_1d(factor[i], target[i])
    return ics

def coverage_by_day(factor: torch.Tensor) -> np.ndarray:
    """计算每日覆盖度（非空资产比例）"""
    n_dates, n_assets = factor.shape
    nan_mask = torch.isnan(factor) | torch.isinf(factor)
    valid_counts = (~nan_mask).sum(dim=1).float()
    return (valid_counts / n_assets).cpu().numpy()

def winsorize_by_day(factor: torch.Tensor, n_mad: float = 5.0) -> torch.Tensor:
    """
    每日截面去极值 (MAD 方法)。
    """
    res = factor.clone()
    for i in range(factor.shape[0]):
        row = factor[i]
        mask = ~(torch.isnan(row) | torch.isinf(row))
        if mask.sum() == 0:
            continue
        valid_data = row[mask]
        median = valid_data.median()
        mad = (valid_data - median).abs().median()
        if mad == 0:
            continue
        
        limit_up = median + n_mad * mad
        limit_down = median - n_mad * mad
        res[i] = torch.clamp(row, limit_down, limit_up)
    return res

def zscore_by_day(factor: torch.Tensor) -> torch.Tensor:
    """
    每日截面标准化。
    """
    res = factor.clone()
    for i in range(factor.shape[0]):
        row = factor[i]
        mask = ~(torch.isnan(row) | torch.isinf(row))
        if mask.sum() < 2:
            continue
        valid_data = row[mask]
        mean = valid_data.mean()
        std = valid_data.std()
        if std == 0:
            res[i][mask] = 0.0
        else:
            res[i][mask] = (valid_data - mean) / std
    return res
