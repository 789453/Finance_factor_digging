import torch
import numpy as np
from typing import Optional, Tuple, List

def valid_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """生成非 NaN 且非 Inf 的掩码"""
    return ~(torch.isnan(x) | torch.isnan(y) | torch.isinf(x) | torch.isinf(y))

def safe_rankdata_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    Safely compute rank on CPU to avoid large OOMs.
    x: shape (N,) or (N, D)
    """
    if x.dim() == 1:
        n = x.size(0)
        sorted_indices = torch.argsort(x)
        ranks = torch.empty_like(x)
        ranks[sorted_indices] = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        return ranks
    elif x.dim() == 2:
        ranks = torch.empty_like(x)
        for i in range(x.size(0)):
            n = x.size(1)
            sorted_indices = torch.argsort(x[i])
            ranks[i, sorted_indices] = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        return ranks
    else:
        raise ValueError("safe_rankdata_cpu only supports 1D or 2D tensors")

def batch_pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate pearson correlation for each row in x and y.
    x, y: (batch, N)
    returns: (batch,)
    """
    mean_x = x.mean(dim=1, keepdim=True)
    mean_y = y.mean(dim=1, keepdim=True)
    xm = x - mean_x
    ym = y - mean_y
    r_num = (xm * ym).sum(dim=1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1)
    r_den = torch.clamp(r_den, min=1e-8)
    r = r_num / r_den
    r = torch.clamp(r, -1.0, 1.0)
    return r

def batch_spearmanr_safe(x: torch.Tensor, y: torch.Tensor, mode: str = "cpu_spearman") -> torch.Tensor:
    """
    Calculate spearman correlation safely.
    x, y: (batch, N)
    """
    orig_device = x.device
    if mode == "cpu_spearman":
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        rank_x = safe_rankdata_cpu(x_cpu)
        rank_y = safe_rankdata_cpu(y_cpu)
        return batch_pearsonr(rank_x, rank_y).to(orig_device)
    elif mode == "pearson_proxy":
        return batch_pearsonr(x, y)
    else:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        rank_x = safe_rankdata_cpu(x_cpu)
        rank_y = safe_rankdata_cpu(y_cpu)
        return batch_pearsonr(rank_x, rank_y).to(orig_device)

def pearson_corr_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算两个 1D tensor 的 Pearson 相关系数"""
    mask = valid_mask(x, y)
    if mask.sum() < 2:
        return 0.0
    x_masked = x[mask]
    y_masked = y[mask]
    
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
    
    x_rank = x_masked.argsort().argsort().float()
    y_rank = y_masked.argsort().argsort().float()
    
    return torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1].item()

def cross_sectional_ic(factor: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """计算每日截面 IC。"""
    n_dates = factor.shape[0]
    ics = np.zeros(n_dates)
    # 使用 batch 处理优化性能
    valid_mask_all = ~(torch.isnan(factor) | torch.isnan(target) | torch.isinf(factor) | torch.isinf(target))
    for i in range(n_dates):
        mask = valid_mask_all[i]
        if mask.sum() < 2:
            ics[i] = 0.0
            continue
        f_v = factor[i, mask].unsqueeze(0)
        t_v = target[i, mask].unsqueeze(0)
        ics[i] = batch_pearsonr(f_v, t_v).item()
    return ics

def cross_sectional_rank_ic(factor: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """计算每日截面 Rank IC。"""
    n_dates = factor.shape[0]
    ics = np.zeros(n_dates)
    valid_mask_all = ~(torch.isnan(factor) | torch.isnan(target) | torch.isinf(factor) | torch.isinf(target))
    for i in range(n_dates):
        mask = valid_mask_all[i]
        if mask.sum() < 2:
            ics[i] = 0.0
            continue
        f_v = factor[i, mask].unsqueeze(0)
        t_v = target[i, mask].unsqueeze(0)
        ics[i] = batch_spearmanr_safe(f_v, t_v).item()
    return ics

def compute_stability_score(ic_s: torch.Tensor, window: int = 20) -> float:
    """计算 IC 序列的稳定性得分"""
    if len(ic_s) < window:
        return 0.0
    
    stability_scores = []
    for i in range(window, len(ic_s)):
        window_ic = ic_s[i-window:i]
        std = window_ic.std().item()
        stability = 1.0 / (std + 1e-8)
        stability_scores.append(stability)
    
    return float(np.mean(stability_scores)) if stability_scores else 0.0

def coverage_by_day(factor: torch.Tensor) -> np.ndarray:
    """计算每日覆盖度（非空资产比例）"""
    n_dates, n_assets = factor.shape
    nan_mask = torch.isnan(factor) | torch.isinf(factor)
    valid_counts = (~nan_mask).sum(dim=1).float()
    return (valid_counts / n_assets).cpu().numpy()

def compute_topk_returns(factor: torch.Tensor, returns: torch.Tensor, k_ratio: float = 0.1) -> np.ndarray:
    """计算 TopK 收益序列"""
    n_dates = factor.shape[0]
    ret_s = np.zeros(n_dates)
    for d in range(n_dates):
        mask = ~(torch.isnan(factor[d]) | torch.isnan(returns[d]) | torch.isinf(factor[d]) | torch.isinf(returns[d]))
        if mask.sum() < 2:
            ret_s[d] = 0.0
            continue
        
        valid_f = factor[d, mask]
        valid_r = returns[d, mask]
        k = max(1, int(k_ratio * valid_f.size(0)))
        _, indices = torch.topk(valid_f, k)
        ret_s[d] = valid_r[indices].mean().item()
    return ret_s

def winsorize_by_day(factor: torch.Tensor, n_mad: float = 5.0) -> torch.Tensor:
    """每日截面去极值 (MAD 方法)。"""
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
    """每日截面标准化。"""
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
