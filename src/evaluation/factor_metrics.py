import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from .panel_ops import cross_sectional_ic, cross_sectional_rank_ic, coverage_by_day

@dataclass
class FactorMetrics:
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    positive_ic_ratio: float = 0.0
    coverage_mean: float = 0.0
    nan_ratio: float = 0.0
    
    # 额外指标
    ic_adj: float = 0.0  # 方向调整后的 IC
    rank_ic_adj: float = 0.0
    
    def to_dict(self):
        return asdict(self)

class FactorMetricsEvaluator:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def evaluate_tensor(self, factor: torch.Tensor, target: torch.Tensor, direction: float = 1.0) -> Dict[str, Any]:
        """
        根据因子值和目标值计算各项指标。
        factor, target: [n_dates, n_assets]
        """
        # 确保在正确的设备上
        factor = factor.to(self.device)
        target = target.to(self.device)
        
        # 1. 计算每日 IC
        daily_ic = cross_sectional_ic(factor, target)
        daily_rank_ic = cross_sectional_rank_ic(factor, target)
        
        # 2. 计算基本统计量
        ic_mean = np.nanmean(daily_ic)
        ic_std = np.nanstd(daily_ic)
        ic_ir = ic_mean / (ic_std + 1e-8)
        
        rank_ic_mean = np.nanmean(daily_rank_ic)
        rank_ic_std = np.nanstd(daily_rank_ic)
        rank_ic_ir = rank_ic_mean / (rank_ic_std + 1e-8)
        
        # 3. 覆盖度与稳定性
        daily_coverage = coverage_by_day(factor)
        coverage_mean = np.nanmean(daily_coverage)
        
        positive_ic_ratio = np.nanmean(daily_ic > 0)
        
        # 4. NaN 比例 (全样本)
        total_elements = factor.numel()
        nan_elements = (torch.isnan(factor) | torch.isinf(factor)).sum().item()
        nan_ratio = nan_elements / total_elements
        
        metrics = FactorMetrics(
            ic_mean=float(ic_mean),
            ic_std=float(ic_std),
            ic_ir=float(ic_ir),
            rank_ic_mean=float(rank_ic_mean),
            rank_ic_std=float(rank_ic_std),
            rank_ic_ir=float(rank_ic_ir),
            positive_ic_ratio=float(positive_ic_ratio),
            coverage_mean=float(coverage_mean),
            nan_ratio=float(nan_ratio),
            ic_adj=float(ic_mean * direction),
            rank_ic_adj=float(rank_ic_mean * direction)
        )
        
        return metrics.to_dict()

    def value_sanity(self, factor: torch.Tensor) -> Dict[str, Any]:
        """检查因子值的健康状况"""
        nan_mask = torch.isnan(factor) | torch.isinf(factor)
        nan_ratio = nan_mask.float().mean().item()
        
        # 检查是否全空
        if nan_ratio >= 1.0:
            return {"ok": False, "reason": "all_nan", "nan_ratio": nan_ratio}
            
        # 检查覆盖度
        daily_coverage = coverage_by_day(factor)
        min_coverage = np.min(daily_coverage)
        if min_coverage < 0.01: # 允许部分日期低覆盖，但不能完全没数
             pass
             
        # 检查是否为常数 (全样本 std)
        # 先去掉 NaN
        valid_vals = factor[~nan_mask]
        if len(valid_vals) < 2:
            return {"ok": False, "reason": "not_enough_valid_values", "nan_ratio": nan_ratio}
            
        if valid_vals.std() == 0:
            return {"ok": False, "reason": "constant_value", "nan_ratio": nan_ratio}
            
        return {"ok": True, "nan_ratio": nan_ratio}
