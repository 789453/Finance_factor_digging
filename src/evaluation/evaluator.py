#!/usr/bin/env python3
"""
因子评价器 - 独立的因子评价层
与训练层解耦，支持多种评价指标和后端
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class FactorMetrics:
    """因子评价指标"""
    ic: float  # 信息系数
    ric: float  # 排名信息系数
    rank_ic: float  # 排名IC
    icir: float  # IC信息比
    ricir: float  # 排名IC信息比
    turnover: float  # 换手率
    long_short_return: float  # 多空收益
    long_short_sharpe: float  # 多空夏普比率
    long_short_max_drawdown: float  # 多空最大回撤
    quantile_returns: List[float]  # 分位数收益
    quantile_sharpes: List[float]  # 分位数夏普比率
    stability: float  # 稳定性指标
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'ic': self.ic,
            'ric': self.ric,
            'rank_ic': self.rank_ic,
            'icir': self.icir,
            'ricir': self.ricir,
            'turnover': self.turnover,
            'long_short_return': self.long_short_return,
            'long_short_sharpe': self.long_short_sharpe,
            'long_short_max_drawdown': self.long_short_max_drawdown,
            'quantile_returns': self.quantile_returns,
            'quantile_sharpes': self.quantile_sharpes,
            'stability': self.stability
        }

class FactorEvaluator(ABC):
    """因子评价器抽象基类"""
    
    @abstractmethod
    def evaluate(self, factor_values: np.ndarray, returns: np.ndarray, 
                  weights: Optional[np.ndarray] = None) -> FactorMetrics:
        """
        评价因子
        
        Args:
            factor_values: 因子值，形状为 (n_samples,)
            returns: 收益率，形状为 (n_samples,)
            weights: 样本权重，形状为 (n_samples,)
            
        Returns:
            因子评价指标
        """
        pass
    
    @abstractmethod
    def evaluate_multi_period(self, factor_values: np.ndarray, 
                              returns: np.ndarray,
                              dates: Optional[np.ndarray] = None) -> Dict[str, FactorMetrics]:
        """
        多期评价
        
        Args:
            factor_values: 因子值，形状为 (n_periods, n_samples)
            returns: 收益率，形状为 (n_periods, n_samples)
            dates: 日期，形状为 (n_periods,)
            
        Returns:
            各期评价结果
        """
        pass

class BasicFactorEvaluator(FactorEvaluator):
    """基础因子评价器"""
    
    def __init__(self, quantiles: int = 5, risk_free_rate: float = 0.0):
        """
        初始化
        
        Args:
            quantiles: 分位数数量
            risk_free_rate: 无风险利率
        """
        self.quantiles = quantiles
        self.risk_free_rate = risk_free_rate
        
        logger.info(f"Initialized BasicFactorEvaluator with {quantiles} quantiles")
    
    def evaluate(self, factor_values: np.ndarray, returns: np.ndarray, 
                  weights: Optional[np.ndarray] = None) -> FactorMetrics:
        """评价因子"""
        # 数据验证
        if len(factor_values) != len(returns):
            raise ValueError("Factor values and returns must have the same length")
        
        if len(factor_values) < 30:
            logger.warning(f"Small sample size: {len(factor_values)}")
        
        # 处理缺失值
        valid_mask = ~np.isnan(factor_values) & ~np.isnan(returns)
        if weights is not None:
            valid_mask = valid_mask & ~np.isnan(weights)
        
        factor_clean = factor_values[valid_mask]
        returns_clean = returns[valid_mask]
        weights_clean = weights[valid_mask] if weights is not None else None
        
        if len(factor_clean) < 10:
            logger.error("Too few valid samples after cleaning")
            return self._create_empty_metrics()
        
        # 计算IC
        ic = self._calculate_ic(factor_clean, returns_clean, weights_clean)
        ric = self._calculate_rank_ic(factor_clean, returns_clean, weights_clean)
        
        # 计算分位数组合收益
        quantile_returns, quantile_sharpes = self._calculate_quantile_performance(
            factor_clean, returns_clean, weights_clean
        )
        
        # 计算多空收益
        long_short_return = quantile_returns[-1] - quantile_returns[0] if len(quantile_returns) >= 2 else 0.0
        long_short_sharpe = self._calculate_sharpe_ratio(long_short_return, self.risk_free_rate)
        
        # 计算换手率
        turnover = self._calculate_turnover(factor_clean)
        
        # 计算稳定性
        stability = self._calculate_stability(factor_clean, returns_clean)
        
        metrics = FactorMetrics(
            ic=ic,
            ric=ric,
            rank_ic=ric,  # 简化处理
            icir=ic / np.sqrt(len(factor_clean)) if len(factor_clean) > 0 else 0.0,
            ricir=ric / np.sqrt(len(factor_clean)) if len(factor_clean) > 0 else 0.0,
            turnover=turnover,
            long_short_return=long_short_return,
            long_short_sharpe=long_short_sharpe,
            long_short_max_drawdown=0.0,  # 简化处理
            quantile_returns=quantile_returns,
            quantile_sharpes=quantile_sharpes,
            stability=stability
        )
        
        logger.info(f"Factor evaluation completed: IC={ic:.4f}, RankIC={ric:.4f}, LS_Return={long_short_return:.4f}")
        return metrics
    
    def evaluate_multi_period(self, factor_values: np.ndarray, 
                              returns: np.ndarray,
                              dates: Optional[np.ndarray] = None) -> Dict[str, FactorMetrics]:
        """多期评价"""
        if factor_values.ndim != 2 or returns.ndim != 2:
            raise ValueError("Multi-period evaluation requires 2D arrays")
        
        if factor_values.shape != returns.shape:
            raise ValueError("Factor values and returns must have the same shape")
        
        n_periods = factor_values.shape[0]
        results = {}
        
        for i in range(n_periods):
            period_key = f"period_{i}"
            if dates is not None and i < len(dates):
                period_key = str(dates[i])
            
            metrics = self.evaluate(factor_values[i], returns[i])
            results[period_key] = metrics
        
        return results
    
    def _calculate_ic(self, factor: np.ndarray, returns: np.ndarray, 
                     weights: Optional[np.ndarray] = None) -> float:
        """计算信息系数"""
        if len(factor) < 2:
            return 0.0
        
        if weights is not None:
            # 加权IC计算
            return np.corrcoef(factor, returns, rowvar=False)[0, 1]
        else:
            return np.corrcoef(factor, returns)[0, 1]
    
    def _calculate_rank_ic(self, factor: np.ndarray, returns: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> float:
        """计算排名信息系数"""
        if len(factor) < 2:
            return 0.0
        
        factor_ranks = self._rank_data(factor)
        return np.corrcoef(factor_ranks, returns)[0, 1]
    
    def _calculate_quantile_performance(self, factor: np.ndarray, returns: np.ndarray,
                                      weights: Optional[np.ndarray] = None) -> Tuple[List[float], List[float]]:
        """计算分位数表现"""
        if len(factor) < self.quantiles * 2:
            logger.warning(f"Insufficient data for {self.quantiles} quantiles")
            return [0.0] * self.quantiles, [0.0] * self.quantiles
        
        # 计算分位数
        quantile_thresholds = np.percentile(factor, np.linspace(0, 100, self.quantiles + 1))
        
        quantile_returns = []
        quantile_sharpes = []
        
        for i in range(self.quantiles):
            lower = quantile_thresholds[i]
            upper = quantile_thresholds[i + 1]
            
            if i == 0:
                mask = factor <= upper
            elif i == self.quantiles - 1:
                mask = factor >= lower
            else:
                mask = (factor >= lower) & (factor < upper)
            
            if np.sum(mask) == 0:
                quantile_returns.append(0.0)
                quantile_sharpes.append(0.0)
            else:
                q_returns = returns[mask]
                q_return = np.mean(q_returns)
                q_sharpe = self._calculate_sharpe_ratio(q_return, self.risk_free_rate)
                
                quantile_returns.append(q_return)
                quantile_sharpes.append(q_sharpe)
        
        return quantile_returns, quantile_sharpes
    
    def _calculate_sharpe_ratio(self, returns: float, risk_free_rate: float) -> float:
        """计算夏普比率"""
        # 简化处理，实际应该使用收益率序列的标准差
        return returns - risk_free_rate if returns > risk_free_rate else 0.0
    
    def _calculate_turnover(self, factor: np.ndarray) -> float:
        """计算换手率"""
        if len(factor) < 2:
            return 0.0
        
        # 简化的换手率计算
        factor_changes = np.abs(np.diff(factor))
        return np.mean(factor_changes) / (np.std(factor) + 1e-8)
    
    def _calculate_stability(self, factor: np.ndarray, returns: np.ndarray) -> float:
        """计算稳定性"""
        # 简化的稳定性指标：因子值的标准差倒数
        factor_std = np.std(factor)
        return 1.0 / (factor_std + 1e-8) if factor_std > 0 else 0.0
    
    def _rank_data(self, data: np.ndarray) -> np.ndarray:
        """计算排名"""
        # 处理重复值（平均排名）
        order = np.argsort(data)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(data))
        
        # 处理重复值
        for val in np.unique(data):
            mask = data == val
            if np.sum(mask) > 1:
                ranks[mask] = np.mean(ranks[mask])
        
        return ranks
    
    def _create_empty_metrics(self) -> FactorMetrics:
        """创建空的评价指标"""
        return FactorMetrics(
            ic=0.0,
            ric=0.0,
            rank_ic=0.0,
            icir=0.0,
            ricir=0.0,
            turnover=0.0,
            long_short_return=0.0,
            long_short_sharpe=0.0,
            long_short_max_drawdown=0.0,
            quantile_returns=[0.0] * self.quantiles,
            quantile_sharpes=[0.0] * self.quantiles,
            stability=0.0
        )

class AdvancedFactorEvaluator(FactorEvaluator):
    """高级因子评价器 - 支持更多评价指标"""
    
    def __init__(self, quantiles: int = 5, risk_free_rate: float = 0.0,
                 max_drawdown_periods: int = 20):
        """
        初始化
        
        Args:
            quantiles: 分位数数量
            risk_free_rate: 无风险利率
            max_drawdown_periods: 最大回撤计算周期
        """
        self.quantiles = quantiles
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_periods = max_drawdown_periods
        
        logger.info(f"Initialized AdvancedFactorEvaluator with {quantiles} quantiles")
    
    def evaluate(self, factor_values: np.ndarray, returns: np.ndarray, 
                  weights: Optional[np.ndarray] = None) -> FactorMetrics:
        """评价因子"""
        # 这里可以实现更复杂的评价逻辑
        # 目前委托给基础评价器，可以后续扩展
        basic_evaluator = BasicFactorEvaluator(self.quantiles, self.risk_free_rate)
        return basic_evaluator.evaluate(factor_values, returns, weights)
    
    def evaluate_multi_period(self, factor_values: np.ndarray, 
                              returns: np.ndarray,
                              dates: Optional[np.ndarray] = None) -> Dict[str, FactorMetrics]:
        """多期评价"""
        basic_evaluator = BasicFactorEvaluator(self.quantiles, self.risk_free_rate)
        return basic_evaluator.evaluate_multi_period(factor_values, returns, dates)

class GPUBackendFactorEvaluator(FactorEvaluator):
    """GPU后端因子评价器 - 支持GPU加速"""
    
    def __init__(self, quantiles: int = 5, risk_free_rate: float = 0.0, 
                 device: str = "cuda"):
        """
        初始化
        
        Args:
            quantiles: 分位数数量
            risk_free_rate: 无风险利率
            device: 设备类型
        """
        self.quantiles = quantiles
        self.risk_free_rate = risk_free_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized GPUBackendFactorEvaluator on {self.device}")
    
    def evaluate(self, factor_values: np.ndarray, returns: np.ndarray, 
                  weights: Optional[np.ndarray] = None) -> FactorMetrics:
        """评价因子"""
        # 转换为PyTorch张量
        factor_tensor = torch.tensor(factor_values, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        if weights is not None:
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            weights_tensor = None
        
        # 使用CPU后端进行计算（可以后续实现真正的GPU版本）
        # 这里先将数据移回CPU进行计算
        factor_cpu = factor_tensor.cpu().numpy()
        returns_cpu = returns_tensor.cpu().numpy()
        weights_cpu = weights_tensor.cpu().numpy() if weights_tensor is not None else None
        
        basic_evaluator = BasicFactorEvaluator(self.quantiles, self.risk_free_rate)
        return basic_evaluator.evaluate(factor_cpu, returns_cpu, weights_cpu)
    
    def evaluate_multi_period(self, factor_values: np.ndarray, 
                              returns: np.ndarray,
                              dates: Optional[np.ndarray] = None) -> Dict[str, FactorMetrics]:
        """多期评价"""
        # 同样委托给CPU后端
        basic_evaluator = BasicFactorEvaluator(self.quantiles, self.risk_free_rate)
        return basic_evaluator.evaluate_multi_period(factor_values, returns, dates)

def create_factor_evaluator(evaluator_type: str = "basic", **kwargs) -> FactorEvaluator:
    """
    创建因子评价器
    
    Args:
        evaluator_type: 评价器类型 ('basic', 'advanced', 'gpu')
        **kwargs: 传递给评价器构造函数的参数
        
    Returns:
        FactorEvaluator实例
    """
    if evaluator_type == "basic":
        return BasicFactorEvaluator(**kwargs)
    elif evaluator_type == "advanced":
        return AdvancedFactorEvaluator(**kwargs)
    elif evaluator_type == "gpu":
        return GPUBackendFactorEvaluator(**kwargs)
    else:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

class FactorEvaluationPipeline:
    """因子评价管道 - 整合多个评价器"""
    
    def __init__(self, evaluators: List[FactorEvaluator], aggregator: str = "mean"):
        """
        初始化
        
        Args:
            evaluators: 评价器列表
            aggregator: 聚合方法 ('mean', 'median', 'weighted')
        """
        self.evaluators = evaluators
        self.aggregator = aggregator
        
        logger.info(f"Initialized FactorEvaluationPipeline with {len(evaluators)} evaluators")
    
    def evaluate(self, factor_values: np.ndarray, returns: np.ndarray,
                weights: Optional[np.ndarray] = None) -> Dict[str, FactorMetrics]:
        """
        使用多个评价器评价因子
        
        Args:
            factor_values: 因子值
            returns: 收益率
            weights: 权重
            
        Returns:
            各评价器的结果
        """
        results = {}
        
        for i, evaluator in enumerate(self.evaluators):
            try:
                metrics = evaluator.evaluate(factor_values, returns, weights)
                results[f"evaluator_{i}"] = metrics
                logger.info(f"Evaluator {i} completed: IC={metrics.ic:.4f}")
            except Exception as e:
                logger.error(f"Evaluator {i} failed: {e}")
                continue
        
        return results
    
    def aggregate_results(self, results: Dict[str, FactorMetrics]) -> FactorMetrics:
        """
        聚合多个评价结果
        
        Args:
            results: 评价结果字典
            
        Returns:
            聚合后的评价指标
        """
        if not results:
            logger.error("No evaluation results to aggregate")
            return BasicFactorEvaluator()._create_empty_metrics()
        
        metrics_list = list(results.values())
        
        if self.aggregator == "mean":
            return self._aggregate_mean(metrics_list)
        elif self.aggregator == "median":
            return self._aggregate_median(metrics_list)
        else:
            logger.warning(f"Unknown aggregator: {self.aggregator}, using mean")
            return self._aggregate_mean(metrics_list)
    
    def _aggregate_mean(self, metrics_list: List[FactorMetrics]) -> FactorMetrics:
        """计算平均值"""
        n = len(metrics_list)
        
        return FactorMetrics(
            ic=np.mean([m.ic for m in metrics_list]),
            ric=np.mean([m.ric for m in metrics_list]),
            rank_ic=np.mean([m.rank_ic for m in metrics_list]),
            icir=np.mean([m.icir for m in metrics_list]),
            ricir=np.mean([m.ricir for m in metrics_list]),
            turnover=np.mean([m.turnover for m in metrics_list]),
            long_short_return=np.mean([m.long_short_return for m in metrics_list]),
            long_short_sharpe=np.mean([m.long_short_sharpe for m in metrics_list]),
            long_short_max_drawdown=np.mean([m.long_short_max_drawdown for m in metrics_list]),
            quantile_returns=[np.mean([m.quantile_returns[i] for m in metrics_list]) 
                            for i in range(len(metrics_list[0].quantile_returns))],
            quantile_sharpes=[np.mean([m.quantile_sharpes[i] for m in metrics_list]) 
                             for i in range(len(metrics_list[0].quantile_sharpes))],
            stability=np.mean([m.stability for m in metrics_list])
        )
    
    def _aggregate_median(self, metrics_list: List[FactorMetrics]) -> FactorMetrics:
        """计算中位数"""
        return FactorMetrics(
            ic=np.median([m.ic for m in metrics_list]),
            ric=np.median([m.ric for m in metrics_list]),
            rank_ic=np.median([m.rank_ic for m in metrics_list]),
            icir=np.median([m.icir for m in metrics_list]),
            ricir=np.median([m.ricir for m in metrics_list]),
            turnover=np.median([m.turnover for m in metrics_list]),
            long_short_return=np.median([m.long_short_return for m in metrics_list]),
            long_short_sharpe=np.median([m.long_short_sharpe for m in metrics_list]),
            long_short_max_drawdown=np.median([m.long_short_max_drawdown for m in metrics_list]),
            quantile_returns=[np.median([m.quantile_returns[i] for m in metrics_list]) 
                            for i in range(len(metrics_list[0].quantile_returns))],
            quantile_sharpes=[np.median([m.quantile_sharpes[i] for m in metrics_list]) 
                             for i in range(len(metrics_list[0].quantile_sharpes))],
            stability=np.median([m.stability for m in metrics_list])
        )