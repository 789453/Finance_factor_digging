#!/usr/bin/env python3
"""
升级版的AlphaPoolGFN，支持优化的入池策略和缓存管理
"""

import hashlib
import logging
import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from alphagen.models.alpha_pool import AlphaPool
from alphagen.data.expression import Expression
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2 as ParquetFeatureLoader
try:
    from alpha_gfn.cache_manager import CacheManager, CacheKeyBuilder
except ImportError:
    from .cache_manager import CacheManager, CacheKeyBuilder

logger = logging.getLogger(__name__)

class AlphaPoolGFN(AlphaPool):
    """
    升级版的AlphaPoolGFN，支持优化的入池策略和缓存管理
    """
    
    def __init__(
        self,
        capacity: int,
        stock_data: ParquetFeatureLoader,
        target: Expression,
        ic_mut_threshold: float = 0.3,
        ssl_k: int = 3,
        ssl_tau: float = 0.1,
        cache_manager: CacheManager = None,
        entry_strategy: str = "ic_ranking",  # ic_ranking, diversity_aware, adaptive
        diversity_weight: float = 0.3,
        min_ic_threshold: float = 0.05,
        max_similarity_threshold: float = 0.95,
        adaptive_threshold_decay: float = 0.99,
        enable_cache: bool = True,
        cache_key_builder: CacheKeyBuilder = None
    ):
        """
        初始化AlphaPoolGFN
        
        Args:
            capacity: 池容量
            stock_data: 股票数据加载器
            target: 目标表达式
            ic_mut_threshold: IC互相关阈值
            ssl_k: SSL最近邻数量
            ssl_tau: SSL温度参数
            cache_manager: 缓存管理器
            entry_strategy: 入池策略
            diversity_weight: 多样性权重
            min_ic_threshold: 最小IC阈值
            max_similarity_threshold: 最大相似度阈值
            adaptive_threshold_decay: 自适应阈值衰减
            enable_cache: 是否启用缓存
            cache_key_builder: 缓存键构建器
        """
        super().__init__(capacity, stock_data, target)
        self.ic_mut_threshold = ic_mut_threshold
        self.ssl_k = ssl_k
        self.ssl_tau = ssl_tau
        self.cache_manager = cache_manager
        self.entry_strategy = entry_strategy
        self.diversity_weight = diversity_weight
        self.min_ic_threshold = min_ic_threshold
        self.max_similarity_threshold = max_similarity_threshold
        self.adaptive_threshold_decay = adaptive_threshold_decay
        self.enable_cache = enable_cache
        self.cache_key_builder = cache_key_builder or CacheKeyBuilder()
        
        # 获取 log_dir 并初始化拒绝日志
        self.log_dir = "."
        # 尝试从 stock_data 或上下文推断 log_dir，或者在外部设置
        self.rejection_file = "rejections.csv"
        
        # 初始化嵌入存储
        self.embeddings: List[Optional[Tensor]] = [None for _ in range(capacity + 1)]
        
        # 统计信息
        self.stats = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'pool_additions': 0,
            'pool_rejections': 0,
            'adaptive_threshold_updates': 0
        }
        
        # 自适应阈值
        self.current_ic_threshold = min_ic_threshold
        self.current_diversity_threshold = 1.0 - max_similarity_threshold
        
        logger.info(f"Initialized AlphaPoolGFN with capacity={capacity}, strategy={entry_strategy}")
    
    def try_new_expr_with_ssl(self, expr: Expression, embedding: Optional[Tensor] = None) -> Tuple[float, float, float]:
        """
        同时计算 IC 奖励、新颖性奖励和 SSL 奖励
        
        Args:
            expr: 表达式
            embedding: 嵌入向量
            
        Returns:
            (ic_reward, nov_reward, ssl_reward) 元组
        """
        # 1. 获取 IC 和新颖性奖励
        ic_reward, nov_reward = self.try_new_expr(expr, embedding)
        
        # 2. 计算 SSL 奖励
        ssl_reward = 0.0
        if embedding is not None and self.size > 1:
            ssl_reward = self.compute_ssl_reward(expr, embedding)
            
        return ic_reward, nov_reward, ssl_reward

    def compute_ssl_reward(self, expr: Expression, embedding: Tensor) -> float:
        """计算 SSL 奖励"""
        # 找到最近邻
        neighbor_indices = self._find_k_nearest_neighbors(embedding, self.ssl_k, exclude_self=True)
        if not neighbor_indices:
            return 0.0
            
        # 计算权重
        weights = self._compute_similarity_weights(embedding, neighbor_indices)
        if len(weights) == 0:
            return 0.0
            
        # 获取表达式值
        try:
            query_value = self._normalize_by_day(expr.evaluate(self.data))
        except:
            return 0.0
            
        # 计算一致性损失
        consistency_loss = self._compute_consistency_loss(query_value, neighbor_indices, weights)
        
        # 转换为奖励
        return float(np.exp(-consistency_loss))

    def _find_k_nearest_neighbors(self, query_embedding: Tensor, k: int, exclude_self: bool = True) -> List[int]:
        """寻找 k 个最近邻"""
        distances = []
        valid_indices = []
        
        for i in range(self.size):
            if self.embeddings[i] is not None:
                dist = torch.norm(query_embedding - self.embeddings[i]).item()
                if exclude_self and dist < 1e-6:
                    continue
                distances.append(dist)
                valid_indices.append(i)
                
        if not distances:
            return []
            
        k = min(k, len(distances))
        indices = np.argsort(distances)[:k]
        return [valid_indices[idx] for idx in indices]

    def _compute_similarity_weights(self, query_embedding: Tensor, neighbor_indices: List[int]) -> Tensor:
        """计算相似度权重"""
        scores = []
        for idx in neighbor_indices:
            if self.embeddings[idx] is not None:
                dist_squared = torch.norm(query_embedding - self.embeddings[idx])**2
                scores.append(-dist_squared / self.ssl_tau)
                
        if not scores:
            return torch.tensor([])
            
        return F.softmax(torch.tensor(scores), dim=0)

    def _compute_consistency_loss(self, query_value: Tensor, neighbor_indices: List[int], weights: Tensor) -> float:
        """计算一致性损失"""
        total_loss = 0.0
        for i, idx in enumerate(neighbor_indices):
            if self.values[idx] is not None:
                neighbor_value = self.values[idx]
                mse = ((query_value - neighbor_value)**2).mean().item()
                total_loss += weights[i].item() * mse
        return total_loss

    def _init_rejection_log(self, log_dir: str):
        """初始化拒绝日志文件"""
        self.log_dir = log_dir
        self.rejection_file = os.path.join(log_dir, "rejections.csv")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.rejection_file):
            with open(self.rejection_file, "w", encoding="utf-8") as f:
                f.write("timestamp,expression,ic,max_mut_corr,reason\n")

    def _log_rejection(self, expr: Expression, ic: float, max_mut: float, reason: str):
        """记录被拒绝的因子"""
        import datetime
        import os
        if not hasattr(self, "rejection_file") or not os.path.exists(os.path.dirname(self.rejection_file)):
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 清理表达式中的换行符
        expr_str = str(expr).replace("\n", " ").replace("\r", "")
        with open(self.rejection_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},\"{expr_str}\",{ic:.6f},{max_mut:.6f},\"{reason}\"\n")

    def _calc_ics(self, value: Tensor, ic_mut_threshold: float = 0.9) -> Tuple[float, np.ndarray]:
        """计算 IC 和互相关性"""
        try:
            # 1. 计算与 Label 的 IC
            # 使用父类已经计算并对齐好的 target
            label = self.target # 形状 [n_days, n_stocks]
            
            # 因子值也需要 normalize (AlphaPool 的 try_new_expr 已经做了，但我们这里是独立的实现)
            value = self._normalize_by_day(value)
            
            v_flat = value.flatten()
            l_flat = label.flatten()
            
            mask = ~(torch.isnan(v_flat) | torch.isnan(l_flat))
            if mask.sum() < 2:
                return 0.0, np.array([])
            
            # 检查是否为常数
            if torch.std(v_flat[mask]) < 1e-8:
                return 0.0, np.array([])

            # 使用 alphagen 提供的 batch_pearsonr 以保持严谨性
            from alphagen.utils.correlation import batch_pearsonr
            ic_per_day = batch_pearsonr(value, label)
            ic = ic_per_day.mean().item()
            if np.isnan(ic): ic = 0.0
            
            # 2. 计算与池中已有因子的互相关性
            ic_mut = []
            for i in range(self.size):
                if self.values[i] is not None:
                    mut_ic = batch_pearsonr(value, self.values[i]).mean().item()
                    if np.isnan(mut_ic): mut_ic = 1.0
                    ic_mut.append(mut_ic)
            
            return ic, np.array(ic_mut)
        except Exception as e:
            logger.error(f"IC calculation failed: {e}")
            return 0.0, np.array([])

    def try_new_expr(self, expr: Expression, embedding: Optional[Tensor] = None) -> Tuple[float, float]:
        """
        尝试添加新表达式到池中
        
        Args:
            expr: 表达式
            embedding: 嵌入向量
            
        Returns:
            (ic_ret, nov_score) 元组
        """
        try:
            # 检查缓存
            if self.enable_cache and self.cache_manager:
                cache_key = self.cache_key_builder.build_key(expr)
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self.stats['cache_hits'] += 1
                    ic_ret, ic_mut, value = cached_result
                    logger.debug(f"Cache hit for expression: {expr}")
                else:
                    self.stats['cache_misses'] += 1
                    value = self._normalize_by_day(expr.evaluate(self.data))
                    ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99)
                    self.cache_manager.put(cache_key, (ic_ret, ic_mut, value))
            else:
                value = self._normalize_by_day(expr.evaluate(self.data))
                ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99)
            
        except Exception as e:
            logger.warning(f"Expression evaluation failed: {expr}, error: {e}")
            return 0.0, 1.0
        
        self.stats['total_evaluations'] += 1
        
        if ic_ret is None or ic_mut is None:
            return 0.0, 1.0
        
        ic_ret = np.abs(ic_ret)
        ic_mut = np.abs(ic_mut)
        
        # 根据入池策略决定是否添加
        should_add = False
        reason = ""
        
        if self.entry_strategy == "ic_ranking":
            should_add, reason = self._should_add_ic_ranking(expr, ic_ret, ic_mut, value)
        elif self.entry_strategy == "diversity_aware":
            should_add, reason = self._should_add_diversity_aware(expr, ic_ret, ic_mut, value, embedding)
        elif self.entry_strategy == "adaptive":
            should_add, reason = self._should_add_adaptive(expr, ic_ret, ic_mut, value, embedding)
        else:
            should_add, reason = self._should_add_default(expr, ic_ret, ic_mut, value)
        
        if should_add:
            self._add_factor(expr, value, ic_ret, ic_mut, embedding)
            self.stats['pool_additions'] += 1
            logger.info(f"[Pool Add] {expr} - {reason}")
        else:
            self.stats['pool_rejections'] += 1
            max_mut = np.max(ic_mut) if ic_mut.size > 0 else 0.0
            self._log_rejection(expr, ic_ret, max_mut, reason)
            logger.debug(f"[Pool Reject] {expr} - {reason}")
        
        # 计算新颖性分数
        nov_score = (1 - np.max(ic_mut)) if ic_mut.size > 0 else 1.0
        
        return ic_ret, nov_score
    
    def _should_add_default(self, expr: Expression, ic_ret: float, ic_mut: np.ndarray, value: Tensor) -> Tuple[bool, str]:
        """默认入池策略"""
        if self.size < self.capacity:
            if ic_mut.size == 0 or np.max(ic_mut) <= self.ic_mut_threshold:
                return True, "Pool not full, IC constraint satisfied"
        else:
            min_ic_idx = np.argmin(self.single_ics[:self.size])
            min_ic = self.single_ics[min_ic_idx]
            if ic_ret > min_ic and (ic_mut.size == 0 or np.max(ic_mut) <= self.ic_mut_threshold):
                return True, "Better than worst factor"
        
        return False, "IC constraint not satisfied or not better than existing"
    
    def _should_add_ic_ranking(self, expr: Expression, ic_ret: float, ic_mut: np.ndarray, value: Tensor) -> Tuple[bool, str]:
        """基于IC排名的入池策略"""
        if ic_ret < self.current_ic_threshold:
            return False, f"IC {ic_ret:.4f} below threshold {self.current_ic_threshold:.4f}"
        
        if ic_mut.size > 0 and np.max(ic_mut) > self.ic_mut_threshold:
            return False, f"IC mutual correlation {np.max(ic_mut):.4f} exceeds threshold {self.ic_mut_threshold}"
        
        if self.size < self.capacity:
            return True, "Pool not full, IC threshold satisfied"
        else:
            min_ic_idx = np.argmin(self.single_ics[:self.size])
            min_ic = self.single_ics[min_ic_idx]
            if ic_ret > min_ic:
                return True, f"IC {ic_ret:.4f} better than worst {min_ic:.4f}"
            else:
                return False, f"IC {ic_ret:.4f} not better than worst {min_ic:.4f}"
    
    def _should_add_diversity_aware(self, expr: Expression, ic_ret: float, ic_mut: np.ndarray, 
                                  value: Tensor, embedding: Optional[Tensor] = None) -> Tuple[bool, str]:
        """多样性感知的入池策略"""
        if ic_ret < self.current_ic_threshold:
            return False, f"IC {ic_ret:.4f} below threshold {self.current_ic_threshold:.4f}"
        
        if ic_mut.size > 0 and np.max(ic_mut) > self.ic_mut_threshold:
            return False, f"IC mutual correlation {np.max(ic_mut):.4f} exceeds threshold {self.ic_mut_threshold}"
        
        # 计算多样性分数
        diversity_score = self._compute_diversity_score(value, embedding)
        
        if self.size < self.capacity:
            if diversity_score >= self.current_diversity_threshold:
                return True, f"Pool not full, diversity score {diversity_score:.4f} sufficient"
            else:
                return False, f"Diversity score {diversity_score:.4f} below threshold {self.current_diversity_threshold:.4f}"
        else:
            # 池已满，需要综合考虑IC和多样性
            combined_scores = []
            for i in range(self.size):
                factor_ic = self.single_ics[i]
                factor_diversity = self._compute_diversity_score(self.values[i], self.embeddings[i])
                combined_score = factor_ic + self.diversity_weight * factor_diversity
                combined_scores.append(combined_score)
            
            new_combined_score = ic_ret + self.diversity_weight * diversity_score
            min_combined_idx = np.argmin(combined_scores)
            min_combined_score = combined_scores[min_combined_idx]
            
            if new_combined_score > min_combined_score:
                return True, f"Combined score {new_combined_score:.4f} better than worst {min_combined_score:.4f}"
            else:
                return False, f"Combined score {new_combined_score:.4f} not better than worst {min_combined_score:.4f}"
    
    def _should_add_adaptive(self, expr: Expression, ic_ret: float, ic_mut: np.ndarray, 
                           value: Tensor, embedding: Optional[Tensor] = None) -> Tuple[bool, str]:
        """自适应入池策略"""
        # 动态调整阈值
        self._update_adaptive_thresholds()
        
        # 使用多样性感知策略作为基础
        return self._should_add_diversity_aware(expr, ic_ret, ic_mut, value, embedding)
    
    def _compute_diversity_score(self, value: Tensor, embedding: Optional[Tensor] = None) -> float:
        """计算多样性分数"""
        if self.size == 0:
            return 1.0
        
        diversity_scores = []
        
        for i in range(self.size):
            if self.values[i] is not None:
                # 计算值相似度
                value_sim = self._compute_value_similarity(value, self.values[i])
                diversity_scores.append(1.0 - value_sim)
        
        if not diversity_scores:
            return 1.0
        
        return np.mean(diversity_scores)
    
    def _compute_value_similarity(self, value1: Tensor, value2: Tensor) -> float:
        """计算值相似度"""
        try:
            # 计算皮尔逊相关系数
            v1_flat = value1.flatten()
            v2_flat = value2.flatten()
            
            # 移除NaN值
            valid_mask = ~(torch.isnan(v1_flat) | torch.isnan(v2_flat))
            if valid_mask.sum() < 2:
                return 0.0
            
            v1_valid = v1_flat[valid_mask]
            v2_valid = v2_flat[valid_mask]
            
            # 计算相关系数
            correlation = torch.corrcoef(torch.stack([v1_valid, v2_valid]))[0, 1].item()
            return abs(correlation)
        except:
            return 0.0
    
    def _update_adaptive_thresholds(self):
        """更新自适应阈值"""
        if self.size == 0:
            return
        
        # 基于池中因子的统计信息调整阈值
        current_ics = self.single_ics[:self.size]
        
        # 如果池中因子质量较高，提高IC阈值
        if len(current_ics) > 5:
            mean_ic = np.mean(current_ics)
            std_ic = np.std(current_ics)
            
            if mean_ic > self.current_ic_threshold:
                self.current_ic_threshold = min(
                    self.current_ic_threshold * self.adaptive_threshold_decay,
                    mean_ic - 0.5 * std_ic
                )
                self.stats['adaptive_threshold_updates'] += 1
        
        # 如果池中因子过于相似，提高多样性要求
        if self.size > 10:
            diversity_scores = []
            for i in range(self.size):
                for j in range(i + 1, self.size):
                    if self.values[i] is not None and self.values[j] is not None:
                        sim = self._compute_value_similarity(self.values[i], self.values[j])
                        diversity_scores.append(1.0 - sim)
            
            if diversity_scores:
                avg_diversity = np.mean(diversity_scores)
                if avg_diversity < 0.3:  # 因子过于相似
                    self.current_diversity_threshold = min(
                        self.current_diversity_threshold * 1.1,
                        0.8
                    )
    
    def _add_factor(
        self,
        expr: Expression,
        value: Tensor,
        ic_ret: float,
        ic_mut: np.ndarray,
        embedding: Optional[Tensor] = None
    ):
        """向池中添加因子，并同步更新基类状态"""
        n = self.size
        if self.size < self.capacity:
            # 池未满，直接追加
            self.exprs.append(expr)
            self.values.append(value)
            self.single_ics[n] = ic_ret
            self.embeddings[n] = embedding
            self.size += 1
        else:
            # 池已满，替换最差的（基于 IC）
            n = np.argmin(self.single_ics[:self.capacity])
            self.exprs[n] = expr
            self.values[n] = value
            self.single_ics[n] = ic_ret
            self.embeddings[n] = embedding
            
        logger.info(f"Factor added to pool at index {n}: {expr} (IC: {ic_ret:.4f})")
    
    def _pop(self) -> None:
        """移除池中IC最低的因子"""
        if self.size <= self.capacity:
            return
        idx = np.argmin(self.single_ics[:self.size])
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity
    
    def _swap_idx(self, i: int, j: int) -> None:
        """交换池中两个位置的因子"""
        if i == j:
            return
        super()._swap_idx(i, j)
        self.embeddings[i], self.embeddings[j] = self.embeddings[j], self.embeddings[i]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池的统计信息"""
        return {
            **self.stats,
            'pool_size': self.size,
            'pool_capacity': self.capacity,
            'current_ic_threshold': self.current_ic_threshold,
            'current_diversity_threshold': self.current_diversity_threshold,
            'best_ic': float(np.max(self.single_ics[:self.size])) if self.size > 0 else 0.0,
            'mean_ic': float(np.mean(self.single_ics[:self.size])) if self.size > 0 else 0.0,
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['total_evaluations']),
        }
    
    def clear_cache(self):
        """清除缓存"""
        if self.cache_manager:
            self.cache_manager.clear()
            logger.info("Cache cleared")
    
    def export_pool(self, output_path: str):
        """导出池到文件"""
        pool_data = {
            'expressions': [str(expr) for expr in self.exprs[:self.size]],
            'ics': self.single_ics[:self.size].tolist() if hasattr(self, 'single_ics') else [],
            'stats': self.get_stats(),
            'embeddings': [e.tolist() if e is not None else None for e in self.embeddings[:self.size]]
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(pool_data, f, indent=2)
        
        logger.info(f"Pool exported to {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            'stats': self.get_stats(),
            'entry_strategy': self.entry_strategy,
            'embeddings': [e.tolist() if e is not None else None for e in self.embeddings[:self.size]]
        })
        return base_dict