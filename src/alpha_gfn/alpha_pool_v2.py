#!/usr/bin/env python3
"""
升级版的AlphaPoolGFN，支持优化的入池策略和缓存管理
"""

import hashlib
import logging
import os
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from alphagen.models.alpha_pool import AlphaPool
from alphagen.data.expression import Expression
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2 as ParquetFeatureLoader
try:
    from alpha_gfn.cache_manager import CacheManager, CacheKeyBuilder
    from alpha_gfn.expression_quality import ExpressionQualityValidator
    from alpha_gfn.expression_canonical import ExpressionCanonicalizer
    from alpha_gfn.semantic_embedding import OllamaExpressionEmbedder
    from evaluation.factor_metrics import FactorMetricsEvaluator
except ImportError:
    from .cache_manager import CacheManager, CacheKeyBuilder
    from .expression_quality import ExpressionQualityValidator
    from .expression_canonical import ExpressionCanonicalizer
    from .semantic_embedding import OllamaExpressionEmbedder
    from ..evaluation.factor_metrics import FactorMetricsEvaluator

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
        entry_strategy: str = "composite",  # ic_ranking, diversity_aware, adaptive, composite
        diversity_weight: float = 0.3,
        min_ic_threshold: float = 0.05,
        max_similarity_threshold: float = 0.95,
        adaptive_threshold_decay: float = 0.99,
        enable_cache: bool = True,
        cache_key_builder: CacheKeyBuilder = None,
        # New components
        valid_data: Optional[ParquetFeatureLoader] = None,
        test_data: Optional[ParquetFeatureLoader] = None,
        metrics_evaluator: Optional[FactorMetricsEvaluator] = None,
        quality_validator: Optional[ExpressionQualityValidator] = None,
        canonicalizer: Optional[ExpressionCanonicalizer] = None,
        semantic_embedder: Optional[OllamaExpressionEmbedder] = None,
        # New thresholds
        min_train_ic: float = 0.015,
        min_valid_ic: float = 0.005,
        min_rank_ic: float = 0.005,
        min_ic_ir: float = 0.05,
        min_coverage: float = 0.65,
        max_nan_ratio: float = 0.35,
        max_pool_corr: float = 0.70,
        semantic_sim_threshold: float = 0.92,
    ):
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
        
        # New components
        self.valid_data = valid_data
        self.test_data = test_data
        self.metrics_evaluator = metrics_evaluator or FactorMetricsEvaluator()
        self.quality_validator = quality_validator
        self.canonicalizer = canonicalizer or ExpressionCanonicalizer()
        self.semantic_embedder = semantic_embedder
        
        # New thresholds
        self.min_train_ic = min_train_ic
        self.min_valid_ic = min_valid_ic
        self.min_rank_ic = min_rank_ic
        self.min_ic_ir = min_ic_ir
        self.min_coverage = min_coverage
        self.max_nan_ratio = max_nan_ratio
        self.max_pool_corr = max_pool_corr
        self.semantic_sim_threshold = semantic_sim_threshold
        
        # 扩展存储
        self.composite_scores = np.zeros(capacity + 1)
        self.metric_records = [None for _ in range(capacity + 1)]
        self.quality_records = [None for _ in range(capacity + 1)]
        self.canonical_exprs = [None for _ in range(capacity + 1)]
        self.canonical_hashes_by_slot = [None for _ in range(capacity + 1)]
        self.canonical_hashes = set()
        self.semantic_embeddings = [None for _ in range(capacity + 1)]
        
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
        if not hasattr(self, "rejection_file"):
            return
            
        # 确保目录存在
        log_dir = os.path.dirname(self.rejection_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception:
                return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 清理表达式中的换行符
        expr_str = str(expr).replace("\n", " ").replace("\r", "")
        
        # 如果文件不存在，写入表头
        if not os.path.exists(self.rejection_file):
            with open(self.rejection_file, "w", encoding="utf-8") as f:
                f.write("timestamp,expression,ic,max_mut_corr,reason\n")
                
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
        尝试添加新表达式到池中 (重构后的逻辑)
        """
        self.stats['total_evaluations'] += 1
        
        # 1. 表达式结构质量验证 (Expression Quality)
        q_report = None
        if self.quality_validator:
            q_report = self.quality_validator.validate(expr)
            if not q_report.accept:
                # 即使质量不合格，如果是为了调试，我们可以尝试计算一下 IC
                # 但为了工程严谨，目前仅记录原因并返回极小奖励以提供梯度反馈
                self._log_rejection(expr, 0.0, 0.0, f"quality:{q_report.reason}")
                return 1e-10, 0.0
                
        # 2. 表达式归一化去重 (Canonical Hash)
        canonical = self.canonicalizer.canonicalize(expr)
        expr_hash = self.canonicalizer.hash(expr)
        if expr_hash in self.canonical_hashes:
            self._log_rejection(expr, 0.0, 1.0, "duplicate:canonical")
            return 1e-10, 0.0
            
        # 3. 评估因子值
        try:
            # 尝试从缓存获取
            value = None
            if self.enable_cache and self.cache_manager:
                cache_key = self.cache_key_builder.build_key(expr)
                cached = self.cache_manager.get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    _, _, value = cached
                else:
                    self.stats['cache_misses'] += 1
            
            if value is None:
                value = self._normalize_by_day(expr.evaluate(self.data))
                if self.enable_cache and self.cache_manager:
                    self.cache_manager.put(cache_key, (None, None, value))
                    
        except Exception as e:
            self._log_rejection(expr, 0.0, 0.0, f"eval_error:{type(e).__name__}")
            return 1e-10, 0.0
            
        # 4. 因子值健康检查 (Value Sanity)
        sanity = self.metrics_evaluator.value_sanity(value)
        if not sanity["ok"]:
            self._log_rejection(expr, 0.0, 0.0, f"value_sanity:{sanity['reason']}")
            return 1e-10, 0.0
            
        # 5. 快速计算 Train Metrics
        train_metrics = self.metrics_evaluator.evaluate_tensor(value, self.target)
        direction = 1.0 if train_metrics["ic_mean"] >= 0 else -1.0
        train_ic_adj = train_metrics["ic_adj"]
        
        if train_ic_adj < self.min_train_ic:
            self._log_rejection(expr, train_ic_adj, 0.0, "low_train_ic")
            # 给模型一个基于 IC 的正反馈，但不入池
            return max(train_ic_adj, 1e-10), 0.0
            
        # 6. 计算 Valid Metrics (只有 Train 合格才算 Valid，省时间)
        valid_metrics = None
        if self.valid_data:
            try:
                valid_value = self._normalize_by_day(expr.evaluate(self.valid_data))
                valid_metrics = self.metrics_evaluator.evaluate_tensor(
                    valid_value, 
                    self.valid_data.target if hasattr(self.valid_data, 'target') else self.target, # Fallback
                    direction=direction
                )
                if valid_metrics["ic_adj"] < self.min_valid_ic:
                    self._log_rejection(expr, train_ic_adj, 0.0, "low_valid_ic")
                    return max(train_ic_adj, 1e-10), 0.0
                if valid_metrics["rank_ic_adj"] < self.min_rank_ic:
                    self._log_rejection(expr, train_ic_adj, 0.0, "low_valid_rank_ic")
                    return max(train_ic_adj, 1e-10), 0.0
            except Exception as e:
                logger.warning(f"Valid evaluation failed for {expr}: {e}")
                
        # 7. 池互相关拒绝 (Pool Correlation)
        ic_mut = []
        max_pool_corr = 0.0
        for i in range(self.size):
            if self.values[i] is not None:
                # 使用 evaluation.panel_ops 中的相关性计算可能更好，但为了速度先用已有的
                from alphagen.utils.correlation import batch_pearsonr
                corr = abs(batch_pearsonr(value, self.values[i]).mean().item())
                ic_mut.append(corr)
        
        if ic_mut:
            max_pool_corr = max(ic_mut)
            if max_pool_corr > self.max_pool_corr:
                self._log_rejection(expr, train_ic_adj, max_pool_corr, "high_pool_corr")
                return max(train_ic_adj, 1e-10), 1.0 - max_pool_corr

        # 8. 语义去重 (Semantic Duplicate)
        semantic_sim = 0.0
        semantic_vec = None
        if self.semantic_embedder:
            semantic_vec = self.semantic_embedder.embed_one(canonical)
            # 计算与池中因子的语义相似度
            for i in range(self.size):
                if self.semantic_embeddings[i] is not None:
                    sim = float(np.dot(semantic_vec, self.semantic_embeddings[i])) # 假设已归一化
                    semantic_sim = max(semantic_sim, sim)
            
            if semantic_sim > self.semantic_sim_threshold:
                self._log_rejection(expr, train_ic_adj, max_pool_corr, "semantic_duplicate")
                return max(train_ic_adj, 1e-10), 1.0 - semantic_sim

        # 9. 计算综合得分 (Composite Score)
        novelty = 1.0 - max(max_pool_corr, semantic_sim)
        score = self._composite_score(train_metrics, valid_metrics, novelty, q_report)
        
        # 10. 入池或替换 (Add / Replace)
        added = self._add_or_replace(
            expr=expr,
            value=value,
            ic_ret=train_ic_adj,
            score=score,
            embedding=embedding,
            semantic_embedding=semantic_vec,
            quality=q_report,
            metrics={"train": train_metrics, "valid": valid_metrics},
            canonical=canonical,
            expr_hash=expr_hash
        )
        
        if added:
            self.stats['pool_additions'] += 1
            logger.info(f"[Pool Add] {expr} (Score: {score:.4f}, IC: {train_ic_adj:.4f})")
        else:
            self.stats['pool_rejections'] += 1
            
        return max(train_ic_adj, 1e-10) + 0.2 * novelty, novelty

    def _composite_score(self, train_metrics, valid_metrics, novelty, q_report) -> float:
        """计算综合得分用于入池排序"""
        # 基础分来自 Valid IC (如果没跑 valid 就用 train)
        ic_score = valid_metrics["ic_adj"] if valid_metrics else train_metrics["ic_adj"]
        rank_ic_score = valid_metrics["rank_ic_adj"] if valid_metrics else train_metrics["rank_ic_adj"]
        
        score = 1.0 * ic_score + 0.5 * rank_ic_score + 0.2 * novelty
        
        # 复杂度奖励
        if q_report:
            if q_report.complexity <= 18:
                score += 0.05 * np.log1p(q_report.complexity)
            else:
                score -= 0.01 * (q_report.complexity - 18) # 过度复杂惩罚
                
        # 稳定性惩罚 (如果 positive_ic_ratio 太低)
        if valid_metrics and valid_metrics.get("positive_ic_ratio", 0) < 0.52:
            score -= 0.1
            
        return score

    def _add_or_replace(self, expr, value, ic_ret, score, **meta) -> bool:
        """重构后的入池与替换逻辑"""
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            # 找到池中得分最低的
            idx = int(np.argmin(self.composite_scores[:self.capacity]))
            if score <= self.composite_scores[idx]:
                self._log_rejection(expr, ic_ret, 0.0, "score_not_better_than_worst")
                return False
            
            # 替换前先移除旧哈希
            old_hash = self.canonical_hashes_by_slot[idx]
            if old_hash in self.canonical_hashes:
                self.canonical_hashes.remove(old_hash)
        
        # 更新存储
        if idx < len(self.exprs):
            self.exprs[idx] = expr
            self.values[idx] = value
        else:
            self.exprs.append(expr)
            self.values.append(value)
            
        self.single_ics[idx] = ic_ret
        self.composite_scores[idx] = score
        self.embeddings[idx] = meta.get("embedding")
        self.semantic_embeddings[idx] = meta.get("semantic_embedding")
        self.metric_records[idx] = meta.get("metrics")
        self.quality_records[idx] = meta.get("quality")
        self.canonical_exprs[idx] = meta.get("canonical")
        self.canonical_hashes_by_slot[idx] = meta.get("expr_hash")
        
        if meta.get("expr_hash"):
            self.canonical_hashes.add(meta.get("expr_hash"))
            
        return True

    def _update_adaptive_thresholds(self):
        """更新自适应阈值 (重构版)"""
        if self.size < max(10, self.capacity // 4):
            return
        
        # 获取池中因子的 Valid IC (如果可用)
        valid_ics = []
        for rec in self.metric_records[:self.size]:
            if rec and "valid" in rec and rec["valid"]:
                valid_ics.append(rec["valid"]["ic_adj"])
            elif rec and "train" in rec:
                valid_ics.append(rec["train"]["ic_adj"])
        
        if len(valid_ics) >= 10:
            # 将门槛设置为当前池中 40% 分位数的 IC
            target = float(np.percentile(valid_ics, 40))
            if target > self.min_train_ic:
                self.min_train_ic = max(self.min_train_ic, target * 0.9)
                self.stats['adaptive_threshold_updates'] = self.stats.get('adaptive_threshold_updates', 0) + 1
    
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
    
    def export_pool(self, output_path: str):
        """导出池到文件 (增强版)"""
        pool_data = {
            'expressions': [str(expr) for expr in self.exprs[:self.size]],
            'canonical_expressions': [str(c) for c in self.canonical_exprs[:self.size]],
            'ics': self.single_ics[:self.size].tolist(),
            'composite_scores': self.composite_scores[:self.size].tolist(),
            'metrics': [m for m in self.metric_records[:self.size]],
            'quality': [asdict(q) if q else None for q in self.quality_records[:self.size]],
            'stats': self.get_stats(),
        }
        
        import json
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return str(obj)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pool_data, f, indent=2, default=default_serializer)
        
        logger.info(f"Pool exported to {output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """获取池的统计信息"""
        stats = {
            **self.stats,
            'pool_size': self.size,
            'pool_capacity': self.capacity,
            'best_ic': float(np.max(self.single_ics[:self.size])) if self.size > 0 else 0.0,
            'mean_ic': float(np.mean(self.single_ics[:self.size])) if self.size > 0 else 0.0,
            'best_score': float(np.max(self.composite_scores[:self.size])) if self.size > 0 else 0.0,
            'mean_score': float(np.mean(self.composite_scores[:self.size])) if self.size > 0 else 0.0,
        }
        if self.semantic_embedder:
            stats.update(self.semantic_embedder.stats)
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            'stats': self.get_stats(),
            'entry_strategy': self.entry_strategy,
            'embeddings': [e.tolist() if e is not None else None for e in self.embeddings[:self.size]]
        })
        return base_dict