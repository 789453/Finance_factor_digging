#!/usr/bin/env python3
"""
因子挖掘完整管道
整合作业规格、因子族配置、数据加载、GFN训练和结果管理
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import numpy as np

# 导入模块
try:
    from mining.job_spec import MiningJobSpec, load_job_spec
    from mining.family_search_space import load_family_spec, build_family_search_space
    from alphagen_generic.dataset_meta import DatasetMeta
    from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    from alphagen.data.expression import Expression, Feature, Ref
    from alpha_gfn.alpha_pool_v2 import AlphaPoolGFN
except ImportError:
    from src.mining.job_spec import MiningJobSpec, load_job_spec
    from src.mining.family_search_space import load_family_spec, build_family_search_space
    from src.alphagen_generic.dataset_meta import DatasetMeta
    from src.alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from src.alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
    from src.alphagen.data.expression import Expression, Feature, Ref
    from src.alpha_gfn.alpha_pool_v2 import AlphaPoolGFN

logger = logging.getLogger(__name__)

class MiningLogger:
    """挖掘过程日志器"""
    
    def __init__(self, job_spec: MiningJobSpec, log_dir: str):
        self.job_spec = job_spec
        self.log_dir = log_dir
        self.start_time = time.time()
        self.checkpoints = []
        
        # 设置日志文件
        log_file = os.path.join(log_dir, "mining.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Mining job started: {job_spec.job_id}")
        logger.info(f"Job name: {job_spec.name}")
        logger.info(f"Dataset: {job_spec.dataset_id}")
        logger.info(f"Family: {job_spec.family_id}")
        logger.info(f"Time range: {job_spec.train_start} to {job_spec.test_end}")
    
    def log_checkpoint(self, episode: int, pool_stats: Dict[str, Any], metrics: Dict[str, float]):
        """记录检查点"""
        elapsed = time.time() - self.start_time
        checkpoint = {
            'episode': episode,
            'elapsed_seconds': elapsed,
            'pool_stats': pool_stats,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoints.append(checkpoint)
        
        # 保存检查点
        checkpoint_file = os.path.join(self.log_dir, f"checkpoint_{episode:06d}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint {episode}: Pool size={pool_stats['pool_size']}, "
                   f"Best IC={metrics.get('best_ic', 0):.4f}, "
                   f"Elapsed={elapsed/60:.1f}min")
    
    def log_completion(self, final_stats: Dict[str, Any]):
        """记录完成信息"""
        total_time = time.time() - self.start_time
        logger.info(f"Mining job completed: {self.job_spec.job_id}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Final pool size: {final_stats.get('pool_size', 0)}")
        logger.info(f"Best IC: {final_stats.get('best_ic', 0):.4f}")
        logger.info(f"Total evaluations: {final_stats.get('total_evaluations', 0)}")
        logger.info(f"Cache hit rate: {final_stats.get('cache_hit_rate', 0):.2%}")
        
        # 保存完成报告
        completion_report = {
            'job_id': self.job_spec.job_id,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_seconds': total_time,
            'final_stats': final_stats,
            'checkpoints': len(self.checkpoints)
        }
        
        report_file = os.path.join(self.log_dir, "completion_report.json")
        with open(report_file, 'w') as f:
            json.dump(completion_report, f, indent=2)

class MiningContext:
    """挖掘上下文"""
    
    def __init__(self, job_spec: MiningJobSpec, family_spec: Dict[str, Any], 
                 dataset_meta: DatasetMeta, log_dir: str):
        self.job_spec = job_spec
        self.family_spec = family_spec
        self.dataset_meta = dataset_meta
        self.log_dir = log_dir
        
        # 构建数据加载器
        self.train_loader, self.test_loader = self._build_data_loaders()
        
        # 构建目标表达式
        self.target_expression = self._build_target_expression()
        
        # 构建搜索空间
        self.operators, self.delta_times, self.constants = self._build_search_space()
        
        logger.info("Mining context initialized successfully")
    
    def _build_data_loaders(self) -> Tuple[ParquetFeatureLoaderV2, ParquetFeatureLoaderV2]:
        """构建数据加载器"""
        logger.info("Building data loaders...")
        
        # 确定层
        layers = self.family_spec.get('enabled_layers', ['raw', 'filled'])
        registry_manager = FeatureRegistryManagerV2()
        
        # 训练数据加载器
        train_loader = ParquetFeatureLoaderV2(
            domain=self.dataset_meta.domain,
            start_time=self.job_spec.train_start,
            end_time=self.job_spec.train_end,
            registry_manager=registry_manager,
            dataset_meta=self.dataset_meta,
            device=torch.device(f'cuda:{self.job_spec.raw.get("cuda", 0)}' if torch.cuda.is_available() else 'cpu'),
            max_backtrack_days=self.job_spec.max_backtrack_days,
            max_future_days=self.job_spec.max_future_days,
            status_filter=self.job_spec.status_filter,
            use_filled=self.job_spec.raw.get("use_filled", True),
            layers=layers,
            cache_root=self.job_spec.raw.get("cache_root", "data/cache")
        )
        
        # 测试数据加载器
        test_loader = ParquetFeatureLoaderV2(
            domain=self.dataset_meta.domain,
            start_time=self.job_spec.test_start,
            end_time=self.job_spec.test_end,
            registry_manager=registry_manager,
            dataset_meta=self.dataset_meta,
            device=torch.device(f'cuda:{self.job_spec.raw.get("cuda", 0)}' if torch.cuda.is_available() else 'cpu'),
            max_backtrack_days=self.job_spec.max_backtrack_days,
            max_future_days=self.job_spec.max_future_days,
            status_filter=self.job_spec.status_filter,
            use_filled=self.job_spec.raw.get("use_filled", True),
            layers=layers,
            cache_root=self.job_spec.raw.get("cache_root", "data/cache")
        )
        
        return train_loader, test_loader
    
    def _build_target_expression(self) -> Expression:
        """构建目标表达式"""
        logger.info("Building target expression...")
        
        registry_manager = FeatureRegistryManagerV2()
        layers = self.family_spec.get('enabled_layers', ['raw', 'filled'])
        domain = self.dataset_meta.domain
        
        feature_enum = registry_manager.create_feature_enum(domain, self.job_spec.status_filter, layers)
        close = Feature(feature_enum.CLOSE)
        target = Ref(close, -self.job_spec.label_days) / close - 1
        
        return target
    
    def _build_search_space(self) -> Tuple[List, List, List]:
        """构建搜索空间"""
        logger.info("Building search space...")
        return build_family_search_space(self.family_spec)

def build_mining_context(job_spec_path: str, dataset_meta_path: Optional[str] = None, 
                        log_dir: Optional[str] = None) -> MiningContext:
    """
    构建挖掘上下文
    
    Args:
        job_spec_path: 作业规格文件路径
        dataset_meta_path: 数据集元数据文件路径
        log_dir: 日志目录
        
    Returns:
        MiningContext实例
    """
    # 加载作业规格
    job_spec = load_job_spec(job_spec_path)
    
    # 验证作业规格
    errors = job_spec.validate()
    if errors:
        raise ValueError(f"Invalid job spec: {errors}")
    
    # 加载因子族规格
    family_spec = load_family_spec(job_spec.family_id)
    
    # 加载数据集元数据
    if dataset_meta_path:
        dataset_meta = DatasetMeta(dataset_meta_path)
    else:
        # 创建默认数据集元数据
        dataset_meta = DatasetMeta({
            "dataset_id": job_spec.dataset_id,
            "domain": "A",  # 默认域，后续可以从job_spec或配置中获取
            "freq_group": "eod",
            "layers_enabled": family_spec.get('enabled_layers', ['raw', 'filled'])
        })
    
    # 验证兼容性
    from mining.job_spec import JobSpecValidator
    compatibility_errors = JobSpecValidator.validate_compatibility(
        job_spec, dataset_meta.to_dict(), family_spec
    )
    if compatibility_errors:
        raise ValueError(f"Compatibility errors: {compatibility_errors}")
    
    # 设置日志目录
    if log_dir is None:
        log_dir = os.path.join(
            job_spec.raw.get("output_dir", "data/mining_logs"),
            f"{job_spec.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志器
    logger_obj = MiningLogger(job_spec, log_dir)
    
    # 创建挖掘上下文
    context = MiningContext(job_spec, family_spec, dataset_meta, log_dir)
    context.logger = logger_obj
    
    return context

def mine_factors(job_spec_path: str, dataset_meta_path: Optional[str] = None, 
                log_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    执行因子挖掘任务
    
    Args:
        job_spec_path: 作业规格文件路径
        dataset_meta_path: 数据集元数据文件路径
        log_dir: 日志目录
        **kwargs: 额外参数
        
    Returns:
        挖掘结果字典
    """
    logger.info("Starting factor mining pipeline...")
    
    # 构建挖掘上下文
    context = build_mining_context(job_spec_path, dataset_meta_path, log_dir)
    
    # 初始化AlphaPoolGFN
    pool = AlphaPoolGFN(
        capacity=context.job_spec.pool_capacity,
        stock_data=context.train_loader,
        target=context.target_expression,
        entry_strategy=context.job_spec.raw.get("entry_strategy", "ic_ranking"),
        diversity_weight=context.job_spec.raw.get("diversity_weight", 0.3),
        min_ic_threshold=context.job_spec.raw.get("min_ic_threshold", 0.05),
        max_similarity_threshold=context.job_spec.raw.get("max_similarity_threshold", 0.95),
        enable_cache=True,
        cache_manager=None  # 可以在这里添加缓存管理器
    )
    
    # 这里可以集成GFN训练逻辑
    # 由于GFN训练需要完整的神经网络设置，这里提供框架
    
    logger.info("Factor mining pipeline setup completed")
    logger.info(f"Pool capacity: {context.job_spec.pool_capacity}")
    logger.info(f"Target episodes: {context.job_spec.n_episodes}")
    logger.info(f"Entry strategy: {context.job_spec.raw.get('entry_strategy', 'ic_ranking')}")
    
    # 返回上下文和池信息
    return {
        'context': context,
        'pool': pool,
        'status': 'initialized',
        'log_dir': context.log_dir
    }

if __name__ == "__main__":
    # 测试功能
    logging.basicConfig(level=logging.INFO)
    
    # 示例用法
    job_spec_path = "config/jobs/a_share_pv_ts_2020_2021.yaml"
    
    if os.path.exists(job_spec_path):
        result = mine_factors(job_spec_path)
        print(f"Mining setup completed: {result['status']}")
        print(f"Log directory: {result['log_dir']}")
    else:
        print(f"Job spec not found: {job_spec_path}")
        print("Please create a job specification file first.")