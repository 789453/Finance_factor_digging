#!/usr/bin/env python3
"""
集成适配器 - 将新的DataHub集成到现有的train_gfn_v2.py中
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# 导入进度管理器 - 使用正确的导入路径
try:
    from src.progress import ProgressManager, RecoveryManager
    from src.evaluation import create_factor_evaluator, FactorMetrics
    from src.datahub.adapter import DuckDBParquetFeatureLoaderV2, create_duckdb_loader
except ImportError:
    # 回退到相对导入
    try:
        from ..progress import ProgressManager, RecoveryManager
        from ..evaluation import create_factor_evaluator, FactorMetrics
        from ..datahub.adapter import DuckDBParquetFeatureLoaderV2, create_duckdb_loader
    except ImportError:
        # 如果都失败，创建兼容的基类
        logger = logging.getLogger(__name__)
        logger.warning("无法导入依赖模块，将创建兼容的基类")
        
        class ProgressManager:
            def __init__(self, *args, **kwargs):
                pass
        
        class RecoveryManager:
            def __init__(self, *args, **kwargs):
                pass
        
        def create_factor_evaluator(*args, **kwargs):
            return None
        
        class FactorMetrics:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class DuckDBParquetFeatureLoaderV2:
            def __init__(self, *args, **kwargs):
                pass
        
        def create_duckdb_loader(*args, **kwargs):
            return DuckDBParquetFeatureLoaderV2()

logger = logging.getLogger(__name__)

class IntegratedMiningPipeline:
    """集成的挖掘管道 - 整合新的数据层和进度管理"""
    
    def __init__(self, job_spec, device='cpu', **kwargs):
        """
        初始化集成管道
        
        Args:
            job_spec: 作业规格
            device: 计算设备
            **kwargs: 其他参数
        """
        self.job_spec = job_spec
        self.device = torch.device(device)
        
        # 初始化进度管理器
        self.progress_manager = ProgressManager(
            job_id=job_spec.job_id,
            store_type="sqlite"
        )
        
        # 初始化因子评价器
        self.factor_evaluator = create_factor_evaluator(
            evaluator_type="basic",
            quantiles=5,
            risk_free_rate=0.0
        )
        
        # 初始化数据加载器
        self.train_loader = None
        self.test_loader = None
        
        logger.info(f"Initialized IntegratedMiningPipeline for job: {job_spec.job_id}")
    
    def setup_data_loaders(self, dataset_meta=None):
        """
        设置数据加载器
        
        Args:
            dataset_meta: 数据集元数据
        """
        try:
            # 创建训练数据加载器
            self.train_loader = create_duckdb_loader(
                domain=self.job_spec.domain,
                start_time=self.job_spec.train_start,
                end_time=self.job_spec.train_end,
                registry_manager=None,  # 使用默认
                dataset_meta=dataset_meta,
                device=self.device,
                max_backtrack_days=self.job_spec.max_backtrack_days,
                max_future_days=self.job_spec.max_future_days,
                status_filter=self.job_spec.status_filter,
                use_filled=self.job_spec.raw.get('use_filled', True),
                layers=['raw', 'filled'],
                cache_root=self.job_spec.cache_root
            )
            
            # 创建测试数据加载器
            self.test_loader = create_duckdb_loader(
                domain=self.job_spec.domain,
                start_time=self.job_spec.test_start,
                end_time=self.job_spec.test_end,
                registry_manager=None,
                dataset_meta=dataset_meta,
                device=self.device,
                max_backtrack_days=self.job_spec.max_backtrack_days,
                max_future_days=self.job_spec.max_future_days,
                status_filter=self.job_spec.status_filter,
                use_filled=self.job_spec.raw.get('use_filled', True),
                layers=['raw', 'filled'],
                cache_root=self.job_spec.cache_root
            )
            
            logger.info(f"Data loaders setup completed")
            logger.info(f"Train data: {len(self.train_loader)} samples")
            logger.info(f"Test data: {len(self.test_loader)} samples")
            
        except Exception as e:
            logger.error(f"Failed to setup data loaders: {e}")
            raise
    
    def initialize_progress(self, total_episodes: int, config_hash: str):
        """
        初始化进度
        
        Args:
            total_episodes: 总训练轮数
            config_hash: 配置哈希
        """
        # 检查是否可以恢复
        recovery_manager = RecoveryManager(self.progress_manager)
        recovery_status = recovery_manager.check_recovery_status()
        
        if recovery_status['can_recover']:
            logger.info(f"Found existing progress, checking recovery options...")
            recovery_plan = recovery_manager.get_recovery_plan()
            
            if recovery_plan['plan'] == 'recover_from_last_checkpoint':
                logger.info(f"Recovering from checkpoint: {recovery_plan['recommended_checkpoint']}")
                recovery_result = recovery_manager.recover_from_checkpoint(
                    recovery_plan['recommended_checkpoint']
                )
                
                if recovery_result['success']:
                    logger.info(f"Successfully recovered from episode: {recovery_result['resumed_episode']}")
                    return recovery_result['resumed_episode']
                else:
                    logger.warning(f"Recovery failed: {recovery_result['error']}")
        
        # 初始化新的进度
        self.progress_manager.initialize_progress(total_episodes, config_hash)
        logger.info(f"Initialized new progress for {total_episodes} episodes")
        
        return 0  # 从第0轮开始
    
    def evaluate_factor(self, factor_values: np.ndarray, returns: np.ndarray) -> FactorMetrics:
        """
        评价因子
        
        Args:
            factor_values: 因子值
            returns: 收益率
            
        Returns:
            因子评价指标
        """
        try:
            metrics = self.factor_evaluator.evaluate(factor_values, returns)
            logger.info(f"Factor evaluation: IC={metrics.ic:.4f}, RankIC={metrics.rank_ic:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Factor evaluation failed: {e}")
            # 返回空的评价指标
            return FactorMetrics(
                ic=0.0, ric=0.0, rank_ic=0.0, icir=0.0, ricir=0.0,
                turnover=0.0, long_short_return=0.0, long_short_sharpe=0.0,
                long_short_max_drawdown=0.0, quantile_returns=[0.0]*5,
                quantile_sharpes=[0.0]*5, stability=0.0
            )
    
    def save_checkpoint(self, episode: int, pool_state: Dict[str, Any], 
                       metrics: Dict[str, float], expressions: List[Dict[str, Any]]):
        """
        保存检查点
        
        Args:
            episode: 当前轮数
            pool_state: 池状态
            metrics: 评价指标
            expressions: 表达式列表
        """
        try:
            checkpoint_id = self.progress_manager.save_checkpoint(
                episode=episode,
                pool_state=pool_state,
                metrics=metrics,
                expressions=expressions,
                config_hash=self._generate_config_hash(),
                metadata={
                    'domain': self.job_spec.domain,
                    'family_id': self.job_spec.family_id,
                    'dataset_id': self.job_spec.dataset_id
                }
            )
            
            logger.info(f"Saved checkpoint: {checkpoint_id}")
            
            # 清理旧的检查点
            self.progress_manager.cleanup_old_checkpoints(keep_last=5)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def update_progress(self, episode: int, metrics: Dict[str, float], 
                       pool_stats: Dict[str, Any], expressions: List[Dict[str, Any]]):
        """
        更新进度
        
        Args:
            episode: 当前轮数
            metrics: 评价指标
            pool_stats: 池统计信息
            expressions: 表达式列表
        """
        try:
            success = self.progress_manager.update_progress(
                episode=episode,
                metrics=metrics,
                pool_stats=pool_stats,
                expressions=expressions
            )
            
            if success:
                logger.info(f"Updated progress: episode {episode}")
            else:
                logger.warning(f"Failed to update progress for episode {episode}")
                
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        获取进度摘要
        
        Returns:
            进度摘要信息
        """
        return self.progress_manager.get_progress_summary()
    
    def complete_mining(self, final_stats: Dict[str, Any]):
        """
        完成挖掘
        
        Args:
            final_stats: 最终统计信息
        """
        try:
            success = self.progress_manager.complete_progress(final_stats)
            
            if success:
                logger.info("Mining completed successfully")
                summary = self.get_progress_summary()
                logger.info(f"Final summary: {summary}")
            else:
                logger.error("Failed to complete mining")
                
        except Exception as e:
            logger.error(f"Error completing mining: {e}")
    
    def fail_mining(self, error_info: Dict[str, Any]):
        """
        标记挖掘失败
        
        Args:
            error_info: 错误信息
        """
        try:
            success = self.progress_manager.fail_progress(error_info)
            
            if success:
                logger.error(f"Mining marked as failed: {error_info}")
            else:
                logger.error("Failed to mark mining as failed")
                
        except Exception as e:
            logger.error(f"Error marking mining as failed: {e}")
    
    def _generate_config_hash(self) -> str:
        """
        生成配置哈希
        
        Returns:
            配置哈希字符串
        """
        import hashlib
        import json
        
        config_dict = {
            'job_id': self.job_spec.job_id,
            'domain': self.job_spec.domain,
            'family_id': self.job_spec.family_id,
            'train_start': self.job_spec.train_start,
            'train_end': self.job_spec.train_end,
            'test_start': self.job_spec.test_start,
            'test_end': self.job_spec.test_end,
            'n_episodes': self.job_spec.n_episodes,
            'pool_capacity': self.job_spec.pool_capacity,
            'encoder_type': self.job_spec.encoder_type
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def __repr__(self):
        return f"IntegratedMiningPipeline(job_id={self.job_spec.job_id}, domain={self.job_spec.domain})"

def create_integrated_pipeline(job_spec, device='cpu', **kwargs) -> IntegratedMiningPipeline:
    """
    创建集成的挖掘管道
    
    Args:
        job_spec: 作业规格
        device: 计算设备
        **kwargs: 其他参数
        
    Returns:
        IntegratedMiningPipeline实例
    """
    return IntegratedMiningPipeline(job_spec, device, **kwargs)