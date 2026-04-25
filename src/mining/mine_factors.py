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
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

# 导入模块
from mining.job_spec import MiningJobSpec, load_job_spec
from mining.family_search_space import load_family_spec, build_family_search_space
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
from alphagen.data.expression import Expression, Feature, Ref, Abs, Log, Sign
from alpha_gfn.alpha_pool_v2 import AlphaPoolGFN
from alpha_gfn.env.core import GFNEnvCore
from alpha_gfn.modules import SequenceEncoder
from alpha_gfn.gflownet import EntropyTBGFlowNet
from alpha_gfn.config import HIDDEN_DIM, MAX_EXPR_LENGTH
from alpha_gfn.cache_manager import CacheManager
from datahub.duckdb_hub import DuckDBDataHub
from datahub.config import DataHubConfig
from gfn.samplers import Sampler
from gfn.modules import DiscretePolicyEstimator

logger = logging.getLogger(__name__)

class SimpleNeuralNet(nn.Module):
    """简单的多层感知机网络"""
    def __init__(self, input_dim, output_dim, n_hidden_layers=0, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        layers = []

        if n_hidden_layers <= 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class WeightScheduler:
    """权重衰减调度器"""
    def __init__(self, initial_ssl_weight, initial_nov_weight, final_ratio, total_steps, scheduler_type='linear'):
        self.initial_ssl_weight = initial_ssl_weight
        self.initial_nov_weight = initial_nov_weight
        self.final_ratio = final_ratio
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.step_idx = 0

    def step(self):
        self.step_idx += 1

    def get_current_weights(self):
        if self.total_steps <= 1:
            ratio = 1.0
        else:
            progress = min(max(self.step_idx / (self.total_steps - 1), 0.0), 1.0)
            if self.scheduler_type == 'exponential':
                ratio = (self.final_ratio ** progress) if self.final_ratio > 0 else 0.0
            else:
                ratio = 1.0 + (self.final_ratio - 1.0) * progress
        return self.initial_ssl_weight * ratio, self.initial_nov_weight * ratio

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
                 dataset_meta: DatasetMeta, log_dir: str, cuda_override: Optional[int] = None):
        self.job_spec = job_spec
        self.family_spec = family_spec
        self.dataset_meta = dataset_meta
        self.log_dir = log_dir
        
        # 确定设备
        cuda_idx = cuda_override if cuda_override is not None else self.job_spec.raw.get("cuda", 0)
        self.device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() and cuda_idx >= 0 else 'cpu')
        
        self.registry_manager = FeatureRegistryManagerV2()
        
        # 初始化 DataHub
        self.datahub = self._init_datahub()
        
        # 确定层
        self.layers = self.family_spec.get('enabled_layers', self.dataset_meta.layers_enabled)
        
        # 构建数据加载器
        self.train_loader, self.test_loader = self._build_data_loaders()
        
        # 构建目标表达式
        self.target_expression = self._build_target_expression()
        
        # 构建搜索空间
        self.operators, self.delta_times, self.constants = self._build_search_space()
        
        # 构建核心组件
        self.pool = self._build_pool()
        self.env = self._build_env()
        self.gfn, self.sampler, self.optimizer = self._build_gfn_chain()
        
        logger.info(f"Mining context initialized successfully on {self.device}")
    
    def _init_datahub(self) -> Optional[DuckDBDataHub]:
        """从配置初始化 DuckDB DataHub"""
        hub_config_path = self.job_spec.raw.get("datahub_config")
        
        try:
            if hub_config_path and os.path.exists(hub_config_path):
                config = DataHubConfig.from_yaml(hub_config_path)
            else:
                # 默认配置，优先从环境变量或硬编码路径获取
                data_root = os.getenv("DATA_ROOT", "D:/Trading/data_ever_26_3_14/data")
                warehouse_path = os.path.join(data_root, "meta/warehouse.duckdb")
                control_db_path = os.path.join(data_root, "meta/control.sqlite3")
                
                config = DataHubConfig(
                    warehouse_path=warehouse_path,
                    control_db_path=control_db_path
                )
            
            hub = DuckDBDataHub(config)
            logger.info(f"DataHub initialized: {config.warehouse_path}")
            return hub
        except Exception as e:
            logger.warning(f"Failed to initialize DataHub: {e}. Falling back to standard Parquet reading.")
            return None

    def _build_data_loaders(self) -> Tuple[ParquetFeatureLoaderV2, ParquetFeatureLoaderV2]:
        """构建数据加载器"""
        logger.info("Building data loaders...")
        
        common_kwargs = self.dataset_meta.to_loader_kwargs()
        # 覆盖部分参数
        common_kwargs.update({
            "registry_manager": self.registry_manager,
            "device": self.device,
            "max_backtrack_days": self.job_spec.max_backtrack_days,
            "max_future_days": self.job_spec.max_future_days,
            "status_filter": self.job_spec.status_filter,
            "layers": self.layers,
            "cache_root": self.job_spec.raw.get("cache_root", "data/cache"),
            "read_mode": self.job_spec.raw.get("read_mode", "prefer_filled"),
            "segment_name": self.job_spec.segment_name,
            "datahub": self.datahub  # 传递 DataHub 实例
        })
        
        # 训练数据加载器
        train_loader = ParquetFeatureLoaderV2(
            start_time=self.job_spec.train_start,
            end_time=self.job_spec.train_end,
            **common_kwargs
        )
        
        # 测试数据加载器
        test_loader = ParquetFeatureLoaderV2(
            start_time=self.job_spec.test_start,
            end_time=self.job_spec.test_end,
            **common_kwargs
        )
        
        return train_loader, test_loader
    
    def _build_target_expression(self) -> Expression:
        """根据 DatasetMeta 构建目标表达式"""
        logger.info("Building target expression from DatasetMeta...")
        
        feature_enum = self.registry_manager.create_feature_enum(
            self.dataset_meta.domain, 
            self.job_spec.status_filter, 
            self.layers
        )
        
        target_cfg = self.dataset_meta.target_config
        price_col = target_cfg.get("price_column", "close").upper()
        
        if not hasattr(feature_enum, price_col):
            fallback = target_cfg.get("fallback_price_columns", ["CLOSE"])
            for col in fallback:
                col_upper = col.upper()
                if hasattr(feature_enum, col_upper):
                    price_col = col_upper
                    break
            else:
                if hasattr(feature_enum, "CLOSE"):
                    price_col = "CLOSE"
                else:
                    raise ValueError(f"Target column {price_col} not found in feature enum")
        
        base = Feature(getattr(feature_enum, price_col))
        horizon = int(target_cfg.get("label_days", self.job_spec.label_days))
        expr_type = target_cfg.get("expression", "forward_return")
        
        if expr_type == "forward_return":
            return Ref(base, -horizon) / base - 1
        elif expr_type == "forward_change":
            return Ref(base, -horizon) - base
        elif expr_type == "forward_log_return":
            # 假设 Log 算子已导入
            return Log(Ref(base, -horizon) / base)
            
        return Ref(base, -horizon) / base - 1
    
    def _build_search_space(self) -> Tuple[List, List, List]:
        """构建搜索空间"""
        logger.info("Building search space...")
        return build_family_search_space(self.family_spec)

    def _build_pool(self) -> AlphaPoolGFN:
        """构建因子池"""
        # 初始化缓存管理器
        cache_manager = None
        if self.job_spec.cache_root:
            from alpha_gfn.cache_manager import CacheManager
            # 构建配置哈希
            config_hash = CacheManager.compute_config_hash(
                operator_names=[op.__name__ for op in self.operators],
                delta_times=self.delta_times,
                constants=self.constants,
                max_expr_length=self.job_spec.max_expr_length
            )
            
            cache_manager = CacheManager(
                base_cache_dir=self.job_spec.cache_root,
                domain=self.dataset_meta.domain,
                date_range=(self.job_spec.train_start, self.job_spec.train_end),
                config_hash=config_hash
            )
            logger.info(f"Initialized cache manager with config hash: {config_hash}")

        return AlphaPoolGFN(
            capacity=self.job_spec.pool_capacity,
            stock_data=self.train_loader,
            target=self.target_expression,
            entry_strategy=self.job_spec.raw.get("entry_strategy", "ic_ranking"),
            diversity_weight=self.job_spec.raw.get("diversity_weight", 0.3),
            min_ic_threshold=self.job_spec.raw.get("min_ic_threshold", 0.05),
            max_similarity_threshold=self.job_spec.raw.get("max_similarity_threshold", 0.95),
            enable_cache=True,
            cache_manager=cache_manager
        )

    def _build_env(self) -> GFNEnvCore:
        """构建 GFN 环境"""
        # 获取特征索引
        feature_enum = self.registry_manager.create_feature_enum(
            self.dataset_meta.domain, self.job_spec.status_filter, self.layers
        )
        
        # 确定需要排除的列（目标计算用的价格列）
        target_cfg = self.dataset_meta.target_config
        exclude_cols = {
            target_cfg.get("price_column", "close").upper(),
            "CLOSE"  # 总是排除默认的 CLOSE
        }
        
        # 获取要排除的 fallback 列
        fallback = target_cfg.get("fallback_price_columns", [])
        for col in fallback:
            exclude_cols.add(col.upper())
            
        # 过滤特征索引，排除用于计算目标的列
        feature_indices = [m for m in feature_enum if m.name not in exclude_cols]
        
        logger.info(f"Environment features: {len(feature_indices)} (excluded: {exclude_cols})")
        
        return GFNEnvCore(
            pool=self.pool,
            device=self.device,
            mask_dropout_prob=self.job_spec.raw.get("mask_dropout_prob", 1.0),
            ssl_weight=self.job_spec.raw.get("ssl_weight", 1.0),
            nov_weight=self.job_spec.raw.get("nov_weight", 0.3),
            custom_features=feature_indices,
            operators=self.operators,
            delta_times=self.delta_times,
            constants=self.constants,
            max_expr_length=self.job_spec.max_expr_length,
            cache_manager=self.pool.cache_manager
        )

    def _build_gfn_chain(self) -> Tuple[EntropyTBGFlowNet, Sampler, Adam]:
        """构建 GFN 链: Model, Policy, Loss, Sampler, Optimizer"""
        n_tokens = self.env.n_tokens
        encoder_type = self.job_spec.raw.get("encoder_type", "gnn")
        
        backbone = SequenceEncoder(n_tokens, encoder_type)
        pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=self.env.n_actions, n_hidden_layers=0)
        pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=self.env.n_actions - 1, n_hidden_layers=0)
        
        pf_module = nn.Sequential(backbone, pf_head)
        pb_module = nn.Sequential(backbone, pb_head)
        
        pf = DiscretePolicyEstimator(pf_module, n_actions=self.env.n_actions, preprocessor=self.env.preprocessor)
        pb = DiscretePolicyEstimator(pb_module, n_actions=self.env.n_actions, preprocessor=self.env.preprocessor, is_backward=True)

        loss_fn = EntropyTBGFlowNet(
            pf=pf,
            pb=pb,
            entropy_coef=self.job_spec.raw.get("entropy_coef", 0.01),
            entropy_temperature=self.job_spec.raw.get("entropy_temperature", 1.0)
        )
        loss_fn.to(self.device)
        sampler = Sampler(estimator=pf)
        
        params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
        optimizer = Adam(params, lr=self.job_spec.raw.get("learning_rate", 1e-4))
        
        return loss_fn, sampler, optimizer

def build_mining_context(job_spec_path: str, dataset_meta_path: Optional[str] = None, 
                        log_dir: Optional[str] = None, cuda_override: Optional[int] = None) -> MiningContext:
    """
    构建挖掘上下文
    
    Args:
        job_spec_path: 作业规格文件路径
        dataset_meta_path: 数据集元数据文件路径
        log_dir: 日志目录
        cuda_override: 覆盖CUDA设备索引
        
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
        raise ValueError(
            "dataset_meta_path is required. "
            "Do not create default A-share DatasetMeta in production pipeline."
        )
    
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
    context = MiningContext(job_spec, family_spec, dataset_meta, log_dir, cuda_override)
    context.logger = logger_obj
    
    return context

def mine_factors(job_spec_path: str, dataset_meta_path: Optional[str] = None, 
                log_dir: Optional[str] = None, cuda_override: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    执行因子挖掘任务
    
    Args:
        job_spec_path: 作业规格文件路径
        dataset_meta_path: 数据集元数据文件路径
        log_dir: 日志目录
        cuda_override: 覆盖CUDA设备索引
        **kwargs: 额外参数
        
    Returns:
        挖掘结果字典
    """
    logger.info("Starting factor mining pipeline...")
    
    # 构建挖掘上下文
    context = build_mining_context(job_spec_path, dataset_meta_path, log_dir, cuda_override)
    
    # 初始化权重调度器
    n_episodes = context.job_spec.n_episodes
    weight_scheduler = WeightScheduler(
        initial_ssl_weight=context.job_spec.raw.get("ssl_weight", 1.0),
        initial_nov_weight=context.job_spec.raw.get("nov_weight", 0.3),
        final_ratio=context.job_spec.raw.get("final_weight_ratio", 0.0),
        total_steps=n_episodes,
        scheduler_type=context.job_spec.raw.get("weight_decay_type", 'linear')
    )
    
    logger.info(f"Starting GFN training for job {context.job_spec.job_id}...")
    
    # 训练循环
    for episode in tqdm(range(n_episodes)):
        # 更新权重
        current_ssl_weight, current_nov_weight = weight_scheduler.get_current_weights()
        context.env.ssl_weight = current_ssl_weight
        context.env.nov_weight = current_nov_weight
        
        # 采样轨迹
        trajectories = context.sampler.sample_trajectories(
            env=context.env, 
            n=context.job_spec.raw.get("batch_size", 1),
            save_estimator_outputs=context.job_spec.raw.get("entropy_coef", 0.01) > 0
        )
        
        # 计算损失
        loss = context.gfn.loss(env=context.env, trajectories=trajectories)

        if loss is not None and torch.isfinite(loss):
            context.optimizer.zero_grad()
            loss.backward()
            context.optimizer.step()

        # 记录日志和检查点
        if episode > 0 and (episode + 1) % context.job_spec.log_freq == 0:
            pool_stats = {
                'pool_size': len(context.pool.exprs),
                'total_evaluations': context.pool.total_evaluations,
                'cache_hit_rate': context.pool.cache_hit_rate if hasattr(context.pool, 'cache_hit_rate') else 0
            }
            
            # 获取池中最佳 IC
            best_ic = 0.0
            if context.pool.exprs:
                best_ic = max(context.pool.ics)
            
            metrics = {
                'loss': float(loss.detach().cpu().item()),
                'best_ic': best_ic,
                'ssl_weight': current_ssl_weight,
                'nov_weight': current_nov_weight
            }
            
            context.logger.log_checkpoint(episode + 1, pool_stats, metrics)
        
        weight_scheduler.step()
    
    # 导出最终结果
    final_pool = context.pool.export_pool() if hasattr(context.pool, "export_pool") else {}
    final_stats = {
        'pool_size': len(context.pool.exprs),
        'best_ic': max(context.pool.ics) if context.pool.exprs else 0,
        'total_evaluations': context.pool.total_evaluations,
        'cache_hit_rate': context.pool.cache_hit_rate if hasattr(context.pool, 'cache_hit_rate') else 0
    }
    
    context.logger.log_completion(final_stats)
    
    logger.info("Factor mining pipeline completed")
    
    # 返回上下文和池信息
    return {
        'status': 'completed',
        'log_dir': context.log_dir,
        'pool_size': len(context.pool.exprs),
        'best_ic': final_stats['best_ic']
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