#!/usr/bin/env python3
"""
增强版的train_gfn_v2.py - 集成新的DataHub和进度管理系统
保持原有GFlowNets算法不变，只改造数据接入和架构层
"""

import torch
import random
import numpy as np
import argparse
import os
import json
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, PolynomialLR
from torch.distributions import Categorical
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

# 路径配置
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# 导入AlphaGen模块
from alphagen.rl.env.wrapper import action2token
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr
from alphagen_generic.features import *
from alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.config import get_date_range
from alphagen_generic.task_config import apply_task_config, parse_str_list
from alphagen_generic.run_artifacts import build_run_dir, save_json

# 导入GFN模块
try:
    from alpha_gfn.config import *
    from alpha_gfn.env.core import GFNEnvCore
    from alpha_gfn.modules import SequenceEncoder
    from alpha_gfn.alpha_pool_v2 import AlphaPoolGFN
    from alpha_gfn.gflownet import EntropyTBGFlowNet
    from alpha_gfn.search_space import build_search_space
    from alpha_gfn.cache_manager import CacheManager
except ImportError:
    from src.alpha_gfn.config import *
    from src.alpha_gfn.env.core import GFNEnvCore
    from src.alpha_gfn.modules import SequenceEncoder
    from src.alpha_gfn.alpha_pool_v2 import AlphaPoolGFN
    from src.alpha_gfn.gflownet import EntropyTBGFlowNet
    from src.alpha_gfn.search_space import build_search_space
    from src.alpha_gfn.cache_manager import CacheManager

# 导入GFN库
try:
    from gfn.samplers import Sampler
    from gfn.modules import DiscretePolicyEstimator
    from gfn.gflownet import TBGFlowNet
except Exception:
    import traceback
    print("GFN library import failed. Please ensure 'torchgfn' is installed correctly.")
    traceback.print_exc()
    sys.exit(1)

# 导入新模块
try:
    from mining.job_spec import MiningJobSpec, load_job_spec
    from mining.family_search_space import load_family_spec, build_family_search_space
    from mining.mine_factors import mine_factors
    # 新增：集成模块
    from integration import create_integrated_pipeline
    from datahub.adapter import DuckDBParquetFeatureLoaderV2
except ImportError:
    from src.mining.job_spec import MiningJobSpec, load_job_spec
    from src.mining.family_search_space import load_family_spec, build_family_search_space
    from src.mining.mine_factors import mine_factors
    from src.integration import create_integrated_pipeline
    from src.datahub.adapter import DuckDBParquetFeatureLoaderV2

load_dotenv(ROOT / ".env")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleNeuralNet(nn.Module):
    """简单的神经网络"""
    def __init__(self, input_dim, output_dim, n_hidden_layers=0, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim or input_dim
        
        layers = []
        if n_hidden_layers == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def resolve_date_ranges(args):
    """解析日期范围"""
    if args.train_start and args.train_end:
        train_start, train_end = args.train_start, args.train_end
    else:
        train_start = f"{args.train_start_year}0101"
        train_end = f"{args.train_end_year}1231"
    
    if args.test_start and args.test_end:
        test_start, test_end = args.test_start, args.test_end
    else:
        test_start = f"{args.test_start_year}0101"
        test_end = f"{args.test_end_year}1231"
    
    return train_start, train_end, test_start, test_end

def build_job_context(args) -> Dict[str, Any]:
    """构建作业上下文 - 增强版，支持新的数据层"""
    logger.info("Building enhanced job context...")
    
    # 基本配置
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集元数据
    dataset_meta = None
    if hasattr(args, "dataset_meta") and args.dataset_meta:
        try:
            dataset_meta = DatasetMeta(args.dataset_meta)
            logger.info(f"Loaded dataset meta from {args.dataset_meta}")
            errors = dataset_meta.validate()
            if errors:
                logger.warning(f"Dataset meta validation errors: {errors}")
        except Exception as e:
            logger.warning(f"Failed to load dataset meta: {e}, using default configuration")
    
    # 加载作业规格
    job_spec = None
    if hasattr(args, "job_spec") and args.job_spec:
        try:
            job_spec = load_job_spec(args.job_spec)
            logger.info(f"Loaded job spec from {args.job_spec}")
        except Exception as e:
            logger.error(f"Failed to load job spec: {e}")
            raise
    
    # 加载因子族规格
    family_spec = None
    if hasattr(args, "family_id") and args.family_id:
        try:
            family_spec = load_family_spec(args.family_id)
            logger.info(f"Loaded family spec for {args.family_id}")
        except Exception as e:
            logger.error(f"Failed to load family spec: {e}")
            raise
    
    # 解析日期范围
    train_start, train_end, test_start, test_end = resolve_date_ranges(args)
    
    # 增强：创建集成管道
    integrated_pipeline = None
    if job_spec:
        try:
            integrated_pipeline = create_integrated_pipeline(
                job_spec=job_spec,
                device=device
            )
            logger.info("Created integrated mining pipeline")
        except Exception as e:
            logger.warning(f"Failed to create integrated pipeline: {e}")
    
    context = {
        'device': device,
        'dataset_meta': dataset_meta,
        'job_spec': job_spec,
        'family_spec': family_spec,
        'integrated_pipeline': integrated_pipeline,  # 新增
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'domain': args.domain,
        'seed': args.seed,
        'max_backtrack_days': getattr(args, "max_backtrack_days", 100),
        'max_future_days': getattr(args, "max_future_days", 30),
        'use_filled': getattr(args, "use_filled", True),
        'status_filter': parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch'],
        'cache_root': getattr(args, "cache_root", "data/cache"),
        'output_dir': getattr(args, "output_dir", None),
        'run_name': getattr(args, "run_name", None),
    }
    
    return context

def build_train_test_loaders_v2(job_ctx: Dict[str, Any]) -> Tuple[DuckDBParquetFeatureLoaderV2, DuckDBParquetFeatureLoaderV2]:
    """构建训练和测试数据加载器 - 使用新的DuckDB适配器"""
    logger.info("Building enhanced train/test loaders with DuckDB...")
    
    # 使用集成管道中的数据加载器
    integrated_pipeline = job_ctx.get('integrated_pipeline')
    if integrated_pipeline and integrated_pipeline.train_loader and integrated_pipeline.test_loader:
        logger.info("Using integrated pipeline data loaders")
        return integrated_pipeline.train_loader, integrated_pipeline.test_loader
    
    # 回退到传统方式
    logger.info("Falling back to traditional ParquetFeatureLoaderV2")
    return build_train_test_loaders_traditional(job_ctx)

def build_train_test_loaders_traditional(job_ctx: Dict[str, Any]) -> Tuple[ParquetFeatureLoaderV2, ParquetFeatureLoaderV2]:
    """传统的数据加载器构建方式"""
    # 确定特征注册管理器版本
    if job_ctx.get('family_spec') and job_ctx['family_spec'].get('feature_scopes'):
        registry_manager = FeatureRegistryManagerV2()
        layers = job_ctx['family_spec'].get('enabled_layers', ['raw', 'filled'])
    else:
        registry_manager = FeatureRegistryManager()
        layers = ['raw', 'filled']
    
    # 构建训练数据加载器
    logger.info(f"Loading training data for domain {job_ctx['domain']} ({job_ctx['train_start']} to {job_ctx['train_end']})...")
    train_loader = ParquetFeatureLoaderV2(
        domain=job_ctx['domain'],
        start_time=job_ctx['train_start'],
        end_time=job_ctx['train_end'],
        registry_manager=registry_manager,
        dataset_meta=job_ctx['dataset_meta'],
        device=job_ctx['device'],
        max_backtrack_days=job_ctx['max_backtrack_days'],
        max_future_days=job_ctx['max_future_days'],
        status_filter=job_ctx['status_filter'],
        use_filled=job_ctx['use_filled'],
        layers=layers,
        cache_root=job_ctx['cache_root']
    )
    
    # 构建测试数据加载器
    logger.info(f"Loading test data for domain {job_ctx['domain']} ({job_ctx['test_start']} to {job_ctx['test_end']})...")
    test_loader = ParquetFeatureLoaderV2(
        domain=job_ctx['domain'],
        start_time=job_ctx['test_start'],
        end_time=job_ctx['test_end'],
        registry_manager=registry_manager,
        dataset_meta=job_ctx['dataset_meta'],
        device=job_ctx['device'],
        max_backtrack_days=job_ctx['max_backtrack_days'],
        max_future_days=job_ctx['max_future_days'],
        status_filter=job_ctx['status_filter'],
        use_filled=job_ctx['use_filled'],
        layers=layers,
        cache_root=job_ctx['cache_root']
    )
    
    return train_loader, test_loader

def build_gfn_components(job_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """构建GFN组件"""
    logger.info("Building GFN components...")
    
    # 构建搜索空间
    selected_operators, selected_delta_times, selected_constants = build_search_space(
        operator_names=getattr(job_ctx, "operator_names", None),
        delta_times=getattr(job_ctx, "delta_times", None),
        constants=getattr(job_ctx, "constants", None),
    )
    
    # 构建数据加载器
    train_loader, test_loader = build_train_test_loaders_v2(job_ctx)
    
    # 创建目标表达式
    close = Feature(train_loader.close)
    target = Ref(close, -job_ctx.get('label_days', 10)) / close - 1
    
    # 初始化AlphaPoolGFN
    pool = AlphaPoolGFN(
        capacity=job_ctx.get('pool_capacity', 50),
        stock_data=train_loader,
        target=target,
        device=job_ctx['device']
    )
    
    # 初始化缓存管理器
    cache_manager = None
    if job_ctx.get('use_cache', True):
        config_hash = f"{job_ctx['domain']}_{job_ctx['train_start']}_{job_ctx['train_end']}"
        cache_manager = CacheManager(
            cache_dir=os.path.join(job_ctx['cache_root'], config_hash),
            max_size=10000
        )
        pool.cache_manager = cache_manager
        logger.info(f"Initialized cache manager with config hash: {config_hash}")
    
    # 创建特征枚举
    if job_ctx.get('family_spec') and job_ctx['family_spec'].get('enabled_layers'):
        registry_manager = FeatureRegistryManagerV2()
        layers = job_ctx['family_spec']['enabled_layers']
        feature_indices = [m for m in registry_manager.create_feature_enum(job_ctx['domain'], job_ctx['status_filter'], layers) if m.name != 'CLOSE']
    else:
        registry_manager = FeatureRegistryManager()
        feature_indices = [m for m in registry_manager.create_feature_enum(job_ctx['domain'], job_ctx['status_filter']) if m.name != 'CLOSE']
    
    # 初始化模型
    n_tokens = 1 + len(selected_operators) + len(feature_indices) + len(selected_delta_times) + len(selected_constants) + 1
    
    # 确定编码器类型
    encoder_type = job_ctx.get('job_spec', {}).get('encoder_type', 'gnn') if job_ctx.get('job_spec') else 'gnn'
    
    backbone = SequenceEncoder(n_tokens, encoder_type)
    
    # 初始化环境
    env_config = {
        'pool': pool,
        'encoder': backbone,
        'device': job_ctx['device'],
        'mask_dropout_prob': job_ctx.get('job_spec', {}).get('mask_dropout_prob', 1.0) if job_ctx.get('job_spec') else 1.0,
        'ssl_weight': job_ctx.get('job_spec', {}).get('ssl_weight', 1.0) if job_ctx.get('job_spec') else 1.0,
        'nov_weight': job_ctx.get('job_spec', {}).get('nov_weight', 0.3) if job_ctx.get('job_spec') else 0.3,
        'custom_features': feature_indices,
        'operators': selected_operators,
        'delta_times': selected_delta_times,
        'constants': selected_constants,
        'max_expr_length': job_ctx.get('job_spec', {}).get('max_expr_length', MAX_EXPR_LENGTH)
    }
    
    env = GFNEnvCore(**env_config)
    
    # 初始化策略网络
    pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0)
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)
    
    # 初始化损失函数
    entropy_coef = job_ctx.get('job_spec', {}).get('entropy_coef', 0.01) if job_ctx.get('job_spec') else 0.01
    entropy_temperature = job_ctx.get('job_spec', {}).get('entropy_temperature', 1.0) if job_ctx.get('job_spec') else 1.0
    
    loss_fn = EntropyTBGFlowNet(
        pf=pf,
        pb=pb,
        entropy_coef=entropy_coef,
        entropy_temperature=entropy_temperature
    )
    loss_fn.to(job_ctx['device'])
    
    sampler = Sampler(estimator=pf)
    
    # 优化器参数
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)
    
    components = {
        'env': env,
        'pool': pool,
        'backbone': backbone,
        'pf': pf,
        'pb': pb,
        'loss_fn': loss_fn,
        'sampler': sampler,
        'optimizer': optimizer,
        'cache_manager': cache_manager,
        'train_loader': train_loader,
        'test_loader': test_loader
    }
    
    return components

def setup_logging(job_ctx: Dict[str, Any]) -> str:
    """设置日志系统"""
    output_root = job_ctx['output_dir'] or os.path.join(ROOT, "data", "gfn_logs")
    os.makedirs(output_root, exist_ok=True)
    
    # 构建运行目录
    prefix = f"gfn_{job_ctx['domain']}"
    if job_ctx.get('family_spec'):
        prefix = f"gfn_{job_ctx['family_spec'].get('family_id', job_ctx['domain'])}"
    
    log_dir = build_run_dir(output_root, prefix=prefix, run_name=job_ctx['run_name'])
    
    # 保存配置
    save_json(os.path.join(log_dir, "resolved_args.json"), job_ctx)
    
    # 保存搜索空间配置
    if job_ctx.get('family_spec'):
        save_json(
            os.path.join(log_dir, "search_space.json"),
            {
                "operators": [op.__name__ for op in job_ctx['family_spec'].get('operators', [])],
                "delta_times": job_ctx['family_spec'].get('delta_times', []),
                "constants": job_ctx['family_spec'].get('constants', [])
            }
        )
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir)
    job_ctx['writer'] = writer
    
    return log_dir

def train_with_job_spec(args):
    """使用作业规格进行训练 - 增强版"""
    logger.info("Starting enhanced training with job spec...")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 构建作业上下文
    job_ctx = build_job_context(args)
    
    # 设置日志
    log_dir = setup_logging(job_ctx)
    logger.info(f"Logging to: {log_dir}")
    
    # 初始化进度管理（新增）
    integrated_pipeline = job_ctx.get('integrated_pipeline')
    if integrated_pipeline:
        # 设置数据加载器
        integrated_pipeline.setup_data_loaders(job_ctx.get('dataset_meta'))
        
        # 初始化进度
        start_episode = integrated_pipeline.initialize_progress(
            total_episodes=job_ctx['job_spec'].n_episodes,
            config_hash=integrated_pipeline._generate_config_hash()
        )
        logger.info(f"Starting from episode: {start_episode}")
    
    # 构建GFN组件
    components = build_gfn_components(job_ctx)
    env = components['env']
    pool = components['pool']
    backbone = components['backbone']
    pf = components['pf']
    pb = components['pb']
    loss_fn = components['loss_fn']
    sampler = components['sampler']
    optimizer = components['optimizer']
    cache_manager = components['cache_manager']
    
    # 训练循环
    n_episodes = job_ctx['job_spec'].n_episodes
    log_freq = job_ctx['job_spec'].raw.get('log_freq', 1000)
    
    logger.info(f"Starting training for {n_episodes} episodes...")
    
    for episode in tqdm(range(start_episode, n_episodes), desc="Training"):
        # 训练步骤（保持原有逻辑）
        optimizer.zero_grad()
        
        # 采样轨迹
        trajectories = sampler.sample(n_trajectories=1)
        
        # 计算损失
        loss = loss_fn(trajectories)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新池（保持原有逻辑）
        expressions = []
        for traj in trajectories:
            if traj.is_complete:
                expr = env.state_to_expression(traj.states[-1])
                expressions.append(expr)
        
        if expressions:
            pool.add_expressions(expressions)
        
        # 增强：更新进度和保存检查点
        if integrated_pipeline and (episode + 1) % log_freq == 0:
            try:
                # 获取当前池状态
                pool_stats = {
                    'pool_size': len(pool),
                    'best_ic': pool.best_ic if hasattr(pool, 'best_ic') else 0.0,
                    'total_expressions': len(pool.expressions) if hasattr(pool, 'expressions') else 0
                }
                
                # 评价当前最佳因子（简化版）
                if pool.expressions:
                    best_expr = pool.expressions[0] if hasattr(pool, 'expressions') else None
                    if best_expr:
                        # 这里可以添加更完整的因子评价
                        metrics = {'best_ic': pool_stats['best_ic']}
                    else:
                        metrics = {'best_ic': 0.0}
                else:
                    metrics = {'best_ic': 0.0}
                
                # 更新进度
                integrated_pipeline.update_progress(
                    episode=episode + 1,
                    metrics=metrics,
                    pool_stats=pool_stats,
                    expressions=[]  # 可以添加表达式序列化
                )
                
                # 保存检查点
                integrated_pipeline.save_checkpoint(
                    episode=episode + 1,
                    pool_state=pool_stats,
                    metrics=metrics,
                    expressions=[]
                )
                
                # 记录日志
                logger.info(f"Episode {episode + 1}: Pool size={pool_stats['pool_size']}, Best IC={metrics['best_ic']:.4f}")
                
            except Exception as e:
                logger.error(f"Error updating progress at episode {episode + 1}: {e}")
        
        # TensorBoard日志
        if job_ctx.get('writer') and (episode + 1) % log_freq == 0:
            try:
                job_ctx['writer'].add_scalar('Loss/train', loss.item(), episode + 1)
                if hasattr(pool, 'best_ic'):
                    job_ctx['writer'].add_scalar('Metrics/best_ic', pool.best_ic, episode + 1)
                job_ctx['writer'].add_scalar('Metrics/pool_size', len(pool), episode + 1)
            except Exception as e:
                logger.warning(f"Failed to write TensorBoard logs: {e}")
    
    # 训练完成
    logger.info("Training completed!")
    
    # 完成进度
    if integrated_pipeline:
        final_stats = {
            'pool_size': len(pool),
            'best_ic': getattr(pool, 'best_ic', 0.0),
            'total_episodes': n_episodes,
            'final_loss': loss.item() if 'loss' in locals() else 0.0
        }
        integrated_pipeline.complete_mining(final_stats)
    
    # 保存最终结果
    try:
        final_results = {
            'job_id': job_ctx['job_spec'].job_id,
            'best_expressions': pool.expressions[:10] if hasattr(pool, 'expressions') else [],
            'best_metrics': {'best_ic': getattr(pool, 'best_ic', 0.0)},
            'total_episodes': n_episodes,
            'final_pool_size': len(pool)
        }
        
        results_file = os.path.join(log_dir, "final_results.json")
        save_json(results_file, final_results)
        logger.info(f"Saved final results to: {results_file}")
        
    except Exception as e:
        logger.error(f"Failed to save final results: {e}")
    
    # 关闭TensorBoard
    if job_ctx.get('writer'):
        job_ctx['writer'].close()

def train_traditional(args):
    """传统训练方式 - 保持不变"""
    logger.info("Starting traditional training...")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备配置
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # 数据集元数据
    dataset_meta = None
    if args.dataset_meta:
        try:
            dataset_meta = DatasetMeta(args.dataset_meta)
            logger.info(f"Loaded dataset meta from {args.dataset_meta}")
        except Exception as e:
            logger.warning(f"Failed to load dataset meta: {e}")
    
    # 解析日期范围
    train_start, train_end, test_start, test_end = resolve_date_ranges(args)
    
    # 构建搜索空间
    selected_operators, selected_delta_times, selected_constants = build_search_space(
        operator_names=getattr(args, "operator_names", None),
        delta_times=getattr(args, "delta_times", None),
        constants=getattr(args, "constants", None),
    )
    
    # 构建数据加载器
    logger.info(f"Loading training data for domain {args.domain} ({train_start} to {train_end})...")
    data = ParquetFeatureLoader(
        domain=args.domain,
        start_time=train_start,
        end_time=train_end,
        registry_manager=FeatureRegistryManager(),
        dataset_meta=dataset_meta,
        pool_path=getattr(args, "sample_pool_path", None),
        daily_path=getattr(args, "daily_path", None),
        device=device,
        max_backtrack_days=getattr(args, "max_backtrack_days", 100),
        max_future_days=getattr(args, "max_future_days", 30),
        status_filter=parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch'],
        use_filled=getattr(args, "use_filled", True),
        feature_file_map_raw=getattr(args, "feature_file_map_raw", None),
        feature_file_map_filled=getattr(args, "feature_file_map_filled", None),
        date_column=getattr(args, "date_column", "trade_date"),
        code_column=getattr(args, "code_column", "ts_code"),
        close_column=getattr(args, "close_column", "close"),
        data_dir=getattr(args, "data_dir", None),
    )
    
    logger.info(f"Loading test data for domain {args.domain} ({test_start} to {test_end})...")
    data_test = ParquetFeatureLoader(
        domain=args.domain,
        start_time=test_start,
        end_time=test_end,
        registry_manager=FeatureRegistryManager(),
        dataset_meta=dataset_meta,
        pool_path=getattr(args, "sample_pool_path", None),
        daily_path=getattr(args, "daily_path", None),
        device=device,
        max_backtrack_days=getattr(args, "max_backtrack_days", 100),
        max_future_days=getattr(args, "max_future_days", 30),
        status_filter=parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch'],
        use_filled=getattr(args, "use_filled", True),
        feature_file_map_raw=getattr(args, "feature_file_map_raw", None),
        feature_file_map_filled=getattr(args, "feature_file_map_filled", None),
        date_column=getattr(args, "date_column", "trade_date"),
        code_column=getattr(args, "code_column", "ts_code"),
        close_column=getattr(args, "close_column", "close"),
        data_dir=getattr(args, "data_dir", None),
    )
    
    # 创建特征枚举
    feature_enum = FeatureRegistryManager().create_feature_enum(args.domain, parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch'])
    feature_indices = [m for m in feature_enum if m.name != 'CLOSE']
    
    # 创建目标表达式
    close = Feature(feature_enum.CLOSE)
    target = Ref(close, -args.label_days) / close - 1
    
    # 初始化AlphaPoolGFN
    pool = AlphaPoolGFN(capacity=args.pool_capacity, stock_data=data, target=target)
    
    # 初始化模型
    n_tokens = 1 + len(selected_operators) + len(feature_indices) + len(selected_delta_times) + len(selected_constants) + 1
    
    backbone = SequenceEncoder(n_tokens, args.encoder_type)
    
    # 初始化环境
    env = GFNEnvCore(
        pool=pool,
        encoder=backbone,
        device=device,
        custom_features=feature_indices,
        operators=selected_operators,
        delta_times=selected_delta_times,
        constants=selected_constants,
        max_expr_length=args.max_expr_length
    )
    
    # 初始化策略网络
    pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0)
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)
    
    # 初始化损失函数
    loss_fn = EntropyTBGFlowNet(pf=pf, pb=pb, entropy_coef=args.entropy_coef)
    loss_fn.to(device)
    
    sampler = Sampler(estimator=pf)
    
    # 优化器
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)
    
    # 设置日志
    output_root = args.output_dir or os.path.join(ROOT, "data", "gfn_logs")
    os.makedirs(output_root, exist_ok=True)
    
    prefix = f"gfn_{args.domain}"
    log_dir = build_run_dir(output_root, prefix=prefix, run_name=args.run_name)
    
    # 训练循环（保持原有逻辑）
    logger.info(f"Starting training for {args.n_episodes} episodes...")
    
    for episode in tqdm(range(args.n_episodes), desc="Training"):
        optimizer.zero_grad()
        
        # 采样轨迹
        trajectories = sampler.sample(n_trajectories=1)
        
        # 计算损失
        loss = loss_fn(trajectories)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新池
        expressions = []
        for traj in trajectories:
            if traj.is_complete:
                expr = env.state_to_expression(traj.states[-1])
                expressions.append(expr)
        
        if expressions:
            pool.add_expressions(expressions)
        
        # 日志记录
        if (episode + 1) % args.log_freq == 0:
            logger.info(f"Episode {episode + 1}: Pool size={len(pool)}, Best IC={getattr(pool, 'best_ic', 0):.4f}")
    
    logger.info("Training completed!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Enhanced GFN Factor Mining')
    
    # 基础参数
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
    
    # 作业规格参数
    parser.add_argument('--job_spec', type=str, default=None, help='Job specification file path')
    parser.add_argument('--family_id', type=str, default=None, help='Factor family ID')
    
    # 数据参数
    parser.add_argument('--domain', type=str, default='A', help='Domain name')
    parser.add_argument('--dataset_meta', type=str, default=None, help='Dataset metadata file')
    
    # 训练参数
    parser.add_argument('--n_episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--pool_capacity', type=int, default=50, help='Alpha pool capacity')
    parser.add_argument('--encoder_type', type=str, default='gnn', choices=['gnn', 'transformer', 'lstm'], help='Encoder type')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--label_days', type=int, default=10, help='Label days')
    parser.add_argument('--max_expr_length', type=int, default=20, help='Maximum expression length')
    parser.add_argument('--log_freq', type=int, default=1000, help='Logging frequency')
    
    # 日期参数
    parser.add_argument('--train_start_year', type=int, default=2020)
    parser.add_argument('--train_end_year', type=int, default=2021)
    parser.add_argument('--test_start_year', type=int, default=2021)
    parser.add_argument('--test_end_year', type=int, default=2022)
    
    # 数据路径参数
    parser.add_argument('--train_end_year', type=int, default=2021)
    parser.add_argument('--test_end_year', type=int, default=2025)
    parser.add_argument('--train_start', type=str, default=None)
    parser.add_argument('--train_end', type=str, default=None)
    parser.add_argument('--test_start', type=str, default=None)
    parser.add_argument('--test_end', type=str, default=None)
    parser.add_argument('--registry_path', type=str, default=None)
    parser.add_argument('--sample_pool_path', type=str, default=None)
    parser.add_argument('--daily_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    
    # 高级参数
    parser.add_argument('--status_filter', type=str, default=None)
    parser.add_argument('--max_backtrack_days', type=int, default=100)
    parser.add_argument('--max_future_days', type=int, default=30)
    parser.add_argument('--use_filled', type=str2bool, default=True)
    parser.add_argument('--date_column', type=str, default='trade_date')
    parser.add_argument('--code_column', type=str, default='ts_code')
    parser.add_argument('--close_column', type=str, default='close')
    parser.add_argument('--max_expr_length', type=int, default=MAX_EXPR_LENGTH)
    parser.add_argument('--operator_names', type=str, default=None)
    parser.add_argument('--delta_times', type=str, default=None)
    parser.add_argument('--constants', type=str, default=None)
    
    # 新增参数
    parser.add_argument('--cache_root', type=str, default='data/cache',
                        help="Root directory for data caching")
    parser.add_argument('--candidate_export_freq', type=int, default=1000,
                        help="Frequency for exporting candidate pools")
    
    # 辅助函数
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument('--use_cache', type=str2bool, default=True,
                        help="Use cache manager")
    
    args = parser.parse_args()
    
    # 应用任务配置
    task_config_raw = apply_task_config(args, parser)
    if task_config_raw:
        setattr(args, "_task_config_raw", task_config_raw)
    
    # 确定训练模式
    if args.job_spec or args.family_id:
        # 使用新的作业规格系统
        train_with_job_spec(args)
    else:
        # 使用传统训练方式
        train_traditional(args)

if __name__ == '__main__':
    main()