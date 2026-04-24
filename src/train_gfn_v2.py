#!/usr/bin/env python3
"""
升级版的GFN训练脚本，支持job_spec和因子族配置
同时保持向后兼容性
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
except ImportError:
    from src.mining.job_spec import MiningJobSpec, load_job_spec
    from src.mining.family_search_space import load_family_spec, build_family_search_space
    from src.mining.mine_factors import mine_factors

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

class GFNLogger:
    """GFN训练日志器"""
    def __init__(self, model: nn.Module, pool: AlphaPoolGFN, log_dir: str, test_data: ParquetFeatureLoader, target: Expression):
        self.model = model
        self.pool = pool
        self.log_dir = log_dir
        self.test_data = test_data
        self.target = target
        self.writer = SummaryWriter(log_dir)
        self.target_test = self.pool._normalize_by_day(self.target.evaluate(self.test_data))

    def log_metrics(self, episode: int):
        self.writer.add_scalar('pool/size', self.pool.size, episode)
        if self.pool.size > 0:
            self.writer.add_scalar('pool/best_single_ic', np.max(self.pool.single_ics[:self.pool.size]), episode)
            # 测试集成指标需要实现或与项目指标对齐
            # 暂时只记录最佳IC
        self.writer.add_scalar('pool/eval_cnt', self.pool.eval_cnt, episode)

    def save_checkpoint(self, episode: int):
        model_path = os.path.join(self.log_dir, f'model_{episode}.pt')
        pool_path = os.path.join(self.log_dir, f'pool_{episode}.json')
        torch.save(self.model.state_dict(), model_path)
        with open(pool_path, 'w') as f:
            json.dump(self.pool.to_dict(), f, indent=4)

    def show_pool_state(self):
        state = self.pool.to_dict()
        exprs = state.get('exprs', [])
        ics = state.get('ics_ret', [])
        n = len(exprs)
        print('---------------------------------------------')
        for i in range(n):
            expr_str = exprs[i]
            ic_ret = ics[i]
            print(f'> Alpha #{i}: ic={ic_ret:.4f}, expr={expr_str}')
        if self.pool.size > 0:
            print(f'>> Best single ic: {np.max(self.pool.single_ics[:self.pool.size]):.4f}')
        print('---------------------------------------------')

    def close(self):
        self.writer.close()

class WeightScheduler:
    """权重调度器"""
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

def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool): 
        return v
    return v.lower() in ("yes", "true", "t", "1")

def resolve_date_ranges(args):
    """解析日期范围"""
    if all(getattr(args, name, None) for name in ["train_start", "train_end", "test_start", "test_end"]):
        return args.train_start, args.train_end, args.test_start, args.test_end
    return get_date_range(
        args.domain,
        args.train_end_year,
        args.test_end_year
    )

def build_job_context(args) -> Dict[str, Any]:
    """构建作业上下文"""
    logger.info("Building job context...")
    
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
    
    context = {
        'device': device,
        'dataset_meta': dataset_meta,
        'job_spec': job_spec,
        'family_spec': family_spec,
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

def build_train_test_loaders(job_ctx: Dict[str, Any]) -> Tuple[ParquetFeatureLoaderV2, ParquetFeatureLoaderV2]:
    """构建训练和测试数据加载器"""
    logger.info("Building train/test loaders...")
    
    # 确定特征注册管理器版本
    if job_ctx.get('family_spec') and job_ctx['family_spec'].get('feature_scopes'):
        # 使用V2版本支持层选择
        registry_manager = FeatureRegistryManagerV2()
        layers = job_ctx['family_spec'].get('enabled_layers', ['raw', 'filled'])
    else:
        # 使用传统版本
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

def build_target_expression(job_ctx: Dict[str, Any]) -> Expression:
    """构建目标表达式"""
    logger.info("Building target expression...")
    
    # 从作业规格获取标签天数
    if job_ctx.get('job_spec'):
        label_days = job_ctx['job_spec'].get('label_days', 10)
    else:
        label_days = 10  # 默认值
    
    # 创建特征枚举
    if job_ctx.get('family_spec') and job_ctx['family_spec'].get('enabled_layers'):
        registry_manager = FeatureRegistryManagerV2()
        layers = job_ctx['family_spec']['enabled_layers']
    else:
        registry_manager = FeatureRegistryManager()
        layers = ['raw', 'filled']
    
    feature_enum = registry_manager.create_feature_enum(job_ctx['domain'], job_ctx['status_filter'], layers)
    close = Feature(feature_enum.CLOSE)
    target = Ref(close, -label_days) / close - 1
    
    return target

def build_family_search_space(job_ctx: Dict[str, Any]) -> Tuple[List, List, List]:
    """构建因子族搜索空间"""
    logger.info("Building family search space...")
    
    if job_ctx.get('family_spec'):
        # 使用因子族规格
        selected_operators, selected_delta_times, selected_constants = build_family_search_space(
            job_ctx['family_spec']
        )
    else:
        # 使用传统方式
        selected_operators, selected_delta_times, selected_constants = build_search_space(
            operator_names=getattr(job_ctx, "operator_names", None),
            delta_times=getattr(job_ctx, "delta_times", None),
            constants=getattr(job_ctx, "constants", None),
        )
    
    return selected_operators, selected_delta_times, selected_constants

def build_gfn_components(job_ctx: Dict[str, Any], train_data: ParquetFeatureLoaderV2, 
                        target: Expression, selected_operators: List, selected_delta_times: List, 
                        selected_constants: List) -> Dict[str, Any]:
    """构建GFN组件"""
    logger.info("Building GFN components...")
    
    # 初始化AlphaPoolGFN v2
    pool_capacity = job_ctx.get('job_spec', {}).get('pool_capacity', 50) if job_ctx.get('job_spec') else 50
    pool = AlphaPoolGFN(
        capacity=pool_capacity,
        stock_data=train_data,
        target=target,
        ic_mut_threshold=0.3,
        ssl_k=3,
        ssl_tau=0.1,
        cache_manager=None,  # 将在下面初始化
        entry_strategy="ic_ranking",
        diversity_weight=0.3,
        min_ic_threshold=0.05,
        max_similarity_threshold=0.95,
        adaptive_threshold_decay=0.99,
        enable_cache=bool(job_ctx.get('cache_root')),
        cache_key_builder=None
    )
    
    # 初始化缓存管理器
    cache_manager = None
    if job_ctx.get('cache_root'):
        # 构建配置哈希
        config_hash = CacheManager.compute_config_hash(
            operator_names=[op.__name__ for op in selected_operators],
            delta_times=selected_delta_times,
            constants=selected_constants,
            max_expr_length=job_ctx.get('job_spec', {}).get('max_expr_length', 20)
        )
        
        cache_manager = CacheManager(
            base_cache_dir=job_ctx['cache_root'],
            domain=job_ctx['domain'],
            date_range=(job_ctx.get('train_start', '20200101'), job_ctx.get('train_end', '20201231')),
            config_hash=config_hash
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
        'cache_manager': cache_manager
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
                "family_id": job_ctx['family_spec'].get('family_id'),
                "operator_names": job_ctx['family_spec'].get('operator_whitelist', []),
                "delta_times": job_ctx['family_spec'].get('delta_times', []),
                "constants": job_ctx['family_spec'].get('constants', []),
                "enabled_layers": job_ctx['family_spec'].get('enabled_layers', []),
                "max_expr_length": job_ctx['family_spec'].get('max_expr_length'),
            }
        )
    
    return log_dir

def run_training_loop(job_ctx: Dict[str, Any], components: Dict[str, Any], log_dir: str):
    """运行训练循环"""
    logger.info("Starting training loop...")
    
    env = components['env']
    pool = components['pool']
    loss_fn = components['loss_fn']
    sampler = components['sampler']
    optimizer = components['optimizer']
    
    # 创建日志器
    test_data = job_ctx.get('test_loader')
    target = job_ctx.get('target_expression')
    logger_obj = GFNLogger(components['pf'], pool, log_dir, test_data, target)
    
    # 权重调度器
    if job_ctx.get('job_spec'):
        n_episodes = job_ctx['job_spec'].get('n_episodes', 10000)
        ssl_weight = job_ctx['job_spec'].get('ssl_weight', 1.0)
        nov_weight = job_ctx['job_spec'].get('nov_weight', 0.3)
        final_weight_ratio = job_ctx['job_spec'].get('final_weight_ratio', 0.0)
        weight_decay_type = job_ctx['job_spec'].get('weight_decay_type', 'linear')
    else:
        n_episodes = 10000
        ssl_weight = 1.0
        nov_weight = 0.3
        final_weight_ratio = 0.0
        weight_decay_type = 'linear'
    
    weight_scheduler = WeightScheduler(
        initial_ssl_weight=ssl_weight,
        initial_nov_weight=nov_weight,
        final_ratio=final_weight_ratio,
        total_steps=n_episodes,
        scheduler_type=weight_decay_type
    )
    
    # 日志频率
    log_freq = job_ctx.get('job_spec', {}).get('log_freq', 1000) if job_ctx.get('job_spec') else 1000
    candidate_export_freq = job_ctx.get('candidate_export_freq', log_freq)
    
    logger.info(f"Starting GFN training for {job_ctx['domain']}...")
    logger.info(f"Total episodes: {n_episodes}, Log frequency: {log_freq}")
    
    for episode in tqdm(range(n_episodes), desc="GFN Training"):
        current_ssl_weight, current_nov_weight = weight_scheduler.get_current_weights()
        env.ssl_weight = current_ssl_weight
        env.nov_weight = current_nov_weight
        
        # 采样轨迹
        trajectories = sampler.sample_trajectories(env=env, n=1, save_estimator_outputs=loss_fn.entropy_coef > 0)
        loss = loss_fn.loss(env=env, trajectories=trajectories)

        if loss is not None and torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 日志和检查点
        if episode > 0 and (episode + 1) % log_freq == 0:
            logger_obj.log_metrics(episode)
            logger_obj.save_checkpoint(episode)
            logger_obj.show_pool_state()
            
            # 导出候选池
            if (episode + 1) % candidate_export_freq == 0:
                export_candidate_pool(pool, log_dir, episode, job_ctx)
        
        weight_scheduler.step()

    logger_obj.close()
    logger.info("GFN Training Completed.")

def export_candidate_pool(pool: AlphaPoolGFN, log_dir: str, episode: int, job_ctx: Dict[str, Any]):
    """导出候选池"""
    candidate_dir = os.path.join(log_dir, "candidate_pool")
    os.makedirs(candidate_dir, exist_ok=True)
    
    # 保存池快照
    pool_path = os.path.join(candidate_dir, f"pool_snapshot_{episode}.json")
    with open(pool_path, 'w') as f:
        json.dump(pool.to_dict(), f, indent=4)
    
    # 保存最佳表达式
    best_exprs_path = os.path.join(candidate_dir, "best_exprs_latest.json")
    best_exprs = {
        "episode": episode,
        "timestamp": datetime.now().isoformat(),
        "exprs": pool.exprs[:pool.size],
        "ics": pool.single_ics[:pool.size].tolist() if hasattr(pool, 'single_ics') else []
    }
    with open(best_exprs_path, 'w') as f:
        json.dump(best_exprs, f, indent=4)
    
    logger.info(f"Exported candidate pool at episode {episode}")

def train_with_job_spec(args):
    """使用作业规格进行训练"""
    logger.info("Starting training with job specification...")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 构建作业上下文
    job_ctx = build_job_context(args)
    
    # 构建数据加载器
    train_loader, test_loader = build_train_test_loaders(job_ctx)
    job_ctx['train_loader'] = train_loader
    job_ctx['test_loader'] = test_loader
    
    # 构建目标表达式
    target = build_target_expression(job_ctx)
    job_ctx['target_expression'] = target
    
    # 构建搜索空间
    selected_operators, selected_delta_times, selected_constants = build_family_search_space(job_ctx)
    
    # 构建GFN组件
    components = build_gfn_components(job_ctx, train_loader, target, selected_operators, 
                                    selected_delta_times, selected_constants)
    
    # 设置日志
    log_dir = setup_logging(job_ctx)
    
    # 运行训练循环
    run_training_loop(job_ctx, components, log_dir)

def train_traditional(args):
    """传统训练方式（向后兼容）"""
    logger.info("Starting traditional training...")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    max_backtrack_days = getattr(args, "max_backtrack_days", 100)
    max_future_days = getattr(args, "max_future_days", 30)
    use_filled = getattr(args, "use_filled", True)
    output_dir = getattr(args, "output_dir", None)
    run_name = getattr(args, "run_name", None)
    
    # 初始化数据集元数据
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
    
    # 初始化注册管理器和特征
    registry_path = args.registry_path or os.path.join(ROOT, "data", "basic", "feature_ready", "feature_registry.csv")
    registry_manager = FeatureRegistryManager(registry_path)
    
    # 获取日期范围
    train_start, train_end, test_start, test_end = resolve_date_ranges(args)
    status_filter = parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch']
    
    # 初始化Parquet数据加载器
    pool_path = args.sample_pool_path or os.path.join(ROOT, "data", "basic", "feature_ready", "sample_pool_200.json")
    daily_path = args.daily_path or os.path.join(ROOT, "data", "basic", "daily.parquet")
    
    logger.info(f"Loading data for domain {args.domain} ({train_start} to {train_end})...")
    data = ParquetFeatureLoader(
        domain=args.domain,
        start_time=train_start,
        end_time=train_end,
        registry_manager=registry_manager,
        dataset_meta=dataset_meta,
        pool_path=pool_path,
        daily_path=daily_path,
        device=device,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        status_filter=status_filter,
        use_filled=use_filled,
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
        registry_manager=registry_manager,
        dataset_meta=dataset_meta,
        pool_path=pool_path,
        daily_path=daily_path,
        device=device,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        status_filter=status_filter,
        use_filled=use_filled,
        feature_file_map_raw=getattr(args, "feature_file_map_raw", None),
        feature_file_map_filled=getattr(args, "feature_file_map_filled", None),
        date_column=getattr(args, "date_column", "trade_date"),
        code_column=getattr(args, "code_column", "ts_code"),
        close_column=getattr(args, "close_column", "close"),
        data_dir=getattr(args, "data_dir", None),
    )
    
    # 创建特征枚举
    feature_enum = registry_manager.create_feature_enum(args.domain, status_filter)
    feature_indices = [m for m in feature_enum if m.name != 'CLOSE']
    
    # 构建搜索空间
    selected_operators, selected_delta_times, selected_constants = build_search_space(
        operator_names=getattr(args, "operator_names", None),
        delta_times=getattr(args, "delta_times", None),
        constants=getattr(args, "constants", None),
    )
    
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
        mask_dropout_prob=args.mask_dropout_prob,
        ssl_weight=args.ssl_weight,
        nov_weight=args.nov_weight,
        custom_features=feature_indices,
        operators=selected_operators,
        delta_times=selected_delta_times,
        constants=selected_constants,
        max_expr_length=getattr(args, "max_expr_length", MAX_EXPR_LENGTH)
    )
    
    # 初始化策略网络
    pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0)
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)

    # 初始化损失函数
    loss_fn = EntropyTBGFlowNet(
        pf=pf,
        pb=pb,
        entropy_coef=args.entropy_coef,
        entropy_temperature=args.entropy_temperature
    )
    loss_fn.to(device)
    sampler = Sampler(estimator=pf)
    
    # 优化器
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)

    # 设置日志
    output_root = output_dir or os.path.join(ROOT, "data", "gfn_logs")
    os.makedirs(output_root, exist_ok=True)
    log_dir = build_run_dir(output_root, prefix=f"gfn_domain_{args.domain}", run_name=run_name)
    save_json(os.path.join(log_dir, "resolved_args.json"), vars(args))
    save_json(
        os.path.join(log_dir, "search_space.json"),
        {
            "operator_names": [op.__name__ for op in selected_operators],
            "delta_times": selected_delta_times,
            "constants": selected_constants,
            "status_filter": status_filter,
        },
    )
    logger_obj = GFNLogger(pf, pool, log_dir, data_test, target)

    # 权重调度器
    weight_scheduler = WeightScheduler(
        initial_ssl_weight=args.ssl_weight,
        initial_nov_weight=args.nov_weight,
        final_ratio=args.final_weight_ratio,
        total_steps=args.n_episodes,
        scheduler_type=args.weight_decay_type
    )
    
    # 训练循环
    logger.info(f"Starting GFN training for domain {args.domain}...")
    for episode in tqdm(range(args.n_episodes)):
        current_ssl_weight, current_nov_weight = weight_scheduler.get_current_weights()
        env.ssl_weight = current_ssl_weight
        env.nov_weight = current_nov_weight
        
        trajectories = sampler.sample_trajectories(env=env, n=1, save_estimator_outputs=args.entropy_coef > 0)
        loss = loss_fn.loss(env=env, trajectories=trajectories)

        if loss is not None and torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode > 0 and (episode + 1) % args.log_freq == 0:
            logger_obj.log_metrics(episode)
            logger_obj.save_checkpoint(episode)
            logger_obj.show_pool_state()
        
        weight_scheduler.step()

    logger_obj.close()
    logger.info("GFN Training Completed.")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AlphaPROBE GFN Training - Enhanced Version")
    
    # 作业规格相关参数
    parser.add_argument('--job_spec', type=str, default=None,
                        help="Path to job specification JSON/YAML file")
    parser.add_argument('--family_id', type=str, default=None,
                        help="Factor family ID for configuration")
    
    # 数据集配置
    parser.add_argument('--task_config', type=str, default=None)
    parser.add_argument('--dataset_meta', type=str, default=None,
                        help="Path to dataset meta JSON file for unified data configuration")
    parser.add_argument('--domain', type=str, default='A', choices=['A', 'B', 'C', 'E'])
    
    # 基础训练参数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pool_capacity', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=1000)
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--encoder_type', type=str, default='gnn', choices=['transformer', 'lstm', 'gnn'])
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--entropy_temperature', type=float, default=1.0)
    
    # 环境参数
    parser.add_argument('--mask_dropout_prob', type=float, default=1.0)
    parser.add_argument('--ssl_weight', type=float, default=1.0)
    parser.add_argument('--nov_weight', type=float, default=0.3)
    parser.add_argument('--weight_decay_type', type=str, default='linear')
    parser.add_argument('--final_weight_ratio', type=float, default=0.0)
    parser.add_argument('--label_days', type=int, default=10)
    
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