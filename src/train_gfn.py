import torch
"""
Legacy GFN training entry.

Do not add new dataset / family / multi-asset logic here.
Use mining.mine_factors with job_spec + dataset_meta for new workflows.
"""
import random
import numpy as np
import argparse
import os
import json
import sys
import time
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, PolynomialLR
from torch.distributions import Categorical
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dotenv import load_dotenv

# Path configuration to handle 'src' module imports
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# AlphaGen imports
from alphagen.rl.env.wrapper import action2token
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_pearsonr
from alphagen_generic.features import *
from alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.config import get_date_range
from alphagen_generic.task_config import apply_task_config, parse_str_list
from alphagen_generic.run_artifacts import build_run_dir, save_json

# GFN specific imports
# Use absolute imports from root if possible, or relative if within package
try:
    from alpha_gfn.config import *
    from alpha_gfn.env.core import GFNEnvCore
    from alpha_gfn.modules import SequenceEncoder
    from alpha_gfn.alpha_pool import AlphaPoolGFN
    from alpha_gfn.gflownet import EntropyTBGFlowNet
    from alpha_gfn.search_space import build_search_space
    from alpha_gfn.cache_manager import CacheManager
except ImportError:
    from src.alpha_gfn.config import *
    from src.alpha_gfn.env.core import GFNEnvCore
    from src.alpha_gfn.modules import SequenceEncoder
    from src.alpha_gfn.alpha_pool import AlphaPoolGFN
    from src.alpha_gfn.gflownet import EntropyTBGFlowNet
    from src.alpha_gfn.search_space import build_search_space
    from src.alpha_gfn.cache_manager import CacheManager

# External gfn library imports (torchgfn 2.4.0 compatibility)
try:
    from gfn.samplers import Sampler
    from gfn.modules import DiscretePolicyEstimator
    from gfn.gflownet import TBGFlowNet
except Exception:
    import traceback
    print("GFN library import failed. Please ensure 'torchgfn' is installed correctly.")
    traceback.print_exc()
    sys.exit(1)

load_dotenv(ROOT / ".env")


class SimpleNeuralNet(nn.Module):
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
            # test_ensemble would need to be implemented or aligned with project metrics
            # For now, we'll just log the best IC
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
    if isinstance(v, bool): return v
    return v.lower() in ("yes", "true", "t", "1")


def resolve_date_ranges(args):
    if all(getattr(args, name, None) for name in ["train_start", "train_end", "test_start", "test_end"]):
        return args.train_start, args.train_end, args.test_start, args.test_end
    return get_date_range(
        args.domain,
        args.train_end_year,
        args.test_end_year
    )

def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    max_backtrack_days = getattr(args, "max_backtrack_days", 100)
    max_future_days = getattr(args, "max_future_days", 30)
    use_filled = getattr(args, "use_filled", True)
    output_dir = getattr(args, "output_dir", None)
    run_name = getattr(args, "run_name", None)
    
    # Initialize DatasetMeta if provided
    dataset_meta = None
    if hasattr(args, "dataset_meta") and args.dataset_meta:
        try:
            dataset_meta = DatasetMeta(args.dataset_meta)
            print(f"Loaded dataset meta from {args.dataset_meta}")
            # Validate meta
            errors = dataset_meta.validate()
            if errors:
                print(f"Warning: Dataset meta validation errors: {errors}")
        except Exception as e:
            print(f"Warning: Failed to load dataset meta: {e}, using default configuration")
    
    # Initialize Registry & Features
    registry_path = args.registry_path or os.path.join(ROOT, "data", "basic", "feature_ready", "feature_registry.csv")
    registry_manager = FeatureRegistryManager(registry_path)
    
    # Get Date Range
    train_start, train_end, test_start, test_end = resolve_date_ranges(args)
    status_filter = parse_str_list(getattr(args, "status_filter", None)) or ['active', 'watch']
    
    # Initialize Parquet Data Loader
    pool_path = args.sample_pool_path or os.path.join(ROOT, "data", "basic", "feature_ready", "sample_pool_200.json")
    daily_path = args.daily_path or os.path.join(ROOT, "data", "basic", "daily.parquet")
    
    print(f"Loading data for domain {args.domain} ({train_start} to {train_end})...")
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
    
    print(f"Loading test data for domain {args.domain} ({test_start} to {test_end})...")
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
    
    feature_enum = registry_manager.create_feature_enum(args.domain, status_filter)
    feature_indices = [m for m in feature_enum if m.name != 'CLOSE']
    selected_operators, selected_delta_times, selected_constants = build_search_space(
        operator_names=getattr(args, "operator_names", None),
        delta_times=getattr(args, "delta_times", None),
        constants=getattr(args, "constants", None),
    )
    
    close = Feature(feature_enum.CLOSE)
    target = Ref(close, -args.label_days) / close - 1
    
    # Initialize AlphaPoolGFN
    pool = AlphaPoolGFN(capacity=args.pool_capacity, stock_data=data, target=target)

    # Initialize model
    # Note: FEATURES, OPERATORS, DELTA_TIMES, CONSTANTS come from config.py
    n_tokens = 1 + len(selected_operators) + len(feature_indices) + len(selected_delta_times) + len(selected_constants) + 1
    
    backbone = SequenceEncoder(n_tokens, args.encoder_type)
    
    # Initialize environment
    env = GFNEnvCore(pool=pool,
                     encoder=backbone, 
                     device=device, 
                     mask_dropout_prob=args.mask_dropout_prob,
                     ssl_weight=args.ssl_weight,
                     nov_weight=args.nov_weight,
                     custom_features=feature_indices,
                     operators=selected_operators,
                     delta_times=selected_delta_times,
                     constants=selected_constants,
                     max_expr_length=getattr(args, "max_expr_length", MAX_EXPR_LENGTH))
    
    pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions, n_hidden_layers=0)
    pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1, n_hidden_layers=0)
    
    pf_module = nn.Sequential(backbone, pf_head)
    pb_module = nn.Sequential(backbone, pb_head)
    
    pf = DiscretePolicyEstimator(pf_module, n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(pb_module, n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)

    loss_fn = EntropyTBGFlowNet(
        pf=pf,
        pb=pb,
        entropy_coef=args.entropy_coef,
        entropy_temperature=args.entropy_temperature
    )
    loss_fn.to(device)
    sampler = Sampler(estimator=pf)
    
    params = list(backbone.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [loss_fn.logZ]
    optimizer = Adam(params, lr=LEARNING_RATE)

    # Setup logging
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
    logger = GFNLogger(pf, pool, log_dir, data_test, target)

    # Training loop
    n_episodes = args.n_episodes
    weight_scheduler = WeightScheduler(
        initial_ssl_weight=args.ssl_weight,
        initial_nov_weight=args.nov_weight,
        final_ratio=args.final_weight_ratio,
        total_steps=n_episodes,
        scheduler_type=args.weight_decay_type
    )
    
    print(f"Starting GFN training for domain {args.domain}...")
    for episode in tqdm(range(n_episodes)):
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
            logger.log_metrics(episode)
            logger.save_checkpoint(episode)
            logger.show_pool_state()
        
        weight_scheduler.step()

    logger.close()
    print("GFN Training Completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str, default=None)
    parser.add_argument('--dataset_meta', type=str, default=None,
                        help="Path to dataset meta JSON file for unified data configuration")
    parser.add_argument('--domain', type=str, default='A', choices=['A', 'B', 'C', 'E'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pool_capacity', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=1000)
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--encoder_type', type=str, default='gnn', choices=['transformer', 'lstm', 'gnn'])
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--entropy_temperature', type=float, default=1.0)
    parser.add_argument('--mask_dropout_prob', type=float, default=1.0)
    parser.add_argument('--ssl_weight', type=float, default=1.0)
    parser.add_argument('--nov_weight', type=float, default=0.3)
    parser.add_argument('--weight_decay_type', type=str, default='linear')
    parser.add_argument('--final_weight_ratio', type=float, default=0.0)
    parser.add_argument('--label_days', type=int, default=10)
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
    
    args = parser.parse_args()
    task_config_raw = apply_task_config(args, parser)
    if task_config_raw:
        setattr(args, "_task_config_raw", task_config_raw)
    train(args)
