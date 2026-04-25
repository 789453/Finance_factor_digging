#!/usr/bin/env python3
"""
AlphaPROBE Factor Mining Pipeline
Unified entry point for factor mining based on MiningJobSpec and DatasetMeta.
Follows the AlphaSage / AlphaGFN research paradigm.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

# Core modules
from mining.job_spec import MiningJobSpec, load_job_spec
from mining.family_search_space import load_family_spec, build_family_search_space
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
from alphagen.data.expression import Expression, Feature, Ref, Abs, Log, Sign
from alpha_gfn.alpha_pool_v2 import AlphaPoolGFN
from alpha_gfn.env.core import GFNEnvCore
from alpha_gfn.expression_quality import ExpressionQualityValidator
from alpha_gfn.semantic_embedding import OllamaExpressionEmbedder
from evaluation.factor_metrics import FactorMetricsEvaluator
from alpha_gfn.modules import SequenceEncoder, SimpleNeuralNet
from alpha_gfn.gflownet import EntropyTBGFlowNet
from alpha_gfn.config import HIDDEN_DIM
from datahub.duckdb_hub import DuckDBDataHub
from datahub.config import DataHubConfig
from gfn.samplers import Sampler
from gfn.modules import DiscretePolicyEstimator

logger = logging.getLogger(__name__)

class TorchGFNCompatSampler:
    """Compatibility layer for torchgfn 2.4.0+ sampler API"""
    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def sample_trajectories(self, env: GFNEnvCore, n: int, save_estimator_outputs: bool = False):
        # torchgfn 2.4.0+ uses 'n' instead of 'n_trajectories'
        if hasattr(self.sampler, "sample_trajectories"):
            try:
                return self.sampler.sample_trajectories(
                    env=env,
                    n=n,
                    save_estimator_outputs=save_estimator_outputs
                )
            except TypeError:
                # Fallback for older 2.x versions if 'n' vs 'n_trajectories' differs
                return self.sampler.sample_trajectories(
                    env=env,
                    n_trajectories=n,
                    save_estimator_outputs=save_estimator_outputs
                )
        raise RuntimeError("Unsupported torchgfn sampler API")

@dataclass
class MiningRuntimeContext:
    """Runtime context for factor mining task"""
    job_spec: MiningJobSpec
    dataset_meta: DatasetMeta
    family_spec: Dict[str, Any]
    registry: FeatureRegistryManagerV2
    datahub: Optional[DuckDBDataHub]
    train_loader: ParquetFeatureLoaderV2
    test_loader: ParquetFeatureLoaderV2
    target: Expression
    operators: List[Any]
    delta_times: List[int]
    constants: List[float]
    pool: AlphaPoolGFN
    env: GFNEnvCore
    gfn: EntropyTBGFlowNet
    sampler: TorchGFNCompatSampler
    optimizer: torch.optim.Optimizer
    log_dir: str
    device: torch.device

class WeightScheduler:
    """Weight decay scheduler for SSL and Novelty weights"""
    def __init__(self, initial_ssl, initial_nov, final_ratio, total_steps, scheduler_type='linear'):
        self.initial_ssl = initial_ssl
        self.initial_nov = initial_nov
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
        return self.initial_ssl * ratio, self.initial_nov * ratio

def load_runtime_specs(job_spec_path: str, dataset_meta_path: Optional[str]) -> Tuple[MiningJobSpec, DatasetMeta, Dict[str, Any]]:
    job_spec = load_job_spec(job_spec_path)
    if not dataset_meta_path:
        raise ValueError("dataset_meta_path is required.")
    dataset_meta = DatasetMeta(dataset_meta_path)
    family_spec = load_family_spec(job_spec.family_id)
    return job_spec, dataset_meta, family_spec

def build_data_context(job_spec: MiningJobSpec, dataset_meta: DatasetMeta, device: torch.device) -> Tuple[FeatureRegistryManagerV2, Optional[DuckDBDataHub], ParquetFeatureLoaderV2, ParquetFeatureLoaderV2, ParquetFeatureLoaderV2]:
    registry = FeatureRegistryManagerV2()
    registry.register_from_dataset_meta(dataset_meta)
    
    # Init DataHub
    datahub = None
    hub_config_path = job_spec.raw.get("datahub_config")
    try:
        if hub_config_path and os.path.exists(hub_config_path):
            config = DataHubConfig.from_yaml(hub_config_path)
        else:
            data_root = os.getenv("DATA_ROOT", "D:/Trading/data_ever_26_3_14/data")
            config = DataHubConfig(
                warehouse_path=os.path.join(data_root, "meta/warehouse.duckdb"),
                control_db_path=os.path.join(data_root, "meta/control.sqlite3")
            )
        datahub = DuckDBDataHub(config)
        logger.info(f"DataHub initialized from {config.warehouse_path}")
    except Exception as e:
        logger.warning(f"DataHub init failed: {e}. Falling back to Parquet.")


    # Build Loaders
    common_kwargs = dataset_meta.to_loader_kwargs()
    common_kwargs.update({
        "registry_manager": registry,
        "device": device,
        "max_backtrack_days": job_spec.max_backtrack_days,
        "max_future_days": job_spec.max_future_days,
        "status_filter": job_spec.status_filter,
        "layers": job_spec.raw.get("layers", dataset_meta.layers_enabled),
        "read_mode": job_spec.raw.get("read_mode", "stack"),
        "segment_name": job_spec.segment_name,
        "datahub": datahub
    })
    
    train_loader = ParquetFeatureLoaderV2(start_time=job_spec.train_start, end_time=job_spec.train_end, **common_kwargs)
    
    valid_start = job_spec.raw.get("valid_start", job_spec.test_start)
    valid_end = job_spec.raw.get("valid_end", job_spec.test_end)
    valid_loader = ParquetFeatureLoaderV2(start_time=valid_start, end_time=valid_end, **common_kwargs)
    
    test_loader = ParquetFeatureLoaderV2(start_time=job_spec.test_start, end_time=job_spec.test_end, **common_kwargs)
    
    return registry, datahub, train_loader, valid_loader, test_loader

def build_alpha_context(job_spec, dataset_meta, family_spec, registry, train_loader, valid_loader=None, test_loader=None):
    # Search Space
    features, operators, delta_times, constants = build_family_search_space(family_spec, dataset_meta)
    
    # Map feature names to Enum members
    feature_enum = registry.create_feature_enum(dataset_meta.domain, job_spec.status_filter, job_spec.raw.get("layers", ["raw", "atomic"]))
    feature_members = []
    for f_name in features:
        try:
            member = feature_enum[f_name.upper()]
            feature_members.append(member)
        except KeyError:
            logger.warning(f"Feature {f_name} not found in registry enum, skipping.")
    
    # Target
    target_cfg = dataset_meta.target_config
    price_col = target_cfg.get("price_column", "close").upper()
    base = Feature(feature_enum[price_col])
    horizon = int(target_cfg.get("label_days", job_spec.label_days))
    target = Ref(base, -horizon) / base - 1
    
    # Quality Validator
    quality_validator = ExpressionQualityValidator(
        min_complexity=job_spec.raw.get("min_complexity", 6),
        min_ts_operators=job_spec.raw.get("min_ts_operators", 1),
        min_operators=job_spec.raw.get("min_operators", 2),
        min_features=job_spec.raw.get("min_features", 1),
    )

    # Semantic Embedder
    semantic_embedder = None
    if job_spec.raw.get("enable_ollama_embedding", False):
        semantic_embedder = OllamaExpressionEmbedder(
            model=job_spec.raw.get("ollama_model", "qwen3-embedding:0.6b"),
            batch_size=job_spec.raw.get("embedding_batch_size", 32)
        )

    # Metrics Evaluator
    metrics_evaluator = FactorMetricsEvaluator(device=str(train_loader.device))

    # Pool
    pool = AlphaPoolGFN(
        capacity=job_spec.pool_capacity,
        stock_data=train_loader,
        target=target,
        valid_data=valid_loader,
        test_data=test_loader,
        metrics_evaluator=metrics_evaluator,
        quality_validator=quality_validator,
        semantic_embedder=semantic_embedder,
        ic_mut_threshold=job_spec.raw.get("ic_mut_threshold", 0.60),
        entry_strategy=job_spec.raw.get("entry_strategy", "composite"),
        min_train_ic=job_spec.raw.get("ic_threshold", 0.015),
        min_valid_ic=job_spec.raw.get("min_valid_ic", 0.005),
        semantic_sim_threshold=job_spec.raw.get("semantic_sim_threshold", 0.92),
    )
    
    return target, feature_members, operators, delta_times, constants, pool, quality_validator

def build_gfn_context(job_spec, pool, registry, dataset_meta, features, operators, delta_times, constants, device, quality_validator=None):
    # Env
    env = GFNEnvCore(
        pool=pool, device=device, custom_features=features,
        operators=operators, delta_times=delta_times, constants=constants,
        max_expr_length=job_spec.max_expr_length,
        mask_dropout_prob=job_spec.raw.get("mask_dropout_prob", 0.0),
        ssl_weight=job_spec.raw.get("ssl_weight", 1.0),
        nov_weight=job_spec.raw.get("nov_weight", 0.3),
        quality_validator=quality_validator,
        min_expr_length=job_spec.raw.get("min_expr_length", 6)
    )
    
    # GFN Chain
    n_tokens = env.n_tokens
    encoder = SequenceEncoder(n_tokens, job_spec.raw.get("encoder_type", "gnn"), tokens=env._tokens)
    pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions)
    pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=env.n_actions - 1)
    
    pf = DiscretePolicyEstimator(nn.Sequential(encoder, pf_head), n_actions=env.n_actions, preprocessor=env.preprocessor)
    pb = DiscretePolicyEstimator(nn.Sequential(encoder, pb_head), n_actions=env.n_actions, preprocessor=env.preprocessor, is_backward=True)
    
    gfn = EntropyTBGFlowNet(pf=pf, pb=pb, entropy_coef=job_spec.raw.get("entropy_coef", 0.01))
    gfn.to(device)
    
    sampler = TorchGFNCompatSampler(Sampler(estimator=pf))
    optimizer = Adam(list(encoder.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [gfn.logZ], 
                     lr=job_spec.raw.get("learning_rate", 1e-4))
    
    return env, gfn, sampler, optimizer

def run_training_loop(ctx: MiningRuntimeContext) -> Dict[str, Any]:
    job_spec = ctx.job_spec
    n_episodes = job_spec.n_episodes
    logger.info(f"Starting GFN training loop for {n_episodes} episodes...")
    
    # Create run manifest
    manifest = {
        "job_id": ctx.job_spec.job_id,
        "dataset_id": ctx.job_spec.dataset_id,
        "family_id": ctx.job_spec.family_id,
        "train_range": [ctx.job_spec.train_start, ctx.job_spec.train_end],
        "test_range": [ctx.job_spec.test_start, ctx.job_spec.test_end],
        "seed": ctx.job_spec.seed,
        "cuda": str(ctx.device),
        "status": "running",
        "start_time": datetime.now().isoformat()
    }
    with open(os.path.join(ctx.log_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    weight_scheduler = WeightScheduler(
        initial_ssl=ctx.job_spec.ssl_weight,
        initial_nov=ctx.job_spec.nov_weight,
        final_ratio=ctx.job_spec.raw.get("final_weight_ratio", 0.0),
        total_steps=ctx.job_spec.n_episodes
    )
    
    pbar = tqdm(range(n_episodes))
    log_freq = ctx.job_spec.log_freq
    checkpoint_freq = ctx.job_spec.raw.get("checkpoint_freq", 1000)

    for episode in pbar:
        ssl, nov = weight_scheduler.get_current_weights()
        ctx.env.ssl_weight, ctx.env.nov_weight = ssl, nov
        
        batch_size = ctx.job_spec.raw.get("batch_size", 32)
        entropy_coef = ctx.job_spec.entropy_coef
        
        trajectories = ctx.sampler.sample_trajectories(
            env=ctx.env, n=batch_size,
            save_estimator_outputs=entropy_coef > 0
        )
        
        loss = ctx.gfn.loss(env=ctx.env, trajectories=trajectories)
        if loss is not None and torch.isfinite(loss):
            ctx.optimizer.zero_grad()
            loss.backward()
            ctx.optimizer.step()
            
            if episode % log_freq == 0:
                stats = ctx.pool.get_stats()
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    pool=f"{stats['pool_size']}",
                    best_ic=f"{stats['best_ic']:.4f}",
                    mean_score=f"{stats.get('mean_score', 0):.4f}"
                )

        # Periodic checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(ctx.log_dir, f"checkpoint_{episode+1}.pt")
            torch.save({
                'episode': episode + 1,
                'gfn_state_dict': ctx.gfn.state_dict(),
                'optimizer_state_dict': ctx.optimizer.state_dict(),
                'pool': ctx.pool.to_dict()
            }, checkpoint_path)
            
            # Export candidates
            ctx.pool.export_pool(os.path.join(ctx.log_dir, f"candidates_{episode+1}.json"))

        weight_scheduler.step()
        
    # Final save
    ctx.pool.export_pool(os.path.join(ctx.log_dir, "final_pool.json"))
    
    # Save best expressions
    stats = ctx.pool.get_stats()
    if ctx.pool.size > 0:
        with open(os.path.join(ctx.log_dir, "best_expressions.txt"), "w") as f:
            for i in range(ctx.pool.size):
                expr = ctx.pool.exprs[i]
                ic = ctx.pool.single_ics[i]
                f.write(f"{ic:.4f} | {expr}\n")

    # Update manifest
    manifest["status"] = "completed"
    manifest["end_time"] = datetime.now().isoformat()
    manifest["best_ic"] = stats["best_ic"]
    manifest["pool_size"] = ctx.pool.size
    manifest["log_dir"] = ctx.log_dir
    with open(os.path.join(ctx.log_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def mine_factors(job_spec_path: str, dataset_meta_path: Optional[str] = None, 
                log_dir: Optional[str] = None, cuda_override: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Unified entry point for factor mining"""
    job_spec, dataset_meta, family_spec = load_runtime_specs(job_spec_path, dataset_meta_path)
    
    if cuda_override is not None and cuda_override < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{cuda_override}' if cuda_override is not None and torch.cuda.is_available() else 
                             torch.device(f'cuda:{job_spec.raw.get("cuda", 0)}' if torch.cuda.is_available() and job_spec.raw.get("cuda", 0) >= 0 else 'cpu'))
    
    registry, datahub, train_loader, valid_loader, test_loader = build_data_context(job_spec, dataset_meta, device)
    
    target, features, operators, delta_times, constants, pool, quality_validator = build_alpha_context(
        job_spec, dataset_meta, family_spec, registry, train_loader, valid_loader, test_loader
    )
    
    # GFN Environment
    env, gfn, sampler, optimizer = build_gfn_context(
        job_spec, pool, registry, dataset_meta, features, operators, delta_times, constants, device,
        quality_validator=quality_validator
    )
    
    if log_dir is None:
        log_dir = os.path.join(job_spec.raw.get("output_dir", "runs"), f"{job_spec.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    ctx = MiningRuntimeContext(
        job_spec=job_spec, dataset_meta=dataset_meta, family_spec=family_spec,
        registry=registry, datahub=datahub, train_loader=train_loader, test_loader=test_loader,
        target=target, operators=operators, delta_times=delta_times, constants=constants,
        pool=pool, env=env, gfn=gfn, sampler=sampler, optimizer=optimizer,
        log_dir=log_dir, device=device
    )
    
    # 初始化拒绝日志
    ctx.pool._init_rejection_log(ctx.log_dir)
    
    return run_training_loop(ctx)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_spec', type=str, required=True)
    parser.add_argument('--dataset_meta', type=str, required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    mine_factors(args.job_spec, args.dataset_meta)
