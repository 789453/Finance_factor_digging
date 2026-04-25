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
from factor_core.expression_quality import ExpressionQualityValidator
from factor_core.semantic_embedding import OllamaExpressionEmbedder
from factor_eval.factor_metrics import FactorMetricsEvaluator
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
    valid_loader: ParquetFeatureLoaderV2
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

def build_data_context(job_spec: MiningJobSpec, dataset_meta: DatasetMeta, device: torch.device, log_dir: Optional[str] = None) -> Tuple[FeatureRegistryManagerV2, Optional[DuckDBDataHub], ParquetFeatureLoaderV2, ParquetFeatureLoaderV2, ParquetFeatureLoaderV2]:
    registry = FeatureRegistryManagerV2()
    registry.register_from_dataset_meta(dataset_meta)

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

    if log_dir:
        diagnostics = {
            "train": train_loader.diagnose(),
            "valid": valid_loader.diagnose(),
            "test": test_loader.diagnose(),
        }
        with open(os.path.join(log_dir, "data_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False, default=str)

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
    
    return target, feature_members, operators, delta_times, constants, pool, quality_validator, feature_enum

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


def run_candidate_eval_loop(job_spec, dataset_meta, family_spec, registry, datahub,
                            train_loader, valid_loader, test_loader, target, pool,
                            quality_validator, operators, delta_times, constants,
                            log_dir, device, feature_enum=None) -> Dict[str, Any]:
    manifest = {
        "job_id": job_spec.job_id,
        "dataset_id": job_spec.dataset_id,
        "family_id": job_spec.family_id,
        "train_range": [job_spec.train_start, job_spec.train_end],
        "valid_range": [job_spec.raw.get("valid_start", job_spec.test_start),
                       job_spec.raw.get("valid_end", job_spec.test_end)],
        "test_range": [job_spec.test_start, job_spec.test_end],
        "generator": job_spec.generator_name,
        "status": "running",
        "start_time": datetime.now().isoformat()
    }
    with open(os.path.join(log_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    candidates = load_candidates_from_job(job_spec, job_spec.generator_name)
    results = []

    for cand in candidates:
        result = evaluate_candidate(cand, pool, train_loader, valid_loader, test_loader,
                                   quality_validator, log_dir, job_spec, feature_enum)
        results.append(result)
        record_candidate(result, log_dir)

    summary = summarize_candidate_results(results)
    with open(os.path.join(log_dir, "candidate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    manifest["status"] = "completed"
    manifest["end_time"] = datetime.now().isoformat()
    manifest["total"] = summary.get("total", 0)
    manifest["accepted"] = summary.get("accepted", 0)
    manifest["rejected"] = summary.get("rejected", 0)
    with open(os.path.join(log_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_candidates_from_job(job_spec, generator_name: str) -> List[Dict[str, Any]]:
    candidates = []
    if generator_name == "manual":
        source = job_spec.raw.get("generator", {}).get("expression_source")
        if source and os.path.exists(source):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for i, expr_str in enumerate(data):
                        candidates.append({
                            "candidate_id": f"manual_{i}",
                            "generator": "manual",
                            "raw_expression": expr_str,
                        })
                elif isinstance(data, dict) and "expressions" in data:
                    for i, expr_str in enumerate(data["expressions"]):
                        candidates.append({
                            "candidate_id": f"manual_{i}",
                            "generator": "manual",
                            "raw_expression": expr_str,
                        })
        else:
            generator_exprs = job_spec.raw.get("generator", {}).get("expressions", [])
            if generator_exprs:
                for i, expr_str in enumerate(generator_exprs):
                    candidates.append({
                        "candidate_id": f"manual_{i}",
                        "generator": "manual",
                        "raw_expression": expr_str,
                    })
            else:
                engine_exprs = job_spec.raw.get("engine", {}).get("params", {}).get("expressions", [])
                for i, expr_str in enumerate(engine_exprs):
                    candidates.append({
                        "candidate_id": f"manual_{i}",
                        "generator": "manual",
                        "raw_expression": expr_str,
                    })
    return candidates


def evaluate_candidate(candidate, pool, train_loader, valid_loader, test_loader,
                      quality_validator, log_dir, job_spec, feature_enum=None) -> Dict[str, Any]:
    from alphagen.data.tree import ExpressionParser
    from dataclasses import asdict

    raw_expr = candidate.get("raw_expression", "")
    result = {
        "candidate_id": candidate.get("candidate_id", "unknown"),
        "generator": candidate.get("generator", "unknown"),
        "raw_expression": raw_expr,
        "canonical_expression": None,
        "accepted": False,
        "stage": "unknown",
        "reason": "",
        "metrics": {},
        "quality": {},
        "exception_type": None,
        "exception_message": None,
    }

    try:
        if feature_enum is not None:
            parser = ExpressionParser(feature_enum)
            expr = parser.parse(raw_expr)
        else:
            from alphagen.data.tree import ExpressionParser
            expr = ExpressionParser().parse(raw_expr)
    except Exception as e:
        result["stage"] = "parse"
        result["reason"] = f"parse_error:{type(e).__name__}"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        return result

    result["stage"] = "quality"
    q_report = quality_validator.validate(expr)
    if not q_report.accept:
        result["reason"] = f"quality:{q_report.reason}"
        result["quality"] = asdict(q_report) if hasattr(q_report, "__dataclass_fields__") else {}
        return result

    result["quality"] = asdict(q_report) if hasattr(q_report, "__dataclass_fields__") else {}
    result["stage"] = "evaluate"

    try:
        value = pool._normalize_by_day(expr.evaluate(train_loader))
    except Exception as e:
        result["reason"] = f"eval_error:{type(e).__name__}"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        return result

    sanity = pool.metrics_evaluator.value_sanity(value)
    if not sanity["ok"]:
        result["stage"] = "value_sanity"
        result["reason"] = f"value_sanity:{sanity['reason']}"
        result["metrics"] = sanity
        return result

    train_metrics = pool.metrics_evaluator.evaluate_tensor(value, train_loader.target)
    direction = 1.0 if train_metrics["ic_mean"] >= 0 else -1.0

    result["metrics"]["train"] = train_metrics

    if train_metrics["ic_adj"] < job_spec.thresholds["min_train_ic"]:
        result["stage"] = "train_metrics"
        result["reason"] = f"low_train_ic:{train_metrics['ic_adj']:.4f} < {job_spec.thresholds['min_train_ic']}"
        return result

    if valid_loader:
        try:
            valid_value = pool._normalize_by_day(expr.evaluate(valid_loader))
            valid_metrics = pool.metrics_evaluator.evaluate_tensor(
                valid_value,
                valid_loader.target if hasattr(valid_loader, 'target') else train_loader.target,
                direction=direction
            )
            result["metrics"]["valid"] = valid_metrics

            if valid_metrics["ic_adj"] < job_spec.thresholds["min_valid_ic"]:
                result["stage"] = "valid_metrics"
                result["reason"] = f"low_valid_ic:{valid_metrics['ic_adj']:.4f} < {job_spec.thresholds['min_valid_ic']}"
                return result
        except Exception as e:
            result["reason"] = f"valid_eval_error:{type(e).__name__}"
            result["exception_type"] = type(e).__name__
            result["exception_message"] = str(e)
            return result

    try:
        canonical = pool.canonicalizer.canonicalize(expr)
        result["canonical_expression"] = str(canonical)
        pool.try_new_expr(expr)
        result["accepted"] = True
        result["stage"] = "accepted"
        result["reason"] = "accepted"
    except Exception as e:
        result["stage"] = "pool_add"
        result["reason"] = f"pool_error:{type(e).__name__}"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)

    return result


def record_candidate(result: Dict[str, Any], log_dir: str):
    import datetime
    event_file = os.path.join(log_dir, "candidate_events.jsonl")
    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        **result
    }
    with open(event_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def summarize_candidate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total": len(results),
        "accepted": sum(1 for r in results if r.get("accepted", False)),
        "rejected": sum(1 for r in results if not r.get("accepted", False)),
        "by_stage": {},
        "best_train_ic": 0.0,
        "best_valid_ic": 0.0,
    }

    for r in results:
        stage = r.get("stage", "unknown")
        summary["by_stage"][stage] = summary["by_stage"].get(stage, 0) + 1

        metrics = r.get("metrics", {})
        if "train" in metrics:
            ic = metrics["train"].get("ic_mean", 0)
            if ic > summary["best_train_ic"]:
                summary["best_train_ic"] = ic
        if "valid" in metrics:
            ic = metrics["valid"].get("ic_mean", 0)
            if ic > summary["best_valid_ic"]:
                summary["best_valid_ic"] = ic

    return summary


def mine_factors(job_spec_path: str, dataset_meta_path: Optional[str] = None,
                log_dir: Optional[str] = None, cuda_override: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    Unified entry point for factor mining.
    Now wraps the new modular orchestrator.
    """
    job_spec, dataset_meta, family_spec = load_runtime_specs(job_spec_path, dataset_meta_path)

    if log_dir is None:
        log_dir = os.path.join(
            job_spec.output_dir,
            f"{job_spec.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(log_dir, exist_ok=True)

    cuda_id = cuda_override if cuda_override is not None else job_spec.cuda
    if torch.cuda.is_available() and cuda_id >= 0:
        device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")

    registry, datahub, train_loader, valid_loader, test_loader = build_data_context(
        job_spec, dataset_meta, device, log_dir=log_dir
    )

    target, feature_members, operators, delta_times, constants, pool, quality_validator, feature_enum = build_alpha_context(
        job_spec, dataset_meta, family_spec, registry, train_loader, valid_loader, test_loader
    )

    pool._init_rejection_log(log_dir, context={
        "job_id": job_spec.job_id,
        "dataset_id": job_spec.dataset_id,
        "family_id": job_spec.family_id,
        "generator": job_spec.generator_name,
    })

    train_target = target.evaluate(train_loader)
    train_loader.target = train_target
    valid_target = target.evaluate(valid_loader)
    valid_loader.target = valid_target
    if test_loader:
        test_target = target.evaluate(test_loader)
        test_loader.target = test_target

    if job_spec.generator_name == "gfn":
        ctx = MiningRuntimeContext(
            job_spec=job_spec,
            dataset_meta=dataset_meta,
            family_spec=family_spec,
            registry=registry,
            datahub=datahub,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            target=target,
            operators=operators,
            delta_times=delta_times,
            constants=constants,
            pool=pool,
            env=None,
            gfn=None,
            sampler=None,
            optimizer=None,
            log_dir=log_dir,
            device=device,
        )
        env, gfn, sampler, optimizer = build_gfn_context(
            job_spec, pool, registry, dataset_meta, feature_members,
            operators, delta_times, constants, device, quality_validator
        )
        ctx.env = env
        ctx.gfn = gfn
        ctx.sampler = sampler
        ctx.optimizer = optimizer
        return run_training_loop(ctx)
    else:
        return run_candidate_eval_loop(job_spec, dataset_meta, family_spec, registry, datahub,
                                       train_loader, valid_loader, test_loader, target, pool,
                                       quality_validator, operators, delta_times, constants, log_dir, device,
                                       feature_enum)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_spec', type=str, required=True)
    parser.add_argument('--dataset_meta', type=str, required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    mine_factors(args.job_spec, args.dataset_meta)
