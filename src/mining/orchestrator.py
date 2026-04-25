import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, List, Iterable
from datetime import datetime
from dataclasses import asdict
from tqdm import tqdm

from factor_io.specs import MiningExperimentSpec, ScreeningSpec
from mining.job_spec import load_experiment_spec
from mining.context import FactorMiningContext
from factor_engines.registry import EngineRegistry
from factor_core.inspector import FactorInspector
from factor_core.pool import FactorPool
from factor_io.output_writer import OutputWriter
from factor_eval.factor_metrics import FactorMetricsEvaluator
from factor_core.expression_quality import ExpressionQualityValidator
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
from datahub.duckdb_hub import DuckDBDataHub
from datahub.config import DataHubConfig
from mining.family_search_space import build_family_search_space, load_family_spec
from alphagen.data.expression import Feature, Ref

logger = logging.getLogger(__name__)

def run_experiment(
    job_spec_path: str,
    dataset_meta_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    cuda_override: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point to run a mining experiment.
    Standard pipeline: Engine -> Candidate -> Inspector -> Pool -> Output
    """
    # 1. Load Spec
    spec = load_experiment_spec(job_spec_path)
    
    # 2. Setup Device
    if cuda_override is not None and cuda_override >= 0:
        device = torch.device(f'cuda:{cuda_override}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        # Force GPU means if not available, we might want to fail or warn
        logger.warning("GPU requested or preferred but not available. Falling back to CPU.")
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")

    # 3. Setup Log Dir
    if log_dir is None:
        log_dir = os.path.join(spec.output_dir, f"{spec.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # 4. Build Context (Data, Loaders, etc.)
    context = build_mining_context(spec, dataset_meta_path, device)
    
    # 5. Initialize Components
    inspector = FactorInspector(
        quality_validator=ExpressionQualityValidator(
            min_complexity=spec.screening.min_complexity if spec.screening else 6,
            min_ts_operators=spec.screening.min_ts_operators if spec.screening else 1,
            min_operators=spec.screening.min_operators if spec.screening else 3,
            min_features=spec.screening.min_features if spec.screening else 1
        ),
        metrics_evaluator=FactorMetricsEvaluator(device=str(device)),
        feature_map={str(f): f for f in context.features},
        screening_spec=spec.screening
    )
    
    pool = FactorPool(
        capacity=spec.engine.params.get("pool_capacity", 50) if spec.engine else 50,
        screening_spec=spec.screening or ScreeningSpec()
    )
    
    writer = OutputWriter(log_dir, spec)
    writer.write_manifest(status="running")
    
    # 6. Initialize Engine
    engine = EngineRegistry.create(spec.engine.type)
    engine.prepare(spec, context)
    
    # 7. Discovery Loop
    logger.info(f"Starting discovery with engine: {engine.engine_name}")
    try:
        candidates = engine.discover(context)
        
        for candidate in candidates:
            # Step A: Write raw candidate
            writer.write_raw_candidate(candidate)
            
            # Step B: Inspect (Syntax, Scope, Metrics)
            report = inspector.inspect(candidate, context)
            
            # Step C: Try adding to Pool (Novelty Check & Scoring)
            decision = pool.add(report)
            
            # Step D: Write checked candidate
            writer.write_checked_candidate(report, decision)
            
            # Step E: If accepted, save values and daily metrics
            if decision.accepted:
                logger.info(f"Accepted factor: {candidate.expression} (Score: {decision.score:.4f})")
                # Save factor values for further analysis if needed
                pass

        # 8. Finalize
        final_pool = pool.export()
        writer.write_pool(final_pool)
        writer.write_report(final_pool)
        writer.write_manifest(status="completed")
        
        return {
            "status": "completed",
            "run_dir": log_dir,
            "n_seen": pool.n_seen,
            "n_selected": pool.size
        }
            
    finally:
        engine.close()

def build_mining_context(spec: MiningExperimentSpec, dataset_meta_path: Optional[str], device: torch.device) -> FactorMiningContext:
    """Build the unified FactorMiningContext."""
    if not dataset_meta_path:
        # Try to infer from spec
        if spec.dataset and spec.dataset.dataset_id:
             dataset_meta_path = f"config/datasets/{spec.dataset.dataset_id}.yaml"
             if not os.path.exists(dataset_meta_path):
                 raise ValueError(f"dataset_meta_path not provided and couldn't infer from {dataset_meta_path}")
        else:
            raise ValueError("dataset_meta_path is required to build context.")
        
    dataset_meta = DatasetMeta(dataset_meta_path)
    
    # 1. Registry & DataHub
    registry = FeatureRegistryManagerV2()
    registry.register_from_dataset_meta(dataset_meta)
    
    # Handle Atomic Layer if enabled
    if spec.dataset and "atomic" in spec.dataset.layers_enabled:
        logger.info("Atomic layer enabled. Loading atomic fields...")
        from atomic.registry import AtomicRegistry
        # AtomicRegistry initializes with CORE_ATOMIC_FIELDS by default
        atomic_registry = AtomicRegistry()
        # Register in feature registry
        registry.register_from_atomic_registry(atomic_registry)
    
    datahub = None
    if spec.dataset and spec.dataset.data_backend:
        backend_type = spec.dataset.data_backend.get("type")
        if backend_type == "duckdb":
            profile = spec.dataset.data_backend.get("profile")
            if profile and os.path.exists(profile):
                config = DataHubConfig.from_yaml(profile)
                datahub = DuckDBDataHub(config)
                logger.info(f"Initialized DataHub from {profile}")

    # 2. Loaders
    common_kwargs = dataset_meta.to_loader_kwargs()
    common_kwargs.update({
        "registry_manager": registry,
        "device": device,
        "max_backtrack_days": spec.constraints.max_ts_window if spec.constraints else 60,
        "layers": spec.dataset.layers_enabled if spec.dataset else ["raw"],
        "datahub": datahub
    })
    
    train_loader = ParquetFeatureLoaderV2(
        start_time=spec.split.train[0], end_time=spec.split.train[1], **common_kwargs
    )
    
    valid_loader = None
    if spec.split and spec.split.valid:
        valid_loader = ParquetFeatureLoaderV2(
            start_time=spec.split.valid[0], end_time=spec.split.valid[1], **common_kwargs
        )
        
    test_loader = None
    if spec.split and spec.split.test:
        test_loader = ParquetFeatureLoaderV2(
            start_time=spec.split.test[0], end_time=spec.split.test[1], **common_kwargs
        )

    # 3. Search Space
    family_spec = load_family_spec(spec.search_space.family_id) if spec.search_space else {}
    features, operators, delta_times, constants = build_family_search_space(family_spec, dataset_meta)
    
    # Map features to registry members
    feature_enum = registry.get_feature_enum(dataset_meta.domain)
    feature_members = []
    for f_name in features:
        try:
            # Check if it's an atomic field or raw field
            feature_members.append(feature_enum[f_name.upper()])
        except KeyError:
            logger.warning(f"Feature {f_name} not found in registry, skipping.")

    # 4. Target
    price_col = spec.target.price_column.upper() if spec.target else "CLOSE"
    try:
        base = Feature(feature_enum[price_col])
    except KeyError:
        logger.warning(f"Price column {price_col} not found in registry, using CLOSE.")
        base = Feature(feature_enum["CLOSE"])
        
    horizon = spec.target.horizon if spec.target else 5
    target = Ref(base, -horizon) / base - 1

    return FactorMiningContext(
        spec=spec,
        dataset_meta=dataset_meta,
        registry=registry,
        datahub=datahub,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        features=feature_members,
        operators=operators,
        delta_times=delta_times,
        constants=constants,
        target=target
    )
