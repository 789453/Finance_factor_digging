from dataclasses import dataclass
from typing import Any, Optional, List, Dict
import torch
from factor_io.specs import MiningExperimentSpec
from alphagen_generic.dataset_meta import DatasetMeta
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2
from datahub.duckdb_hub import DuckDBDataHub

@dataclass
class FactorMiningContext:
    """
    Context object holding all necessary resources for a mining experiment.
    """
    spec: MiningExperimentSpec
    dataset_meta: DatasetMeta
    registry: FeatureRegistryManagerV2
    datahub: Optional[DuckDBDataHub]
    train_loader: ParquetFeatureLoaderV2
    valid_loader: Optional[ParquetFeatureLoaderV2]
    test_loader: Optional[ParquetFeatureLoaderV2]
    device: torch.device
    
    # Mapped search space components
    features: List[Any]
    operators: List[Any]
    delta_times: List[int]
    constants: List[float]
    target: Any # Expression
