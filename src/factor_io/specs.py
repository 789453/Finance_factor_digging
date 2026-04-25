from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Literal

@dataclass
class DatasetSpec:
    dataset_id: str
    name: str
    domain: str
    frequency: str
    date_column: str
    asset_column: str
    columns: Dict[str, str]
    layers_enabled: List[str] = field(default_factory=lambda: ["raw"])
    data_backend: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UniverseSpec:
    universe_id: str
    domain: str
    asset_filter: Dict[str, Any] = field(default_factory=dict)
    liquidity_filter: Optional[Dict[str, Any]] = None
    status_filter: List[str] = field(default_factory=lambda: ["active"])
    benchmark: Optional[str] = None

@dataclass
class TargetSpec:
    target_id: str
    type: str = "forward_return"
    price_column: str = "close"
    horizon: int = 5
    neutralization: Optional[Dict[str, Any]] = None
    direction: Literal["predictive", "contrarian"] = "predictive"

@dataclass
class ExperimentSplit:
    train: Tuple[str, str]
    valid: Optional[Tuple[str, str]] = None
    test: Optional[Tuple[str, str]] = None

@dataclass
class SearchSpaceSpec:
    family_id: str
    max_expr_length: int = 20
    min_expr_length: int = 4
    operators: List[str] = field(default_factory=list)
    constants: List[float] = field(default_factory=list)
    windows: List[int] = field(default_factory=list)

@dataclass
class EngineSpec:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationSpec:
    metrics: List[str] = field(default_factory=lambda: ["ic", "rank_ic", "ic_ir"])
    topk: List[int] = field(default_factory=lambda: [20, 50])
    rolling_windows: List[int] = field(default_factory=lambda: [20, 60])
    min_coverage: float = 0.65
    max_nan_ratio: float = 0.35

@dataclass
class OutputSpec:
    formats: List[str] = field(default_factory=lambda: ["json", "parquet"])
    save_factor_values: bool = True
    save_daily_metrics: bool = True
    save_rejections: bool = True

@dataclass
class ScreeningSpec:
    """因子筛选条件，支持按区域/Domain定制"""
    min_ic: float = 0.02
    min_rank_ic: float = 0.02
    min_icir: float = 0.5
    min_stability: float = 0.1
    max_nan_ratio: float = 0.3
    min_coverage: float = 0.7
    max_complexity: int = 10
    max_depth: int = 5
    # 区域特定覆盖
    domain_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ConstraintSpec:
    """算子组合约束"""
    allow_nested_ts: bool = True
    max_ts_window: int = 60
    forbidden_combinations: List[Tuple[str, str]] = field(default_factory=list)
    required_operators: List[str] = field(default_factory=list)

@dataclass
class MiningExperimentSpec:
    job_id: str
    name: str
    seed: int = 42
    output_dir: str = "runs"
    dataset: Optional[DatasetSpec] = None
    universe: Optional[UniverseSpec] = None
    target: Optional[TargetSpec] = None
    split: Optional[ExperimentSplit] = None
    search_space: Optional[SearchSpaceSpec] = None
    engine: Optional[EngineSpec] = None
    evaluation: Optional[EvaluationSpec] = None
    screening: Optional[ScreeningSpec] = field(default_factory=ScreeningSpec)
    constraints: Optional[ConstraintSpec] = field(default_factory=ConstraintSpec)
    output: Optional[OutputSpec] = None
