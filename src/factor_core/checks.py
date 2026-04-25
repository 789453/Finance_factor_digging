from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class SyntaxCheckResult:
    ok: bool
    canonical_expr: Optional[str] = None
    normalized_expr: Optional[str] = None
    used_features: List[str] = field(default_factory=list)
    used_operators: List[str] = field(default_factory=list)
    windows: List[int] = field(default_factory=list)
    complexity: int = 0
    depth: int = 0
    reason: Optional[str] = None

@dataclass
class ScopeCheckResult:
    ok: bool
    dataset_id: str
    universe_id: str
    start_date: str
    end_date: str
    missing_features: List[str] = field(default_factory=list)
    unsupported_operators: List[str] = field(default_factory=list)
    max_backtrack_required: int = 0
    max_future_required: int = 0
    reason: Optional[str] = None

@dataclass
class MetricCheckResult:
    ok: bool
    split: str
    start_date: str
    end_date: str
    horizon: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    daily_metrics_path: Optional[str] = None
    factor_values_path: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class FactorInspectionReport:
    candidate_id: str
    syntax: SyntaxCheckResult
    scope: Optional[ScopeCheckResult] = None
    metrics: Dict[str, MetricCheckResult] = field(default_factory=dict)
    novelty: Dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"
    rejection_reason: Optional[str] = None
