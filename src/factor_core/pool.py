import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import torch

from factor_core.checks import FactorInspectionReport
from factor_io.specs import ScreeningSpec, EvaluationSpec
from factor_core.novelty import NoveltyChecker

logger = logging.getLogger(__name__)

@dataclass
class PoolDecision:
    accepted: bool
    action: Literal["add", "replace", "reject"]
    score: float
    rejection_reason: Optional[str] = None
    replaced_factor_id: Optional[str] = None
    novelty_score: float = 1.0

class PoolPolicy:
    """
    Standard policy for factor pool management.
    Handles screening, scoring, and replacement logic.
    """
    def __init__(self, screening_spec: ScreeningSpec):
        self.spec = screening_spec
        self.novelty_checker = NoveltyChecker(
            max_pool_corr=screening_spec.max_pool_corr,
            max_semantic_sim=getattr(screening_spec, "max_semantic_sim", 0.95)
        )

    def decide(self, report: FactorInspectionReport, current_pool: List[FactorInspectionReport]) -> PoolDecision:
        """
        Decide whether to add, replace or reject a factor based on its inspection report.
        """
        # 1. Hard Reject based on ScreeningSpec
        if report.decision == "rejected":
            return PoolDecision(accepted=False, action="reject", score=0.0, rejection_reason=report.rejection_reason)

        # 2. Metric Screening (Valid set)
        valid_metrics = report.metrics.get("valid")
        if not valid_metrics or not valid_metrics.ok:
            return PoolDecision(accepted=False, action="reject", score=0.0, rejection_reason="missing_valid_metrics")

        # IC screening (using adjusted IC from metrics)
        ic_adj = valid_metrics.metrics.get("ic_adj", 0)
        if ic_adj < self.spec.min_valid_abs_ic:
            return PoolDecision(accepted=False, action="reject", score=0.0, rejection_reason=f"low_valid_ic_adj:{ic_adj:.4f}")

        # 3. Novelty Check (Canonical Duplicate)
        if self.spec.max_canonical_duplicate:
            if not self.novelty_checker.check_canonical(report.syntax.canonical_expr):
                return PoolDecision(accepted=False, action="reject", score=0.0, rejection_reason="canonical_duplicate")
        
        # 4. Semantic Novelty Check (Ollama Embedding)
        embedding = None
        if getattr(self.spec, "use_semantic_novelty", False):
            embedding = self.novelty_checker.get_embedding(report.syntax.normalized_expr)
            if embedding is not None:
                ok, max_sim = self.novelty_checker.check_semantic_novelty(embedding)
                if not ok:
                    return PoolDecision(
                        accepted=False, action="reject", score=0.0, 
                        rejection_reason=f"semantic_duplicate:sim={max_sim:.4f}"
                    )
                report.metadata["embedding"] = embedding.tolist() # Store for later

        # 5. Composite Scoring
        score = self._calculate_composite_score(report)
        
        return PoolDecision(accepted=True, action="add", score=score)

    def _calculate_composite_score(self, report: FactorInspectionReport) -> float:
        """
        Calculate composite score for ranking factors.
        score = 1.00 * valid_ic_adj + 0.60 * valid_rank_ic_adj + ...
        As specified in 工程修改文档425-20.md
        """
        valid_metrics = report.metrics.get("valid")
        if not valid_metrics or not valid_metrics.ok:
            return 0.0
        
        m = valid_metrics.metrics
        
        # Base components (already direction-adjusted if evaluator handled it)
        # Note: FactorMetricsEvaluator adds ic_adj and rank_ic_adj
        ic_adj = m.get("ic_adj", 0)
        rank_ic_adj = m.get("rank_ic_adj", 0)
        
        # ICIR needs adjustment if not already
        ic_ir = m.get("ic_ir", 0)
        rank_ic_ir = m.get("rank_ic_ir", 0)
        
        # Composite formula from document
        score = (
            1.00 * ic_adj +
            0.60 * rank_ic_adj +
            0.25 * abs(ic_ir) +  # Using abs since direction is handled by ic_adj
            0.25 * abs(rank_ic_ir)
        )
        
        # Stability
        stability = m.get("stability", 0)
        score += 0.05 * stability
        
        # Complexity penalty (prefer simpler factors)
        complexity = report.syntax.complexity
        if complexity > 20:
            score -= 0.01 * (complexity - 20)
            
        return float(score)

class FactorPool:
    """
    Unified Factor Pool for managing discovered factors.
    """
    def __init__(self, capacity: int, screening_spec: ScreeningSpec):
        self.capacity = capacity
        self.policy = PoolPolicy(screening_spec)
        self.items: List[FactorInspectionReport] = []
        self.scores: List[float] = []
        self.n_seen = 0

    def add(self, report: FactorInspectionReport) -> PoolDecision:
        self.n_seen += 1
        decision = self.policy.decide(report, self.items)
        
        if decision.accepted:
            if len(self.items) < self.capacity:
                self.items.append(report)
                self.scores.append(decision.score)
                # Register in novelty checker
                emb = report.metadata.get("embedding")
                self.policy.novelty_checker.add_to_pool(
                    report.syntax.canonical_expr, 
                    np.array(emb) if emb else None
                )
                decision.action = "add"
            else:
                # Find worst factor in pool
                worst_idx = np.argmin(self.scores)
                if decision.score > self.scores[worst_idx]:
                    decision.replaced_factor_id = self.items[worst_idx].candidate_id
                    decision.action = "replace"
                    
                    # Update pool and novelty checker
                    self.items[worst_idx] = report
                    self.scores[worst_idx] = decision.score
                    
                    # Note: Replacement in NoveltyChecker is tricky, for now we just add new ones
                    emb = report.metadata.get("embedding")
                    self.policy.novelty_checker.add_to_pool(
                        report.syntax.canonical_expr, 
                        np.array(emb) if emb else None
                    )
                else:
                    decision.accepted = False
                    decision.action = "reject"
                    decision.rejection_reason = f"score_too_low:{decision.score:.4f}<{self.scores[worst_idx]:.4f}"
        
        return decision

    def export(self) -> List[Dict[str, Any]]:
        """Export pool to list of dicts."""
        return [asdict(item) for item in self.items]

    @property
    def size(self) -> int:
        return len(self.items)
