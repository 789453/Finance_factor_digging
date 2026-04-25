from __future__ import annotations
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from mining.context import FactorMiningContext

from factor_core.candidate import FactorCandidate
from factor_core.checks import (
    SyntaxCheckResult, ScopeCheckResult, MetricCheckResult, FactorInspectionReport
)
from factor_core.expression_quality import ExpressionQualityValidator, ExpressionQualityReport
from alphagen.data.tree import ExpressionParser, InvalidExpressionException
from alphagen.data.expression import Expression, Feature, Operator
from factor_eval.factor_metrics import FactorMetricsEvaluator

logger = logging.getLogger(__name__)

class FactorInspector:
    """
    Unified inspection entry point for factor candidates.
    Coordinates syntax, quality, scope, metric, and novelty checks.
    """
    def __init__(
        self,
        quality_validator: Optional[ExpressionQualityValidator] = None,
        metrics_evaluator: Optional[FactorMetricsEvaluator] = None,
        feature_map: Optional[Dict[str, Any]] = None,
        screening_spec: Optional[Any] = None,
    ):
        self.parser = ExpressionParser(feature_map=feature_map)
        self.quality_validator = quality_validator or ExpressionQualityValidator()
        self.metrics_evaluator = metrics_evaluator or FactorMetricsEvaluator()
        self.screening_spec = screening_spec
        self.feature_map = feature_map or {}

    def inspect(
        self,
        candidate: FactorCandidate,
        context: 'FactorMiningContext',
        splits: List[str] = ["train", "valid", "test"]
    ) -> FactorInspectionReport:
        """
        Perform full inspection on a factor candidate.
        """
        report = FactorInspectionReport(candidate_id=candidate.factor_id, syntax=SyntaxCheckResult(ok=False))
        
        # 1. Syntax Check
        try:
            expr_obj = self.parser.parse(candidate.expression)
            candidate.canonical_expr = str(expr_obj)
            
            # Extract features and operators
            used_features = []
            used_operators = []
            windows = []
            
            def _extract_stats(node):
                if isinstance(node, Feature):
                    used_features.append(str(node))
                elif isinstance(node, Expression):
                    used_operators.append(node.__class__.__name__)
                    # Handle windows if it's a rolling operator
                    if hasattr(node, "_delta_time"):
                        windows.append(int(node._delta_time))
                
                if hasattr(node, "operands"):
                    for op in node.operands: _extract_stats(op)
                elif hasattr(node, "_operand"):
                    _extract_stats(node._operand)
                elif hasattr(node, "_lhs") and hasattr(node, "_rhs"):
                    _extract_stats(node._lhs)
                    _extract_stats(node._rhs)

            _extract_stats(expr_obj)
            
            # Get complexity and depth from quality validator
            q_report = self.quality_validator.analyze(expr_obj)
            
            report.syntax = SyntaxCheckResult(
                ok=True,
                canonical_expr=candidate.canonical_expr,
                normalized_expr=str(expr_obj),
                used_features=list(set(used_features)),
                used_operators=list(set(used_operators)),
                windows=list(set(windows)),
                complexity=q_report.complexity,
                depth=q_report.depth
            )
        except Exception as e:
            report.syntax = SyntaxCheckResult(ok=False, reason=f"syntax_error: {str(e)}")
            report.decision = "rejected"
            report.rejection_reason = "syntax_error"
            return report

        # 2. Quality Check (Expression-based)
        if not q_report.accept:
            report.decision = "rejected"
            report.rejection_reason = q_report.reason
            return report

        # 3. Scope Check (Context-based)
        # Check if all features exist in registry
        missing = []
        for feat in report.syntax.used_features:
            # Normalize feature name: handle $ prefix or case
            clean_feat = feat.replace("$", "").upper()
            if clean_feat not in self.feature_map and clean_feat.lower() not in self.feature_map:
                 # Check registry directly if map is just based on context.features
                 if clean_feat not in context.registry.features:
                    missing.append(feat)
        
        report.scope = ScopeCheckResult(
            ok=len(missing) == 0,
            dataset_id=context.spec.dataset.dataset_id if context.spec.dataset else "unknown",
            universe_id=context.spec.universe.universe_id if context.spec.universe else "unknown",
            start_date=context.spec.split.train[0] if context.spec.split else "unknown",
            end_date=context.spec.split.test[1] if context.spec.split and context.spec.split.test else "unknown",
            missing_features=missing,
            reason=f"missing_features: {missing}" if missing else None
        )
        
        if not report.scope.ok:
            report.decision = "rejected"
            report.rejection_reason = report.scope.reason
            return report

        # 4. Metric Evaluation (Value-based)
        try:
            # First evaluate TRAIN to get direction
            train_result = self._evaluate_split(expr_obj, "train", context)
            report.metrics["train"] = train_result
            
            if not train_result.ok:
                report.decision = "rejected"
                report.rejection_reason = f"train_eval_error: {train_result.reason}"
                return report
                
            # Determine direction from train IC
            direction = 1.0 if train_result.metrics.get("ic_mean", 0) >= 0 else -1.0
            
            # Apply train screening
            if self.screening_spec:
                ic = train_result.metrics.get("ic_mean", 0)
                if abs(ic) < self.screening_spec.min_train_abs_ic:
                    report.decision = "rejected"
                    report.rejection_reason = f"low_train_ic: {abs(ic):.4f} < {self.screening_spec.min_train_abs_ic}"
                    return report

            # Evaluate other splits with the SAME direction
            for split_name in splits:
                if split_name == "train": continue
                
                metric_result = self._evaluate_split(expr_obj, split_name, context, direction=direction)
                report.metrics[split_name] = metric_result
                
                # Screening on VALID split
                if self.screening_spec and split_name == "valid":
                    ic_adj = metric_result.metrics.get("ic_adj", 0)
                    if ic_adj < self.screening_spec.min_valid_abs_ic:
                        report.decision = "rejected"
                        report.rejection_reason = f"low_valid_ic_adj: {ic_adj:.4f} < {self.screening_spec.min_valid_abs_ic}"
                        return report
                    
                    icir_adj = metric_result.metrics.get("ic_ir", 0) * direction # Or use icir_adj if provided
                    if icir_adj < self.screening_spec.min_valid_icir:
                        report.decision = "rejected"
                        report.rejection_reason = f"low_valid_icir_adj: {icir_adj:.4f} < {self.screening_spec.min_valid_icir}"
                        return report

            report.decision = "passed"
        except Exception as e:
            logger.exception(f"Error evaluating metrics for {candidate.factor_id}")
            report.decision = "rejected"
            report.rejection_reason = f"evaluation_error: {str(e)}"

        return report

    def _evaluate_split(self, expr: Expression, split_name: str, context: 'FactorMiningContext', direction: float = 1.0) -> MetricCheckResult:
        """Evaluate factor on a specific split."""
        loader = getattr(context, f"{split_name}_loader", None)
        if loader is None:
            return MetricCheckResult(ok=False, split=split_name, start_date="", end_date="", horizon=0, reason="loader_not_found")
        
        try:
            # Generate factor values
            factor_values = loader.evaluate_expression(expr)
            
            # Get target values (already on device)
            target_values = loader.get_target()
            
            # Evaluate metrics
            metrics = self.metrics_evaluator.evaluate_tensor(
                factor_values, 
                target_values, 
                direction=direction
            )
            
            return MetricCheckResult(
                ok=True,
                split=split_name,
                start_date=loader.start_date,
                end_date=loader.end_date,
                horizon=getattr(context.spec.target, "horizon", 5),
                metrics=metrics
            )
        except Exception as e:
            return MetricCheckResult(
                ok=False,
                split=split_name,
                start_date="error",
                end_date="error",
                horizon=0,
                reason=str(e)
            )
