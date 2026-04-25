from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import torch
import numpy as np

try:
    from alphagen.data.expression import Expression, Feature, Constant, DeltaTime, Operator
    from alpha_gfn.operator_complexity import get_op_complexity
except ImportError:
    from src.alphagen.data.expression import Expression, Feature, Constant, DeltaTime, Operator
    from src.alpha_gfn.operator_complexity import get_op_complexity

@dataclass
class ExpressionQualityReport:
    accept: bool
    reason: str
    complexity: int = 0
    depth: int = 0
    n_features: int = 0
    n_constants: int = 0
    n_operators: int = 0
    n_ts_operators: int = 0
    n_binary_operators: int = 0
    n_cross_feature_ops: int = 0
    has_rank: bool = False
    has_ts_rank: bool = False
    has_pair_ts: bool = False
    canonical: str = ""
    flags: Dict[str, Any] = field(default_factory=dict)

class ExpressionQualityValidator:
    """
    表达式质量验证器，用于拒绝过于简单、无效、恒等或近似恒等的表达式。
    """
    
    TS_OPS = {"Ref", "TsMean", "TsStd", "TsRank", "TsDelta", "TsCorr", "TsCov", "TsSum", "TsMax", "TsMin", "TsIr", "TsVar", "TsSkew", "TsKurt", "TsMed", "TsMad"}
    BINARY_OPS = {"Add", "Sub", "Mul", "Div", "TsCorr", "TsCov", "Pow", "Greater", "Less"}
    RANK_OPS = {"Rank", "TsRank"}

    def __init__(
        self,
        min_complexity: int = 4,
        min_depth: int = 2,
        min_operators: int = 2,
        min_ts_operators: int = 1,
        min_features: int = 1,
        reject_constant_only: bool = True,
        reject_raw_feature_only: bool = True,
        reject_single_ref: bool = True,
        reject_identity: bool = True,
        max_constant_ratio: float = 0.4,
        max_depth: int = 10,
    ):
        self.min_complexity = min_complexity
        self.min_depth = min_depth
        self.min_operators = min_operators
        self.min_ts_operators = min_ts_operators
        self.min_features = min_features
        self.reject_constant_only = reject_constant_only
        self.reject_raw_feature_only = reject_raw_feature_only
        self.reject_single_ref = reject_single_ref
        self.reject_identity = reject_identity
        self.max_constant_ratio = max_constant_ratio
        self.max_depth = max_depth

    def analyze(self, expr: Expression) -> ExpressionQualityReport:
        """分析表达式结构并生成报告"""
        expr_str = str(expr)
        
        # 基础统计
        stats = {
            "n_features": 0,
            "n_constants": 0,
            "n_operators": 0,
            "n_ts_operators": 0,
            "n_binary_operators": 0,
            "n_cross_feature_ops": 0,
            "has_rank": False,
            "has_ts_rank": False,
            "has_pair_ts": False,
            "max_depth": 0,
            "complexity": 0
        }
        
        self._recursive_analyze(expr, stats, depth=1)
        
        report = ExpressionQualityReport(
            accept=True,
            reason="",
            complexity=stats["complexity"],
            depth=stats["max_depth"],
            n_features=stats["n_features"],
            n_constants=stats["n_constants"],
            n_operators=stats["n_operators"],
            n_ts_operators=stats["n_ts_operators"],
            n_binary_operators=stats["n_binary_operators"],
            n_cross_feature_ops=stats["n_cross_feature_ops"],
            has_rank=stats["has_rank"],
            has_ts_rank=stats["has_ts_rank"],
            has_pair_ts=stats["has_pair_ts"],
            canonical=expr_str
        )
        
        # 执行校验逻辑
        self._check_rules(report)
        
        return report

    def _recursive_analyze(self, expr: Any, stats: Dict[str, Any], depth: int):
        stats["max_depth"] = max(stats["max_depth"], depth)
        
        if isinstance(expr, Feature):
            stats["n_features"] += 1
            stats["complexity"] += get_op_complexity("Feature")
        elif isinstance(expr, Constant):
            stats["n_constants"] += 1
            stats["complexity"] += get_op_complexity("Constant")
        elif isinstance(expr, Expression):
            # 这是一个算子
            op_name = expr.__class__.__name__
            stats["n_operators"] += 1
            stats["complexity"] += get_op_complexity(op_name)
            
            if op_name in self.TS_OPS:
                stats["n_ts_operators"] += 1
            if op_name in self.BINARY_OPS:
                stats["n_binary_operators"] += 1
            if op_name in self.RANK_OPS:
                stats["has_rank"] = stats["has_rank"] or (op_name == "Rank")
                stats["has_ts_rank"] = stats["has_ts_rank"] or (op_name == "TsRank")
            if op_name in {"TsCorr", "TsCov"}:
                stats["has_pair_ts"] = True
                
            # 递归处理子表达式
            # alphagen 表达式通常有 operands 属性或者类似的
            if hasattr(expr, "operands"):
                for operand in expr.operands:
                    self._recursive_analyze(operand, stats, depth + 1)
            elif hasattr(expr, "_operand"): # Unary
                self._recursive_analyze(expr._operand, stats, depth + 1)
            elif hasattr(expr, "_lhs") and hasattr(expr, "_rhs"): # Binary
                self._recursive_analyze(expr._lhs, stats, depth + 1)
                self._recursive_analyze(expr._rhs, stats, depth + 1)
        elif isinstance(expr, (int, float)):
            stats["n_constants"] += 1
            stats["complexity"] += 1

    def _check_rules(self, report: ExpressionQualityReport):
        # 1. 基础硬拒绝
        if self.reject_constant_only and report.n_features == 0 and report.n_operators > 0:
            report.accept = False
            report.reason = "constant_only"
            return

        if self.reject_raw_feature_only and report.n_operators == 0 and report.n_features == 1:
            report.accept = False
            report.reason = "raw_feature_only"
            return

        if self.reject_single_ref and report.n_operators == 1 and report.canonical.startswith("Ref("):
            report.accept = False
            report.reason = "single_ref"
            return

        # 2. 复杂度与深度
        if report.complexity < self.min_complexity:
            report.accept = False
            report.reason = f"low_complexity:{report.complexity}<{self.min_complexity}"
            return

        if report.depth < self.min_depth:
            report.accept = False
            report.reason = f"low_depth:{report.depth}<{self.min_depth}"
            return

        if report.n_operators < self.min_operators:
            report.accept = False
            report.reason = f"low_operators:{report.n_operators}<{self.min_operators}"
            return

        if report.n_ts_operators < self.min_ts_operators:
            report.accept = False
            report.reason = f"low_ts_operators:{report.n_ts_operators}<{self.min_ts_operators}"
            return

        if report.n_features < self.min_features:
            report.accept = False
            report.reason = f"low_features:{report.n_features}<{self.min_features}"
            return

        # 3. 常数比例
        total_nodes = max(1, report.n_features + report.n_operators)
        const_ratio = report.n_constants / total_nodes
        if const_ratio > self.max_constant_ratio:
            report.accept = False
            report.reason = f"high_constant_ratio:{const_ratio:.2f}>{self.max_constant_ratio}"
            return

        # 4. 恒等式 (简单检查)
        if self.reject_identity:
            s = report.canonical.lower()
            identities = ["sub(x,x)", "div(x,x)", "rank(rank(", "add(x,0)", "mul(x,1)"]
            # 这是一个非常简化的检查，实际可能需要更复杂的解析
            if "rank(rank(" in s:
                report.accept = False
                report.reason = "identity:nested_rank"
                return

    def validate(self, expr: Expression) -> ExpressionQualityReport:
        return self.analyze(expr)
