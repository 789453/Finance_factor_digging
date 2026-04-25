import pytest
import torch
import numpy as np
from typing import Any

from alphagen.data.expression import Feature, Constant, Add, Sub, Mul, Div, Rank, Ref, TsMean, TsStd, TsCorr, TsCov
from alphagen.data.stock_data import FeatureType
from alpha_gfn.expression_quality import ExpressionQualityValidator
from alpha_gfn.expression_canonical import ExpressionCanonicalizer
from evaluation.factor_metrics import FactorMetricsEvaluator

# Mock FeatureType for testing
class MockFeatureType:
    def __init__(self, name):
        self.name = name
    def __int__(self):
        return 0

F_CLOSE = Feature(MockFeatureType("CLOSE"))
F_OPEN = Feature(MockFeatureType("OPEN"))
F_HIGH = Feature(MockFeatureType("HIGH"))

def test_quality_validator():
    validator = ExpressionQualityValidator(
        min_complexity=5,
        min_ts_operators=1,
        min_operators=2,
        min_features=1
    )
    
    # 1. Reject too simple
    expr_simple = Add(F_CLOSE, Constant(1.0))
    report = validator.validate(expr_simple)
    assert not report.accept
    assert "low_complexity" in report.reason or "low_ts_operators" in report.reason
    
    # 2. Reject constant only
    expr_const = Add(Constant(1.0), Constant(2.0))
    report = validator.validate(expr_const)
    assert not report.accept
    assert report.reason == "constant_only"
    
    # 3. Reject raw feature
    report = validator.validate(F_CLOSE)
    assert not report.accept
    assert report.reason == "raw_feature_only"
    
    # 4. Accept complex enough
    # TsMean(F_CLOSE, 10) + TsStd(F_OPEN, 10)
    expr_ok = Add(TsMean(F_CLOSE, 10), TsStd(F_OPEN, 10))
    report = validator.validate(expr_ok)
    assert report.accept
    assert report.n_ts_operators == 2
    assert report.n_features == 2

def test_canonicalizer():
    canonicalizer = ExpressionCanonicalizer()
    
    # 1. Commutative Add
    expr1 = Add(F_CLOSE, F_OPEN)
    expr2 = Add(F_OPEN, F_CLOSE)
    assert canonicalizer.canonicalize(expr1) == canonicalizer.canonicalize(expr2)
    
    # 2. Commutative Mul
    expr3 = Mul(F_CLOSE, Constant(2.0))
    expr4 = Mul(Constant(2.0), F_CLOSE)
    assert canonicalizer.canonicalize(expr3) == canonicalizer.canonicalize(expr4)
    
    # 3. Nested Rank simplification
    expr_rank = Rank(Rank(F_CLOSE))
    assert canonicalizer.canonicalize(expr_rank) == "rank(close)"
    
    # 4. Alias mapping
    assert canonicalizer.canonicalize("Mean(Close, 10)") == "tsmean(close,10)"

def test_metrics_evaluator():
    evaluator = FactorMetricsEvaluator(device="cpu")
    
    # Create fake factor and target [n_dates, n_assets]
    n_dates, n_assets = 10, 50
    factor = torch.randn(n_dates, n_assets)
    target = 0.5 * factor + 0.1 * torch.randn(n_dates, n_assets) # Positively correlated
    
    metrics = evaluator.evaluate_tensor(factor, target)
    
    assert metrics["ic_mean"] > 0
    assert metrics["positive_ic_ratio"] > 0.5
    assert metrics["coverage_mean"] == 1.0
    assert metrics["nan_ratio"] == 0.0
    
    # Test sanity check
    sanity_ok = evaluator.value_sanity(factor)
    assert sanity_ok["ok"]
    
    # Test constant factor failure
    const_factor = torch.ones(n_dates, n_assets)
    sanity_fail = evaluator.value_sanity(const_factor)
    assert not sanity_fail["ok"]
    assert sanity_fail["reason"] == "constant_value"

if __name__ == "__main__":
    # If run directly, just run the tests
    test_quality_validator()
    test_canonicalizer()
    test_metrics_evaluator()
    print("All tests passed!")
