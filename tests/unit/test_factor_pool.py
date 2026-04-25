import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from alphagen.data.expression import Feature, Constant, Add, TsMean, Ref
from alphagen.data.stock_data import FeatureType
from alpha_gfn.alpha_pool_v2 import AlphaPoolGFN
from factor_core.expression_quality import ExpressionQualityValidator

class MockFeatureType:
    def __init__(self, name):
        self.name = name
    def __int__(self):
        return 0

F_CLOSE = Feature(MockFeatureType("CLOSE"))
F_OPEN = Feature(MockFeatureType("OPEN"))

class MockStockData:
    def __init__(self, n_days=100, n_stocks=50, device="cpu"):
        self.n_days = n_days
        self.n_stocks = n_stocks
        self.device = torch.device(device)
        self.max_backtrack_days = 60
        self.max_future_days = 20
        # Dummy data tensor
        self.data = torch.randn(n_days + self.max_backtrack_days + self.max_future_days, 10, n_stocks)

def test_pool_composite_score():
    mock_data = MockStockData()
    
    # Mock target evaluation
    target_expr = MagicMock()
    target_expr.evaluate.return_value = torch.randn(100, 50)
    
    # Validator
    validator = ExpressionQualityValidator(min_complexity=4, min_ts_operators=1)
    
    pool = AlphaPoolGFN(
        capacity=5,
        stock_data=mock_data,
        target=target_expr,
        quality_validator=validator,
        min_train_ic=0.01,
        enable_cache=False
    )
    
    # Case 1: Rejected by Quality
    bad_expr = Add(F_CLOSE, Constant(1.0))
    ic, nov = pool.try_new_expr(bad_expr)
    assert ic == 1e-10
    assert pool.size == 0
    
    # Case 2: Accepted and Added
    # We need to mock evaluate for the expression
    # Use Add(TsMean, TsMean) to satisfy min_operators=2
    ok_expr = Add(TsMean(F_CLOSE, 10), TsMean(F_OPEN, 10))
    # Set up the expression so it can be evaluated
    with MagicMock() as mock_eval:
        ok_expr.evaluate = MagicMock(return_value=pool.target * 0.5 + torch.randn(100, 50) * 0.1)
        
        ic, nov = pool.try_new_expr(ok_expr)
        assert ic > 0.01
        assert pool.size == 1
        assert pool.exprs[0] == ok_expr

def test_pool_duplicate_canonical():
    mock_data = MockStockData()
    target_expr = MagicMock()
    target_expr.evaluate.return_value = torch.randn(100, 50)
    
    pool = AlphaPoolGFN(
        capacity=5,
        stock_data=mock_data,
        target=target_expr,
        min_train_ic=0.01,
        enable_cache=False
    )
    
    expr1 = TsMean(Add(F_CLOSE, F_OPEN), 10)
    expr2 = TsMean(Add(F_OPEN, F_CLOSE), 10) # Canonical duplicate
    
    expr1.evaluate = MagicMock(return_value=pool.target * 0.5)
    expr2.evaluate = MagicMock(return_value=pool.target * 0.5)
    
    ic1, nov1 = pool.try_new_expr(expr1)
    assert pool.size == 1
    
    ic2, nov2 = pool.try_new_expr(expr2)
    assert ic2 == 1e-10
    assert pool.size == 1 # Still 1

if __name__ == "__main__":
    test_pool_composite_score()
    test_pool_duplicate_canonical()
    print("Pool tests passed!")
