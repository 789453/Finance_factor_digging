import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append('/root/Desktop/factor_mining/src')

from alphagen.data.expression import Expression, Feature
from alphagen.data.tree import ExpressionParser
from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen_generic.features import target
from alphagen.data.expression_knowledge_graph import ExpressionKnowledgeGraph
from alpha_knowledge.alpha_pool import AlphaKnowledgePool

# Mocking data and environment
class MockData:
    def __init__(self):
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        self.n_days = 100
        self.n_stocks = 50
        self.max_backtrack_days = 60
        self.max_future_days = 30
        
        # Random data
        self.data = torch.randn(200, 100, self.n_stocks, device=self.device, dtype=self.dtype)

def test_add_expr():
    print("Initializing parser and graph...")
    parser = ExpressionParser()
    graph = ExpressionKnowledgeGraph(parser=parser)
    
    # Using real domain A features if possible
    from enum import IntEnum
    class DummyFeature(IntEnum):
        RET_CC_1D = 0
        RET_OC_1D = 1
        GAP_OPEN = 2
        INTRADAY_RANGE = 3
        UPPER_SHADOW_RATIO = 4
        LOWER_SHADOW_RATIO = 5
        TURNOVER_FREE = 6
        VOLUME_RATIO_LN = 7
        
    feature_map = {
        "$ret_cc_1d": DummyFeature.RET_CC_1D, 
        "$ret_oc_1d": DummyFeature.RET_OC_1D, 
        "$gap_open": DummyFeature.GAP_OPEN, 
        "$intraday_range": DummyFeature.INTRADAY_RANGE, 
        "$upper_shadow_ratio": DummyFeature.UPPER_SHADOW_RATIO, 
        "$lower_shadow_ratio": DummyFeature.LOWER_SHADOW_RATIO, 
        "$turnover_free": DummyFeature.TURNOVER_FREE, 
        "$volume_ratio_ln": DummyFeature.VOLUME_RATIO_LN
    }
    parser = ExpressionParser(feature_map=feature_map)
    graph = ExpressionKnowledgeGraph(parser=parser)

    # Mock AlphaKnowledgePool
    mock_data = MockData()
    # Need a valid target for pool
    import alphagen.data.expression as exp_module
    mock_target = exp_module.Feature(0)
    
    pool = AlphaKnowledgePool(
        capacity=10,
        stock_data=mock_data,
        target=mock_target,
        use_semantic_similarity=False,  # Disable sentence transformer for fast test
        device='cpu'
    )
    pool.attach_knowledge_graph(graph)
    
    # 1. Test parsing a new expression with constants
    test_expr_str = "Less(TsMean($ret_oc_1d, 30), -0.01)"
    print(f"\n1. Testing Parse: {test_expr_str}")
    try:
        expr_obj = parser.parse(test_expr_str)
        print(f"Parsed successfully: {expr_obj}")
    except Exception as e:
        print(f"Parse failed: {e}")
        return

    # 2. Test pool.try_new_expr directly
    print("\n2. Testing pool.try_new_expr...")
    node = graph.build_expression_node(
        test_expr_str, "Test Topic", "Test Desc", 50, 0.05, 0.5
    )
    
    if node:
        print("Node built successfully.")
        # We need to bypass actual data evaluation because MockData doesn't align with feature enums perfectly
        # Let's mock _calc_ics_and_icir
        pool._calc_ics_and_icir = lambda v, ic_mut_threshold: (0.05, 0.5, np.array([0.1]))
        pool._normalize_by_day = lambda v: v
        
        # Test try_new_expr
        res = pool.try_new_expr(expr_obj, "Test Topic", "Test Desc", node)
        print(f"try_new_expr result: {res}")
        
        if res:
            graph.insert(node, None)
            print("Node inserted to graph.")
    else:
        print("Failed to build node.")
        
    print(f"\nPool size: {pool.size}")
    if pool.size > 0:
        print(f"Factor 0: {pool.exprs[0]}")
        print(f"Node 0: {pool.expr2node[0]}")

if __name__ == "__main__":
    test_add_expr()
