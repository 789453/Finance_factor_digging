import os
import sys
import torch
import pandas as pd
from types import SimpleNamespace

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen.data.expression import Feature, Ref
from alphagen.models.alpha_pool import AlphaPoolBase
from alpha_knowledge.alpha_knowledge_trainer import AlphaKnowledgeLogger

def test_feature_loading_and_target():
    print("Testing data loader and target calculation...")
    
    # Mock args
    args = SimpleNamespace(
        domain='E',
        train_end_year=2021,
        test_end_year=2025,
        registry_path='/mnt/data/factor_mining/factor_ready/feature_registry.csv',
        sample_pool_path='/mnt/data/factor_mining/factor_ready/sample_pool_200.json',
        daily_path='/mnt/data/factor_mining/basic/daily.parquet',
        dynamic_pool=True,
        data_device='cpu',
        log_device='cpu',
        cuda=0,
        label_days=10,
        adaptive_eval_batch_size=5,
        adaptive_cache_device="cpu",
        adaptive_no_grad=True,
        adaptive_dedup_method="none",
        adaptive_solver="ridge",
        ridge_alpha=1e-4,
        n_factors=10,
        search_time=1,
        resume_iteration=0,
        combo_mode="Every Round",
        log_combination=True,
        adaptive_mode="local",
        adaptive_window_type="rolling",
        adaptive_window_size=252,
        combo_interval=5,
        ric_mode="cpu_spearman",
        ric_chunk_days=20
    )
    
    # 1. Test DataLoader
    print("\n1. Initializing ParquetFeatureLoader...")
    data = ParquetFeatureLoader(
        domain=args.domain,
        start_time="2020-01-01",
        end_time="2021-12-31",
        registry_manager=FeatureRegistryManager(args.registry_path),
        daily_path=args.daily_path,
        pool_path=args.sample_pool_path,
        device=args.data_device
    )
    
    # Check tensor shape
    print(f"Data tensor shape: {data.data.shape}")
    print(f"Number of features in tensor: {data.data.shape[1]}")
    
    # 2. Test Registry and Enum
    print("\n2. Testing Registry and Feature Enum...")
    feature_type_enum = data.registry_manager.create_feature_enum(args.domain)
    print(f"Enum members: {list(feature_type_enum.__members__.keys())}")
    
    assert "CLOSE" in feature_type_enum.__members__, "CLOSE must be in Enum"
    assert feature_type_enum.CLOSE == 0, "CLOSE must be at index 0"
    
    # 3. Test Feature Map (Isolation)
    print("\n3. Testing Feature Map...")
    feature_map = data.feature_map
    print(f"Feature map keys: {list(feature_map.keys())}")
    
    assert "$close" not in feature_map, "$close should be hidden from LLM"
    assert "CLOSE" not in [v.name for v in feature_map.values()], "CLOSE should not be mapped"
    
    # 4. Test Target Calculation
    print("\n4. Testing Target Calculation...")
    close = Feature(feature_type_enum.CLOSE)
    target = Ref(close, -args.label_days) / close - 1
    print(f"Target expression: {target}")
    
    # Try evaluating the target on data
    from alphagen.utils.correlation import batch_pearsonr
    from gan.utils.builder import exprs2tensor
    try:
        tgt_tensor = exprs2tensor([target], data, normalize=False)
        print(f"Target tensor evaluated successfully. Shape: {tgt_tensor.shape}")
        
        # Check if CLOSE data is actually valid (not all NaNs)
        close_tensor = data.data[:, 0, :] # Index 0 is CLOSE
        valid_close_count = torch.isfinite(close_tensor).sum().item()
        print(f"Valid CLOSE values in tensor: {valid_close_count}")
        assert valid_close_count > 0, "CLOSE data is all NaN, meaning it wasn't loaded correctly from daily.parquet"
        
    except Exception as e:
        print(f"Failed to evaluate target: {e}")
        
    print("\n5. Testing AlphaKnowledgeLogger (Adaptive Combination Fix)...")
    logger = AlphaKnowledgeLogger(test_data=data, target=target, log_dir=".", args=args)
    
    # Mock a pool with a simple expression
    class MockPool:
        def __init__(self):
            self.state = {'exprs': [Ref(Feature(feature_type_enum.HOUR1_RET), -1)]}
            
    try:
        # We write to a dummy file
        with open("/dev/null", "w") as f:
            logger.load_test_res(MockPool(), f)
        print("Logger load_test_res executed successfully!")
    except Exception as e:
        print(f"Logger test failed: {e}")
        
    print("\nAll tests finished!")

if __name__ == "__main__":
    test_feature_loading_and_target()