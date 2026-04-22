import torch
import os
import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.train_gfn import train
import argparse

def test_gfn_setup():
    print("Testing GFN Setup...")
    
    # Define dummy arguments for a quick test
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    
    # Override with test-specific parameters
    args.domain = 'A'
    args.seed = 0
    args.cuda = 0
    args.pool_capacity = 5
    args.log_freq = 1
    args.n_episodes = 2 # Only 2 episodes for testing
    args.encoder_type = 'lstm' # LSTM is usually faster to init than GNN for tests
    args.entropy_coef = 0.01
    args.entropy_temperature = 1.0
    args.mask_dropout_prob = 0.5
    args.ssl_weight = 1.0
    args.nov_weight = 0.3
    args.weight_decay_type = 'linear'
    args.final_weight_ratio = 0.1
    args.label_days = 10
    args.train_end_year = 2011 # Small range for test speed
    args.test_end_year = 2012
    
    # Paths (using default project structure)
    args.registry_path = os.path.join(ROOT, "data", "factor_ready", "feature_registry.csv")
    args.sample_pool_path = os.path.join(ROOT, "data", "factor_ready", "sample_pool_200.json")
    args.daily_path = os.path.join(ROOT, "data", "basic", "daily.parquet")
    
    if not os.path.exists(args.registry_path):
        print(f"Skipping test: Registry not found at {args.registry_path}")
        return

    try:
        train(args)
        print("GFN Setup and short training run SUCCESSFUL!")
    except Exception as e:
        print(f"GFN Setup FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_gfn_setup()
