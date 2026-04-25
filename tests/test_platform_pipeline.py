import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mining.orchestrator import run_experiment

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # We use a smoke test config
    # Since we need a real dataset_meta and parquet data for the inspector to work, 
    # we point to the existing moneyflow smoke test.
    job_spec_path = "config/mining/test_moneyflow.yaml"
    dataset_meta_path = "config/datasets/cn_stock_moneyflow_daily_v1.yaml"
    
    # Override engine to 'random' for smoke test
    # We can pass kwargs to run_experiment if we add support for engine override, 
    # but for now let's just use the job_spec as is or create a temp one.
    
    print("Testing Modular Factor Mining Platform Pipeline...")
    try:
        manifest = run_experiment(
            job_spec_path=job_spec_path,
            dataset_meta_path=dataset_meta_path,
            cuda_override=0 # Force use GPU 0
        )
        print("\nPipeline completed successfully!")
        print(f"Status: {manifest['status']}")
        print(f"Run Directory: {manifest['run_dir']}")
        print(f"Factors Seen: {manifest['n_seen']}")
        print(f"Factors Selected: {manifest['n_selected']}")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
