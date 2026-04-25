import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from mining.orchestrator import run_experiment

def main():
    # Setup detailed logging to see inspector/pool decisions
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    job_spec_path = "config/mining/manual_test.yaml"
    dataset_meta_path = "config/datasets/cn_stock_moneyflow_daily_v1.yaml"
    
    print("Testing Platform Pipeline with Manual Factors...")
    print(f"Spec: {job_spec_path}")
    
    try:
        manifest = run_experiment(
            job_spec_path=job_spec_path,
            dataset_meta_path=dataset_meta_path,
            cuda_override=0 # Use GPU
        )
        print("\nManual Test Pipeline completed!")
        print(f"Status: {manifest['status']}")
        print(f"Run Directory: {manifest['run_dir']}")
        print(f"Factors Processed: {manifest['n_seen']}")
        print(f"Factors Accepted: {manifest['n_selected']}")
        
        # Check output CSV
        csv_path = os.path.join(manifest['run_dir'], "candidates_checked.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            print("\nInspection Results Summary:")
            print(df[['factor_id', 'expression', 'decision', 'rejection_reason', 'composite_score']])
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
