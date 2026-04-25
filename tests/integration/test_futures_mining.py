import os
import sys
import logging
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from mining.mine_factors import mine_factors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_futures_mining():
    job_spec = "config/jobs/fut_smoke_test.yaml"
    dataset_meta = "config/datasets/cn_futures_daily_v1.yaml"
    
    # 确保目录存在
    os.makedirs("data/mining_logs/smoke", exist_ok=True)
    
    print("="*60)
    print(" Starting Futures Mining Smoke Test ")
    print("="*60)
    
    try:
        result = mine_factors(
            job_spec_path=job_spec,
            dataset_meta_path=dataset_meta
        )
        print("\n" + "="*60)
        print(f"Mining completed successfully!")
        print(f"Status: {result['status']}")
        print(f"Best IC: {result.get('best_ic', 0):.4f}")
        print(f"Pool Size: {result.get('pool_size', 0)}")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: Mining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_futures_mining()
