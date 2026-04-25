import os
import sys
import logging
from pathlib import Path
import torch

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from mining.mine_factors import mine_factors

def test_smoke_mining():
    # 使用 A 股 2020-2021 的配置进行冒烟测试
    job_spec = "config/jobs/a_share_pv_ts_2020_2021.yaml"
    dataset_meta = "config/datasets/cn_stock_daily_v1.yaml"
    
    # 覆盖部分参数以缩短测试时间
    # 注意：mine_factors 内部会读取 YAML，这里我们先手动读取并修改
    import yaml
    with open(job_spec, 'r', encoding='utf-8') as f:
        spec_dict = yaml.safe_load(f)
    
    spec_dict['n_episodes'] = 5  # 只跑 5 个 episode
    spec_dict['batch_size'] = 2
    spec_dict['pool_capacity'] = 5
    spec_dict['train_start'] = "20200101"
    spec_dict['train_end'] = "20200131" # 缩短日期范围
    spec_dict['test_start'] = "20200201"
    spec_dict['test_end'] = "20200210"
    
    temp_job_spec = "config/jobs/smoke_temp.yaml"
    with open(temp_job_spec, 'w', encoding='utf-8') as f:
        yaml.dump(spec_dict, f)
    
    print("="*60)
    print(" Starting Factor Mining Smoke Test ")
    print("="*60)
    
    try:
        result = mine_factors(
            job_spec_path=temp_job_spec,
            dataset_meta_path=dataset_meta,
            cuda_override=-1 # 使用 CPU
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
    finally:
        if os.path.exists(temp_job_spec):
            os.remove(temp_job_spec)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_smoke_mining()
