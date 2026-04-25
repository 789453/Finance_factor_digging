#!/usr/bin/env python3
"""
AlphaPROBE GFN Training CLI
基于 MiningJobSpec 和 DatasetMeta 的统一入口包装器
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# 路径配置
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# 导入统一入口
try:
    from mining.mine_factors import mine_factors
except ImportError:
    from src.mining.mine_factors import mine_factors

def main():
    parser = argparse.ArgumentParser(description="AlphaPROBE GFN Training CLI")
    
    # 核心参数
    parser.add_argument('--job_spec', type=str, required=True,
                        help="Path to job specification YAML/JSON file")
    parser.add_argument('--dataset_meta', type=str, required=True,
                        help="Path to dataset meta YAML file")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Override output directory")
    parser.add_argument('--cuda', type=int, default=None,
                        help="Override CUDA device index")
    parser.add_argument('--log_level', type=str, default='INFO',
                        help="Logging level")

    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 打印欢迎信息
    print("="*60)
    print(" AlphaPROBE Factor Mining Pipeline ")
    print("="*60)
    print(f"Job Spec: {args.job_spec}")
    print(f"Dataset Meta: {args.dataset_meta}")
    if args.output_dir:
        print(f"Output Dir: {args.output_dir}")
    print("="*60)

    try:
        # 调用统一挖掘入口
        result = mine_factors(
            job_spec_path=args.job_spec,
            dataset_meta_path=args.dataset_meta,
            log_dir=args.output_dir,
            cuda_override=args.cuda
        )
        
        print("\n" + "="*60)
        print(f"Mining completed successfully!")
        print(f"Status: {result['status']}")
        print(f"Best IC: {result.get('best_ic', 0):.4f}")
        print(f"Pool Size: {result.get('pool_size', 0)}")
        print(f"Logs saved to: {result['log_dir']}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Mining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
