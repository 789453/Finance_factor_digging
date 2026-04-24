#!/usr/bin/env python3
"""
参数化预处理脚本 - 支持配置文件驱动的特征预处理
"""

import pandas as pd
import numpy as np
import os
import json
import yaml
import argparse
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from utils.path_utils import map_path
except ImportError:
    try:
        from src.utils.path_utils import map_path
    except ImportError:
        def map_path(path: str) -> str:
            return path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreprocessConfig:
    """预处理配置类"""
    
    DEFAULT_CONFIG = {
        "domains": {
            "A": "feature_A_price_volume.parquet",
            "B": "feature_B_moneyflow.parquet", 
            "C": "feature_C_chip.parquet",
            "E": "feature_E_intraday_summary.parquet"
        },
        "input_dir": "data/factor_ready",
        "output_dir": "data/factor_ready_filled",
        "stats_dir": "data/feature_digged",
        "column_mapping": {
            "date": "trade_date",
            "code": "ts_code"
        },
        "filling_config": {
            "ffill_limit": 30,
            "bfill_limit": 15,
            "cross_sectional_fill": True,
            "global_fallback": 0.0
        },
        "compression": "snappy",
        "save_stats": True
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        else:
            self.config = self.DEFAULT_CONFIG.copy()
            logger.info("Using default config")
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置有效性"""
        required_keys = ["domains", "input_dir", "output_dir"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # 设置默认值
        self.config.setdefault("column_mapping", self.DEFAULT_CONFIG["column_mapping"])
        self.config.setdefault("filling_config", self.DEFAULT_CONFIG["filling_config"])
        self.config.setdefault("compression", self.DEFAULT_CONFIG["compression"])
        self.config.setdefault("save_stats", self.DEFAULT_CONFIG["save_stats"])
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    @property
    def domains(self) -> Dict[str, str]:
        """获取域配置"""
        return self.config["domains"]
    
    @property
    def input_dir(self) -> str:
        """获取输入目录"""
        return map_path(self.config["input_dir"])
    
    @property
    def output_dir(self) -> str:
        """获取输出目录"""
        return map_path(self.config["output_dir"])
    
    @property
    def stats_dir(self) -> str:
        """获取统计目录"""
        return map_path(self.config.get("stats_dir", "data/feature_digged"))
    
    @property
    def filling_config(self) -> Dict[str, Any]:
        """获取填充配置"""
        return self.config["filling_config"]
    
    @property
    def column_mapping(self) -> Dict[str, str]:
        """获取列映射"""
        return self.config["column_mapping"]


def preprocess_domain_features(
    domain: str, 
    input_path: str, 
    output_dir: str, 
    config: PreprocessConfig
) -> dict:
    """
    预处理域特征
    
    Args:
        domain: 域标识
        input_path: 输入文件路径
        output_dir: 输出目录
        config: 预处理配置
        
    Returns:
        处理统计信息
    """
    logger.info(f"Processing domain {domain} from {input_path}...")
    
    # 获取配置
    filling_config = config.filling_config
    column_mapping = config.column_mapping
    
    date_col = column_mapping.get("date", "trade_date")
    code_col = column_mapping.get("code", "ts_code")
    
    # 1. 高效加载
    logger.info("Loading data...")
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    # 识别特征列
    cols_to_skip = [date_col, code_col]
    feature_cols = [c for c in df.columns if c not in cols_to_skip]
    
    logger.info(f"Found {len(feature_cols)} feature columns")
    
    initial_nan_counts = df[feature_cols].isna().sum()
    total_rows = len(df)
    
    # 2. 时间序列填充
    logger.info("Performing temporal filling...")
    df = df.sort_values([code_col, date_col])
    
    ffill_limit = filling_config.get("ffill_limit", 30)
    bfill_limit = filling_config.get("bfill_limit", 15)
    
    df[feature_cols] = df.groupby(code_col)[feature_cols].ffill(limit=ffill_limit).bfill(limit=bfill_limit)
    
    # 3. 横截面填充
    if filling_config.get("cross_sectional_fill", True):
        logger.info("Performing cross-sectional median filling...")
        day_medians = df.groupby(date_col)[feature_cols].transform('median')
        df[feature_cols] = df[feature_cols].fillna(day_medians)
    
    # 4. 全局回退
    global_fallback = filling_config.get("global_fallback", 0.0)
    logger.info(f"Final global fallback to {global_fallback}...")
    df[feature_cols] = df[feature_cols].fillna(global_fallback)
    
    # 5. 计算统计信息
    final_nan_counts = df[feature_cols].isna().sum()
    stats = {}
    for col in feature_cols:
        stats[col] = {
            "initial_nan_rate": float(initial_nan_counts[col] / total_rows),
            "final_nan_rate": float(final_nan_counts[col] / total_rows),
            "filled_count": int(initial_nan_counts[col] - final_nan_counts[col]),
            "fill_rate": float((initial_nan_counts[col] - final_nan_counts[col]) / initial_nan_counts[col]) if initial_nan_counts[col] > 0 else 0.0
        }
    
    # 6. 高效保存
    output_path = os.path.join(output_dir, f"feature_{domain}_filled.parquet")
    logger.info(f"Saving to {output_path}...")
    
    compression = config.get("compression", "snappy")
    new_table = pa.Table.from_pandas(df)
    pq.write_table(new_table, output_path, compression=compression)
    
    logger.info(f"Domain {domain} processing completed")
    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="参数化特征预处理脚本")
    parser.add_argument("--config", type=str, help="配置文件路径 (YAML或JSON)")
    parser.add_argument("--domains", type=str, nargs="+", help="要处理的域列表")
    parser.add_argument("--input-dir", type=str, help="输入目录")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--stats-dir", type=str, help="统计目录")
    parser.add_argument("--save-config", type=str, help="保存默认配置到文件")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 保存默认配置
    if args.save_config:
        with open(args.save_config, 'w', encoding='utf-8') as f:
            yaml.dump(PreprocessConfig.DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Default config saved to {args.save_config}")
        return
    
    # 加载配置
    config = PreprocessConfig(args.config)
    
    # 确定要处理的域
    domains_to_process = args.domains or list(config.domains.keys())
    
    # 创建必要的目录
    os.makedirs(config.output_dir, exist_ok=True)
    if config.get("save_stats", True):
        os.makedirs(config.stats_dir, exist_ok=True)
    
    logger.info(f"Starting preprocessing with config: {json.dumps(config.config, indent=2)}")
    logger.info(f"Processing domains: {domains_to_process}")
    
    all_stats = {}
    
    for domain in domains_to_process:
        if domain not in config.domains:
            logger.warning(f"Domain {domain} not found in config, skipping")
            continue
        
        filename = config.domains[domain]
        input_file = os.path.join(config.input_dir, filename)
        
        if not os.path.exists(input_file):
            logger.warning(f"Input file not found: {input_file}, skipping")
            continue
        
        try:
            domain_stats = preprocess_domain_features(domain, input_file, config.output_dir, config)
            all_stats[domain] = domain_stats
            logger.info(f"Domain {domain} completed successfully")
        except Exception as e:
            logger.error(f"Error processing domain {domain}: {e}")
            continue
    
    # 保存统计信息
    if config.get("save_stats", True) and all_stats:
        stats_file = os.path.join(config.stats_dir, "preprocessing_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {stats_file}")
    
    # 生成清单文件
    manifest = {
        "processed_at": pd.Timestamp.now().isoformat(),
        "config": config.config,
        "domains_processed": list(all_stats.keys()),
        "total_domains": len(all_stats),
        "statistics": {
            domain: {
                "total_features": len(stats),
                "avg_fill_rate": sum(s["fill_rate"] for s in stats.values()) / len(stats) if stats else 0
            }
            for domain, stats in all_stats.items()
        }
    }
    
    manifest_file = os.path.join(config.stats_dir, "preprocessing_manifest.json")
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Preprocessing completed. Manifest saved to {manifest_file}")
    logger.info(f"Processed {len(all_stats)} domains successfully")


if __name__ == "__main__":
    main()