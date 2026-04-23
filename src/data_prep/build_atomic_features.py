#!/usr/bin/env python3
"""
构建原子特征脚本
根据工程文档要求，构建原子派生特征并生成对应的注册表
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到路径
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen_generic.dataset_meta import DatasetMeta

logger = logging.getLogger(__name__)

class AtomicFeatureBuilder:
    """原子特征构建器"""
    
    # 推荐的原子特征集合
    ATOMIC_FEATURES = {
        # 价格动量
        "ret_1": {
            "formula": "Ref($close, -1) / $close - 1",
            "source_cols": ["close"],
            "description": "1日收益率",
            "category": "momentum"
        },
        "ret_5": {
            "formula": "Ref($close, -5) / $close - 1", 
            "source_cols": ["close"],
            "description": "5日收益率",
            "category": "momentum"
        },
        "ret_10": {
            "formula": "Ref($close, -10) / $close - 1",
            "source_cols": ["close"], 
            "description": "10日收益率",
            "category": "momentum"
        },
        "ret_20": {
            "formula": "Ref($close, -20) / $close - 1",
            "source_cols": ["close"],
            "description": "20日收益率", 
            "category": "momentum"
        },
        
        # 波动率
        "vol_5": {
            "formula": "TsStd(Ref($close, -1) / $close - 1, 5)",
            "source_cols": ["close"],
            "description": "5日波动率",
            "category": "volatility"
        },
        "vol_10": {
            "formula": "TsStd(Ref($close, -1) / $close - 1, 10)",
            "source_cols": ["close"],
            "description": "10日波动率",
            "category": "volatility"
        },
        "vol_20": {
            "formula": "TsStd(Ref($close, -1) / $close - 1, 20)",
            "source_cols": ["close"],
            "description": "20日波动率",
            "category": "volatility"
        },
        "hl_spread": {
            "formula": "($high - $low) / $close",
            "source_cols": ["high", "low", "close"],
            "description": "高低价差",
            "category": "volatility"
        },
        "oc_gap": {
            "formula": "($open - Ref($close, -1)) / Ref($close, -1)",
            "source_cols": ["open", "close"],
            "description": "开盘缺口",
            "category": "volatility"
        },
        
        # 量价偏离
        "vwap_dev": {
            "formula": "($close - TsMean($close, 20)) / TsStd($close, 20)",
            "source_cols": ["close"],
            "description": "收盘价相对20日均值偏离度",
            "category": "deviation"
        },
        "amt_z20": {
            "formula": "($amount - TsMean($amount, 20)) / TsStd($amount, 20)",
            "source_cols": ["amount"],
            "description": "成交额20日标准化",
            "category": "deviation"
        },
        "turnover_z20": {
            "formula": "($turnover - TsMean($turnover, 20)) / TsStd($turnover, 20)",
            "source_cols": ["turnover"],
            "description": "换手率20日标准化",
            "category": "deviation"
        },
        "volume_cv_20": {
            "formula": "TsStd($volume, 20) / TsMean($volume, 20)",
            "source_cols": ["volume"],
            "description": "成交量20日变异系数",
            "category": "deviation"
        },
        
        # 排名型
        "ret_rank_20": {
            "formula": "Rank(Ref($close, -20) / $close - 1)",
            "source_cols": ["close"],
            "description": "20日收益率排名",
            "category": "rank"
        },
        "turnover_rank_20": {
            "formula": "Rank($turnover)",
            "source_cols": ["turnover"],
            "description": "换手率排名",
            "category": "rank"
        },
        "amount_rank_20": {
            "formula": "Rank($amount)",
            "source_cols": ["amount"],
            "description": "成交额排名",
            "category": "rank"
        },
        
        # 时序关系
        "corr_close_volume_20": {
            "formula": "TsCorr($close, $volume, 20)",
            "source_cols": ["close", "volume"],
            "description": "收盘价与成交量20日相关性",
            "category": "correlation"
        },
        "corr_ret_amount_20": {
            "formula": "TsCorr(Ref($close, -1) / $close - 1, $amount, 20)",
            "source_cols": ["close", "amount"],
            "description": "收益率与成交额20日相关性",
            "category": "correlation"
        },
        
        # 稳定性
        "skew_ret_20": {
            "formula": "TsSkew(Ref($close, -1) / $close - 1, 20)",
            "source_cols": ["close"],
            "description": "20日收益率偏度",
            "category": "stability"
        },
        "kurt_ret_20": {
            "formula": "TsKurt(Ref($close, -1) / $close - 1, 20)",
            "source_cols": ["close"],
            "description": "20日收益率峰度",
            "category": "stability"
        },
        "downside_vol_20": {
            "formula": "TsStd(Min(Ref($close, -1) / $close - 1, 0), 20)",
            "source_cols": ["close"],
            "description": "20日下行波动率",
            "category": "stability"
        }
    }
    
    def __init__(self, 
                 domain: str,
                 dataset_meta: Optional[DatasetMeta] = None,
                 registry_manager: Optional[FeatureRegistryManager] = None,
                 custom_features: Optional[Dict[str, Dict]] = None):
        """
        初始化原子特征构建器
        
        Args:
            domain: 数据域
            dataset_meta: 数据集元数据
            registry_manager: 特征注册管理器
            custom_features: 自定义原子特征
        """
        self.domain = domain
        self.dataset_meta = dataset_meta
        self.registry_manager = registry_manager or FeatureRegistryManager()
        self.custom_features = custom_features or {}
        
        # 合并特征定义
        self.all_features = {**self.ATOMIC_FEATURES, **self.custom_features}
        
    def validate_source_columns(self, available_columns: List[str]) -> Dict[str, List[str]]:
        """
        验证源列是否可用
        
        Args:
            available_columns: 可用列列表
            
        Returns:
            Dict: 每个特征的缺失列
        """
        missing_cols = {}
        available_set = set(available_columns)
        
        for feature_name, feature_def in self.all_features.items():
            required_cols = feature_def["source_cols"]
            missing = [col for col in required_cols if col not in available_set]
            if missing:
                missing_cols[feature_name] = missing
                
        return missing_cols
    
    def build_atomic_features(self, 
                            raw_data: pd.DataFrame,
                            output_path: str,
                            registry_output_path: str,
                            stats_output_path: str,
                            chunk_size: int = 100000) -> Dict[str, any]:
        """
        构建原子特征
        
        Args:
            raw_data: 原始数据DataFrame
            output_path: 输出parquet路径
            registry_output_path: 注册表输出路径
            stats_output_path: 统计信息输出路径
            chunk_size: 处理块大小
            
        Returns:
            Dict: 构建结果统计
        """
        logger.info(f"开始构建原子特征，数据形状: {raw_data.shape}")
        
        # 验证源列
        available_columns = raw_data.columns.tolist()
        missing_cols = self.validate_source_columns(available_columns)
        
        if missing_cols:
            logger.warning(f"以下特征缺少源列: {missing_cols}")
            # 过滤掉无法构建的特征
            available_features = {
                name: def_ for name, def_ in self.all_features.items() 
                if name not in missing_cols
            }
        else:
            available_features = self.all_features.copy()
        
        logger.info(f"可用原子特征数量: {len(available_features)}")
        
        # 按股票代码分组处理
        if self.code_column not in raw_data.columns:
            raise ValueError(f"数据中缺少股票代码列: {self.code_column}")
            
        # 获取唯一股票代码
        stock_codes = raw_data[self.code_column].unique()
        logger.info(f"股票数量: {len(stock_codes)}")
        
        # 准备结果DataFrame
        result_dfs = []
        
        # 分批处理股票
        for i in tqdm(range(0, len(stock_codes), chunk_size), desc="处理股票"):
            batch_codes = stock_codes[i:i+chunk_size]
            batch_data = raw_data[raw_data[self.code_column].isin(batch_codes)].copy()
            
            # 构建原子特征
            atomic_df = self._build_features_for_batch(batch_data, available_features)
            result_dfs.append(atomic_df)
            
            # 内存清理
            del batch_data
            
        # 合并所有结果
        logger.info("合并所有股票的原子特征...")
        final_df = pd.concat(result_dfs, ignore_index=True)
        
        # 保存结果
        logger.info(f"保存原子特征到: {output_path}")
        final_df.to_parquet(output_path, index=False)
        
        # 生成注册表
        registry_df = self._generate_registry(available_features)
        registry_df.to_csv(registry_output_path, index=False)
        logger.info(f"保存注册表到: {registry_output_path}")
        
        # 生成统计信息
        stats = self._generate_stats(final_df, available_features)
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"保存统计信息到: {stats_output_path}")
        
        logger.info("原子特征构建完成")
        return stats
    
    def _build_features_for_batch(self, batch_data: pd.DataFrame, features: Dict[str, Dict]) -> pd.DataFrame:
        """为一批数据构建特征"""
        # 确保数据按日期排序
        batch_data = batch_data.sort_values([self.code_column, self.date_column])
        
        # 基础列
        base_columns = [self.date_column, self.code_column]
        if self.close_column in batch_data.columns:
            base_columns.append(self.close_column)
            
        result_df = batch_data[base_columns].copy()
        
        # 构建每个原子特征
        for feature_name, feature_def in features.items():
            try:
                # 这里需要实现具体的特征计算逻辑
                # 由于涉及复杂的表达式解析，这里提供框架
                feature_values = self._compute_atomic_feature(batch_data, feature_def)
                result_df[feature_name] = feature_values
                
            except Exception as e:
                logger.warning(f"构建特征 {feature_name} 失败: {e}")
                result_df[feature_name] = np.nan
        
        return result_df
    
    def _compute_atomic_feature(self, data: pd.DataFrame, feature_def: Dict) -> pd.Series:
        """计算单个原子特征"""
        # 这里需要根据特征定义实现具体的计算逻辑
        # 由于涉及复杂的时序计算，这里提供框架实现
        
        formula = feature_def["formula"]
        source_cols = feature_def["source_cols"]
        
        # 简单的示例实现 - 实际需要根据公式解析
        if feature_def["category"] == "momentum":
            if "ret_1" in feature_def["formula"]:
                return data[self.close_column].pct_change()
            elif "ret_5" in feature_def["formula"]:
                return data[self.close_column].pct_change(5)
            elif "ret_10" in feature_def["formula"]:
                return data[self.close_column].pct_change(10)
            elif "ret_20" in feature_def["formula"]:
                return data[self.close_column].pct_change(20)
                
        elif feature_def["category"] == "volatility":
            if "vol_" in feature_def["formula"]:
                window = int(feature_def["formula"].split("TsStd")[1].split(",")[1].split(")")[0])
                returns = data[self.close_column].pct_change()
                return returns.rolling(window=window).std()
            elif "hl_spread" in feature_def["formula"]:
                return (data["high"] - data["low"]) / data["close"]
            elif "oc_gap" in feature_def["formula"]:
                return (data["open"] - data[self.close_column].shift(1)) / data[self.close_column].shift(1)
                
        # 默认返回NaN
        return pd.Series(np.nan, index=data.index)
    
    def _generate_registry(self, features: Dict[str, Dict]) -> pd.DataFrame:
        """生成特征注册表"""
        registry_data = []
        
        for feature_name, feature_def in features.items():
            registry_entry = {
                "feature_name": feature_name,
                "formula": feature_def["formula"],
                "source_cols": ",".join(feature_def["source_cols"]),
                "description": feature_def.get("description", ""),
                "category": feature_def.get("category", "unknown"),
                "domain": self.domain,
                "freq_group": "eod",  # 假设日频
                "fill_policy": "forward",  # 默认前向填充
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            registry_data.append(registry_entry)
        
        return pd.DataFrame(registry_data)
    
    def _generate_stats(self, final_df: pd.DataFrame, features: Dict[str, Dict]) -> Dict:
        """生成统计信息"""
        stats = {
            "build_info": {
                "domain": self.domain,
                "total_features": len(features),
                "build_time": datetime.now().isoformat(),
                "data_shape": final_df.shape
            },
            "feature_stats": {}
        }
        
        # 计算每个特征的统计信息
        for feature_name in features.keys():
            if feature_name in final_df.columns:
                feature_data = final_df[feature_name]
                feature_stats = {
                    "count": int(feature_data.count()),
                    "mean": float(feature_data.mean()),
                    "std": float(feature_data.std()),
                    "min": float(feature_data.min()),
                    "max": float(feature_data.max()),
                    "nan_ratio": float(feature_data.isna().sum() / len(feature_data)),
                    "skewness": float(feature_data.skew()),
                    "kurtosis": float(feature_data.kurtosis())
                }
                stats["feature_stats"][feature_name] = feature_stats
        
        return stats

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"build_atomic_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建原子特征")
    parser.add_argument("--domain", type=str, default="A", choices=["A", "B", "C", "E"],
                       help="数据域")
    parser.add_argument("--input_path", type=str, required=True,
                       help="输入原始数据路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出原子特征parquet路径")
    parser.add_argument("--registry_output", type=str, required=True,
                       help="输出注册表CSV路径")
    parser.add_argument("--stats_output", type=str, required=True,
                       help="输出统计信息JSON路径")
    parser.add_argument("--dataset_meta", type=str,
                       help="数据集元数据路径")
    parser.add_argument("--registry_path", type=str,
                       help="特征注册表路径")
    parser.add_argument("--chunk_size", type=int, default=100000,
                       help="处理块大小")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger.info("开始原子特征构建任务")
    logger.info(f"参数: {args}")
    
    try:
        # 加载数据集元数据
        dataset_meta = None
        if args.dataset_meta and os.path.exists(args.dataset_meta):
            dataset_meta = DatasetMeta(args.dataset_meta)
            logger.info(f"加载数据集元数据: {dataset_meta.dataset_id}")
        
        # 创建注册管理器
        registry_manager = None
        if args.registry_path:
            registry_manager = FeatureRegistryManager(args.registry_path)
            logger.info(f"加载特征注册表: {args.registry_path}")
        
        # 创建原子特征构建器
        builder = AtomicFeatureBuilder(
            domain=args.domain,
            dataset_meta=dataset_meta,
            registry_manager=registry_manager
        )
        
        # 加载原始数据
        logger.info(f"加载原始数据: {args.input_path}")
        raw_data = pd.read_parquet(args.input_path)
        logger.info(f"原始数据形状: {raw_data.shape}")
        logger.info(f"原始数据列: {raw_data.columns.tolist()}")
        
        # 构建原子特征
        stats = builder.build_atomic_features(
            raw_data=raw_data,
            output_path=args.output_path,
            registry_output_path=args.registry_output,
            stats_output_path=args.stats_output,
            chunk_size=args.chunk_size
        )
        
        logger.info("原子特征构建完成")
        logger.info(f"统计信息: {stats}")
        
    except Exception as e:
        logger.error(f"构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()