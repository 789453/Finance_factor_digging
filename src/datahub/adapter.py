#!/usr/bin/env python3
"""
DuckDB数据适配器 - 将新的DataHub接口适配到现有的ParquetFeatureLoaderV2
"""

import logging
import pandas as pd
import numpy as np
import pyarrow as pa
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path
import sqlite3
import duckdb
from datetime import datetime, timedelta

# 导入现有的数据加载器 - 使用正确的导入路径
try:
    from src.alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
    from src.alphagen_generic.dataset_meta import DatasetMeta
except ImportError:
    # 回退到相对导入
    try:
        from ..alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
        from ..alphagen_generic.dataset_meta import DatasetMeta
    except ImportError:
        # 如果都失败，创建兼容的基类
        logger = logging.getLogger(__name__)
        logger.warning("无法导入ParquetFeatureLoaderV2，将创建兼容的基类")
        
        class ParquetFeatureLoaderV2:
            """兼容的ParquetFeatureLoaderV2基类"""
            def __init__(self, *args, **kwargs):
                pass
                
        class DatasetMeta:
            """兼容的DatasetMeta基类"""
            def __init__(self, *args, **kwargs):
                pass

# 导入新的DataHub
from .duckdb_hub import DuckDBDataHub
from .config import DataHubConfig

logger = logging.getLogger(__name__)

class DuckDBAdapter:
    """DuckDB数据适配器"""
    
    def __init__(self, datahub: DuckDBDataHub, config: DataHubConfig):
        """
        初始化适配器
        
        Args:
            datahub: DuckDB数据枢纽
            config: 数据枢纽配置
        """
        self.datahub = datahub
        self.config = config
        self._cache = {}
        
        logger.info(f"Initialized DuckDBAdapter with datahub: {config.warehouse_path}")
    
    def get_feature_data(self, domain: str, feature_name: str, 
                        start_date: str, end_date: str,
                        status_filter: List[str] = None) -> pd.DataFrame:
        """
        获取特征数据
        
        Args:
            domain: 域名称
            feature_name: 特征名称
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            status_filter: 状态过滤列表
            
        Returns:
            特征数据DataFrame
        """
        # 构建查询
        query = self._build_feature_query(domain, feature_name, start_date, end_date, status_filter)
        
        try:
            # 执行查询
            result = self.datahub.execute_sql(query)
            
            if result is None:
                logger.warning(f"No data returned for query: {query}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            if isinstance(result, pa.RecordBatch):
                df = result.to_pandas()
            elif isinstance(result, pa.Table):
                df = result.to_pandas()
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                logger.error(f"Unsupported result type: {type(result)}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(df)} rows for {feature_name} in {domain}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get feature data: {e}")
            return pd.DataFrame()
    
    def _build_feature_query(self, domain: str, feature_name: str,
                           start_date: str, end_date: str,
                           status_filter: List[str] = None) -> str:
        """构建特征查询"""
        # 获取域配置
        domain_config = self._get_domain_config(domain)
        if not domain_config:
            logger.error(f"Domain config not found: {domain}")
            return ""
        
        # 构建基础查询
        table_name = domain_config.get('table_name', f'{domain}_daily')
        date_column = domain_config.get('date_column', 'trade_date')
        code_column = domain_config.get('code_column', 'ts_code')
        status_column = domain_config.get('status_column', 'status')
        
        # 构建WHERE子句
        where_conditions = [
            f"{date_column} >= '{start_date}'",
            f"{date_column} <= '{end_date}'"
        ]
        
        # 添加状态过滤
        if status_filter and status_column:
            status_list = "','".join(status_filter)
            where_conditions.append(f"{status_column} IN ('{status_list}')")
        
        where_clause = " AND ".join(where_conditions)
        
        # 构建完整查询
        query = f"""
        SELECT {code_column}, {date_column}, {feature_name}
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY {code_column}, {date_column}
        """
        
        return query.strip()
    
    def _get_domain_config(self, domain: str) -> Dict[str, Any]:
        """获取域配置"""
        # 这里可以根据实际需求从配置文件中读取域配置
        # 目前使用默认配置
        default_configs = {
            'A': {
                'table_name': 'A_share_daily',
                'date_column': 'trade_date',
                'code_column': 'ts_code',
                'status_column': 'status'
            },
            'pv_daily': {
                'table_name': 'pv_daily',
                'date_column': 'trade_date',
                'code_column': 'ts_code',
                'status_column': 'status'
            },
            'moneyflow': {
                'table_name': 'moneyflow_daily',
                'date_column': 'trade_date',
                'code_column': 'ts_code',
                'status_column': 'status'
            }
        }
        
        return default_configs.get(domain, default_configs['A'])
    
    def get_available_features(self, domain: str) -> List[str]:
        """
        获取可用的特征列表
        
        Args:
            domain: 域名称
            
        Returns:
            特征名称列表
        """
        try:
            # 获取表结构
            domain_config = self._get_domain_config(domain)
            table_name = domain_config.get('table_name', f'{domain}_daily')
            
            # 查询表结构
            query = f"PRAGMA table_info({table_name})"
            result = self.datahub.execute_sql(query)
            
            if result is None:
                return []
            
            # 转换为DataFrame并提取列名
            if isinstance(result, (pa.RecordBatch, pa.Table)):
                df = result.to_pandas()
            else:
                df = result
            
            # 获取列名（排除标识列）
            exclude_columns = {'ts_code', 'trade_date', 'symbol', 'status'}
            features = [col for col in df['name'].tolist() if col not in exclude_columns]
            
            logger.info(f"Found {len(features)} features in {domain}: {features[:10]}...")
            return features
            
        except Exception as e:
            logger.error(f"Failed to get available features: {e}")
            return []
    
    def get_date_range(self, domain: str) -> tuple:
        """
        获取日期范围
        
        Args:
            domain: 域名称
            
        Returns:
            (min_date, max_date) 元组
        """
        try:
            domain_config = self._get_domain_config(domain)
            table_name = domain_config.get('table_name', f'{domain}_daily')
            date_column = domain_config.get('date_column', 'trade_date')
            
            query = f"""
            SELECT MIN({date_column}) as min_date, MAX({date_column}) as max_date
            FROM {table_name}
            """
            
            result = self.datahub.execute_sql(query)
            if result is None:
                return None, None
            
            if isinstance(result, (pa.RecordBatch, pa.Table)):
                df = result.to_pandas()
            else:
                df = result
            
            if len(df) > 0:
                min_date = df.iloc[0]['min_date']
                max_date = df.iloc[0]['max_date']
                logger.info(f"Date range for {domain}: {min_date} to {max_date}")
                return min_date, max_date
            
            return None, None
            
        except Exception as e:
            logger.error(f"Failed to get date range: {e}")
            return None, None

class DuckDBParquetFeatureLoaderV2(ParquetFeatureLoaderV2):
    """适配到DuckDB的ParquetFeatureLoaderV2"""
    
    def __init__(self, domain: str, start_time: str, end_time: str,
                 registry_manager=None, dataset_meta=None, device=None,
                 max_backtrack_days: int = 100, max_future_days: int = 30,
                 status_filter: List[str] = None, use_filled: bool = True,
                 layers: List[str] = None, cache_root: str = "data/cache",
                 datahub_config_path: str = None):
        """
        初始化适配的加载器
        
        Args:
            domain: 域名称
            start_time: 开始时间 (YYYYMMDD)
            end_time: 结束时间 (YYYYMMDD)
            registry_manager: 特征注册管理器
            dataset_meta: 数据集元数据
            device: 设备
            max_backtrack_days: 最大回溯天数
            max_future_days: 最大未来天数
            status_filter: 状态过滤
            use_filled: 是否使用填充数据
            layers: 层列表
            cache_root: 缓存根目录
            datahub_config_path: DataHub配置文件路径
        """
        # 初始化DataHub
        if datahub_config_path:
            hub_config = DataHubConfig.from_yaml(datahub_config_path)
        else:
            # 使用默认配置
            hub_config = DataHubConfig(
                warehouse_path="D:/Trading/data_ever_26_3_14/data",
                control_db_path="D:/Trading/data_ever_26_3_14/data/meta/control.sqlite3"
            )
        
        self.datahub = DuckDBDataHub(hub_config)
        self.adapter = DuckDBAdapter(self.datahub, hub_config)
        
        # 保存参数
        self.domain = domain
        self.start_time = start_time
        self.end_time = end_time
        self.status_filter = status_filter or ['active', 'watch']
        self.use_filled = use_filled
        self.layers = layers or ['raw', 'filled']
        self.cache_root = cache_root
        
        # 调用父类初始化（简化版本）
        # 注意：这里不调用完整的父类初始化，而是提供兼容的接口
        self.registry_manager = registry_manager
        self.dataset_meta = dataset_meta
        self.device = device or "cpu"
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        
        # 初始化数据存储
        self._feature_data = {}
        self._close_data = None
        self._date_index = None
        self._stock_codes = None
        
        logger.info(f"Initialized DuckDBParquetFeatureLoaderV2 for {domain}")
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        try:
            # 获取可用特征
            available_features = self.adapter.get_available_features(self.domain)
            logger.info(f"Available features in {self.domain}: {len(available_features)}")
            
            # 获取收盘价数据（用于计算收益率）
            close_data = self.adapter.get_feature_data(
                self.domain, 'close', self.start_time, self.end_time, self.status_filter
            )
            
            if close_data.empty:
                logger.error("No close data available")
                return
            
            self._close_data = close_data
            self._date_index = sorted(close_data['trade_date'].unique())
            self._stock_codes = sorted(close_data['ts_code'].unique())
            
            # 加载其他特征
            for feature in available_features[:20]:  # 限制特征数量
                if feature == 'close':
                    continue
                    
                feature_data = self.adapter.get_feature_data(
                    self.domain, feature, self.start_time, self.end_time, self.status_filter
                )
                
                if not feature_data.empty:
                    self._feature_data[feature] = feature_data
                    logger.info(f"Loaded feature: {feature} ({len(feature_data)} rows)")
            
            logger.info(f"Loaded {len(self._feature_data)} features for {self.domain}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
    
    def get_feature(self, feature_name: str) -> Optional[np.ndarray]:
        """
        获取特征数据
        
        Args:
            feature_name: 特征名称
            
        Returns:
            特征数据数组
        """
        if feature_name in self._feature_data:
            return self._feature_data[feature_name][feature_name].values
        elif feature_name == 'close':
            return self._close_data['close'].values if self._close_data is not None else None
        else:
            logger.warning(f"Feature not found: {feature_name}")
            return None
    
    def get_batch(self, feature_names: List[str]) -> Optional[np.ndarray]:
        """
        获取批量特征数据
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            批量特征数据数组，形状为 (n_samples, n_features)
        """
        if not feature_names:
            return None
        
        features = []
        for name in feature_names:
            feature_data = self.get_feature(name)
            if feature_data is not None:
                features.append(feature_data)
            else:
                logger.warning(f"Feature data not available: {name}")
                return None
        
        if not features:
            return None
        
        # 堆叠特征数据
        try:
            return np.column_stack(features)
        except Exception as e:
            logger.error(f"Failed to stack features: {e}")
            return None
    
    @property
    def close(self) -> Optional[np.ndarray]:
        """获取收盘价数据"""
        return self.get_feature('close')
    
    @property
    def dates(self) -> Optional[List[str]]:
        """获取日期列表"""
        return self._date_index
    
    @property
    def stock_codes(self) -> Optional[List[str]]:
        """获取股票代码列表"""
        return self._stock_codes
    
    def __len__(self):
        """获取数据长度"""
        if self._close_data is not None:
            return len(self._close_data)
        return 0
    
    def __repr__(self):
        return f"DuckDBParquetFeatureLoaderV2(domain={self.domain}, time_range={self.start_time}-{self.end_time})"

def create_duckdb_loader(domain: str, start_time: str, end_time: str, **kwargs) -> DuckDBParquetFeatureLoaderV2:
    """
    创建DuckDB数据加载器
    
    Args:
        domain: 域名称
        start_time: 开始时间
        end_time: 结束时间
        **kwargs: 其他参数
        
    Returns:
        DuckDBParquetFeatureLoaderV2实例
    """
    return DuckDBParquetFeatureLoaderV2(
        domain=domain,
        start_time=start_time,
        end_time=end_time,
        **kwargs
    )