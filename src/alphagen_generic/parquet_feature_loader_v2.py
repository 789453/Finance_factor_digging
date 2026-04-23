import pandas as pd
import torch
import numpy as np
import os
import pyarrow.parquet as pq
import hashlib
import json
import logging
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
from enum import IntEnum
from tqdm import tqdm
from .feature_registry_manager import FeatureRegistryManager
from .sample_pool_builder import SamplePoolBuilder
from .dataset_meta import DatasetMeta

logger = logging.getLogger(__name__)

class ParquetFeatureLoaderV2:
    """
    升级版ParquetFeatureLoader，支持多层数据视图和缓存
    支持raw/filled/atomic三层数据架构
    """
    
    def __init__(self,
                 domain: str,
                 start_time: str,
                 end_time: str,
                 registry_manager: FeatureRegistryManager = None,
                 dataset_meta: DatasetMeta = None,
                 data_dir: str = None,
                 daily_path: str = None,
                 pool_path: str = None,
                 device: torch.device = torch.device('cuda:0'),
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 status_filter: List[str] = ['active', 'watch'],
                 use_filled: bool = True,
                 feature_file_map_raw: Optional[Dict[str, str]] = None,
                 feature_file_map_filled: Optional[Dict[str, str]] = None,
                 atomic_path: Optional[str] = None,
                 date_column: str = "trade_date",
                 code_column: str = "ts_code",
                 close_column: str = "close",
                 dataset_id: Optional[str] = None,
                 freq_group: str = "eod",
                 layers: List[str] = None,
                 feature_names: Optional[List[str]] = None,
                 segment_name: Optional[str] = None,
                 read_mode: str = "mixed",
                 cache_root: Optional[str] = None,
                 return_close_only: bool = False,
                 ):
        """
        初始化ParquetFeatureLoaderV2
        
        Args:
            domain: 数据域
            start_time: 开始时间
            end_time: 结束时间
            registry_manager: 特征注册管理器
            dataset_meta: 数据集元数据
            data_dir: 数据目录
            daily_path: daily数据路径
            pool_path: 样本池路径
            device: 计算设备
            max_backtrack_days: 最大回溯天数
            max_future_days: 最大未来天数
            status_filter: 状态过滤
            use_filled: 是否使用填充数据
            feature_file_map_raw: 原始特征文件映射
            feature_file_map_filled: 填充特征文件映射
            atomic_path: 原子特征路径
            date_column: 日期列名
            code_column: 代码列名
            close_column: 收盘价列名
            dataset_id: 数据集ID
            freq_group: 频率组
            layers: 启用的数据层
            feature_names: 指定特征名列表
            segment_name: 区段名称
            read_mode: 读取模式 (raw|filled|atomic|mixed)
            cache_root: 缓存根目录
            return_close_only: 是否只返回收盘价
        """
        self.domain = domain
        self.start_time = str(start_time).replace("-", "")
        self.end_time = str(end_time).replace("-", "")
        self.device = device
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.status_filter = status_filter
        self.use_filled = use_filled
        self.dataset_meta = dataset_meta
        self.dataset_id = dataset_id or f"{domain}.{freq_group}"
        self.freq_group = freq_group
        self.layers = layers or ["raw", "filled"]
        self.feature_names = feature_names
        self.segment_name = segment_name
        self.read_mode = read_mode
        self.cache_root = cache_root
        self.return_close_only = return_close_only
        
        # 优先从 dataset_meta 获取配置
        if dataset_meta is not None:
            meta_kwargs = dataset_meta.to_loader_kwargs()
            self._apply_meta_config(meta_kwargs)
        
        self.date_column = date_column
        self.code_column = code_column
        self.close_column = close_column

        # 初始化注册管理器
        if registry_manager is None:
            self.registry_manager = FeatureRegistryManager()
        else:
            self.registry_manager = registry_manager

        # 设置数据目录
        self.data_dir = self._resolve_data_dir(data_dir)
        self.daily_path = self._resolve_daily_path(daily_path)
        self.atomic_path = atomic_path

        # 设置文件映射
        self._setup_file_maps(feature_file_map_raw, feature_file_map_filled)
        
        # 加载样本池
        self.sample_pool = self._load_sample_pool(pool_path)
        
        # 获取特征列表
        self.features = self._get_features()
        
        # 检查缓存或加载数据
        self.data, self._dates, self._stock_ids = self._load_or_cache_data()
    
    def _apply_meta_config(self, meta_kwargs: Dict[str, Any]) -> None:
        """应用数据集元数据配置"""
        if "data_dir" in meta_kwargs:
            self.data_dir = meta_kwargs["data_dir"]
        if "date_column" in meta_kwargs:
            self.date_column = meta_kwargs["date_column"]
        if "code_column" in meta_kwargs:
            self.code_column = meta_kwargs["code_column"]
        if "close_column" in meta_kwargs:
            self.close_column = meta_kwargs["close_column"]
        if "pool_path" in meta_kwargs:
            self.pool_path = meta_kwargs["pool_path"]
        if "feature_file_map_raw" in meta_kwargs:
            self.feature_file_map_raw = meta_kwargs["feature_file_map_raw"]
        if "feature_file_map_filled" in meta_kwargs:
            self.feature_file_map_filled = meta_kwargs["feature_file_map_filled"]
        if "atomic_path" in meta_kwargs:
            self.atomic_path = meta_kwargs["atomic_path"]
        if "dataset_id" in meta_kwargs:
            self.dataset_id = meta_kwargs["dataset_id"]
        if "freq_group" in meta_kwargs:
            self.freq_group = meta_kwargs["freq_group"]
        if "layers" in meta_kwargs:
            self.layers = meta_kwargs["layers"]
    
    def _resolve_data_dir(self, data_dir: Optional[str]) -> str:
        """解析数据目录"""
        if data_dir is not None:
            return data_dir
        
        # 默认路径
        from utils.path_utils import FACTOR_READY_DIR
        return FACTOR_READY_DIR
    
    def _resolve_daily_path(self, daily_path: Optional[str]) -> str:
        """解析daily数据路径"""
        if daily_path is not None:
            return daily_path
        
        from utils.path_utils import map_path
        return map_path(r"data/basic/daily.parquet")
    
    def _setup_file_maps(self, 
                        feature_file_map_raw: Optional[Dict[str, str]], 
                        feature_file_map_filled: Optional[Dict[str, str]]) -> None:
        """设置文件映射"""
        self.feature_file_map_filled = feature_file_map_filled or {
            'A': 'feature_A_filled.parquet',
            'B': 'feature_B_filled.parquet', 
            'C': 'feature_C_filled.parquet',
            'E': 'feature_E_filled.parquet'
        }
        self.feature_file_map_raw = feature_file_map_raw or {
            'A': 'feature_A_price_volume.parquet',
            'B': 'feature_B_moneyflow.parquet',
            'C': 'feature_C_chip.parquet',
            'E': 'feature_E_intraday_summary.parquet'
        }
    
    def _load_sample_pool(self, pool_path: Optional[str]) -> Optional[Union[List, Dict]]:
        """加载样本池"""
        if pool_path and os.path.exists(pool_path):
            with open(pool_path, 'r') as f:
                import json
                return json.load(f)
        else:
            logger.warning("No sample pool provided. Using all stocks from data.")
            return None
    
    def _get_features(self) -> List[str]:
        """获取特征列表"""
        if self.feature_names:
            return self.feature_names
        
        features = self.registry_manager.get_features_by_domain(self.domain, self.status_filter)
        if not features:
            raise ValueError(f"No active features found for domain {self.domain}")
        return features
    
    def _get_cache_key(self) -> str:
        """生成缓存键"""
        # 构建缓存键的组成部分
        key_parts = [
            self.dataset_id,
            self.domain,
            self.freq_group,
            self.start_time,
            self.end_time,
            str(self.max_backtrack_days),
            str(self.max_future_days),
            str(sorted(self.status_filter)),
            str(sorted(self.layers)),
            self.read_mode
        ]
        
        # 如果有样本池，加入哈希
        if self.sample_pool:
            if isinstance(self.sample_pool, list):
                key_parts.append(str(len(self.sample_pool)))
            elif isinstance(self.sample_pool, dict):
                key_parts.append(str(hash(str(sorted(self.sample_pool.keys())))))
        
        # 如果有指定特征，加入特征哈希
        if self.feature_names:
            key_parts.append(str(hash(str(sorted(self.feature_names)))))
        
        # 生成最终哈希
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_paths(self, cache_key: str) -> Dict[str, str]:
        """获取缓存路径"""
        if not self.cache_root:
            return {}
        
        cache_dir = os.path.join(self.cache_root, "data_views", cache_key[:2], cache_key)
        os.makedirs(cache_dir, exist_ok=True)
        
        return {
            "tensor": os.path.join(cache_dir, "tensor.pt"),
            "dates": os.path.join(cache_dir, "dates.npy"),
            "stocks": os.path.join(cache_dir, "stocks.npy"),
            "feature_index": os.path.join(cache_dir, "feature_index.json"),
            "view_manifest": os.path.join(cache_dir, "view_manifest.json")
        }
    
    def _load_or_cache_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """加载或缓存数据"""
        cache_key = self._get_cache_key()
        cache_paths = self._get_cache_paths(cache_key)
        
        # 尝试从缓存加载
        if cache_paths and self._is_cache_valid(cache_paths):
            logger.info(f"Loading data view from cache: {cache_key}")
            return self._load_from_cache(cache_paths)
        
        # 重新构建数据
        logger.info(f"Building data view for cache key: {cache_key}")
        data, dates, stocks = self._build_data_view()
        
        # 保存到缓存
        if cache_paths:
            logger.info(f"Saving data view to cache: {cache_key}")
            self._save_to_cache(data, dates, stocks, cache_paths)
        
        return data, dates, stocks
    
    def _is_cache_valid(self, cache_paths: Dict[str, str]) -> bool:
        """检查缓存是否有效"""
        required_files = ["tensor", "dates", "stocks", "feature_index", "view_manifest"]
        return all(os.path.exists(cache_paths[key]) for key in required_files)
    
    def _load_from_cache(self, cache_paths: Dict[str, str]) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """从缓存加载"""
        data = torch.load(cache_paths["tensor"])
        dates = pd.Index(np.load(cache_paths["dates"]))
        stocks = pd.Index(np.load(cache_paths["stocks"]))
        
        # 验证设备
        if str(data.device) != str(self.device):
            data = data.to(self.device)
        
        return data, dates, stocks
    
    def _save_to_cache(self, data: torch.Tensor, dates: pd.Index, stocks: pd.Index, cache_paths: Dict[str, str]) -> None:
        """保存到缓存"""
        # 保存张量（移到CPU）
        torch.save(data.cpu(), cache_paths["tensor"])
        
        # 保存索引
        np.save(cache_paths["dates"], dates.values)
        np.save(cache_paths["stocks"], stocks.values)
        
        # 保存特征索引
        feature_index = {feature: i for i, feature in enumerate(self.features)}
        with open(cache_paths["feature_index"], "w") as f:
            json.dump(feature_index, f, indent=2)
        
        # 保存视图清单
        manifest = {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "freq_group": self.freq_group,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "n_features": len(self.features),
            "n_days": len(dates),
            "n_stocks": len(stocks),
            "layers": self.layers,
            "read_mode": self.read_mode,
            "cache_version": "2.0",
            "created_at": pd.Timestamp.now().isoformat()
        }
        with open(cache_paths["view_manifest"], "w") as f:
            json.dump(manifest, f, indent=2)
    
    def _build_data_view(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """构建数据视图"""
        # 根据读取模式选择数据源
        if self.read_mode == "raw":
            return self._build_from_raw_layer()
        elif self.read_mode == "filled":
            return self._build_from_filled_layer()
        elif self.read_mode == "atomic":
            return self._build_from_atomic_layer()
        elif self.read_mode == "mixed":
            return self._build_mixed_view()
        else:
            raise ValueError(f"Unsupported read_mode: {self.read_mode}")
    
    def _resolve_input_files(self) -> Dict[str, str]:
        """解析输入文件"""
        input_files = {}
        
        # 从dataset_meta获取文件路径
        if self.dataset_meta:
            paths = self.dataset_meta.resolve_parquet_paths()
            for layer, path in paths.items():
                if layer in self.layers and os.path.exists(path):
                    input_files[layer] = path
        
        # 从CLI参数获取文件路径
        if "raw" in self.layers and self.domain in self.feature_file_map_raw:
            raw_path = os.path.join(self.data_dir, self.feature_file_map_raw[self.domain])
            if os.path.exists(raw_path):
                input_files["raw"] = raw_path
        
        if "filled" in self.layers and self.domain in self.feature_file_map_filled:
            filled_dir = os.path.join(os.path.dirname(self.data_dir), "factor_ready_filled")
            filled_path = os.path.join(filled_dir, self.feature_file_map_filled[self.domain])
            if os.path.exists(filled_path):
                input_files["filled"] = filled_path
        
        if "atomic" in self.layers and self.atomic_path and os.path.exists(self.atomic_path):
            input_files["atomic"] = self.atomic_path
        
        return input_files
    
    def _build_from_raw_layer(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """从原始层构建数据"""
        input_files = self._resolve_input_files()
        if "raw" not in input_files:
            raise FileNotFoundError(f"No raw layer file found for domain {self.domain}")
        
        return self._load_parquet_data(input_files["raw"])
    
    def _build_from_filled_layer(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """从填充层构建数据"""
        input_files = self._resolve_input_files()
        if "filled" not in input_files:
            raise FileNotFoundError(f"No filled layer file found for domain {self.domain}")
        
        return self._load_parquet_data(input_files["filled"])
    
    def _build_from_atomic_layer(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """从原子层构建数据"""
        input_files = self._resolve_input_files()
        if "atomic" not in input_files:
            raise FileNotFoundError(f"No atomic layer file found for domain {self.domain}")
        
        return self._load_parquet_data(input_files["atomic"])
    
    def _build_mixed_view(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """构建混合视图"""
        input_files = self._resolve_input_files()
        
        # 优先顺序: filled > raw > atomic
        if "filled" in input_files:
            return self._load_parquet_data(input_files["filled"])
        elif "raw" in input_files:
            return self._load_parquet_data(input_files["raw"])
        elif "atomic" in input_files:
            return self._load_parquet_data(input_files["atomic"])
        else:
            raise FileNotFoundError(f"No data files found for layers {self.layers}")
    
    def _load_parquet_data(self, parquet_path: str) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """加载parquet数据"""
        logger.info(f"Loading data from {parquet_path}")
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        # 读取所有日期以确定范围
        df_dates = pd.read_parquet(parquet_path, columns=[self.date_column])
        all_dates = df_dates[self.date_column].unique()
        all_dates.sort()
        all_dates_str = all_dates.astype(str)
        
        # 找到索引
        try:
            start_idx = next(i for i, d in enumerate(all_dates_str) if d >= self.start_time)
            end_idx = next(i for i, d in enumerate(all_dates_str) if d > self.end_time) - 1
        except StopIteration:
            raise ValueError(f"Date range {self.start_time}-{self.end_time} not covered by data")
        
        if end_idx < start_idx:
            raise ValueError("end_time before start_time")
        
        # 扩展范围
        real_start_idx = max(0, start_idx - self.max_backtrack_days)
        real_end_idx = min(len(all_dates), end_idx + self.max_future_days + 1)
        
        real_start_date = all_dates[real_start_idx]
        real_end_date = all_dates[real_end_idx - 1]
        
        logger.info(f"Loading expanded range: {real_start_date} to {real_end_date} (Request: {self.start_time}-{self.end_time})")
        
        # 记录完整日期范围
        self._dates = pd.Index(all_dates[real_start_idx:real_end_idx])
        
        # 获取可用列
        available_columns = self._available_columns(parquet_path)
        resolved_feature_columns = self._resolve_feature_columns(available_columns)
        
        # 构建所需列
        required_columns = [self.date_column, self.code_column]
        required_columns.extend(col for col in resolved_feature_columns.values() if col is not None)
        
        # 如果只需要收盘价，只加载必要列
        if self.return_close_only and self.close_column in available_columns:
            required_columns = [self.date_column, self.code_column, self.close_column]
        
        # 读取数据
        df = pd.read_parquet(parquet_path, columns=sorted(set(required_columns)))
        df = df[(df[self.date_column] >= real_start_date) & (df[self.date_column] <= real_end_date)]
        
        # 合并daily数据用于目标计算
        if self.daily_path and os.path.exists(self.daily_path) and not self.return_close_only:
            df = self._merge_daily_data(df, real_start_date, real_end_date)
        
        # 应用样本池过滤
        df = self._apply_sample_pool_filter(df)
        
        # 获取股票ID并排序
        stock_ids = df[self.code_column].unique()
        stock_ids.sort()
        self._stock_ids = pd.Index(stock_ids)
        
        # 重构索引
        full_idx = pd.MultiIndex.from_product([self._dates, self._stock_ids], 
                                              names=[self.date_column, self.code_column])
        df = df.set_index([self.date_column, self.code_column])
        df = df.reindex(full_idx)
        
        # 如果只返回收盘价，直接返回
        if self.return_close_only:
            close_data = df[self.close_column].unstack(level=1)
            close_tensor = torch.tensor(close_data.values, dtype=torch.float32)
            return close_tensor.unsqueeze(0), self._dates, self._stock_ids
        
        # 构建特征张量
        return self._build_feature_tensors(df, resolved_feature_columns)
    
    def _available_columns(self, parquet_path: str) -> List[str]:
        """获取可用列"""
        return pq.ParquetFile(parquet_path).schema.names
    
    def _resolve_feature_columns(self, available_columns: List[str]) -> Dict[str, Optional[str]]:
        """解析特征列"""
        resolved = {}
        available = set(available_columns)
        
        for feature in self.features:
            candidates = [
                f"{feature}_robust",
                f"{feature}_valid", 
                f"{feature}_raw",
                feature,
            ]
            resolved[feature] = next((col for col in candidates if col in available), None)
        return resolved
    
    def _merge_daily_data(self, df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        """合并daily数据"""
        logger.info(f"Merging daily data from {self.daily_path}")
        df_market = pd.read_parquet(self.daily_path, 
                                   columns=[self.code_column, self.date_column, self.close_column])
        df_market[self.date_column] = df_market[self.date_column].astype(str)
        df_market = df_market[(df_market[self.date_column] >= start_date) & 
                             (df_market[self.date_column] <= end_date)]
        
        if self.sample_pool:
            if isinstance(self.sample_pool, list):
                df_market = df_market[df_market[self.code_column].isin(self.sample_pool)]
            elif isinstance(self.sample_pool, dict) and self.sample_pool.get("type") == "dynamic":
                all_pool_stocks = set()
                for p in self.sample_pool["pools"].values():
                    all_pool_stocks.update(p)
                df_market = df_market[df_market[self.code_column].isin(all_pool_stocks)]
        
        return pd.merge(df, df_market, on=[self.code_column, self.date_column], how='left', suffixes=('', '_daily'))
    
    def _apply_sample_pool_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用样本池过滤"""
        if not self.sample_pool:
            return df
        
        if isinstance(self.sample_pool, list):
            # 静态池
            return df[df[self.code_column].isin(self.sample_pool)]
        
        elif isinstance(self.sample_pool, dict) and self.sample_pool.get("type") == "dynamic":
            logger.info("Applying dynamic pool filtering...")
            pools = self.sample_pool["pools"]
            
            # 获取所有池中的股票
            all_pool_stocks = set()
            for p in pools.values():
                all_pool_stocks.update(p)
            df = df[df[self.code_column].isin(all_pool_stocks)]
            
            # 按日期分组处理
            pool_dates = sorted(pools.keys())
            df_dates_str = df[self.date_column].astype(str)
            mask = pd.Series(False, index=df.index)
            
            for i in range(len(pool_dates)):
                current_rebal = pool_dates[i]
                next_rebal = pool_dates[i+1] if i + 1 < len(pool_dates) else "99991231"
                
                period_mask = (df_dates_str >= current_rebal) & (df_dates_str < next_rebal)
                if period_mask.any():
                    stocks_in_pool = pools[current_rebal]
                    stock_mask = df[self.code_column].isin(stocks_in_pool)
                    mask |= (period_mask & stock_mask)
            
            # 对不在池中的股票设置NaN
            cols_to_mask = [c for c in df.columns if c not in [self.code_column, self.date_column]]
            df.loc[~mask, cols_to_mask] = np.nan
            logger.info("Dynamic filtering applied as NaN mask.")
            
        return df
    
    def _build_feature_tensors(self, df: pd.DataFrame, resolved_feature_columns: Dict[str, Optional[str]]) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """构建特征张量"""
        feature_tensors = {}
        dates = self._dates
        stocks = self._stock_ids
        
        # 处理特征
        for feature in self.features:
            col_name = resolved_feature_columns.get(feature)
            
            if col_name not in df.columns:
                logger.warning(f"Feature {feature} not found in parquet columns. Filling with NaN.")
                tensor = torch.full((len(dates), len(stocks)), float('nan'))
            else:
                feature_data = df[col_name]
                feature_df = feature_data.unstack(level=1)
                # 应用安全填充
                feature_df = feature_df.ffill(limit=5)
                tensor = torch.tensor(feature_df.values, dtype=torch.float32)
            
            feature_tensors[feature.upper()] = tensor
        
        # 特殊处理CLOSE
        if 'CLOSE' not in feature_tensors and self.close_column in df.columns:
            close_data = df[self.close_column].unstack(level=1)
            feature_tensors['CLOSE'] = torch.tensor(close_data.values, dtype=torch.float32)
        elif 'CLOSE' not in feature_tensors and 'ret_cc_1d' in df.columns:
            # 从收益率合成价格
            ret_data = df['ret_cc_1d'].unstack(level=1).fillna(0.0)
            price_index = (1 + ret_data).cumprod()
            feature_tensors['CLOSE'] = torch.tensor(price_index.values, dtype=torch.float32)
        
        # 构建最终张量
        return self._assemble_final_tensor(feature_tensors, dates, stocks)
    
    def _assemble_final_tensor(self, feature_tensors: Dict[str, torch.Tensor], 
                             dates: pd.Index, stocks: pd.Index) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """组装最终张量"""
        # 创建特征枚举
        feature_enum = self.registry_manager.create_feature_enum(self.domain, self.status_filter)
        
        # 按枚举顺序创建张量列表
        sorted_features = sorted(feature_enum.__members__.items(), key=lambda x: x[1])
        final_tensor_list = []
        
        for name, idx in sorted_features:
            if name in feature_tensors:
                final_tensor_list.append(feature_tensors[name])
            elif name == 'CLOSE' and self.close_column:
                # 标准CLOSE处理
                if 'CLOSE' in feature_tensors:
                    final_tensor_list.append(feature_tensors['CLOSE'])
                else:
                    logger.warning(f"CLOSE not found, filling with NaN")
                    final_tensor_list.append(torch.full((len(dates), len(stocks)), float('nan')))
            else:
                logger.warning(f"Feature {name} from Enum not found, filling with NaN")
                final_tensor_list.append(torch.full((len(dates), len(stocks)), float('nan')))
        
        # 堆叠张量 (n_features, n_days, n_stocks)
        data = torch.stack(final_tensor_list, dim=0)
        # 转置为 (n_days, n_features, n_stocks)
        data = data.permute(1, 0, 2)
        
        return data.to(self.device), dates, stocks
    
    # 属性访问
    @property
    def n_features(self) -> int:
        return len(self.features)
    
    @property
    def n_stocks(self) -> int:
        return len(self._stock_ids)
    
    @property
    def n_days(self) -> int:
        return len(self._dates) - self.max_backtrack_days - self.max_future_days
    
    @property
    def dates(self) -> pd.Index:
        return self._dates
    
    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids
    
    @property
    def mask(self) -> torch.Tensor:
        """
        返回布尔掩码，形状为 (n_days, n_stocks)
        True表示数据可用
        """
        return ~torch.isnan(self.data[:, 0, :])
    
    @property
    def feature_map(self) -> Dict[str, IntEnum]:
        """
        返回特征映射，用于表达式解析器
        排除保留的CLOSE成员以防止在表达式中使用
        """
        enum_cls = self.registry_manager.create_feature_enum(self.domain, self.status_filter)
        mapping = {}
        for member in enum_cls:
            if member.name != 'CLOSE':
                mapping[f"${member.name.lower()}"] = member
        return mapping
    
    # 数据转换方法
    def make_dataframe(self, data: Union[torch.Tensor, List[torch.Tensor]], 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """将张量转换回DataFrame"""
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        
        n_days, n_stocks, n_columns = data.shape
        
        # 处理日期对齐
        date_index = self._dates
        if len(date_index) != n_days:
            if len(date_index) > n_days:
                date_index = date_index[-n_days:]
            else:
                raise ValueError(f"Data length {n_days} > Dates length {len(date_index)}")
        
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)