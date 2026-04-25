import pandas as pd
import torch
import numpy as np
import os
import pyarrow.parquet as pq
import pyarrow as pa
import hashlib
import json
import logging
from datetime import datetime
from typing import List, Optional, Union, Tuple, Dict, Any, Literal, Iterator, Set
from dataclasses import dataclass
from pathlib import Path
from enum import IntEnum
from tqdm import tqdm
from .feature_registry_manager_v2 import FeatureRegistryManagerV2
from .dataset_meta import DatasetMeta

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LayerSource:
    layer: Literal["raw", "filled", "atomic"]
    source_type: Literal["duckdb_table", "parquet_file"]
    name: str
    date_col: str
    code_col: str
    available_columns: Set[str]

class ParquetFeatureLoaderV2:
    """
    升级版ParquetFeatureLoader，支持多层数据视图和缓存。
    作为“数据视图构造器”，支持 raw/filled/atomic 三层架构。
    默认使用 DuckDBDataHub 进行高性能数据读取。
    """
    
    def __init__(self,
                 domain: str,
                 start_time: str,
                 end_time: str,
                 registry_manager: FeatureRegistryManagerV2 = None,
                 dataset_meta: DatasetMeta = None,
                 device: torch.device = torch.device('cuda:0'),
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 status_filter: List[str] = ['active', 'watch'],
                 layers: List[str] = None,
                 read_mode: str = "stack",
                 cache_root: Optional[str] = None,
                 datahub: Optional[Any] = None,
                 segment_name: Optional[str] = None,
                 **kwargs
                 ):
        self.domain = domain
        self.start_time = str(start_time).replace("-", "")
        self.end_time = str(end_time).replace("-", "")
        self.device = device
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.status_filter = status_filter
        self.dataset_meta = dataset_meta
        self.layers = layers or ["raw", "atomic"]
        self.read_mode = read_mode
        self.cache_root = cache_root
        self.datahub = datahub
        self.segment_name = segment_name
        
        # 初始化注册管理器
        self.registry_manager = registry_manager or FeatureRegistryManagerV2()

        # 解析层源
        self.layer_sources = self._resolve_layer_sources()
        
        # 构建特征计划
        self.feature_plan = self._resolve_feature_plan()
        
        # 加载数据 (优先从缓存)
        self.data, self._dates, self._stock_ids = self._load_or_cache_data()

    def _resolve_layer_sources(self) -> Dict[str, LayerSource]:
        """解析每一层的数据源"""
        sources = {}
        if not self.dataset_meta:
            return sources
            
        meta_files = self.dataset_meta.raw.get("files", {})
        cols_config = self.dataset_meta.raw.get("columns", {})
        date_col = cols_config.get("date", "trade_date")
        code_col = cols_config.get("code", "ts_code")
        
        for layer in ["raw", "filled", "atomic"]:
            layer_info = meta_files.get(layer)
            if not layer_info:
                logger.debug(f"Layer {layer} not found in meta_files")
                continue
            
            name = layer_info.get(self.domain) if isinstance(layer_info, dict) else layer_info
            if not name:
                logger.debug(f"No name found for domain {self.domain} in layer {layer}")
                continue
            
            # 优先检查 DuckDB
            if self.datahub:
                try:
                    tables = self.datahub.list_tables()
                    logger.debug(f"Searching for {name} in DuckDB tables (total {len(tables)})")
                    
                    matched_table = None
                    search_names = [name]
                    if "." in name:
                        # 尝试去掉前缀
                        short_name = name.split(".")[-1]
                        search_names.append(short_name)
                    
                    # 尝试大小写敏感和不敏感匹配
                    for sn in search_names:
                        # 精确匹配
                        if sn in tables:
                            matched_table = sn
                            break
                        # 大小写不敏感匹配
                        for t in tables:
                            if t.lower() == sn.lower():
                                matched_table = t
                                break
                        if matched_table: break

                    if matched_table:
                        # 检查列
                        try:
                            # 处理带 Schema 的名称：要么不加引号，要么分开加
                            if "." in matched_table:
                                parts = matched_table.split(".")
                                safe_name = ".".join([f'"{p}"' for p in parts])
                            else:
                                safe_name = f'"{matched_table}"'
                                
                            res = self.datahub.execute_sql(f"SELECT * FROM {safe_name} LIMIT 0")
                            available_cols = set(res.column_names)
                            sources[layer] = LayerSource(
                                layer=layer, source_type="duckdb_table", name=matched_table,
                                date_col=date_col, code_col=code_col, available_columns=available_cols
                            )
                            logger.info(f"Resolved layer {layer} from DuckDB table {matched_table}")
                        except Exception as e:
                            logger.warning(f"Failed to get columns for table {matched_table}: {e}")
                    else:
                        logger.debug(f"Table {name} not found in DuckDB tables")
                except Exception as e:
                    logger.warning(f"Error checking DuckDB tables: {e}")

            if layer not in sources:
                # 检查 Parquet 文件
                # 尝试多个可能的位置
                data_dir = Path(self.dataset_meta.raw.get("data_dir", "."))
                potential_names = [name]
                if "." in name: potential_names.append(name.split(".")[-1])
                
                path = None
                for pn in potential_names:
                    p = data_dir / pn
                    if not p.suffix: p = p.with_suffix(".parquet")
                    if p.exists():
                        path = p
                        break
                
                if path:
                    try:
                        available_cols = set(pq.ParquetFile(path).schema.names)
                        sources[layer] = LayerSource(
                            layer=layer, source_type="parquet_file", name=str(path),
                            date_col=date_col, code_col=code_col, available_columns=available_cols
                        )
                        logger.info(f"Resolved layer {layer} from Parquet file {path}")
                    except Exception as e:
                        logger.warning(f"Error reading Parquet file {path}: {e}")
                else:
                    logger.debug(f"Parquet file for {name} does not exist in {data_dir}")
        
        if not sources:
            logger.error(f"Failed to resolve any layer sources for domain {self.domain}")
        return sources


    def _resolve_feature_plan(self) -> List[Tuple[str, str, str]]:
        """
        确定每个特征从哪个层、哪个列读取。
        返回 [(feature_name, layer, source_column)]
        """
        plan = []
        feature_enum = self.registry_manager.create_feature_enum(
            self.domain, self.status_filter, self.layers
        )
        
        # 获取所有特征名
        all_features = [m.name for m in feature_enum]
        
        for feat in all_features:
            found = False
            # 根据 read_mode 决定搜索顺序
            search_layers = []
            if self.read_mode == "stack":
                # stack 模式：优先匹配注册表中的层，如果不匹配则搜索所有启用层
                feat_spec = self.registry_manager.get_feature_spec(feat)
                preferred_layer = feat_spec.layer if feat_spec else None
                
                search_layers = []
                if preferred_layer and preferred_layer in self.layers:
                    search_layers.append(preferred_layer)
                
                for layer in self.layers:
                    if layer not in search_layers:
                        search_layers.append(layer)
            elif self.read_mode == "raw":
                search_layers = ["raw"]
            elif self.read_mode == "filled":
                search_layers = ["filled"]
            elif self.read_mode == "atomic":
                search_layers = ["atomic"]
            else: # legacy mixed
                search_layers = ["filled", "raw", "atomic"]
                
            for layer in search_layers:
                if layer not in self.layer_sources: continue
                src = self.layer_sources[layer]
                
                # 匹配列名：优先使用 DatasetMeta 中的映射
                actual_col = None
                if self.dataset_meta:
                    actual_col = self.dataset_meta.columns.get(feat.lower())
                
                candidates = []
                if actual_col: candidates.append(actual_col)
                candidates.extend([feat, feat.lower(), f"{feat}_robust", f"{feat}_raw"])
                
                # 准备可用列名的映射（小写 -> 原始名）
                col_map = {c.lower(): c for c in src.available_columns}
                
                for cand in candidates:
                    cand_lower = cand.lower()
                    if cand_lower in col_map:
                        plan.append((feat, layer, col_map[cand_lower]))
                        found = True
                        break
                if found: break
            
            if not found:
                logger.warning(f"Feature {feat} not found in any enabled layers {self.layers}")
                plan.append((feat, "none", "none"))
                
        return plan

    def _load_or_cache_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """加载或缓存数据"""
        cache_key = self._get_cache_key()
        cache_paths = self._get_cache_paths(cache_key)
        
        if cache_paths and self._is_cache_valid(cache_paths):
            logger.info(f"Loading data view from cache: {cache_key}")
            return self._load_from_cache(cache_paths)
        
        logger.info(f"Building data view for cache key: {cache_key}")
        data, dates, stocks = self._build_data_view()
        
        if cache_paths:
            self._save_to_cache(data, dates, stocks, cache_paths)
            
        return data, dates, stocks

    def _get_cache_key(self) -> str:
        key_parts = [self.domain, self.start_time, self.end_time, self.read_mode, str(sorted(self.layers))]
        if self.segment_name: key_parts.append(self.segment_name)
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _get_cache_paths(self, cache_key: str) -> Dict[str, str]:
        if not self.cache_root: return {}
        root = Path(self.cache_root) / "data_views" / cache_key[:2] / cache_key
        root.mkdir(parents=True, exist_ok=True)
        return {
            "tensor": str(root / "tensor.pt"),
            "dates": str(root / "dates.npy"),
            "stocks": str(root / "stocks.npy"),
            "manifest": str(root / "manifest.json")
        }

    def _is_cache_valid(self, paths: Dict[str, str]) -> bool:
        return all(os.path.exists(p) for p in paths.values())

    def _load_from_cache(self, paths: Dict[str, str]) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        data = torch.load(paths["tensor"]).to(self.device)
        dates = pd.Index(np.load(paths["dates"]))
        stocks = pd.Index(np.load(paths["stocks"]))
        return data, dates, stocks

    def _save_to_cache(self, data: torch.Tensor, dates: pd.Index, stocks: pd.Index, paths: Dict[str, str]):
        torch.save(data.cpu(), paths["tensor"])
        np.save(paths["dates"], dates.values)
        np.save(paths["stocks"], stocks.values)
        with open(paths["manifest"], "w") as f:
            json.dump({"created_at": datetime.now().isoformat()}, f)

    def _build_data_view(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        """通过执行计划加载数据并组装张量"""
        # 1. 确定所有需要的层和列
        layer_to_cols = {}
        for feat, layer, col in self.feature_plan:
            if layer == "none": continue
            if layer not in layer_to_cols: layer_to_cols[layer] = set()
            layer_to_cols[layer].add(col)
            
        # 2. 确定日期范围
        all_dates = self._resolve_all_dates()
        if not all_dates:
            raise ValueError(f"No dates found in domain {self.domain}. Please check datahub or parquet files.")

        # 确定目标区间的索引
        try:
            start_idx = next(i for i, d in enumerate(all_dates) if d >= self.start_time)
        except StopIteration:
            start_idx = len(all_dates) - 1
            
        try:
            end_idx = next(i for i, d in enumerate(all_dates) if d > self.end_time) - 1
        except StopIteration:
            end_idx = len(all_dates) - 1
            
        if start_idx > end_idx:
            start_idx = end_idx
        
        # 目标日期序列
        target_dates = all_dates[start_idx : end_idx + 1]
        self._n_target_days = len(target_dates)
        
        # 构建完整的 buffer 日期序列 (backtrack + target + future)
        # 必须严格保证长度，即使数据源中没有那么多日期，也要补足（用 dummy 日期或直接 nan）
        # 这里我们直接从 all_dates 中切片，如果不足则向两端延伸
        
        real_start_idx = start_idx - self.max_backtrack_days
        real_end_idx = end_idx + self.max_future_days + 1
        
        buffer_dates = []
        # 处理左边界
        if real_start_idx < 0:
            # 补齐缺少的 backtrack 天数
            buffer_dates.extend([f"PRE_{i:04d}" for i in range(abs(real_start_idx))])
            buffer_dates.extend(all_dates[0 : end_idx + 1])
        else:
            buffer_dates.extend(all_dates[real_start_idx : end_idx + 1])
            
        # 处理右边界
        if real_end_idx > len(all_dates):
            # 补齐缺少的 future 天数
            curr_len = len(buffer_dates)
            needed = (self.max_backtrack_days + self._n_target_days + self.max_future_days) - curr_len
            if needed > 0:
                buffer_dates.extend(all_dates[end_idx + 1 :])
                buffer_dates.extend([f"POST_{i:04d}" for i in range(needed - (len(all_dates) - (end_idx + 1)))])
        else:
            buffer_dates = buffer_dates[:self.max_backtrack_days + self._n_target_days]
            buffer_dates.extend(all_dates[end_idx + 1 : real_end_idx])

        self._dates = pd.Index(buffer_dates)
        real_start_date = all_dates[max(0, real_start_idx)]
        real_end_date = all_dates[min(len(all_dates)-1, real_end_idx-1)]
        
        # 3. 加载每一层的数据
        layer_dfs = {}
        all_stocks = set()
        
        for layer, cols in layer_to_cols.items():
            src = self.layer_sources[layer]
            cols.add(src.date_col)
            cols.add(src.code_col)
            
            df = self._read_source(src, list(cols), real_start_date, real_end_date)
            if df.empty: continue
                
            df[src.date_col] = df[src.date_col].astype(str)
            all_stocks.update(df[src.code_col].unique())
            layer_dfs[layer] = df.set_index([src.date_col, src.code_col])
            
        if not all_stocks:
            self._stock_ids = pd.Index([])
        else:
            self._stock_ids = pd.Index(sorted(all_stocks))
            
        full_idx = pd.MultiIndex.from_product([self._dates, self._stock_ids])
        
        # 4. 组装特征张量
        feature_tensors = []
        for feat, layer, col in self.feature_plan:
            if layer == "none" or layer not in layer_dfs or self._stock_ids.empty:
                tensor = torch.full((len(self._dates), len(self._stock_ids)), float('nan'))
            else:
                df_layer = layer_dfs[layer]
                if col in df_layer.columns:
                    # reindex 会自动处理 buffer 中补充的 PRE/POST 日期为 NaN
                    feat_series = df_layer[col].reindex(full_idx)
                    feat_df = feat_series.unstack(level=1)
                    feat_df = feat_df.ffill(limit=5)
                    tensor = torch.tensor(feat_df.values, dtype=torch.float32)
                else:
                    tensor = torch.full((len(self._dates), len(self._stock_ids)), float('nan'))
            feature_tensors.append(tensor)
            
        if not feature_tensors:
             data = torch.empty((len(self._dates), 0, len(self._stock_ids)))
        else:
             data = torch.stack(feature_tensors, dim=0).permute(1, 0, 2)
             
        return data.to(self.device), self._dates, self._stock_ids

    def _quote_ident(self, name: str) -> str:
        if "." in name:
            return ".".join(f'"{p}"' for p in name.split("."))
        return f'"{name}"'

    def _resolve_all_dates(self) -> List[str]:
        if not self.layer_sources: return []
        src = next(iter(self.layer_sources.values()))
        try:
            if src.source_type == "duckdb_table":
                table = self._quote_ident(src.name)
                date_col = self._quote_ident(src.date_col)
                df = self.datahub.execute_sql(
                    f"SELECT DISTINCT {date_col} FROM {table} ORDER BY {date_col}"
                ).to_pandas()
                dates = df[src.date_col].astype(str).tolist()
            else:
                df = pd.read_parquet(src.name, columns=[src.date_col])
                dates = sorted(df[src.date_col].unique().astype(str).tolist())

            return [d for d in dates if len(d) >= 8]
        except Exception as e:
            logger.error(f"Error resolving all dates from {src.name}: {e}")
            return []

    def _read_source(self, src: LayerSource, cols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        safe_cols = ", ".join(self._quote_ident(c) for c in cols)
        safe_table = self._quote_ident(src.name)
        safe_date = self._quote_ident(src.date_col)

        if src.source_type == "duckdb_table":
            sql = f"SELECT {safe_cols} FROM {safe_table} WHERE {safe_date} BETWEEN '{start_date}' AND '{end_date}'"
            return self.datahub.execute_sql(sql).to_pandas()
        else:
            if self.datahub:
                sql = f"SELECT {safe_cols} FROM read_parquet('{src.name}') WHERE {safe_date} BETWEEN '{start_date}' AND '{end_date}'"
                return self.datahub.execute_sql(sql).to_pandas()
            else:
                df = pd.read_parquet(src.name, columns=cols)
                df[src.date_col] = df[src.date_col].astype(str)
                return df[(df[src.date_col] >= start_date) & (df[src.date_col] <= end_date)]

    @property
    def dates(self) -> pd.Index: return self._dates
    @property
    def stock_ids(self) -> pd.Index: return self._stock_ids
    @property
    def mask(self) -> torch.Tensor: return ~torch.isnan(self.data[:, 0, :])

    @property
    def n_days(self) -> int:
        """返回目标区间的长度，供 alphagen.Expression 使用"""
        return self._n_target_days

    @property
    def n_stocks(self) -> int:
        return len(self._stock_ids)

    def __getitem__(self, item: Union[int, IntEnum]) -> torch.Tensor:
        """支持通过索引或枚举获取特征张量"""
        if isinstance(item, IntEnum):
            idx = item.value
        else:
            idx = item
        return self.data[:, idx, :]

    def diagnose(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "layers": self.layers,
            "read_mode": self.read_mode,
            "n_dates": len(self._dates),
            "n_stocks": len(self._stock_ids),
            "n_features": len(self.feature_plan),
            "missing_features": [
                feat for feat, layer, col in self.feature_plan if layer == "none"
            ],
            "layer_sources": {
                layer: {
                    "source_type": src.source_type,
                    "name": src.name,
                    "date_col": src.date_col,
                    "code_col": src.code_col,
                    "n_cols": len(src.available_columns),
                }
                for layer, src in self.layer_sources.items()
            },
        }
