import pandas as pd
import torch
import numpy as np
import os
import pyarrow.parquet as pq
from utils.path_utils import map_path, FACTOR_READY_DIR
from typing import List, Optional, Union, Tuple, Dict
from enum import IntEnum
from .feature_registry_manager import FeatureRegistryManager
from .sample_pool_builder import SamplePoolBuilder
from .dataset_meta import DatasetMeta

class ParquetFeatureLoader:
    """
    Loads feature data from Parquet files based on domain and registry.
    Mimics the interface of alphagen_qlib.stock_data.StockData.
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
                 date_column: str = "trade_date",
                 code_column: str = "ts_code",
                 close_column: str = "close",
                 ):
        self.domain = domain
        self.start_time = str(start_time).replace("-", "")
        self.end_time = str(end_time).replace("-", "")
        self.device = device
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.status_filter = status_filter
        self.use_filled = use_filled
        self.dataset_meta = dataset_meta
        
        # 优先从 dataset_meta 获取配置
        if dataset_meta is not None:
            meta_kwargs = dataset_meta.to_loader_kwargs()
            if data_dir is None and "data_dir" in meta_kwargs:
                data_dir = meta_kwargs["data_dir"]
            if date_column == "trade_date" and "date_column" in meta_kwargs:
                date_column = meta_kwargs["date_column"]
            if code_column == "ts_code" and "code_column" in meta_kwargs:
                code_column = meta_kwargs["code_column"]
            if close_column == "close" and "close_column" in meta_kwargs:
                close_column = meta_kwargs["close_column"]
            if pool_path is None and "pool_path" in meta_kwargs:
                pool_path = meta_kwargs["pool_path"]
            if feature_file_map_raw is None and "feature_file_map_raw" in meta_kwargs:
                feature_file_map_raw = meta_kwargs["feature_file_map_raw"]
            if feature_file_map_filled is None and "feature_file_map_filled" in meta_kwargs:
                feature_file_map_filled = meta_kwargs["feature_file_map_filled"]
        
        self.date_column = date_column
        self.code_column = code_column
        self.close_column = close_column

        if registry_manager is None:
            self.registry_manager = FeatureRegistryManager()
        else:
            self.registry_manager = registry_manager

        if data_dir is None:
             # Default path
             self.data_dir = FACTOR_READY_DIR
        else:
            self.data_dir = data_dir

        self.daily_path = daily_path
        if self.daily_path is None:
             # Default path
             self.daily_path = map_path(r"data/basic/daily.parquet")

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

        # Load Pool
        if pool_path and os.path.exists(pool_path):
             with open(pool_path, 'r') as f:
                 import json
                 self.sample_pool = json.load(f)
        else:
             # Fallback: build a pool or use all
             print("Warning: No sample pool provided. Using all stocks from data.")
             self.sample_pool = None

        # Get features
        self.features = self.registry_manager.get_features_by_domain(domain, status_filter)
        if not self.features:
            raise ValueError(f"No active features found for domain {domain}")

        self.data, self._dates, self._stock_ids = self._load_data()

    def _resolve_parquet_path(self) -> str:
        filled_dir = os.path.join(os.path.dirname(self.data_dir), "factor_ready_filled")
        filled_path = os.path.join(filled_dir, self.feature_file_map_filled.get(self.domain, ""))

        if self.use_filled and os.path.exists(filled_path):
            print(f"Using PRE-FILLED data from {filled_path}")
            return filled_path

        raw_path = os.path.join(self.data_dir, self.feature_file_map_raw.get(self.domain, ""))
        print(f"Using RAW data from {raw_path}")
        return raw_path

    def _available_columns(self, parquet_path: str) -> List[str]:
        return pq.ParquetFile(parquet_path).schema.names

    def _resolve_feature_columns(self, available_columns: List[str]) -> Dict[str, Optional[str]]:
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

    def _load_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        parquet_path = self._resolve_parquet_path()
            
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        print(f"Loading data from {parquet_path}...")
        # Read all dates first to determine range
        df_dates = pd.read_parquet(parquet_path, columns=[self.date_column])
        all_dates = df_dates[self.date_column].unique()
        all_dates.sort()
        all_dates_str = all_dates.astype(str)
        
        # Find indices
        # Ensure start_time and end_time are in format
        try:
            start_idx = next(i for i, d in enumerate(all_dates_str) if d >= self.start_time)
            end_idx = next(i for i, d in enumerate(all_dates_str) if d > self.end_time) - 1
        except StopIteration:
            raise ValueError(f"Date range {self.start_time}-{self.end_time} not covered by data")

        if end_idx < start_idx:
             raise ValueError("end_time before start_time")

        # Expand range
        real_start_idx = max(0, start_idx - self.max_backtrack_days)
        real_end_idx = min(len(all_dates), end_idx + self.max_future_days + 1)
        
        real_start_date = all_dates[real_start_idx]
        real_end_date = all_dates[real_end_idx - 1] # Inclusive for comparison
        
        print(f"Loading expanded range: {real_start_date} to {real_end_date} (Request: {self.start_time}-{self.end_time})")
        
        # Keep track of the full date range for alignment
        self._dates = pd.Index(all_dates[real_start_idx : real_end_idx])

        available_columns = self._available_columns(parquet_path)
        resolved_feature_columns = self._resolve_feature_columns(available_columns)
        required_columns = [self.date_column, self.code_column]
        required_columns.extend(
            col for col in resolved_feature_columns.values() if col is not None
        )
        df = pd.read_parquet(parquet_path, columns=sorted(set(required_columns)))
        # Filter by expanded date range
        df = df[(df[self.date_column] >= real_start_date) & (df[self.date_column] <= real_end_date)]
        
        # 2. Load Market Data (Specifically CLOSE from daily.parquet for target calculation)
        if self.daily_path and os.path.exists(self.daily_path):
            print(f"Loading market data from {self.daily_path} for target calculation...")
            df_market = pd.read_parquet(self.daily_path, columns=[self.code_column, self.date_column, self.close_column])
            df_market[self.date_column] = df_market[self.date_column].astype(str)
            df_market = df_market[(df_market[self.date_column] >= real_start_date) & (df_market[self.date_column] <= real_end_date)]
            if self.sample_pool:
                if isinstance(self.sample_pool, list):
                    df_market = df_market[df_market[self.code_column].isin(self.sample_pool)]
                elif isinstance(self.sample_pool, dict) and self.sample_pool.get("type") == "dynamic":
                    all_pool_stocks = set()
                    for p in self.sample_pool["pools"].values():
                        all_pool_stocks.update(p)
                    df_market = df_market[df_market[self.code_column].isin(all_pool_stocks)]
            
            # Merge close into main df
            df = pd.merge(df, df_market, on=[self.code_column, self.date_column], how='left')
        elif self.domain != 'A':
            # Fallback to domain A if daily.parquet is missing (though daily.parquet is preferred)
            path_a = os.path.join(self.data_dir, self.feature_file_map_raw['A'])
            if os.path.exists(path_a):
                print(f"Loading auxiliary data from {path_a} for target calculation...")
                df_a = pd.read_parquet(path_a, columns=[self.code_column, self.date_column, 'ret_cc_1d'])
                df = pd.merge(df, df_a, on=[self.code_column, self.date_column], how='left')
            else:
                print("Warning: Domain A data not found. Target calculation may fail (missing price reference).")
        
        # Filter by pool
        if self.sample_pool:
            if isinstance(self.sample_pool, list):
                # Static Pool
                df = df[df[self.code_column].isin(self.sample_pool)]
            elif isinstance(self.sample_pool, dict) and self.sample_pool.get("type") == "dynamic":
                print("Applying dynamic pool filtering...")
                pools = self.sample_pool["pools"]
                
                # First, keep only stocks that appear in ANY pool to save memory
                all_pool_stocks = set()
                for p in pools.values():
                    all_pool_stocks.update(p)
                df = df[df[self.code_column].isin(all_pool_stocks)]
                
                # Handle date type
                if len(df) > 0:
                    # Sort pools by date key to ensure correct temporal matching
                    pool_dates = sorted(pools.keys())
                    
                    # Convert df['trade_date'] to string if needed for comparison
                    df_dates_str = df[self.date_column].astype(str)
                    mask = pd.Series(False, index=df.index)
                    
                    # Group by rebalance intervals
                    for i in range(len(pool_dates)):
                        current_rebal = pool_dates[i]
                        next_rebal = pool_dates[i+1] if i + 1 < len(pool_dates) else "99991231"
                        
                        # Filter for dates within this rebalance period
                        period_mask = (df_dates_str >= current_rebal) & (df_dates_str < next_rebal)
                        if period_mask.any():
                             stocks_in_pool = pools[current_rebal]
                             stock_mask = df[self.code_column].isin(stocks_in_pool)
                             mask |= (period_mask & stock_mask)
                
                # Instead of dropping rows, we set values to NaN for stocks NOT in the pool
                # This ensures we keep all dates and stocks in the resulting tensor
                # and align correctly with backtrack/future logic.
                cols_to_mask = [c for c in df.columns if c not in [self.code_column, self.date_column]]
                df.loc[~mask, cols_to_mask] = np.nan
                print(f"Dynamic filtering (Jan/Jul) applied as NaN mask.")
            else:
                 # Fallback for other dict formats (if any) or assume it's a list wrapped in dict?
                 # If it is a simple dict not marked dynamic, treat keys as something else?
                 # For now, if it's not dynamic dict, ignore or warn.
                 # Actually, old code might assume list.
                 if isinstance(self.sample_pool, list):
                      df = df[df[self.code_column].isin(self.sample_pool)]
        
        # Sort and get unique stock_ids from df
        # (df has been filtered by domain range, and potentially by sample pool)
        stock_ids = df[self.code_column].unique()
        stock_ids.sort()
        self._stock_ids = pd.Index(stock_ids)
        
        # Use the FULL date range for the pivot to ensure tensor alignment
        dates = self._dates
        
        full_idx = pd.MultiIndex.from_product([dates, stock_ids], names=[self.date_column, self.code_column])
        df = df.set_index([self.date_column, self.code_column])
        df = df.reindex(full_idx) 

        # 3. Collect Feature Tensors
        feature_tensors = {}
        
        # Domain Features
        for feature in self.features:
            col_name = resolved_feature_columns.get(feature)
                
            if col_name not in df.columns:
                print(f"Warning: Feature {feature} (or its robust/valid/raw variants) not in parquet columns. Filling with NaN.")
                tensor = torch.full((len(dates), len(stock_ids)), float('nan'))
            else:
                feature_data = df[col_name]
                feature_df = feature_data.unstack(level=1)
                # If using pre-filled data, the NaNs should already be minimal.
                # However, we still apply a small safety fill here to handle any remaining edge cases.
                feature_df = feature_df.ffill(limit=5)
                tensor = torch.tensor(feature_df.values, dtype=torch.float32)
            
            feature_tensors[feature.upper()] = tensor
            
        # Special case: CLOSE might be needed for target calculation in scripts, 
        # even if not in registry for domain. 
        # But per user request, we don't include it in the features tensor unless in registry.
        if 'CLOSE' not in feature_tensors and self.close_column in df.columns:
             feature_tensors['CLOSE'] = torch.tensor(df[self.close_column].unstack(level=1).values, dtype=torch.float32)
        elif 'CLOSE' not in feature_tensors and 'ret_cc_1d' in df.columns:
             ret_data = df['ret_cc_1d'].unstack(level=1).fillna(0.0)
             price_index = (1 + ret_data).cumprod()
             feature_tensors['CLOSE'] = torch.tensor(price_index.values, dtype=torch.float32)

        # 4. Assemble Final Tensor
        # We respect the order defined by the dynamic FeatureType Enum
        feature_enum = self.registry_manager.create_feature_enum(self.domain, ['active', 'watch'])
        
        # Create a list of tensors in order of Enum values
        sorted_features = sorted(feature_enum.__members__.items(), key=lambda x: x[1])
        final_tensor_list = []
        
        for name, idx in sorted_features:
            if name in feature_tensors:
                final_tensor_list.append(feature_tensors[name])
            elif name == 'CLOSE' and self.close_column in df.columns:
                # Standard CLOSE handled here
                tensor = torch.tensor(df[self.close_column].unstack(level=1).values, dtype=torch.float32)
                final_tensor_list.append(tensor)
            elif name == 'CLOSE' and 'ret_cc_1d' in df.columns:
                # Fallback to synthesis for CLOSE if missing from daily.parquet
                ret_data = df['ret_cc_1d'].unstack(level=1).fillna(0.0)
                price_index = (1 + ret_data).cumprod()
                final_tensor_list.append(torch.tensor(price_index.values, dtype=torch.float32))
            else:
                print(f"Warning: Feature {name} from Enum not found in tensors. Filling with NaN.")
                final_tensor_list.append(torch.full((len(dates), len(stock_ids)), float('nan')))
                
        data = torch.stack(final_tensor_list, dim=0) # (n_features, n_days, n_stocks)
        data = data.permute(1, 0, 2) # (n_days, n_features, n_stocks)
        
        return data.to(self.device), self._dates, self._stock_ids

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_stocks(self) -> int:
        return len(self._stock_ids)

    @property
    def n_days(self) -> int:
        # Number of trading days in the requested [start_time, end_time] range
        return len(self._dates) - self.max_backtrack_days - self.max_future_days

    @property
    def dates(self) -> pd.Index:
        # Return ALL dates (including backtrack/future) to match StockData interface
        return self._dates

    @property
    def mask(self) -> torch.Tensor:
        """
        Return a boolean mask of shape (n_days, n_stocks).
        True where data is available (listed days).
        """
        # We use the first feature as a proxy for 'is listed'
        # self.data has shape (n_days, n_features, n_stocks)
        return ~torch.isnan(self.data[:, 0, :])
        
    @property
    def feature_map(self) -> Dict[str, IntEnum]:
        """
        Return a map of '$feature_name' -> FeatureType member.
        Used by ExpressionParser.
        This only exposes features defined in the registry for the current domain.
        EXCLUDES the reserved 'CLOSE' member to prevent LLM from using it in expressions.
        """
        enum_cls = self.registry_manager.create_feature_enum(self.domain, ['active', 'watch'])
        mapping = {}
        for member in enum_cls:
            if member.name != 'CLOSE':
                mapping[f"${member.name.lower()}"] = member
        return mapping

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert tensor back to DataFrame, mimicking StockData.make_dataframe.
        """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        
        # Note: self.n_days property is based on _dates length
        # StockData uses shape[0] - backtrack - future
        # We need to be careful with alignment
        
        if self.n_stocks != n_stocks:
             # This check might fail if data is subset
             pass

        # Reconstruct index
        # We need to slice _dates if we want to match exactly like StockData
        # But for now, we assume data aligns with _dates
        date_index = self._dates
        
        # If lengths mismatch, truncate or error
        if len(date_index) != n_days:
            # Try to align if backtrack/future logic was used in StockData (we didn't implement it fully)
            # We just return based on what we have
            if len(date_index) > n_days:
                 date_index = date_index[-n_days:]
            else:
                 raise ValueError(f"Data length {n_days} > Dates length {len(date_index)}")

        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
