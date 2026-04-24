"""Arrow stream utilities for efficient data transfer."""

import pyarrow as pa
from typing import Iterator, Optional, Dict, Any, List
import numpy as np
from datetime import datetime

class ArrowStreamProcessor:
    """Process Arrow RecordBatches for factor mining workflows."""
    
    def __init__(self, date_col: str = "trade_date", symbol_col: str = "ts_code"):
        self.date_col = date_col
        self.symbol_col = symbol_col
    
    def batches_to_panels(self, batches: Iterator[pa.RecordBatch], 
                         feature_cols: List[str]) -> Iterator[Dict[str, Any]]:
        """Convert Arrow batches to panel format for factor computation."""
        
        for batch in batches:
            if batch.num_rows == 0:
                continue
            
            # Extract dates and symbols
            dates = batch.column(self.date_col).to_numpy()
            symbols = batch.column(self.symbol_col).to_numpy()
            
            # Convert dates to int32 YYYYMMDD format
            if dates.dtype == np.object_ or dates.dtype.type == str:
                # String dates, convert to int
                date_ints = np.array([int(d.replace("-", "")) for d in dates])
            else:
                # Already numeric
                date_ints = dates.astype(np.int32)
            
            # Extract features
            features = {}
            for col in feature_cols:
                if col in batch.column_names:
                    col_data = batch.column(col).to_numpy()
                    # Handle NaN values
                    if col_data.dtype in [np.float32, np.float64]:
                        features[col] = np.nan_to_num(col_data, nan=0.0)
                    else:
                        features[col] = col_data
            
            yield {
                "dates": date_ints,
                "symbols": symbols,
                "features": features,
                "batch_size": batch.num_rows
            }
    
    def create_market_panel_batch(self, batch: pa.RecordBatch,
                                 feature_cols: List[str]) -> Dict[str, Any]:
        """Create standardized market panel batch from Arrow batch."""
        
        # Get unique dates and symbols
        dates_array = batch.column(self.date_col).to_numpy()
        symbols_array = batch.column(self.symbol_col).to_numpy()
        
        unique_dates = np.unique(dates_array)
        unique_symbols = np.unique(symbols_array)
        
        # Create date-to-index and symbol-to-index mappings
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        symbol_to_idx = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
        
        # Initialize panel data structure
        n_dates = len(unique_dates)
        n_symbols = len(unique_symbols)
        n_features = len(feature_cols)
        
        # Create 3D tensor: [dates, symbols, features]
        panel_data = np.full((n_dates, n_symbols, n_features), np.nan, dtype=np.float32)
        mask = np.zeros((n_dates, n_symbols), dtype=bool)
        
        # Fill panel data
        for row_idx in range(batch.num_rows):
            date_val = dates_array[row_idx]
            symbol_val = symbols_array[row_idx]
            
            if date_val in date_to_idx and symbol_val in symbol_to_idx:
                date_idx = date_to_idx[date_val]
                symbol_idx = symbol_to_idx[symbol_val]
                
                # Set mask to True (valid data point)
                mask[date_idx, symbol_idx] = True
                
                # Fill feature values
                for feat_idx, col in enumerate(feature_cols):
                    if col in batch.column_names:
                        col_data = batch.column(col).to_numpy()
                        value = col_data[row_idx]
                        
                        # Handle different data types
                        if isinstance(value, (int, float)):
                            panel_data[date_idx, symbol_idx, feat_idx] = float(value)
                        elif isinstance(value, str):
                            try:
                                panel_data[date_idx, symbol_idx, feat_idx] = float(value)
                            except (ValueError, TypeError):
                                panel_data[date_idx, symbol_idx, feat_idx] = np.nan
                        else:
                            panel_data[date_idx, symbol_idx, feat_idx] = np.nan
        
        return {
            "dates": unique_dates,
            "symbols": unique_symbols,
            "values": panel_data,
            "mask": mask,
            "fields": feature_cols,
            "shape": (n_dates, n_symbols, n_features)
        }
    
    def validate_batch_consistency(self, batch: pa.RecordBatch,
                                 expected_schema: Optional[Dict[str, str]] = None) -> List[str]:
        """Validate Arrow batch consistency."""
        
        errors = []
        
        # Check required columns
        required_cols = [self.date_col, self.symbol_col]
        for col in required_cols:
            if col not in batch.column_names:
                errors.append(f"Missing required column: {col}")
        
        # Check for null values in key columns
        for col in required_cols:
            if col in batch.column_names:
                col_data = batch.column(col)
                if col_data.null_count > 0:
                    errors.append(f"Null values found in key column: {col}")
        
        # Check data types
        if expected_schema:
            for col_name, expected_type in expected_schema.items():
                if col_name in batch.column_names:
                    actual_type = batch.column(col_name).type
                    # Simple type checking (can be enhanced)
                    if str(actual_type) != expected_type:
                        errors.append(f"Type mismatch for {col_name}: expected {expected_type}, got {actual_type}")
        
        # Check for empty batch
        if batch.num_rows == 0:
            errors.append("Empty batch")
        
        return errors
    
    def estimate_memory_usage(self, batch: pa.RecordBatch) -> Dict[str, int]:
        """Estimate memory usage of Arrow batch."""
        
        total_bytes = 0
        column_usage = {}
        
        for col_name in batch.column_names:
            col = batch.column(col_name)
            col_bytes = col.nbytes
            total_bytes += col_bytes
            column_usage[col_name] = col_bytes
        
        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "column_usage": column_usage,
            "num_rows": batch.num_rows,
            "num_columns": batch.num_columns
        }
    
    def convert_date_format(self, dates: np.ndarray, target_format: str = "int32") -> np.ndarray:
        """Convert date array to specified format."""
        
        if target_format == "int32":
            if dates.dtype == np.object_ or dates.dtype.type == str:
                # Convert string dates to int32 YYYYMMDD
                return np.array([int(d.replace("-", "")) for d in dates], dtype=np.int32)
            else:
                return dates.astype(np.int32)
        
        elif target_format == "datetime":
            if dates.dtype == np.object_ or dates.dtype.type == str:
                return np.array([datetime.strptime(d, "%Y-%m-%d") for d in dates])
            else:
                # Assume dates are already in YYYYMMDD format
                return np.array([datetime.strptime(str(d), "%Y%m%d") for d in dates])
        
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

class BatchOptimizer:
    """Optimize batch sizes for memory and performance."""
    
    def __init__(self, target_memory_mb: int = 512, max_rows_per_batch: int = 500_000):
        self.target_memory_mb = target_memory_mb
        self.max_rows_per_batch = max_rows_per_batch
        self.target_memory_bytes = target_memory_mb * 1024 * 1024
    
    def calculate_optimal_batch_size(self, sample_batch: pa.RecordBatch) -> int:
        """Calculate optimal batch size based on memory usage."""
        
        if sample_batch.num_rows == 0:
            return self.max_rows_per_batch
        
        # Estimate memory per row
        memory_usage = 0
        for col_name in sample_batch.column_names:
            col = sample_batch.column(col_name)
            memory_usage += col.nbytes
        
        bytes_per_row = memory_usage / sample_batch.num_rows
        
        # Calculate optimal batch size
        optimal_size = int(self.target_memory_bytes / bytes_per_row)
        
        # Apply constraints
        optimal_size = min(optimal_size, self.max_rows_per_batch)
        optimal_size = max(optimal_size, 1000)  # Minimum batch size
        
        return optimal_size
    
    def split_large_batch(self, batch: pa.RecordBatch, target_size: int) -> List[pa.RecordBatch]:
        """Split large batch into smaller batches."""
        
        if batch.num_rows <= target_size:
            return [batch]
        
        batches = []
        num_batches = (batch.num_rows + target_size - 1) // target_size
        
        for i in range(num_batches):
            start_idx = i * target_size
            end_idx = min((i + 1) * target_size, batch.num_rows)
            
            # Slice the batch
            sliced_batch = batch.slice(start_idx, end_idx - start_idx)
            batches.append(sliced_batch)
        
        return batches