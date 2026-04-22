import os
import json
import torch
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

# Configure matplotlib for Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese characters
plt.rcParams['axes.unicode_minus'] = False     # Ensure minus sign is displayed correctly

# Project internal imports
from alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from alphagen_generic.feature_registry_manager import FeatureRegistryManager
from alphagen.data.tree import ExpressionParser
from utils.path_utils import map_path, PROJECT_ROOT

def calculate_stats(data_np: np.ndarray, valid_mask: np.ndarray = None) -> Dict:
    """
    Calculate statistics for a factor: range, 99% range, 95% range, and missing rate.
    If valid_mask is provided, missing_rate is calculated relative to valid entries.
    """
    if valid_mask is not None:
        # data_np and valid_mask should have shape (n_days, n_stocks)
        target_data = data_np[valid_mask]
    else:
        target_data = data_np.flatten()
        
    valid_data = target_data[~np.isnan(target_data)]
    
    if len(valid_data) == 0:
        return {
            "full_range": [None, None],
            "range_99": [None, None],
            "range_95": [None, None],
            "missing_rate": 1.0,
            "kurtosis": 0.0
        }
    
    # Missing rate within the valid mask (e.g. only listed days)
    missing_rate = np.isnan(target_data).mean()
    full_range = [float(valid_data.min()), float(valid_data.max())]
    
    p0_5, p99_5 = np.percentile(valid_data, [0.5, 99.5])
    range_99 = [float(p0_5), float(p99_5)]
    
    p2_5, p97_5 = np.percentile(valid_data, [2.5, 97.5])
    range_95 = [float(p2_5), float(p97_5)]
    
    # Calculate kurtosis
    mean = valid_data.mean()
    std = valid_data.std()
    if std > 1e-9:
        kurtosis = float(((valid_data - mean)**4).mean() / (std**4))
    else:
        kurtosis = 0.0
        
    return {
        "full_range": full_range,
        "range_99": range_99,
        "range_95": range_95,
        "missing_rate": float(missing_rate),
        "kurtosis": kurtosis
    }

def build_digged_factors(
    domain: str,
    json_path: str,
    output_dir: str,
    plot_dir: str,
    start_time: str = "20100101",
    end_time: str = "20251231",
    device: str = "cuda:0",
    do_plot: bool = False
):
    """
    Load expressions from JSON, evaluate them on full market data, and write to Parquet.
    """
    print(f"\n{'='*50}")
    print(f"Processing Domain: {domain}")
    print(f"{'='*50}")
    
    # 1. Setup paths
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"factor_data_{domain}.parquet")
    
    # 2. Load expressions and metadata
    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        pool_data = json.load(f)
        expr_strs = pool_data.get("exprs", [])
        weights = pool_data.get("weights", [])
        topics = pool_data.get("topics", [])
        descriptions = pool_data.get("descriptions", [])
        ics_ret = pool_data.get("ics_ret", [])
        icir = pool_data.get("icir", [])
        
    if not expr_strs:
        print(f"No expressions found in {json_path}")
        return
        
    print(f"Loaded {len(expr_strs)} expressions.")
    
    # 3. Initialize Data Loader (Full Market)
    registry_manager = FeatureRegistryManager()
    data_device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing ParquetFeatureLoader for {domain} (Full Market)...")
    loader = ParquetFeatureLoader(
        domain=domain,
        start_time=start_time,
        end_time=end_time,
        registry_manager=registry_manager,
        pool_path=None, # None means full market
        device=data_device
    )
    
    # 4. Setup Parser
    feature_map = loader.feature_map
    parser = ExpressionParser(feature_map=feature_map)
    
    # 5. Evaluate and Write
    all_dates = loader._dates
    stock_ids = loader._stock_ids
    
    results_data = {}
    stats_list = []
    
    # Determine alignment dynamically based on returned factor shape
    final_dates = None
    # We use a liberal mask: True if ANY feature is present (proxy for 'listed')
    # loader.data shape: (n_days, n_features, n_stocks)
    liberal_mask_torch = ~torch.isnan(loader.data).all(dim=1)
    liberal_mask_np = liberal_mask_torch.detach().cpu().numpy()
    current_liberal_mask = None

    print("Evaluating expressions...")
    for i, expr_str in enumerate(tqdm(expr_strs)):
        try:
            expr_obj = parser.parse(expr_str)
            # evaluate returns (n_days, n_stocks) tensor corresponding to the valid range
            factor_tensor = expr_obj.evaluate(loader)
            factor_np = factor_tensor.detach().cpu().numpy()
            
            # Per user request: minimize missing data. 
            # Aggressive two-step filling for the final factor:
            factor_df_temp = pd.DataFrame(factor_np)
            # 1. Temporal ffill (long limit to bridge gaps)
            factor_df_temp = factor_df_temp.ffill(limit=30).bfill(limit=10)
            # 2. Cross-sectional median fill (crucial for < 1% missing rate)
            # This ensures that even if a stock has no data, it gets the market average
            day_median = factor_df_temp.median(axis=1)
            factor_df_temp = factor_df_temp.T.fillna(day_median).T.fillna(0.0)
            
            factor_np = factor_df_temp.values
            
            # Align dates and mask based on actual returned days
            if final_dates is None:
                n_days_factor = factor_np.shape[0]
                # Assuming evaluate returns the period just before the future buffer
                # Standard: total_days - backtrack - future
                # Precision alignment:
                end_idx = len(all_dates) - loader.max_future_days
                start_idx = end_idx - n_days_factor
                final_dates = all_dates[start_idx:end_idx]
                current_liberal_mask = liberal_mask_np[start_idx:end_idx, :]
                print(f"Aligned range: {n_days_factor} days, {len(stock_ids)} stocks")

            # Column naming: factor_{domain}_{idx:03d}
            col_name = f"factor_{domain}_{i:03d}"
            results_data[col_name] = factor_np.flatten()
            
            # Calculate stats relative to the liberal mask (summary only)
            stats = calculate_stats(factor_np, valid_mask=current_liberal_mask)
            stats["col_name"] = col_name
            stats["expr"] = expr_str
            # Add metadata from input JSON if available
            stats["weight"] = weights[i] if i < len(weights) else None
            stats["topic"] = topics[i] if i < len(topics) else None
            stats["description"] = descriptions[i] if i < len(descriptions) else None
            stats["ic_ret"] = ics_ret[i] if i < len(ics_ret) else None
            stats["icir"] = icir[i] if i < len(icir) else None
            
            stats_list.append(stats)
            
        except Exception as e:
            print(f"Error evaluating expression {i}: {expr_str}")
            print(f"Reason: {e}")
            continue
            
    if not results_data:
        print("No factors successfully evaluated.")
        return
        
    # 6. Create Index and Save Parquet
    print("Creating final DataFrame...")
    n_days = len(final_dates)
    n_stocks = len(stock_ids)
    all_dates_flattened = np.repeat(final_dates.values, n_stocks)
    all_stocks_flattened = np.tile(stock_ids.values, n_days)
    
    df_final = pd.DataFrame({
        "trade_date": all_dates_flattened,
        "ts_code": all_stocks_flattened
    })
    
    for col_name, data_np in results_data.items():
        df_final[col_name] = data_np
    
    # Per user request: keep all data, do not filter out non-valid rows
    print(f"Final DataFrame shape: {df_final.shape}")
        
    print(f"Writing to {output_path} using pyarrow...")
    table = pa.Table.from_pandas(df_final)
    pq.write_table(table, output_path, compression='snappy', row_group_size=100000)
    
    # 7. Generate Summary JSON
    summary_path = os.path.join(output_dir, f"summary_{domain}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats_list, f, indent=4, ensure_ascii=False)
    print(f"Summary JSON saved to {summary_path}")
    
    # 8. Plot Distribution (Optional)
    if do_plot:
        num_to_plot = max(1, int(len(stats_list) * 0.1))
        # Sort by kurtosis descending
        sorted_stats = sorted(stats_list, key=lambda x: x["kurtosis"], reverse=True)
        top_extreme = sorted_stats[:num_to_plot]
        
        print(f"Plotting top {num_to_plot} extreme factors...")
        for item in top_extreme:
            col_name = item["col_name"]
            data_to_plot = results_data[col_name]
            valid_data = data_to_plot[~np.isnan(data_to_plot)]
            
            if len(valid_data) == 0: continue
            
            plt.figure(figsize=(10, 6))
            # Use 99% range for better visualization
            p0_5, p99_5 = item["range_99"]
            filtered_data = valid_data[(valid_data >= p0_5) & (valid_data <= p99_5)]
            
            if len(filtered_data) > 0:
                sns.histplot(filtered_data, kde=True, bins=50)
                plt.title(f"因子分布: {col_name}\n表达式: {item['expr'][:80]}...", fontsize=10)
                plt.xlabel("因子值")
                plt.ylabel("频率")
                
                plot_path = os.path.join(plot_dir, f"{col_name}_dist.png")
                plt.savefig(plot_path)
            plt.close()
    
    print(f"Successfully processed Domain {domain}")

if __name__ == "__main__":
    # Configure parameters
    DOMAINS = ["A", "B", "C", "E"]
    JSON_DIR = map_path("data/factor_format_json")
    OUTPUT_DIR = map_path("data/feature_digged")
    PLOT_DIR = map_path("data/feature_digged/feature_digged_dist_plt")
    
    # Time range for evaluation (full range)
    START_TIME = "20100101"
    END_TIME = "20251231"
    
    for domain in DOMAINS:
        json_file = os.path.join(JSON_DIR, f"pool_{domain}.json")
        build_digged_factors(
            domain=domain,
            json_path=json_file,
            output_dir=OUTPUT_DIR,
            plot_dir=PLOT_DIR,
            start_time=START_TIME,
            end_time=END_TIME,
            do_plot=False # Default to False to speed up
        )
