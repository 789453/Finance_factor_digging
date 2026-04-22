import pandas as pd
import numpy as np
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from utils.path_utils import map_path

def preprocess_domain_features(domain: str, input_path: str, output_dir: str) -> dict:
    """
    Load raw features efficiently, perform vectorized filling, and save.
    Uses groupby operations instead of expensive unstack/stack loops.
    """
    print(f"\nPreprocessing Domain {domain} from {input_path}...")
    
    # 1. Efficient Load using pyarrow
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    # Identify feature columns
    cols_to_skip = ['trade_date', 'ts_code']
    feature_cols = [c for c in df.columns if c not in cols_to_skip]
    
    initial_nan_counts = df[feature_cols].isna().sum()
    total_rows = len(df)
    
    # 2. Vectorized Temporal Filling (Per Stock)
    # Sorting is crucial for groupby ffill/bfill
    print(f"  Sorting data for temporal filling...")
    df = df.sort_values(['ts_code', 'trade_date'])
    
    print(f"  Performing temporal filling (ffill=30, bfill=15) for all features...")
    # Group by stock and fill. This is vectorized across all columns.
    # Note: we use limit to avoid filling too much into delisted/pre-listed periods
    df[feature_cols] = df.groupby('ts_code')[feature_cols].ffill(limit=30).bfill(limit=15)
    
    # 3. Vectorized Cross-Sectional Filling (Per Date)
    print(f"  Performing cross-sectional median filling...")
    # Calculate median per day for all columns
    day_medians = df.groupby('trade_date')[feature_cols].transform('median')
    # Fill remaining NaNs with the day's median
    df[feature_cols] = df[feature_cols].fillna(day_medians)
    
    # 4. Final Global Fallback
    print(f"  Final global fallback to 0.0...")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    # 5. Calculate Stats
    final_nan_counts = df[feature_cols].isna().sum()
    stats = {}
    for col in feature_cols:
        stats[col] = {
            "initial_nan_rate": float(initial_nan_counts[col] / total_rows),
            "final_nan_rate": float(final_nan_counts[col] / total_rows),
            "filled_count": int(initial_nan_counts[col] - final_nan_counts[col])
        }
    
    # 6. Efficient Save using pyarrow
    output_path = os.path.join(output_dir, f"feature_{domain}_filled.parquet")
    print(f"  Saving to {output_path}...")
    
    # Ensure correct types and convert back to arrow table for fast writing
    new_table = pa.Table.from_pandas(df)
    pq.write_table(new_table, output_path, compression='snappy')
    
    return stats

if __name__ == "__main__":
    DOMAINS = {
        "A": "feature_A_price_volume.parquet",
        "B": "feature_B_moneyflow.parquet",
        "C": "feature_C_chip.parquet",
        "E": "feature_E_intraday_summary.parquet"
    }
    
    INPUT_DIR = map_path("data/factor_ready")
    OUTPUT_DIR = map_path("data/factor_ready_filled")
    STATS_DIR = map_path("data/feature_digged")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    all_stats = {}
    
    for domain, filename in DOMAINS.items():
        input_file = os.path.join(INPUT_DIR, filename)
        if os.path.exists(input_file):
            domain_stats = preprocess_domain_features(domain, input_file, OUTPUT_DIR)
            all_stats[domain] = domain_stats
        else:
            print(f"Warning: File not found {input_file}")
            
    stats_json_path = os.path.join(STATS_DIR, "filling_stats.json")
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False)
        
    print(f"\nAll domains processed efficiently. Stats saved to {stats_json_path}")
