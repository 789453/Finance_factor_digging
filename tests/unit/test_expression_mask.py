import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from alphagen_generic.sample_pool_builder import SamplePoolBuilder

def test_dynamic_pool_mask():
    print("Testing dynamic pool filtering and visualizing mask retention...")
    
    # 1. Paths
    daily_path = r"D:\Trading\data_ever_26_3_14\data\Raw_data\daily.parquet"
    index_path = str(ROOT / "data" / "basic" / "index_daily_basic_circ_mv.parquet")
    output_path = str(ROOT / "data" / "test_dynamic_pool_mask.json")
    
    # 2. Build a test pool for 2021-2022
    print(f"Building sample pool from 2021 to 2022...")
    builder = SamplePoolBuilder(daily_path=daily_path, index_path=index_path)
    
    # We use a smaller target size for testing speed, but 200 is fine
    pool_data = builder.build_pool(
        target_size=200,
        output_path=output_path,
        mode="dynamic",
        start_year=2021,
        end_year=2022
    )
    
    print(f"Pool built successfully. Keys: {list(pool_data['pools'].keys())}")
    
    # 3. Load daily data to see the exact mask effect
    print("Loading daily data to apply mask...")
    df_daily = pd.read_parquet(daily_path, columns=['trade_date', 'ts_code', 'close'])
    
    # Filter to 2021-2022
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'].astype(str))
    df_daily = df_daily[(df_daily['trade_date'] >= '2021-01-01') & (df_daily['trade_date'] <= '2022-12-31')]
    
    # Pivot to get a matrix of (dates, stocks)
    df_pivot = df_daily.pivot(index='trade_date', columns='ts_code', values='close')
    
    # Apply dynamic mask
    mask = pd.DataFrame(False, index=df_pivot.index, columns=df_pivot.columns)
    
    for date_str, codes in pool_data['pools'].items():
        pool_date = pd.Timestamp(date_str)
        
        # Next pool date (or end of data)
        # Assuming pools are updated at the end of Jan and Jul
        if pool_date.month == 1:
            next_date = pd.Timestamp(f"{pool_date.year}-07-31")
        else:
            next_date = pd.Timestamp(f"{pool_date.year + 1}-01-31")
            
        # Select rows in this period
        period_mask = (mask.index > pool_date) & (mask.index <= next_date)
        valid_codes = [c for c in codes if c in mask.columns]
        mask.loc[period_mask, valid_codes] = True
    
    # Apply mask to data
    df_masked = df_pivot.where(mask)
    
    # 4. Calculate retention statistics
    total_stocks_per_day = df_pivot.notna().sum(axis=1)
    pool_stocks_per_day = df_masked.notna().sum(axis=1)
    
    retention_rate = pool_stocks_per_day / 200.0  # relative to target size 200
    
    print("\n--- Mask Statistics ---")
    print(f"Average stocks in pool per day: {pool_stocks_per_day.mean():.1f}")
    print(f"Min stocks in pool per day: {pool_stocks_per_day.min()}")
    print(f"Max stocks in pool per day: {pool_stocks_per_day.max()}")
    
    # 5. Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(14, 6))
    
    plt.plot(pool_stocks_per_day.index, pool_stocks_per_day.values, label='Active Stocks in Pool (Target=200)', color='blue', linewidth=2)
    
    # Mark the pool update dates
    for date_str in pool_data['pools'].keys():
        update_date = pd.Timestamp(date_str)
        if update_date in pool_stocks_per_day.index:
            plt.axvline(x=update_date, color='red', linestyle='--', alpha=0.5)
            
    plt.title('Dynamic Pool Size Over Time (2021-2022)')
    plt.xlabel('Date')
    plt.ylabel('Number of Active Stocks')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = ROOT / "data" / "dynamic_pool_mask_viz.png"
    plt.savefig(plot_path)
    print(f"Visualization saved to: {plot_path}")

if __name__ == "__main__":
    test_dynamic_pool_mask()
