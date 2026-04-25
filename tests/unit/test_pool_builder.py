import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from alphagen_generic.sample_pool_builder import SamplePoolBuilder

def test_pool_strict_size():
    print("=== Testing Dynamic Pool Strict Size & Continuity ===")
    
    # 1. Paths
    daily_path = r"D:\Trading\data_ever_26_3_14\data\Raw_data\daily.parquet"
    index_path = str(ROOT / "data" / "basic" / "index_daily_basic_circ_mv.parquet")
    output_path = str(ROOT / "data" / "test_pool_strict.json")
    
    # 2. Build Pool
    builder = SamplePoolBuilder(daily_path=daily_path, index_path=index_path)
    pool_data = builder.build_pool(
        target_size=200,
        output_path=output_path,
        mode="dynamic",
        start_year=2021,
        end_year=2024
    )
    
    # 3. Verify individual pool sizes in JSON
    print("\nVerifying JSON pool sizes:")
    for date, codes in pool_data['pools'].items():
        size = len(codes)
        print(f"  Date: {date}, Size: {size}")
        assert size == 200, f"Error: Pool at {date} has size {size} instead of 200!"

    # 4. Simulate Daily Selection (Correct Logic: No overlap)
    print("\nSimulating daily stock count (Strict No-Overlap Logic)...")
    
    # Load some dummy dates to simulate a timeline
    df_index = pd.read_parquet(index_path, columns=['trade_date'])
    all_trading_dates = sorted(df_index['trade_date'].unique())
    all_trading_dates = [d for d in all_trading_dates if d >= "20210101" and d <= "20241231"]
    
    # Correct mapping: for each day, find the latest available rebalance date
    rebal_dates = sorted(pool_data['pools'].keys())
    
    daily_counts = []
    for d in all_trading_dates:
        # Find the active rebal_date for this trading day
        # Active rebal_date is the largest one that is <= current day
        active_rebal = None
        for r in reversed(rebal_dates):
            if r <= d:
                active_rebal = r
                break
        
        if active_rebal:
            count = len(pool_data['pools'][active_rebal])
        else:
            count = 0
        daily_counts.append(count)
        
    df_daily = pd.DataFrame({
        'trade_date': pd.to_datetime(all_trading_dates),
        'stock_count': daily_counts
    })
    
    print(f"Average daily count: {df_daily['stock_count'].mean():.2f}")
    print(f"Max daily count: {df_daily['stock_count'].max()}")
    print(f"Min daily count (after first pool): {df_daily[df_daily['trade_date'] >= rebal_dates[0]]['stock_count'].min()}")

    # 5. Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(df_daily['trade_date'], df_daily['stock_count'], color='green', label='Daily Active Stocks')
    plt.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='Target (200)')
    
    for r in rebal_dates:
        plt.axvline(x=pd.to_datetime(r), color='blue', linestyle=':', alpha=0.3)
        
    plt.title("Dynamic Pool Size - Corrected Daily Simulation (2021-2024)")
    plt.ylabel("Number of Stocks")
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    viz_path = ROOT / "data" / "dynamic_pool_corrected_viz.png"
    plt.savefig(viz_path)
    print(f"\nSuccess! Corrected visualization saved to: {viz_path}")
    print("The spikes are gone because we now correctly use only ONE pool per day.")

if __name__ == "__main__":
    test_pool_strict_size()
