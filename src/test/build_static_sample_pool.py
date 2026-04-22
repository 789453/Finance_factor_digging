import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union

# Set base directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class StaticSamplePoolBuilder:
    """
    Builds a STATIC stock sample pool for training.
    Follows the stratification rules (Industry + Market Cap) and basic filtering.
    """
    def __init__(self, data_dir: str = None, daily_path: str = None, index_path: str = None):
        # Default paths (based on SamplePoolBuilder)
        self.data_dir = data_dir or str(PROJECT_ROOT / "data" / "factor_ready")
        self.daily_path = daily_path or r"D:\Trading\data_ever_26_3_14\data\Raw_data\daily.parquet"
        self.index_path = index_path or str(PROJECT_ROOT / "data" / "basic" / "index_daily_basic_circ_mv.parquet")

    def build_static_pool(self, target_size: int = 200, selection_date: str = "20240101", output_path: str = None) -> List[str]:
        """
        Build a single static list of stocks valid at a specific date.
        """
        print(f"Building static sample pool for date: {selection_date} (Target size: {target_size})")
        
        # 1. Load Data
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index data not found: {self.index_path}")
        if not os.path.exists(self.daily_path):
            raise FileNotFoundError(f"Daily data not found: {self.daily_path}")
            
        print("Loading index data...")
        df_index = pd.read_parquet(self.index_path)
        df_index['trade_date'] = df_index['trade_date'].astype(str)
        
        print("Loading daily data (Liquidity check)...")
        # Load close as well for potential fallback, but amount is needed for score
        df_daily = pd.read_parquet(self.daily_path, columns=['ts_code', 'trade_date', 'amount'])
        df_daily['trade_date'] = df_daily['trade_date'].astype(str)
        
        # Load Feature Data for Validity Check (Domain B as proxy)
        feature_path = os.path.join(self.data_dir, "feature_B_moneyflow.parquet")
        if not os.path.exists(feature_path):
             feature_path = os.path.join(self.data_dir, "feature_A_price_volume.parquet")
        
        print(f"Loading feature data for validity check: {feature_path}...")
        df_feat = pd.read_parquet(feature_path, columns=['ts_code', 'trade_date']) if os.path.exists(feature_path) else None
        if df_feat is not None:
            df_feat['trade_date'] = df_feat['trade_date'].astype(str)

        all_dates = sorted(df_index['trade_date'].unique())
        
        # Find the actual trading date closest to selection_date
        valid_selection_dates = [d for d in all_dates if d >= selection_date]
        if not valid_selection_dates:
            # Fallback to last available
            rebal_date = all_dates[-1]
        else:
            rebal_date = valid_selection_dates[0]
            
        print(f"Actual selection date used: {rebal_date}")

        # --- STEP 1: Filter Available Universe ---
        univ = df_index[df_index['trade_date'] == rebal_date].copy()
        
        # Industry Labeling
        if 'industry' in univ.columns:
             univ['industry_label'] = univ['industry'].fillna(univ['l1_name'])
        else:
             univ['industry_label'] = univ['l1_name']
        
        univ = univ.dropna(subset=['industry_label', 'circ_mv', 'list_date'])
        
        # 1.1 Basic Filtering (Not ST, listed >= 252 days)
        if 'name' in univ.columns:
            univ = univ[~univ['name'].str.contains('ST', na=False)]
        
        rebal_dt = pd.to_datetime(rebal_date)
        univ['list_dt'] = pd.to_datetime(univ['list_date'].astype(str))
        univ = univ[(rebal_dt - univ['list_dt']).dt.days >= 252]
        
        # 1.2 Feature Coverage > 95% in past 120 days
        if df_feat is not None:
            idx_date = all_dates.index(rebal_date)
            valid_range = all_dates[max(0, idx_date-120) : idx_date]
            if valid_range:
                feat_sub = df_feat[df_feat['trade_date'].isin(valid_range)]
                counts = feat_sub['ts_code'].value_counts()
                threshold = len(valid_range) * 0.95
                valid_codes = counts[counts >= threshold].index
                univ = univ[univ['ts_code'].isin(valid_codes)]

        # 1.3 Liquidity (Avg amount 60d > 0)
        idx_date = all_dates.index(rebal_date)
        daily_range = all_dates[max(0, idx_date-60) : idx_date]
        if daily_range:
            daily_sub = df_daily[df_daily['trade_date'].isin(daily_range)]
            avg_amt = daily_sub.groupby('ts_code')['amount'].mean()
            univ = univ.merge(avg_amt.rename('score'), on='ts_code', how='inner')
            univ = univ[univ['score'] > 0]
        else:
            univ['score'] = 0

        # --- STEP 2: Market Cap Buckets ---
        univ['mv_rank'] = univ['circ_mv'].rank(pct=True)
        univ['bucket'] = pd.cut(univ['mv_rank'], bins=[0, 0.35, 0.80, 1.0], labels=['small', 'mid', 'big'], include_lowest=True)

        # --- STEP 3: Industry Allocation ---
        ind_counts = univ['industry_label'].value_counts()
        alloc = pd.DataFrame({'N': ind_counts})
        alloc['base'] = 2
        
        remaining = target_size - alloc['base'].sum()
        if remaining < 0:
            alloc['total'] = (alloc['N'] / alloc['N'].sum() * target_size).round().astype(int)
        else:
            alloc['sqrt_N'] = np.sqrt(alloc['N'])
            alloc['weight'] = alloc['sqrt_N'] / alloc['sqrt_N'].sum()
            alloc['extra'] = (alloc['weight'] * remaining).round().astype(int)
            alloc['total'] = alloc['base'] + alloc['extra']
        
        # Cap at 12 and ensure not more than available
        alloc['total'] = np.minimum(alloc['total'], 12)
        alloc['total'] = np.minimum(alloc['total'], alloc['N'])
        alloc['total'] = alloc['total'].astype(int)
        
        # Final adjustment to match exactly target_size
        diff = target_size - alloc['total'].sum()
        if diff != 0:
            adj_idx = alloc.sort_values('N', ascending=False).index[:abs(diff)]
            alloc.loc[adj_idx, 'total'] += (1 if diff > 0 else -1)

        # --- STEP 4: Stratified Selection ---
        final_pool = []
        # Use a fixed random seed for reproducibility in static pool
        seed_base = int(selection_date) % 1000000
        
        for ind, group in univ.groupby('industry_label'):
            if ind not in alloc.index: continue
            n_target = int(alloc.loc[ind, 'total'])
            if n_target <= 0: continue
            
            b_counts = group['bucket'].value_counts()
            b_target = (b_counts / b_counts.sum() * n_target).round().astype(int)
            
            b_diff = n_target - b_target.sum()
            if b_diff != 0 and not b_target.empty:
                b_target.iloc[0] += b_diff
            
            for b, b_n in b_target.items():
                if b_n <= 0: continue
                b_group = group[group['bucket'] == b]
                if b_group.empty: continue
                
                # Pick top 2x candidates, then random sample
                n_candidates = min(len(b_group), b_n * 2)
                candidates = b_group.sort_values('score', ascending=False).head(n_candidates)
                
                selected = candidates.sample(n=min(len(candidates), b_n), random_state=seed_base)
                final_pool.extend(selected['ts_code'].tolist())

        # Clamp to target_size
        if len(final_pool) > target_size:
            final_pool = final_pool[:target_size]
        elif len(final_pool) < target_size:
            remaining_n = target_size - len(final_pool)
            candidates = univ[~univ['ts_code'].isin(final_pool)].sort_values('score', ascending=False).head(remaining_n)
            final_pool.extend(candidates['ts_code'].tolist())

        print(f"Final pool size: {len(final_pool)}")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_pool, f, indent=4, ensure_ascii=False)
            print(f"Static pool saved to {output_path}")
            
        return final_pool

if __name__ == "__main__":
    # Parameters requested by user
    TARGET_SIZE = 800
    SELECTION_DATE = "20230101" # Start of 2024-2025 period
    
    OUTPUT_FILE = str(PROJECT_ROOT / "data" / "test_temp_data" / f"static_pool_800_{SELECTION_DATE}.json")
    
    builder = StaticSamplePoolBuilder()
    pool = builder.build_static_pool(target_size=TARGET_SIZE, selection_date=SELECTION_DATE, output_path=OUTPUT_FILE)
    print("Execution complete.")
