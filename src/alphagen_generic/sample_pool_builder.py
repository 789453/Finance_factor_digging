import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SamplePoolBuilder:
    """
    Builds and manages the stock sample pool.
    Strictly follows the rules in 股票池子筛选.md:
    - Two-layer stratified (Industry + Market Cap)
    - Half-year reconstruction (Jan & Jul)
    - 200 stocks total
    - Quality sorting + Random sampling
    """
    def __init__(self, data_dir: str = None, daily_path: str = None, index_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.data_dir = data_dir or os.path.join(base_dir, "data", "factor_ready")
        self.daily_path = daily_path or r"D:\Trading\data_ever_26_3_14\data\Raw_data\daily.parquet"
        self.index_path = index_path or os.path.join(base_dir, "data", "basic", "index_daily_basic_circ_mv.parquet")

    def build_pool(self, target_size: int = 200, output_path: str = None, mode: str = "dynamic", start_year: int = 2020, end_year: int = 2025) -> Union[List[str], Dict[str, List[str]]]:
        """
        Build a sample pool of stocks.
        """
        # Always build dynamic pool as per requirements
        return self._build_dynamic_pool(start_year, end_year, target_size, output_path)

    def _build_static_pool(self, target_size: int, output_path: str) -> List[str]:
        # Simple fallback to most recent data
        df_index = pd.read_parquet(self.index_path)
        last_date = df_index['trade_date'].max()
        univ = df_index[df_index['trade_date'] == last_date].copy()
        selected_stocks = univ.sort_values('circ_mv', ascending=False).head(target_size)['ts_code'].tolist()
        
        if output_path:
            self.save_pool(selected_stocks, output_path)
        return selected_stocks

    def _build_dynamic_pool(self, start_year: int, end_year: int, target_size: int, output_path: str) -> Dict[str, List[str]]:
        print(f"Building dynamic sample pool ({start_year}-{end_year})...")
        
        # 1. Load Data
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index data not found: {self.index_path}")
        if not os.path.exists(self.daily_path):
            raise FileNotFoundError(f"Daily data not found: {self.daily_path}")
            
        print("Loading index data...")
        df_index = pd.read_parquet(self.index_path)
        df_index['trade_date'] = df_index['trade_date'].astype(str)
        
        print("Loading daily data (Liquidity check)...")
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

        pools = {}
        all_dates = sorted(df_index['trade_date'].unique())
        
        # 2. Loop Half-Years (Jan and Jul)
        for year in range(start_year, end_year + 1):
            for month in ["01", "07"]:
                rebal_prefix = f"{year}{month}"
                # Find first trading day on or after rebal_prefix
                rebal_dates = [d for d in all_dates if d.startswith(rebal_prefix)]
                if not rebal_dates:
                    continue
                rebal_date = rebal_dates[0]
                print(f"Processing rebalance date: {rebal_date}")

                # Step 1: Filter Available Universe U(T)
                # 1.1 Basic & Industry/Cap not null
                univ = df_index[df_index['trade_date'] == rebal_date].copy()
                
                # Use 'industry' column added in previous step, fallback to 'l1_name'
                if 'industry' in univ.columns:
                     univ['industry_label'] = univ['industry'].fillna(univ['l1_name'])
                else:
                     univ['industry_label'] = univ['l1_name']
                
                univ = univ.dropna(subset=['industry_label', 'circ_mv', 'list_date'])
                
                # 1.2 Filtering Conditions (ST, Listed days, Liquidity)
                # a. Not ST (name does not contain ST or *ST)
                if 'name' in univ.columns:
                    univ = univ[~univ['name'].str.contains('ST', na=False)]
                
                # b. Listed for at least 252 days
                # Ensure types are correct
                rebal_dt = pd.to_datetime(rebal_date)
                univ['list_dt'] = pd.to_datetime(univ['list_date'].astype(str))
                univ = univ[(rebal_dt - univ['list_dt']).dt.days >= 252]
                
                # c. Feature Coverage > 95% in past 120 days
                if df_feat is not None:
                    valid_range = all_dates[max(0, all_dates.index(rebal_date)-120) : all_dates.index(rebal_date)]
                    if valid_range:
                        feat_sub = df_feat[df_feat['trade_date'].isin(valid_range)]
                        counts = feat_sub['ts_code'].value_counts()
                        threshold = len(valid_range) * 0.95
                        valid_codes = counts[counts >= threshold].index
                        univ = univ[univ['ts_code'].isin(valid_codes)]

                # d. Liquidity (Avg amount 60d > 0 and Score calculation)
                daily_range = all_dates[max(0, all_dates.index(rebal_date)-60) : all_dates.index(rebal_date)]
                if daily_range:
                    daily_sub = df_daily[df_daily['trade_date'].isin(daily_range)]
                    avg_amt = daily_sub.groupby('ts_code')['amount'].mean()
                    # Keep stocks with positive liquidity
                    univ = univ.merge(avg_amt.rename('score'), on='ts_code', how='inner')
                    univ = univ[univ['score'] > 0]
                else:
                    univ['score'] = 0

                # e. Not delisted before rebal_date
                if 'delist_date' in univ.columns:
                     # Handle NaNs properly after string conversion
                     delist_s = univ['delist_date'].astype(str)
                     # Treat "None", "nan" or empty as not delisted
                     not_delisted = delist_s.isin(["None", "nan", "<NA>", ""]) | delist_s.isna()
                     univ = univ[not_delisted | (delist_s > rebal_date)]

                # f. Market filter (optional, e.g., exclude some if needed)
                # For now, we use all available but we can filter by 'market' column if requested.
                
                # Step 2: Market Cap Buckets (0-35%, 35-80%, 80-100%)
                univ['mv_rank'] = univ['circ_mv'].rank(pct=True)
                univ['bucket'] = pd.cut(univ['mv_rank'], bins=[0, 0.35, 0.80, 1.0], labels=['small', 'mid', 'big'], include_lowest=True)

                # Step 3 & 4: Industry Allocation
                ind_counts = univ['industry_label'].value_counts()
                # Use a more robust DataFrame construction
                alloc = pd.DataFrame({'N': ind_counts})
                alloc['base'] = 2
                
                # Remaining quota
                remaining = target_size - alloc['base'].sum()
                if remaining < 0:
                    # Target size too small, scale base
                    alloc['total'] = (alloc['N'] / alloc['N'].sum() * target_size).round().astype(int)
                else:
                    alloc['sqrt_N'] = np.sqrt(alloc['N'])
                    alloc['weight'] = alloc['sqrt_N'] / alloc['sqrt_N'].sum()
                    alloc['extra'] = (alloc['weight'] * remaining).round().astype(int)
                    alloc['total'] = alloc['base'] + alloc['extra']
                
                # Apply Cap at 12 (Vectorized to avoid ValueError)
                alloc['total'] = np.minimum(alloc['total'], 12)
                alloc['total'] = np.minimum(alloc['total'], alloc['N'])
                alloc['total'] = alloc['total'].astype(int)
                
                # Adjust if sum != target due to rounding or caps
                diff = target_size - alloc['total'].sum()
                if diff != 0:
                    # Add/Subtract from largest industries
                    adj_idx = alloc.sort_values('N', ascending=False).index[:abs(diff)]
                    alloc.loc[adj_idx, 'total'] += (1 if diff > 0 else -1)

                # Step 5 & 6: Stratified Selection
                final_pool = []
                for ind, group in univ.groupby('industry_label'):
                    if ind not in alloc.index: continue
                    n_target = int(alloc.loc[ind, 'total'])
                    if n_target <= 0: continue
                    
                    # Distribute n_target into buckets based on available counts in this industry
                    b_counts = group['bucket'].value_counts()
                    b_target = (b_counts / b_counts.sum() * n_target).round().astype(int)
                    
                    # Fix rounding for bucket targets
                    b_diff = n_target - b_target.sum()
                    if b_diff != 0 and not b_target.empty:
                        b_target.iloc[0] += b_diff
                    
                    for b, b_n in b_target.items():
                        if b_n <= 0: continue
                        b_group = group[group['bucket'] == b]
                        if b_group.empty: continue
                        
                        # Selection: Pick top 2x candidates by score, then random sample
                        n_candidates = min(len(b_group), b_n * 2)
                        candidates = b_group.sort_values('score', ascending=False).head(n_candidates)
                        
                        selected = candidates.sample(n=min(len(candidates), b_n), random_state=42 + year + int(month))
                        final_pool.extend(selected['ts_code'].tolist())

                # Final Clamp: Ensure exactly target_size
                if len(final_pool) != target_size:
                    print(f"  Final adjustment: {len(final_pool)} -> {target_size}")
                    if len(final_pool) > target_size:
                        final_pool = final_pool[:target_size]
                    else:
                        # If somehow less, pad with top score from univ not in pool
                        remaining_n = target_size - len(final_pool)
                        candidates = univ[~univ['ts_code'].isin(final_pool)].sort_values('score', ascending=False).head(remaining_n)
                        final_pool.extend(candidates['ts_code'].tolist())

                print(f"  Selected {len(final_pool)} stocks for {rebal_date}")
                pools[rebal_date] = final_pool

        # Save result
        result = {"type": "dynamic", "pools": pools}
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"Dynamic pool saved to {output_path}")
            
        return result

    def visualize_pool(self, pool_result: Dict, output_dir: str = None):
        """
        Visualize the sample pool distribution over time.
        """
        if not pool_result or "pools" not in pool_result:
            print("No pool data to visualize.")
            return

        print("Generating visualizations...")
        df_index = pd.read_parquet(self.index_path)
        
        data_list = []
        for date, codes in pool_result['pools'].items():
            sub = df_index[(df_index['trade_date'] == date) & (df_index['ts_code'].isin(codes))].copy()
            if 'industry' in sub.columns:
                 sub['industry_label'] = sub['industry'].fillna(sub['l1_name'])
            else:
                 sub['industry_label'] = sub['l1_name']
            # Re-calculate buckets for visualization
            sub['mv_rank'] = sub['circ_mv'].rank(pct=True)
            sub['bucket'] = pd.cut(sub['mv_rank'], bins=[0, 0.35, 0.80, 1.0], labels=['Small', 'Mid', 'Big'], include_lowest=True)
            sub['date'] = date
            data_list.append(sub)
        
        df_plot = pd.concat(data_list)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 1. Industry Distribution (Latest date)
        latest_date = max(df_plot['date'])
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_plot[df_plot['date'] == latest_date], x='industry_label', order=df_plot[df_plot['date'] == latest_date]['industry_label'].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f"Industry Distribution in Sample Pool ({latest_date})")
        plt.tight_layout()
        if output_dir: plt.savefig(os.path.join(output_dir, "industry_dist.png"))
        plt.show()

        # 2. Market Cap Bucket Distribution (Latest date)
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_plot[df_plot['date'] == latest_date], x='bucket')
        plt.title(f"Market Cap Bucket Distribution ({latest_date})")
        if output_dir: plt.savefig(os.path.join(output_dir, "mkt_cap_dist.png"))
        plt.show()

        # 3. Pool Size Over Time
        size_series = df_plot.groupby('date').size()
        plt.figure(figsize=(10, 4))
        size_series.plot(kind='line', marker='o')
        plt.title("Sample Pool Size Over Time")
        plt.ylim(0, max(size_series) * 1.2)
        if output_dir: plt.savefig(os.path.join(output_dir, "pool_size_time.png"))
        plt.show()

    def save_pool(self, pool: Union[List[str], Dict], path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(pool, f, indent=4, ensure_ascii=False)

    def load_pool(self, path: str) -> Union[List[str], Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

if __name__ == "__main__":
    # Test execution
    builder = SamplePoolBuilder()
    output_json = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "factor_ready", "sample_pool_200.json")
    pool = builder.build_pool(target_size=200, output_path=output_json, start_year=2010, end_year=2025)
    builder.visualize_pool(pool, output_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "plots"))
