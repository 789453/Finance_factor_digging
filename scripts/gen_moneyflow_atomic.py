import pandas as pd
import numpy as np
import os

input_path = 'D:/Trading/data_ever_26_3_14/data/silver/stock_moneyflow.parquet'
output_path = 'D:/Trading/data_ever_26_3_14/data/silver/stock_moneyflow_atomic.parquet'

print(f"Reading {input_path}...")
df = pd.read_parquet(input_path)

# 确保索引正确以便计算加速度
df = df.sort_values(['ts_code', 'trade_date'])

print("Calculating atomic features...")
df['mf_imbalance_lg'] = (df['buy_lg_amount'] - df['sell_lg_amount']) / (df['buy_lg_amount'] + df['sell_lg_amount'] + 1e-8)
df['mf_imbalance_elg'] = (df['buy_elg_amount'] - df['sell_elg_amount']) / (df['buy_elg_amount'] + df['sell_elg_amount'] + 1e-8)

total_amount = (
    df['buy_sm_amount'] + df['buy_md_amount'] + df['buy_lg_amount'] + df['buy_elg_amount'] +
    df['sell_sm_amount'] + df['sell_md_amount'] + df['sell_lg_amount'] + df['sell_elg_amount'] + 1e-8
)
df['mf_net_ratio'] = df['net_mf_amount'] / total_amount

df['mf_pressure'] = (df['buy_lg_amount'] + df['buy_elg_amount']) / (df['sell_lg_amount'] + df['sell_elg_amount'] + 1e-8)

# 资金流入加速度 (截面或序列？通常是序列差分)
df['mf_acceleration'] = df.groupby('ts_code')['net_mf_amount'].diff()

# 只保留需要的列
cols_to_keep = ['ts_code', 'trade_date', 'mf_imbalance_lg', 'mf_imbalance_elg', 'mf_net_ratio', 'mf_pressure', 'mf_acceleration']
atomic_df = df[cols_to_keep]

print(f"Saving to {output_path}...")
atomic_df.to_parquet(output_path)
print("Done!")
