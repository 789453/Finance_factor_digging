import os
from datetime import datetime

# Centralized Date Configuration
DEFAULT_TRAIN_START = "20100101"
DEFAULT_TRAIN_END = "20211231"
DEFAULT_TEST_START = "20220101"
DEFAULT_TEST_END = "20250630"

# Domain-specific earliest possible start dates
DOMAIN_START_DATES = {
    'A': '20100101',
    'B': '20100101',
    'C': '20180101',
    'E': '20100101'
}

def get_date_range(domain, train_end_year=2021, test_end_year=2025):
    """
    Generate train/test date ranges based on domain and training end year.
    Ensures continuity.
    """
    base_start = DOMAIN_START_DATES.get(domain, DEFAULT_TRAIN_START)
    
    train_start = base_start
    train_end = f"{train_end_year}1231"
    
    test_start = f"{train_end_year + 1}0101"
    test_end = f"{test_end_year}0630" # Usually ending in 0630 for half year
    
    return train_start, train_end, test_start, test_end

from utils.path_utils import map_path, FACTOR_READY_DIR, BASIC_DIR

# Shared Defaults
DEFAULT_LABEL_DAYS = 10
DEFAULT_CHUNK_SIZE = 250
DEFAULT_N_FACTORS = 15

# Path Defaults
DEFAULT_REGISTRY_PATH = os.path.join(FACTOR_READY_DIR, "feature_registry.csv")
DEFAULT_POOL_PATH = os.path.join(FACTOR_READY_DIR, "sample_pool_200.json")
DEFAULT_DAILY_PATH = map_path(r"data/basic/daily.parquet")
DEFAULT_INDEX_PATH = os.path.join(BASIC_DIR, "index_daily_basic_circ_mv.parquet")
