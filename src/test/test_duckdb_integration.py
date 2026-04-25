import os
import sys
import logging
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from datahub.duckdb_hub import DuckDBDataHub
from datahub.config import DataHubConfig
from alphagen_generic.parquet_feature_loader_v2 import ParquetFeatureLoaderV2
from alphagen_generic.feature_registry_manager_v2 import FeatureRegistryManagerV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_duckdb_integration():
    # 1. Setup DataHub
    data_root = "D:/Trading/data_ever_26_3_14/data"
    warehouse_path = os.path.join(data_root, "meta/warehouse.duckdb")
    control_db_path = os.path.join(data_root, "meta/control.sqlite3")
    
    if not os.path.exists(data_root):
        logger.error(f"Data root not found: {data_root}")
        return

    config = DataHubConfig(
        warehouse_path=warehouse_path,
        control_db_path=control_db_path
    )
    datahub = DuckDBDataHub(config)
    logger.info("DataHub initialized successfully")

    # 2. Setup Loader
    # Using a known date range from the README
    start_date = "20260101"
    end_date = "20260110"
    
    registry = FeatureRegistryManagerV2()
    
    loader = ParquetFeatureLoaderV2(
        domain="A",
        start_time=start_date,
        end_time=end_date,
        registry_manager=registry,
        datahub=datahub,
        read_mode="mixed",
        layers=["raw"],
        device="cpu"
    )
    
    logger.info(f"Loader initialized. Data shape: {loader.data.shape}")
    logger.info(f"Dates: {len(loader.dates)}, Stocks: {len(loader.stock_ids)}")
    
    # 3. Verify data
    if loader.data.shape[0] > 0:
        logger.info("Successfully loaded data using DuckDB integration!")
    else:
        logger.error("Failed to load data")

if __name__ == "__main__":
    test_duckdb_integration()
