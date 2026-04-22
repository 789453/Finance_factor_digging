import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Environment detection: Check if we are in the cloud environment with /mnt mounted
MNT_DATA_BASE = Path("/mnt/data/factor_mining")
MNT_OUTPUT_BASE = Path("/mnt/output/factor_mining")

# We use IS_CLOUD to decide whether to use /mnt paths
IS_CLOUD = MNT_DATA_BASE.exists() and MNT_OUTPUT_BASE.exists()

def get_data_path(sub_path: str = "") -> str:
    """
    Get the mapped data path.
    Example: get_data_path("factor_ready") -> "/mnt/data/factor_mining/factor_ready" (cloud) 
                                           or "PROJECT_ROOT/data/factor_ready" (local)
    """
    if IS_CLOUD:
        return str(MNT_DATA_BASE / sub_path)
    else:
        return str(PROJECT_ROOT / "data" / sub_path)

def get_output_path(sub_path: str = "") -> str:
    """
    Get the mapped output path.
    Example: get_output_path("knowledge_logs") -> "/mnt/output/factor_mining/knowledge_logs" (cloud)
                                               or "PROJECT_ROOT/data/knowledge_logs" (local)
    """
    if IS_CLOUD:
        return str(MNT_OUTPUT_BASE / sub_path)
    else:
        # Locally, outputs also go to the data directory
        return str(PROJECT_ROOT / "data" / sub_path)

# Pre-defined directories for common use
FACTOR_READY_DIR = get_data_path("factor_ready")
BASIC_DIR = get_data_path("basic")
KNOWLEDGE_LOGS_DIR = get_output_path("knowledge_logs")
ADAPTIVE_LOGS_DIR = get_output_path("adaptive_combination_logs")
TEST_ADAPTIVE_LOGS_DIR = get_output_path("test_adaptive_logs")
# Embedding model configuration
CLOUD_EMBEDDING_PATH = "/mnt/models/Qwen_Qwen3-Embedding-4B"
LOCAL_EMBEDDING_PATH = r"C:\Users\Purple_Born\.cache\torch\sentence_transformers\Qwen_Qwen3-Embedding-4B"

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
if not EMBEDDING_MODEL_NAME:
    EMBEDDING_MODEL_NAME = CLOUD_EMBEDDING_PATH if IS_CLOUD else LOCAL_EMBEDDING_PATH

def map_path(original_path: str) -> str:
    """
    Maps an original path (starting with 'data/') to the correct environment-specific path.
    """
    if not original_path:
        return original_path
        
    path_str = str(original_path).replace("\\", "/")
    
    if path_str.startswith("data/factor_ready"):
        return path_str.replace("data/factor_ready", FACTOR_READY_DIR)
    elif path_str.startswith("data/basic"):
        return path_str.replace("data/basic", BASIC_DIR)
    elif path_str.startswith("data/knowledge_logs"):
        return path_str.replace("data/knowledge_logs", KNOWLEDGE_LOGS_DIR)
    elif path_str.startswith("data/adaptive_combination_logs"):
        return path_str.replace("data/adaptive_combination_logs", ADAPTIVE_LOGS_DIR)
    elif path_str.startswith("data/test_adaptive_logs"):
        return path_str.replace("data/test_adaptive_logs", TEST_ADAPTIVE_LOGS_DIR)
    
    # Generic mapping if it starts with data/ but not one of the specific ones
    if path_str.startswith("data/"):
        return get_data_path(path_str[5:])
        
    return original_path

if __name__ == "__main__":
    print(f"IS_CLOUD: {IS_CLOUD}")
    print(f"FACTOR_READY_DIR: {FACTOR_READY_DIR}")
    print(f"KNOWLEDGE_LOGS_DIR: {KNOWLEDGE_LOGS_DIR}")
    print(f"Mapped sample pool: {map_path('data/factor_ready/sample_pool_200.json')}")
