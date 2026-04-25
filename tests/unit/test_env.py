
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.path_utils import EMBEDDING_MODEL_NAME, IS_CLOUD

# Load .env
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

print(f"EMBEDDING_MODEL_NAME from env: {os.getenv('EMBEDDING_MODEL_NAME')}")
print(f"EMBEDDING_MODEL_NAME from path_utils: {EMBEDDING_MODEL_NAME}")
print(f"IS_CLOUD: {IS_CLOUD}")
