import json
import os
from datetime import datetime
from typing import Any

import numpy as np


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            k: _to_serializable(v)
            for k, v in value.__dict__.items()
            if not k.startswith("_")
        }
    return value


def build_run_dir(
    output_root: str,
    prefix: str,
    run_name: str | None = None,
    suffix: str | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [prefix]
    if run_name:
        name_parts.append(run_name)
    if suffix:
        name_parts.append(suffix)
    name_parts.append(timestamp)
    run_dir = os.path.join(output_root, "_".join(str(p) for p in name_parts if p))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, ensure_ascii=False, indent=2)


def save_numpy(path: str, array: Any) -> None:
    np.save(path, np.asarray(array))
