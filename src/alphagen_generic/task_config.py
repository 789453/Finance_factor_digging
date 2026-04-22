import argparse
import json
import os
from typing import Any, Dict, Iterable


def load_task_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Task config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_config(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=prefix))
        else:
            flat[key] = value
    return flat


def _known_parser_dests(parser: argparse.ArgumentParser) -> Iterable[str]:
    for action in parser._actions:
        if action.dest != "help":
            yield action.dest


def apply_task_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    config_path_attr: str = "task_config",
) -> Dict[str, Any]:
    config_path = getattr(args, config_path_attr, None)
    config = load_task_config(config_path)
    if not config:
        return {}

    flat_config = _flatten_config(config)
    parser_dests = set(_known_parser_dests(parser)) if parser is not None else set()

    for key, value in flat_config.items():
        if parser is None:
            setattr(args, key, value)
            continue

        if key not in parser_dests:
            setattr(args, key, value)
            continue

        current_value = getattr(args, key, None)
        default_value = parser.get_default(key)
        if current_value == default_value:
            setattr(args, key, value)

    return config


def parse_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    return [str(value)]


def parse_int_list(value: Any) -> list[int] | None:
    items = parse_str_list(value)
    if items is None:
        return None
    return [int(item) for item in items]


def parse_float_list(value: Any) -> list[float] | None:
    items = parse_str_list(value)
    if items is None:
        return None
    return [float(item) for item in items]
