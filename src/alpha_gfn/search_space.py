from typing import Any, Sequence

try:
    from alpha_gfn.config import CONSTANTS, DELTA_TIMES, OPERATORS
except ImportError:
    from .config import CONSTANTS, DELTA_TIMES, OPERATORS
from alphagen_generic.task_config import parse_float_list, parse_int_list, parse_str_list


DEFAULT_OPERATOR_MAP = {op.__name__: op for op in OPERATORS}


def build_search_space(
    operator_names: Any = None,
    delta_times: Any = None,
    constants: Any = None,
):
    operator_list = OPERATORS
    if operator_names:
        parsed_names = parse_str_list(operator_names) or []
        unknown = [name for name in parsed_names if name not in DEFAULT_OPERATOR_MAP]
        if unknown:
            raise ValueError(f"Unknown operators in search space: {unknown}")
        operator_list = [DEFAULT_OPERATOR_MAP[name] for name in parsed_names]

    delta_time_list = parse_int_list(delta_times) if delta_times is not None else list(DELTA_TIMES)
    constant_list = parse_float_list(constants) if constants is not None else list(CONSTANTS)

    return operator_list, delta_time_list, constant_list
