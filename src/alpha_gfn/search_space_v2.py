from typing import Any, Sequence, Dict, List, Optional
import logging

try:
    from alpha_gfn.config import CONSTANTS, DELTA_TIMES, OPERATORS
except ImportError:
    from .config import CONSTANTS, DELTA_TIMES, OPERATORS
from alphagen_generic.task_config import parse_float_list, parse_int_list, parse_str_list
from alphagen_generic.factor_family_config import FactorFamilyConfig, FactorFamilyRegistry

logger = logging.getLogger(__name__)

DEFAULT_OPERATOR_MAP = {op.__name__: op for op in OPERATORS}

def build_search_space(
    operator_names: Any = None,
    delta_times: Any = None,
    constants: Any = None,
    family_spec: Optional[Dict[str, Any]] = None,
):
    """
    构建搜索空间
    
    Args:
        operator_names: 操作符名称列表
        delta_times: 时间窗口列表
        constants: 常数列表
        family_spec: 因子族配置（优先级最高）
        
    Returns:
        Tuple: (operator_list, delta_time_list, constant_list)
    """
    # 如果提供了因子族配置，优先使用
    if family_spec:
        return _build_search_space_from_family(family_spec)
    
    # 否则使用传统参数
    return _build_search_space_from_params(operator_names, delta_times, constants)

def _build_search_space_from_family(family_spec: Dict[str, Any]) -> tuple:
    """从因子族配置构建搜索空间"""
    
    # 操作符
    operator_names = family_spec.get('operator_names', [])
    operator_list = []
    
    if operator_names:
        unknown = [name for name in operator_names if name not in DEFAULT_OPERATOR_MAP]
        if unknown:
            raise ValueError(f"未知操作符在搜索空间中: {unknown}")
        operator_list = [DEFAULT_OPERATOR_MAP[name] for name in operator_names]
    else:
        # 如果没有指定，使用默认操作符
        operator_list = OPERATORS
    
    # 过滤禁止的操作符
    forbid_operators = family_spec.get('forbid_operators', [])
    if forbid_operators:
        operator_list = [op for op in operator_list if op.__name__ not in forbid_operators]
        logger.info(f"过滤了 {len(forbid_operators)} 个禁止的操作符: {forbid_operators}")
    
    # 时间窗口
    delta_time_list = family_spec.get('delta_times', list(DELTA_TIMES))
    
    # 常数
    constant_list = family_spec.get('constants', list(CONSTANTS))
    
    # 验证最小时间序列操作符数量
    min_ts_operator_count = family_spec.get('min_ts_operator_count', 1)
    ts_operators = [op for op in operator_list if 'Ts' in op.__name__]
    
    if len(ts_operators) < min_ts_operator_count:
        logger.warning(f"时间序列操作符数量 {len(ts_operators)} 小于最小要求 {min_ts_operator_count}")
    
    logger.info(f"从因子族配置构建搜索空间:")
    logger.info(f"  操作符: {len(operator_list)} 个")
    logger.info(f"  时间窗口: {len(delta_time_list)} 个")
    logger.info(f"  常数: {len(constant_list)} 个")
    logger.info(f"  最小时间序列操作符: {min_ts_operator_count}")
    
    return operator_list, delta_time_list, constant_list

def _build_search_space_from_params(operator_names: Any, delta_times: Any, constants: Any) -> tuple:
    """从参数构建搜索空间（向后兼容）"""
    
    # 操作符
    operator_list = OPERATORS
    if operator_names:
        parsed_names = parse_str_list(operator_names) or []
        unknown = [name for name in parsed_names if name not in DEFAULT_OPERATOR_MAP]
        if unknown:
            raise ValueError(f"未知操作符在搜索空间中: {unknown}")
        operator_list = [DEFAULT_OPERATOR_MAP[name] for name in parsed_names]
    
    # 时间窗口
    delta_time_list = parse_int_list(delta_times) if delta_times is not None else list(DELTA_TIMES)
    
    # 常数
    constant_list = parse_float_list(constants) if constants is not None else list(CONSTANTS)
    
    logger.info(f"从参数构建搜索空间:")
    logger.info(f"  操作符: {len(operator_list)} 个")
    logger.info(f"  时间窗口: {len(delta_time_list)} 个")
    logger.info(f"  常数: {len(constant_list)} 个")
    
    return operator_list, delta_time_list, constant_list

def build_search_space_from_family_config(family_config_path: str, family_id: str) -> tuple:
    """
    从因子族配置文件构建搜索空间
    
    Args:
        family_config_path: 配置文件路径
        family_id: 因子族ID
        
    Returns:
        Tuple: (operator_list, delta_time_list, constant_list)
    """
    try:
        family_config = FactorFamilyConfig(family_config_path)
        family_spec = family_config.get_search_space_config(family_id)
        
        return build_search_space(family_spec=family_spec)
        
    except Exception as e:
        logger.error(f"从因子族配置构建搜索空间失败: {e}")
        raise

def build_search_space_from_family_registry(registry_dir: str, family_id: str) -> tuple:
    """
    从因子族注册表构建搜索空间
    
    Args:
        registry_dir: 注册表目录
        family_id: 因子族ID
        
    Returns:
        Tuple: (operator_list, delta_time_list, constant_list)
    """
    try:
        registry = FactorFamilyRegistry(registry_dir)
        family_spec = registry.get_family(family_id)
        
        return build_search_space(family_spec=family_spec)
        
    except Exception as e:
        logger.error(f"从因子族注册表构建搜索空间失败: {e}")
        raise

def validate_search_space_config(operator_list: List, delta_time_list: List, constant_list: List) -> List[str]:
    """
    验证搜索空间配置
    
    Args:
        operator_list: 操作符列表
        delta_time_list: 时间窗口列表
        constant_list: 常数列表
        
    Returns:
        验证问题列表
    """
    issues = []
    
    if not operator_list:
        issues.append("操作符列表为空")
    
    if not delta_time_list:
        issues.append("时间窗口列表为空")
    
    if not constant_list:
        issues.append("常数列表为空")
    
    # 检查时间窗口是否为正整数
    for dt in delta_time_list:
        if not isinstance(dt, int) or dt <= 0:
            issues.append(f"无效的时间窗口: {dt}")
    
    # 检查常数是否为数字
    for const in constant_list:
        if not isinstance(const, (int, float)):
            issues.append(f"无效的常数: {const}")
    
    return issues

def get_search_space_summary(operator_list: List, delta_time_list: List, constant_list: List) -> Dict[str, Any]:
    """
    获取搜索空间摘要信息
    
    Args:
        operator_list: 操作符列表
        delta_time_list: 时间窗口列表
        constant_list: 常数列表
        
    Returns:
        摘要信息字典
    """
    ts_operators = [op for op in operator_list if 'Ts' in op.__name__]
    
    return {
        'n_operators': len(operator_list),
        'n_delta_times': len(delta_time_list),
        'n_constants': len(constant_list),
        'n_ts_operators': len(ts_operators),
        'operators': [op.__name__ for op in operator_list],
        'delta_times': delta_time_list,
        'constants': constant_list,
        'ts_operators': [op.__name__ for op in ts_operators]
    }