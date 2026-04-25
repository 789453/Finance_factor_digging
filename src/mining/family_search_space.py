#!/usr/bin/env python3
"""
因子族搜索空间构建器
支持YAML配置文件和动态搜索空间生成
"""

import yaml
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_family_spec(family_id: str, config_dir: str = "config/factor_families") -> Dict[str, Any]:
    """
    加载因子族规格
    
    Args:
        family_id: 因子族ID
        config_dir: 配置文件目录
        
    Returns:
        因子族规格字典
    """
    # 尝试不同的文件扩展名
    for ext in ['.yaml', '.yml', '.json']:
        config_path = os.path.join(config_dir, f"{family_id}{ext}")
        if os.path.exists(config_path):
            if ext in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
    
    raise FileNotFoundError(f"Family spec not found for {family_id} in {config_dir}")

def validate_family_spec(spec: Dict[str, Any]) -> List[str]:
    """
    验证因子族规格
    
    Args:
        spec: 因子族规格
        
    Returns:
        错误列表
    """
    errors = []
    
    # 必填字段
    required_fields = ['family_id', 'enabled_layers', 'allowed_domains', 'operator_whitelist']
    for field in required_fields:
        if field not in spec:
            errors.append(f"Missing required field: {field}")
    
    # 验证字段类型
    if 'enabled_layers' in spec and not isinstance(spec['enabled_layers'], list):
        errors.append("enabled_layers must be a list")
    
    if 'allowed_domains' in spec and not isinstance(spec['allowed_domains'], list):
        errors.append("allowed_domains must be a list")
    
    if 'operator_whitelist' in spec and not isinstance(spec['operator_whitelist'], list):
        errors.append("operator_whitelist must be a list")
    
    if 'delta_times' in spec and not isinstance(spec['delta_times'], list):
        errors.append("delta_times must be a list")
    
    if 'constants' in spec and not isinstance(spec['constants'], list):
        errors.append("constants must be a list")
    
    # 验证数值合理性
    if 'max_expr_length' in spec:
        max_len = spec['max_expr_length']
        if not isinstance(max_len, int) or max_len <= 0:
            errors.append("max_expr_length must be a positive integer")
    
    if 'min_ts_operator_count' in spec:
        min_count = spec['min_ts_operator_count']
        if not isinstance(min_count, int) or min_count < 0:
            errors.append("min_ts_operator_count must be a non-negative integer")
    
    if 'sample_budget' in spec:
        budget = spec['sample_budget']
        if not isinstance(budget, int) or budget <= 0:
            errors.append("sample_budget must be a positive integer")
    
    if 'pool_capacity' in spec:
        capacity = spec['pool_capacity']
        if not isinstance(capacity, int) or capacity <= 0:
            errors.append("pool_capacity must be a positive integer")
    
    return errors

def build_family_search_space(family_spec: Dict[str, Any], dataset_meta: Optional[Any] = None) -> Tuple[List, List, List, List]:
    """
    根据因子族规格构建搜索空间
    
    Args:
        family_spec: 因子族规格
        dataset_meta: 数据集元数据对象
        
    Returns:
        (features, operators, delta_times, constants) 元组
    """
    # 验证规格
    errors = validate_family_spec(family_spec)
    if errors:
        raise ValueError(f"Invalid family spec: {errors}")
    
    # 使用 alpha_gfn.search_space_v2 统一构建
    from alpha_gfn.search_space_v2 import build_search_space
        
    # 转换字段名以匹配 search_space_v2
    if 'operator_whitelist' in family_spec and 'operator_names' not in family_spec:
        family_spec['operator_names'] = family_spec['operator_whitelist']
        
    features, operators, delta_times, constants = build_search_space(
        family_spec=family_spec, 
        dataset_meta=dataset_meta
    )
    
    # 二次过滤 (如果 family_spec 有额外要求)
    forbid_operators = family_spec.get('forbid_operators', [])
    if forbid_operators:
        operators = [op for op in operators if (getattr(op, "__name__", op.__class__.__name__)) not in forbid_operators]
        
    logger.info(f"Built search space for family {family_spec.get('family_id', 'unknown')}: "
                f"{len(features)} features, {len(operators)} operators, {len(delta_times)} delta_times, {len(constants)} constants")
    
    return features, operators, delta_times, constants

def create_default_family_spec(family_id: str, domain: str = "A") -> Dict[str, Any]:
    """
    创建默认的因子族规格
    
    Args:
        family_id: 因子族ID
        domain: 数据域
        
    Returns:
        默认因子族规格
    """
    return {
        "family_id": family_id,
        "enabled_layers": ["raw", "atomic"],
        "allowed_domains": [domain],
        "allowed_freq_groups": ["eod"],
        "operator_whitelist": [
            "Ref", "TsMean", "TsStd", "TsRank", "TsDelta", 
            "TsCorr", "TsCov", "Add", "Sub", "Mul", "Div", "Rank"
        ],
        "forbid_operators": ["Greater", "Less", "Pow"],
        "delta_times": [5, 10, 20, 40, 60],
        "constants": [0.5, 1.0, 2.0],
        "max_expr_length": 18,
        "min_ts_operator_count": 1,
        "feature_scopes": {
            "raw": ["open", "high", "low", "close", "vwap", "amount", "volume", "turnover"],
            "atomic": ["ret_5", "ret_10", "vwap_dev", "amount_z20", "corr_close_volume_20"]
        },
        "sample_budget": 20000,
        "pool_capacity": 200
    }

def save_family_spec(spec: Dict[str, Any], output_path: str):
    """
    保存因子族规格到文件
    
    Args:
        spec: 因子族规格
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.yaml', '.yml']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved family spec to {output_path}")

def list_available_families(config_dir: str = "config/factor_families") -> List[str]:
    """
    列出可用的因子族
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        可用的因子族ID列表
    """
    if not os.path.exists(config_dir):
        return []
    
    families = []
    for filename in os.listdir(config_dir):
        if filename.endswith(('.yaml', '.yml', '.json')):
            family_id = os.path.splitext(filename)[0]
            families.append(family_id)
    
    return sorted(families)

def validate_family_compatibility(family_spec: Dict[str, Any], dataset_meta: Dict[str, Any]) -> List[str]:
    """
    验证因子族规格与数据集元数据的兼容性
    
    Args:
        family_spec: 因子族规格
        dataset_meta: 数据集元数据
        
    Returns:
        兼容性错误列表
    """
    errors = []
    
    # 检查域兼容性
    if 'allowed_domains' in family_spec and 'domain' in dataset_meta:
        allowed_domains = family_spec['allowed_domains']
        dataset_domain = dataset_meta['domain']
        if dataset_domain not in allowed_domains:
            errors.append(f"Dataset domain '{dataset_domain}' not in allowed domains {allowed_domains}")
    
    # 检查频率组兼容性
    if 'allowed_freq_groups' in family_spec and 'freq_group' in dataset_meta:
        allowed_freq_groups = family_spec['allowed_freq_groups']
        dataset_freq_group = dataset_meta['freq_group']
        if dataset_freq_group not in allowed_freq_groups:
            errors.append(f"Dataset freq_group '{dataset_freq_group}' not in allowed freq_groups {allowed_freq_groups}")
    
    # 检查层兼容性
    if 'enabled_layers' in family_spec and 'layers_enabled' in dataset_meta:
        enabled_layers = family_spec['enabled_layers']
        dataset_layers = dataset_meta['layers_enabled']
        missing_layers = set(enabled_layers) - set(dataset_layers)
        if missing_layers:
            errors.append(f"Family requires layers {missing_layers} not enabled in dataset")
    
    return errors

if __name__ == "__main__":
    # 测试功能
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例配置
    example_spec = create_default_family_spec("pv_ts_core", "A")
    print("Example family spec:")
    print(yaml.dump(example_spec, default_flow_style=False))
    
    # 验证规格
    errors = validate_family_spec(example_spec)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Specification is valid")
    
    # 构建搜索空间
    operators, delta_times, constants = build_family_search_space(example_spec)
    print(f"Built search space: {len(operators)} operators, {len(delta_times)} delta_times, {len(constants)} constants")