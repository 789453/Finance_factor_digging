import yaml
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class FactorFamilyConfig:
    """因子族配置管理器"""
    
    def __init__(self, config_path: str):
        """
        初始化因子族配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"因子族配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # 支持多个文档（---分隔）
                configs = list(yaml.safe_load_all(f))
                
                if not configs:
                    raise ValueError("配置文件为空")
                
                # 如果只有一个文档，直接返回
                if len(configs) == 1:
                    return configs[0]
                
                # 如果有多个文档，合并它们
                merged_config = {}
                for config in configs:
                    if config and 'family_id' in config:
                        family_id = config['family_id']
                        merged_config[family_id] = config
                
                return merged_config
                
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _validate_config(self):
        """验证配置有效性"""
        if not self.config:
            raise ValueError("配置为空")
        
        # 如果是单因子族配置
        if 'family_id' in self.config:
            self._validate_single_family(self.config)
        else:
            # 如果是多因子族配置
            for family_id, family_config in self.config.items():
                self._validate_single_family(family_config, family_id)
    
    def _validate_single_family(self, family_config: Dict[str, Any], family_id: str = None):
        """验证单个因子族配置"""
        if family_id is None:
            family_id = family_config.get('family_id')
        
        if not family_id:
            raise ValueError("缺少family_id字段")
        
        # 必需字段验证
        required_fields = ['enabled_layers', 'allowed_domains', 'operator_whitelist', 'delta_times', 'constants']
        for field in required_fields:
            if field not in family_config:
                raise ValueError(f"因子族 {family_id} 缺少必需字段: {field}")
        
        # 验证层类型
        valid_layers = ['raw', 'atomic']
        for layer in family_config['enabled_layers']:
            if layer not in valid_layers:
                raise ValueError(f"因子族 {family_id} 包含无效的层类型: {layer}")
        
        # 验证域
        valid_domains = ['A', 'B', 'C', 'E']
        for domain in family_config['allowed_domains']:
            if domain not in valid_domains:
                raise ValueError(f"因子族 {family_id} 包含无效的域: {domain}")
        
        # 验证频率组
        if 'allowed_freq_groups' in family_config:
            valid_freq_groups = ['eod', 'intraday', 'weekly', 'monthly']
            for freq_group in family_config['allowed_freq_groups']:
                if freq_group not in valid_freq_groups:
                    raise ValueError(f"因子族 {family_id} 包含无效的频率组: {freq_group}")
        
        # 验证操作符
        if not isinstance(family_config['operator_whitelist'], list) or not family_config['operator_whitelist']:
            raise ValueError(f"因子族 {family_id} 的operator_whitelist必须是包含至少一个操作符的列表")
        
        # 验证时间窗口
        if not isinstance(family_config['delta_times'], list) or not family_config['delta_times']:
            raise ValueError(f"因子族 {family_id} 的delta_times必须是包含至少一个时间窗口的列表")
        
        # 验证常数
        if not isinstance(family_config['constants'], list) or not family_config['constants']:
            raise ValueError(f"因子族 {family_id} 的constants必须是包含至少一个常数的列表")
        
        logger.info(f"验证因子族配置: {family_id} ✓")
    
    def get_family_config(self, family_id: str) -> Dict[str, Any]:
        """
        获取指定因子族的配置
        
        Args:
            family_id: 因子族ID
            
        Returns:
            因子族配置字典
        """
        if 'family_id' in self.config and self.config['family_id'] == family_id:
            return self.config.copy()
        elif family_id in self.config:
            return self.config[family_id].copy()
        else:
            raise ValueError(f"因子族 {family_id} 不存在")
    
    def get_all_families(self) -> List[str]:
        """
        获取所有因子族ID
        
        Returns:
            因子族ID列表
        """
        if 'family_id' in self.config:
            return [self.config['family_id']]
        else:
            return list(self.config.keys())
    
    def get_search_space_config(self, family_id: str) -> Dict[str, Any]:
        """
        获取搜索空间配置
        
        Args:
            family_id: 因子族ID
            
        Returns:
            搜索空间配置
        """
        family_config = self.get_family_config(family_id)
        
        return {
            'operator_names': family_config['operator_whitelist'],
            'delta_times': family_config['delta_times'],
            'constants': family_config['constants'],
            'max_expr_length': family_config.get('max_expr_length', 20),
            'min_ts_operator_count': family_config.get('min_ts_operator_count', 1),
            'forbid_operators': family_config.get('forbid_operators', [])
        }
    
    def get_feature_scope(self, family_id: str, layer: str) -> List[str]:
        """
        获取特征范围
        
        Args:
            family_id: 因子族ID
            layer: 层类型 (raw, atomic)
            
        Returns:
            特征名称列表
        """
        family_config = self.get_family_config(family_id)
        
        if 'feature_scopes' not in family_config:
            return []  # 如果没有定义特征范围，返回空列表
        
        feature_scopes = family_config['feature_scopes']
        
        if layer not in feature_scopes:
            return []  # 如果指定层没有特征范围，返回空列表
        
        return feature_scopes[layer]
    
    def get_pool_config(self, family_id: str) -> Dict[str, Any]:
        """
        获取池配置
        
        Args:
            family_id: 因子族ID
            
        Returns:
            池配置
        """
        family_config = self.get_family_config(family_id)
        
        return {
            'sample_budget': family_config.get('sample_budget', 10000),
            'pool_capacity': family_config.get('pool_capacity', 100)
        }
    
    def validate_family_for_domain(self, family_id: str, domain: str, freq_group: str = 'eod') -> List[str]:
        """
        验证因子族是否适用于指定域和频率组
        
        Args:
            family_id: 因子族ID
            domain: 域标识
            freq_group: 频率组
            
        Returns:
            验证问题列表
        """
        issues = []
        
        try:
            family_config = self.get_family_config(family_id)
        except ValueError as e:
            return [str(e)]
        
        # 检查域
        if domain not in family_config['allowed_domains']:
            issues.append(f"域 {domain} 不在允许列表中: {family_config['allowed_domains']}")
        
        # 检查频率组
        if 'allowed_freq_groups' in family_config:
            if freq_group not in family_config['allowed_freq_groups']:
                issues.append(f"频率组 {freq_group} 不在允许列表中: {family_config['allowed_freq_groups']}")
        
        return issues
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置，override_config中的值会覆盖base_config中的值
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                # 如果是列表，合并而不是覆盖
                merged[key] = list(set(merged[key] + value))
            else:
                merged[key] = value
        
        return merged

class FactorFamilyRegistry:
    """因子族注册表，管理多个因子族配置"""
    
    def __init__(self, config_dir: str):
        """
        初始化因子族注册表
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = config_dir
        self.family_configs = {}
        self._load_all_families()
    
    def _load_all_families(self):
        """加载所有因子族配置"""
        if not os.path.exists(self.config_dir):
            logger.warning(f"配置目录不存在: {self.config_dir}")
            return
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                config_path = os.path.join(self.config_dir, filename)
                try:
                    config = FactorFamilyConfig(config_path)
                    for family_id in config.get_all_families():
                        self.family_configs[family_id] = config.get_family_config(family_id)
                        logger.info(f"加载因子族: {family_id} from {filename}")
                except Exception as e:
                    logger.error(f"加载配置文件失败 {filename}: {e}")
    
    def get_family(self, family_id: str) -> Dict[str, Any]:
        """
        获取指定因子族
        
        Args:
            family_id: 因子族ID
            
        Returns:
            因子族配置
        """
        if family_id not in self.family_configs:
            raise ValueError(f"因子族 {family_id} 不存在")
        
        return self.family_configs[family_id].copy()
    
    def get_all_families(self) -> List[str]:
        """
        获取所有因子族ID
        
        Returns:
            因子族ID列表
        """
        return list(self.family_configs.keys())
    
    def get_families_by_domain(self, domain: str) -> List[str]:
        """
        获取指定域的所有因子族
        
        Args:
            domain: 域标识
            
        Returns:
            因子族ID列表
        """
        compatible_families = []
        
        for family_id, config in self.family_configs.items():
            if domain in config.get('allowed_domains', []):
                compatible_families.append(family_id)
        
        return compatible_families

def create_default_family_config(family_id: str, 
                                enabled_layers: List[str] = None,
                                allowed_domains: List[str] = None,
                                operator_whitelist: List[str] = None) -> Dict[str, Any]:
    """
    创建默认的因子族配置
    
    Args:
        family_id: 因子族ID
        enabled_layers: 启用的层
        allowed_domains: 允许的域
        operator_whitelist: 操作符白名单
        
    Returns:
        因子族配置字典
    """
    return {
        'family_id': family_id,
        'enabled_layers': enabled_layers or ['raw', 'atomic'],
        'allowed_domains': allowed_domains or ['A'],
        'allowed_freq_groups': ['eod'],
        'operator_whitelist': operator_whitelist or ['Ref', 'TsMean', 'TsStd', 'TsRank', 'Add', 'Sub', 'Mul', 'Div', 'Rank'],
        'delta_times': [5, 10, 20, 40, 60],
        'constants': [0.5, 1.0, 2.0],
        'max_expr_length': 15,
        'min_ts_operator_count': 1,
        'forbid_operators': [],
        'feature_scopes': {
            'raw': ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'],
            'atomic': ['ret_5', 'ret_10', 'vol_20']
        },
        'sample_budget': 10000,
        'pool_capacity': 100
    }