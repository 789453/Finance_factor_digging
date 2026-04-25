import pandas as pd
from enum import IntEnum
import os
import logging
import sys
from pathlib import Path
try:
    from utils.path_utils import map_path, FACTOR_READY_DIR
except ImportError:
    try:
        from src.utils.path_utils import map_path, FACTOR_READY_DIR
    except ImportError:
        # Fallback definitions if path_utils is not available
        FACTOR_READY_DIR = "data/factor_ready"
        def map_path(path: str) -> str:
            return path
from typing import List, Dict, Optional, Set, Union
from datetime import datetime

# 定义ROOT路径
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logger = logging.getLogger(__name__)

class FeatureRegistryManagerV2:
    """
    升级版特征注册管理器，支持两层输入设计
    支持raw层和atomic层的特征管理
    """
    
    # 支持的层类型
    VALID_LAYERS = ["raw", "atomic"]
    
    # 特征分类
    FEATURE_CATEGORIES = {
        "momentum": "价格动量",
        "volatility": "波动率", 
        "deviation": "量价偏离",
        "rank": "排名型",
        "correlation": "时序关系",
        "stability": "稳定性",
        "neutral": "结构中性"
    }
    
    def __init__(self, 
                 raw_registry_path: str = None,
                 atomic_registry_path: str = None,
                 unified_registry_path: str = None):
        """
        初始化特征注册管理器
        
        Args:
            raw_registry_path: 原始特征注册表路径
            atomic_registry_path: 原子特征注册表路径
            unified_registry_path: 统一注册表路径（如果不提供，则分别管理）
        """
        self.raw_registry_path = raw_registry_path or os.path.join(FACTOR_READY_DIR, "feature_registry.csv")
        self.atomic_registry_path = atomic_registry_path or os.path.join(ROOT, "data", "registry", "feature_registry_atomic.csv")
        self.unified_registry_path = unified_registry_path
        
        # 加载注册表
        self.raw_df = None
        self.atomic_df = None
        self.unified_df = None
        
        self._load_registries()
        
        # 状态映射
        self.status_map = {}
        self._initialize_status_maps()
        
        # 层配置
        self.layer_config = {
            "raw": {"enabled": True, "priority": 1},
            "atomic": {"enabled": True, "priority": 2}
        }
    
    def _load_registries(self):
        """加载注册表"""
        # 加载原始特征注册表
        if os.path.exists(self.raw_registry_path):
            try:
                self.raw_df = pd.read_csv(self.raw_registry_path)
                self.raw_df.columns = [c.lower() for c in self.raw_df.columns]
                logger.info(f"加载原始特征注册表: {len(self.raw_df)} 个特征")
            except Exception as e:
                logger.warning(f"加载原始特征注册表失败: {e}")
                self.raw_df = self._create_default_raw_registry()
        else:
            logger.warning(f"原始特征注册表不存在: {self.raw_registry_path}")
            self.raw_df = self._create_default_raw_registry()
        
        # 加载原子特征注册表
        if os.path.exists(self.atomic_registry_path):
            try:
                self.atomic_df = pd.read_csv(self.atomic_registry_path)
                self.atomic_df.columns = [c.lower() for c in self.atomic_df.columns]
                logger.info(f"加载原子特征注册表: {len(self.atomic_df)} 个特征")
            except Exception as e:
                logger.warning(f"加载原子特征注册表失败: {e}")
                self.atomic_df = self._create_default_atomic_registry()
        else:
            logger.warning(f"原子特征注册表不存在: {self.atomic_registry_path}")
            self.atomic_df = self._create_default_atomic_registry()
        
        # 如果提供统一注册表路径，尝试加载
        if self.unified_registry_path and os.path.exists(self.unified_registry_path):
            try:
                self.unified_df = pd.read_csv(self.unified_registry_path)
                logger.info(f"加载统一特征注册表: {len(self.unified_df)} 个特征")
            except Exception as e:
                logger.warning(f"加载统一特征注册表失败: {e}")
    
    def _create_default_raw_registry(self) -> pd.DataFrame:
        """创建默认原始特征注册表"""
        default_data = [
            {"feature_name": "open", "domain": "A", "category": "price", "description": "开盘价", "allow_in_alphaprobe": 1},
            {"feature_name": "high", "domain": "A", "category": "price", "description": "最高价", "allow_in_alphaprobe": 1},
            {"feature_name": "low", "domain": "A", "category": "price", "description": "最低价", "allow_in_alphaprobe": 1},
            {"feature_name": "close", "domain": "A", "category": "price", "description": "收盘价", "allow_in_alphaprobe": 1},
            {"feature_name": "volume", "domain": "A", "category": "volume", "description": "成交量", "allow_in_alphaprobe": 1},
            {"feature_name": "amount", "domain": "A", "category": "volume", "description": "成交额", "allow_in_alphaprobe": 1},
            {"feature_name": "turnover", "domain": "A", "category": "volume", "description": "换手率", "allow_in_alphaprobe": 1},
            {"feature_name": "vwap", "domain": "A", "category": "price", "description": "成交量加权平均价", "allow_in_alphaprobe": 1},
        ]
        return pd.DataFrame(default_data)
    
    def _create_default_atomic_registry(self) -> pd.DataFrame:
        """创建默认原子特征注册表"""
        default_data = [
            {"feature_name": "ret_1", "domain": "A", "category": "momentum", "description": "1日收益率", "formula": "Ref($close, -1) / $close - 1", "source_cols": "close", "allow_in_alphaprobe": 1},
            {"feature_name": "ret_5", "domain": "A", "category": "momentum", "description": "5日收益率", "formula": "Ref($close, -5) / $close - 1", "source_cols": "close", "allow_in_alphaprobe": 1},
            {"feature_name": "ret_10", "domain": "A", "category": "momentum", "description": "10日收益率", "formula": "Ref($close, -10) / $close - 1", "source_cols": "close", "allow_in_alphaprobe": 1},
            {"feature_name": "ret_20", "domain": "A", "category": "momentum", "description": "20日收益率", "formula": "Ref($close, -20) / $close - 1", "source_cols": "close", "allow_in_alphaprobe": 1},
            {"feature_name": "vol_5", "domain": "A", "category": "volatility", "description": "5日波动率", "formula": "TsStd(Ref($close, -1) / $close - 1, 5)", "source_cols": "close", "allow_in_alphaprobe": 1},
            {"feature_name": "vol_10", "domain": "A", "category": "volatility", "description": "10日波动率", "formula": "TsStd(Ref($close, -1) / $close - 1, 10)", "source_cols": "close", "allow_in_alphaprobe": 1},
            {"feature_name": "vol_20", "domain": "A", "category": "volatility", "description": "20日波动率", "formula": "TsStd(Ref($close, -1) / $close - 1, 20)", "source_cols": "close", "allow_in_alphaprobe": 1},
        ]
        return pd.DataFrame(default_data)
    
    def _status_key(self, domain: str, feature_name: str, layer: str) -> str:
        """生成状态映射的统一键名"""
        if layer == "raw" or layer == "filled":
            return f"{domain}:{feature_name}"
        return f"{domain}:{feature_name}:{layer}"

    def _initialize_status_maps(self):
        """初始化状态映射"""
        # 原始特征状态映射
        if self.raw_df is not None:
            for _, row in self.raw_df.iterrows():
                feature = row['feature_name']
                domain = row['domain']
                key = self._status_key(domain, feature, "raw")
                
                # 默认状态：如果允许使用则为active
                if row.get('allow_in_alphaprobe', 0) == 1:
                    self.status_map[key] = 'active'
                else:
                    self.status_map[key] = 'inactive'
        
        # 原子特征状态映射
        if self.atomic_df is not None:
            for _, row in self.atomic_df.iterrows():
                feature = row['feature_name']
                domain = row['domain']
                key = self._status_key(domain, feature, "atomic")
                
                if row.get('allow_in_alphaprobe', 0) == 1:
                    self.status_map[key] = 'active'
                else:
                    self.status_map[key] = 'inactive'
    
    def get_features_by_domain_and_layer(self, 
                                       domain: str, 
                                       layer: str,
                                       status_filter: List[str] = ['active', 'watch']) -> List[str]:
        """
        获取指定域和层的特征列表
        
        Args:
            domain: 域标识 (A, B, C, E, FUT, etc.)
            layer: 层类型 (raw, atomic)
            status_filter: 允许的状态列表
            
        Returns:
            特征名称列表
        """
        if layer not in self.VALID_LAYERS:
            raise ValueError(f"不支持的层类型: {layer}，支持: {self.VALID_LAYERS}")
        
        if layer == "raw":
            if self.raw_df is None:
                return []
            domain_features = self.raw_df[self.raw_df['domain'] == domain]['feature_name'].tolist()
        elif layer == "atomic":
            if self.atomic_df is None:
                return []
            domain_features = self.atomic_df[self.atomic_df['domain'] == domain]['feature_name'].tolist()
        
        # 过滤状态
        filtered_features = []
        for feature in domain_features:
            key = self._status_key(domain, feature, layer)
            if self.status_map.get(key) in status_filter:
                filtered_features.append(feature)
        
        return filtered_features
    
    def get_features_by_domain(self, domain: str, status_filter: List[str] = ['active', 'watch']) -> List[str]:
        """
        获取指定域的所有特征（合并所有层）
        
        Args:
            domain: 域标识
            status_filter: 允许的状态列表
            
        Returns:
            特征名称列表
        """
        all_features = []
        
        # 获取原始层特征
        if self.layer_config["raw"]["enabled"]:
            raw_features = self.get_features_by_domain_and_layer(domain, "raw", status_filter)
            all_features.extend(raw_features)
        
        # 获取原子层特征
        if self.layer_config["atomic"]["enabled"]:
            atomic_features = self.get_features_by_domain_and_layer(domain, "atomic", status_filter)
            # 为原子特征添加前缀以避免冲突
            atomic_features_prefixed = [f"atomic_{f}" for f in atomic_features]
            all_features.extend(atomic_features_prefixed)
        
        return all_features
    
    def get_feature_info(self, feature_name: str, layer: str = None) -> Dict:
        """
        获取特征信息
        
        Args:
            feature_name: 特征名称
            layer: 层类型（可选）
            
        Returns:
            特征信息字典
        """
        if layer is None:
            # 尝试从所有层中查找
            for layer_type in self.VALID_LAYERS:
                info = self._get_feature_info_from_layer(feature_name, layer_type)
                if info:
                    return info
            return {}
        else:
            return self._get_feature_info_from_layer(feature_name, layer)
    
    def _get_feature_info_from_layer(self, feature_name: str, layer: str) -> Dict:
        """从指定层获取特征信息"""
        if layer == "raw" and self.raw_df is not None:
            row = self.raw_df[self.raw_df['feature_name'] == feature_name]
            if not row.empty:
                return row.iloc[0].to_dict()
        elif layer == "atomic" and self.atomic_df is not None:
            # 去掉原子特征的前缀
            clean_name = feature_name.replace("atomic_", "", 1)
            row = self.atomic_df[self.atomic_df['feature_name'] == clean_name]
            if not row.empty:
                return row.iloc[0].to_dict()
        
        return {}
    
    def get_fill_policy(self, feature_name: str, layer: str = "raw") -> str:
        """获取特征的填充策略"""
        info = self.get_feature_info(feature_name, layer)
        return info.get('fill_policy', 'none')
    
    def create_feature_enum(self, domain: str, status_filter: List[str] = ['active', 'watch'], layers: List[str] = None) -> IntEnum:
        """
        动态创建FeatureType IntEnum
        
        Args:
            domain: 域标识
            status_filter: 允许的状态列表
            layers: 指定要包含的层，如果为None则使用所有启用的层
            
        Returns:
            IntEnum类
        """
        # 如果没有指定层，使用所有启用的层
        if layers is None:
            layers = []
            if self.layer_config["raw"]["enabled"]:
                layers.append("raw")
            if self.layer_config["atomic"]["enabled"]:
                layers.append("atomic")
        
        features = []
        for layer in layers:
            if layer == "raw" and self.layer_config["raw"]["enabled"]:
                raw_features = self.get_features_by_domain_and_layer(domain, "raw", status_filter)
                features.extend(raw_features)
            elif layer == "atomic" and self.layer_config["atomic"]["enabled"]:
                atomic_features = self.get_features_by_domain_and_layer(domain, "atomic", status_filter)
                # 为原子特征添加前缀以避免冲突
                atomic_features_prefixed = [f"atomic_{f}" for f in atomic_features]
                features.extend(atomic_features_prefixed)
        
        enum_dict = {'CLOSE': 0}  # 保留CLOSE作为目标计算
        idx = 1
        
        # 添加域特征
        for name in features:
            upper_name = name.upper()
            if upper_name not in enum_dict:
                enum_dict[upper_name] = idx
                idx += 1
        
        return IntEnum('FeatureType', enum_dict)
    
    def create_feature_enum_by_layer(self, domain: str, layer: str, status_filter: List[str] = ['active', 'watch']) -> IntEnum:
        """
        为指定层创建FeatureType IntEnum
        
        Args:
            domain: 域标识
            layer: 层类型
            status_filter: 允许的状态列表
            
        Returns:
            IntEnum类
        """
        features = self.get_features_by_domain_and_layer(domain, layer, status_filter)
        
        enum_dict = {'CLOSE': 0}  # 保留CLOSE
        idx = 1
        
        # 添加层特征
        for name in features:
            upper_name = name.upper()
            if upper_name not in enum_dict:
                enum_dict[upper_name] = idx
                idx += 1
        
        return IntEnum('FeatureType', enum_dict)
    
    def update_feature_status(self, domain: str, feature_name: str, layer: str, status: str):
        """
        更新特征状态
        
        Args:
            domain: 域标识
            feature_name: 特征名称
            layer: 层类型
            status: 新状态
        """
        key = self._status_key(domain, feature_name, layer)
        self.status_map[key] = status
        logger.info(f"更新特征状态: {key} -> {status}")
    
    def add_atomic_feature(self, feature_def: Dict):
        """
        添加原子特征
        
        Args:
            feature_def: 特征定义字典
        """
        if self.atomic_df is None:
            self.atomic_df = pd.DataFrame()
        
        # 验证必需字段
        required_fields = ['feature_name', 'domain', 'formula', 'source_cols']
        for field in required_fields:
            if field not in feature_def:
                raise ValueError(f"缺少必需字段: {field}")
        
        # 添加默认值
        feature_def.setdefault('category', 'unknown')
        feature_def.setdefault('description', '')
        feature_def.setdefault('allow_in_alphaprobe', 1)
        feature_def.setdefault('created_at', datetime.now().isoformat())
        feature_def.setdefault('version', '1.0')
        
        # 添加到DataFrame
        new_row = pd.DataFrame([feature_def])
        self.atomic_df = pd.concat([self.atomic_df, new_row], ignore_index=True)
        
        # 更新状态映射
        key = f"{feature_def['domain']}:{feature_def['feature_name']}:atomic"
        self.status_map[key] = 'active'
        
        logger.info(f"添加原子特征: {feature_def['feature_name']}")
    
    def get_all_domains(self) -> List[str]:
        """获取所有域"""
        domains = set()
        
        if self.raw_df is not None:
            domains.update(self.raw_df['domain'].unique())
        
        if self.atomic_df is not None:
            domains.update(self.atomic_df['domain'].unique())
        
        return sorted(list(domains))
    
    def get_layer_config(self) -> Dict:
        """获取层配置"""
        return self.layer_config.copy()
    
    def set_layer_config(self, layer: str, enabled: bool, priority: int = None):
        """
        设置层配置
        
        Args:
            layer: 层类型
            enabled: 是否启用
            priority: 优先级（可选）
        """
        if layer not in self.VALID_LAYERS:
            raise ValueError(f"不支持的层类型: {layer}")
        
        self.layer_config[layer]["enabled"] = enabled
        if priority is not None:
            self.layer_config[layer]["priority"] = priority
        
        logger.info(f"更新层配置: {layer} -> enabled={enabled}, priority={priority}")
    
    def save_registries(self):
        """保存注册表到文件"""
        if self.raw_df is not None:
            self.raw_df.to_csv(self.raw_registry_path, index=False)
            logger.info(f"保存原始特征注册表: {self.raw_registry_path}")
        
        if self.atomic_df is not None:
            self.atomic_df.to_csv(self.atomic_registry_path, index=False)
            logger.info(f"保存原子特征注册表: {self.atomic_registry_path}")
    
    def validate_feature_compatibility(self, feature_name: str, layer: str, domain: str) -> List[str]:
        """
        验证特征兼容性
        
        Args:
            feature_name: 特征名称
            layer: 层类型
            domain: 域
            
        Returns:
            兼容性问题列表
        """
        issues = []
        
        # 检查层是否启用
        if not self.layer_config.get(layer, {}).get("enabled", False):
            issues.append(f"层 {layer} 未启用")
        
        # 检查特征是否存在
        features = self.get_features_by_domain_and_layer(domain, layer)
        if feature_name not in features:
            issues.append(f"特征 {feature_name} 在 {layer} 层中不存在")
        
        # 检查特征状态
        key = self._status_key(domain, feature_name, layer)
        if self.status_map.get(key) not in ['active', 'watch']:
            issues.append(f"特征 {feature_name} 状态为 {self.status_map.get(key)}")
        
        return issues

# 向后兼容的包装器
class FeatureRegistryManager(FeatureRegistryManagerV2):
    """向后兼容的包装器"""
    
    def __init__(self, registry_path: str = None):
        """
        向后兼容的初始化
        
        Args:
            registry_path: 注册表路径（仅用于原始特征）
        """
        # 调用父类初始化，只提供原始注册表路径
        super().__init__(raw_registry_path=registry_path)
    
    def get_features_by_domain(self, domain: str, status_filter: List[str] = ['active', 'watch']) -> List[str]:
        """
        向后兼容的方法，只返回原始特征
        """
        return self.get_features_by_domain_and_layer(domain, "raw", status_filter)