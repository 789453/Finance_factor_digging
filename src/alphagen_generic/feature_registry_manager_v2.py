import pandas as pd
from enum import IntEnum
import os
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Union, Literal, Any
from datetime import datetime

# 定义ROOT路径
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    domain: str
    layer: Literal["raw", "filled", "atomic"]
    category: str
    description: str
    status: str = "active"
    formula: Optional[str] = None
    source_cols: Optional[str] = None

class FeatureRegistryManagerV2:
    """
    升级版特征注册管理器，支持两层输入设计。
    负责特征的元数据管理、准入控制和枚举生成。
    """
    
    VALID_LAYERS = ["raw", "filled", "atomic"]
    
    def __init__(self, 
                 raw_registry_path: str = None,
                 atomic_registry_path: str = None):
        self.raw_registry_path = raw_registry_path or os.path.join(ROOT, "data", "registry", "feature_registry.csv")
        self.atomic_registry_path = atomic_registry_path or os.path.join(ROOT, "data", "registry", "feature_registry_atomic.csv")
        
        # 加载数据
        self._load_registries()

    def register_feature(self, spec: FeatureSpec):
        """手动注册一个特征"""
        self.features[spec.name.upper()] = spec

    def register_from_dataset_meta(self, dataset_meta: Any):
        """从数据集元数据中批量注册特征"""
        cols = dataset_meta.columns # 这是一个字典 {alias: actual_col_name}
        domain = dataset_meta.domain
        
        for alias, col_name in cols.items():
            if alias in ['date', 'code']: continue
            
            # 同时注册 alias 和物理名 (如 volume 和 vol)
            names_to_register = {alias, col_name}
            for name in names_to_register:
                spec = FeatureSpec(
                    name=name,
                    domain=domain,
                    layer="raw",
                    category="price" if name.lower() in ['open', 'high', 'low', 'close', 'settle', 'vwap'] else "volume",
                    description=f"Auto-registered: {name}"
                )
                self.register_feature(spec)
        logger.info(f"Registered features from dataset meta for domain {domain}")

    def _load_registries(self):
        # 简化版：如果文件不存在则使用默认值
        self.features: Dict[str, FeatureSpec] = {}
        
        # 默认原始特征
        default_raw = [
            ("open", "price"), ("high", "price"), ("low", "price"), ("close", "price"),
            ("volume", "volume"), ("amount", "volume"), ("turnover", "volume"), ("vwap", "price")
        ]
        for name, cat in default_raw:
            self.features[name.upper()] = FeatureSpec(
                name=name, domain="A", layer="raw", category=cat, description=name
            )
            
        # 默认原子特征
        default_atomic = [
            ("ret_1", "momentum"), ("ret_5", "momentum"), ("ret_10", "momentum"), ("ret_20", "momentum")
        ]
        for name, cat in default_atomic:
            self.features[name.upper()] = FeatureSpec(
                name=name, domain="A", layer="atomic", category=cat, description=name
            )

    def get_feature_spec(self, feature_name: str) -> Optional[FeatureSpec]:
        return self.features.get(feature_name.upper())

    def get_features_by_domain_and_layer(self, domain: str, layer: str, status_filter: List[str] = ['active']) -> List[str]:
        return [f.name for f in self.features.values() if f.domain == domain and f.layer == layer and f.status in status_filter]

    def create_feature_enum(self, domain: str, status_filter: List[str] = ['active'], layers: List[str] = None) -> IntEnum:
        if layers is None: layers = ["raw", "atomic"]
        
        features = []
        for layer in layers:
            features.extend(self.get_features_by_domain_and_layer(domain, layer, status_filter))
            
        enum_dict = {'CLOSE': 0}
        idx = 1
        for name in sorted(features):
            upper_name = name.upper()
            if upper_name not in enum_dict:
                enum_dict[upper_name] = idx
                idx += 1
        return IntEnum('FeatureType', enum_dict)

# Legacy compatibility
class FeatureRegistryManager(FeatureRegistryManagerV2):
    def __init__(self, registry_path: str = None):
        super().__init__(raw_registry_path=registry_path)
    def get_features_by_domain(self, domain: str, status_filter: List[str] = ['active']) -> List[str]:
        return self.get_features_by_domain_and_layer(domain, "raw", status_filter)
