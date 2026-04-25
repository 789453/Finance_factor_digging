import pandas as pd
from enum import IntEnum
import os
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Union, Literal, Any
from datetime import datetime

# 定义ROOT路径为项目根目录
ROOT = Path(__file__).parent.parent.parent
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
        """
        初始化特征注册表。
        不再使用硬编码的默认特征，特征应通过 register_from_dataset_meta 动态注册，
        或从 CSV 注册表文件加载。
        """
        self.features: Dict[str, FeatureSpec] = {}
        
        # 1. 尝试加载原始特征注册表
        if os.path.exists(self.raw_registry_path):
            try:
                # 尝试多种编码加载
                df = None
                for encoding in ['utf-8', 'gbk', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(self.raw_registry_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is not None:
                    # 处理列名差异
                    name_col = 'feature_name' if 'feature_name' in df.columns else 'name'
                    for _, row in df.iterrows():
                        spec = FeatureSpec(
                            name=row[name_col],
                            domain=row.get('domain', 'A'),
                            layer='raw',
                            category=row.get('category', 'unknown'),
                            description=row.get('description', ''),
                            status=row.get('status', 'active')
                        )
                        self.register_feature(spec)
                    logger.info(f"Loaded {len(df)} features from {self.raw_registry_path}")
                else:
                    logger.warning(f"Could not load registry {self.raw_registry_path} with any supported encoding.")
            except Exception as e:
                logger.warning(f"Failed to load raw registry from {self.raw_registry_path}: {e}")

        # 2. 尝试加载原子特征注册表
        if os.path.exists(self.atomic_registry_path):
            try:
                df = None
                for encoding in ['utf-8', 'gbk', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(self.atomic_registry_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is not None:
                    # 处理列名差异
                    name_col = 'feature_name' if 'feature_name' in df.columns else 'name'
                    for _, row in df.iterrows():
                        spec = FeatureSpec(
                            name=row[name_col],
                            domain=row.get('domain', 'A'),
                            layer='atomic',
                            category=row.get('category', 'momentum'),
                            description=row.get('description', ''),
                            status=row.get('status', 'active')
                        )
                        self.register_feature(spec)
                    logger.info(f"Loaded {len(df)} atomic features from {self.atomic_registry_path}")
                else:
                    logger.warning(f"Could not load atomic registry {self.atomic_registry_path} with any supported encoding.")
            except Exception as e:
                logger.warning(f"Failed to load atomic registry from {self.atomic_registry_path}: {e}")

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

    def register_from_atomic_registry(self, atomic_registry: Any):
        """Register features from an AtomicRegistry instance."""
        for field in atomic_registry.get_all_fields():
            spec = FeatureSpec(
                name=field.name,
                domain=field.domain.value,
                layer="atomic",
                category="atomic",
                description=field.economic_meaning or f"Atomic field: {field.name}"
            )
            self.register_feature(spec)
        logger.info(f"Registered {len(atomic_registry.get_all_fields())} atomic features from registry")

    def get_feature_enum(self, domain: str, status_filter: List[str] = ['active'], layers: List[str] = None) -> IntEnum:
        """Helper to get IntEnum for a specific domain, compatible with AlphaGen."""
        return self.create_feature_enum(domain, status_filter, layers)

# Legacy compatibility
class FeatureRegistryManager(FeatureRegistryManagerV2):
    def __init__(self, registry_path: str = None):
        super().__init__(raw_registry_path=registry_path)
    def get_features_by_domain(self, domain: str, status_filter: List[str] = ['active']) -> List[str]:
        return self.get_features_by_domain_and_layer(domain, "raw", status_filter)
