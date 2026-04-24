#!/usr/bin/env python3
"""
DataHub配置模块 - 修复版本
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DataHubConfig:
    """DataHub配置"""
    warehouse_path: str
    control_db_path: str
    schema_metadata_path: Optional[str] = None
    integrity_summary_path: Optional[str] = None
    
    def __post_init__(self):
        """后处理"""
        # 设置默认值
        if self.schema_metadata_path is None:
            self.schema_metadata_path = os.path.join(self.warehouse_path, "meta", "schema_metadata.yaml")
        
        if self.integrity_summary_path is None:
            self.integrity_summary_path = os.path.join(self.warehouse_path, "meta", "integrity_summary.json")
    
    def validate_paths(self):
        """验证路径是否存在"""
        if not os.path.exists(self.warehouse_path):
            raise FileNotFoundError(f"Warehouse path does not exist: {self.warehouse_path}")
        
        if not os.path.exists(self.control_db_path):
            raise FileNotFoundError(f"Control database path does not exist: {self.control_db_path}")
        
        # 检查元数据目录是否存在，如果不存在则创建
        meta_dir = os.path.dirname(self.schema_metadata_path)
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir, exist_ok=True)
        
        meta_dir = os.path.dirname(self.integrity_summary_path)
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DataHubConfig':
        """从YAML文件创建配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            warehouse_path=config_data.get('warehouse_path'),
            control_db_path=config_data.get('control_db_path'),
            schema_metadata_path=config_data.get('schema_metadata_path'),
            integrity_summary_path=config_data.get('integrity_summary_path')
        )