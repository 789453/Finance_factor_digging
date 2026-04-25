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
        # 获取基础目录 (如果 warehouse_path 是文件，则获取其父目录的父目录，假设结构为 data/meta/warehouse.duckdb)
        # 如果 warehouse_path 是目录，则直接使用
        w_path = Path(self.warehouse_path)
        if w_path.is_file() or w_path.suffix == '.duckdb':
            base_dir = w_path.parent.parent
        else:
            base_dir = w_path
            
        # 设置默认值
        if self.schema_metadata_path is None:
            self.schema_metadata_path = str(base_dir / "meta" / "schema_metadata.yaml")
        
        if self.integrity_summary_path is None:
            self.integrity_summary_path = str(base_dir / "meta" / "integrity_summary.json")
    
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