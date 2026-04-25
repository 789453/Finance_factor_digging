#!/usr/bin/env python3
"""
作业规格系统 - 定义完整的挖掘任务配置
"""

import yaml
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MiningJobSpec:
    """挖掘作业规格"""
    
    REQUIRED_FIELDS = [
        "job_id", "name", "dataset_id", "family_id", "segment_name",
        "train_start", "train_end", "test_start", "test_end"
    ]
    
    OPTIONAL_FIELDS = [
        "version", "description", "tags", "priority", "owner",
        "n_episodes", "pool_capacity", "encoder_type", "entropy_coef",
        "ssl_weight", "nov_weight", "weight_decay_type", "final_weight_ratio",
        "label_days", "max_expr_length", "mask_dropout_prob", "log_freq",
        "status_filter", "max_backtrack_days", "max_future_days", "use_filled",
        "cache_root", "output_dir", "run_name", "seed", "cuda", "domain"
    ]
    
    def __init__(self, spec_dict: Dict[str, Any]):
        """初始化作业规格"""
        self.raw = spec_dict.copy()
        self._validate_required_fields()
        self._normalize_spec()
    
    def _validate_required_fields(self):
        """验证必填字段"""
        missing = [field for field in self.REQUIRED_FIELDS if field not in self.raw]
        if missing:
            raise ValueError(f"Missing required fields in job spec: {missing}")
    
    def _normalize_spec(self):
        """标准化规格"""
        # 设置默认值
        self.raw.setdefault("version", "1.0")
        self.raw.setdefault("n_episodes", 10000)
        self.raw.setdefault("pool_capacity", 50)
        self.raw.setdefault("encoder_type", "gnn")
        self.raw.setdefault("entropy_coef", 0.01)
        self.raw.setdefault("ssl_weight", 1.0)
        self.raw.setdefault("nov_weight", 0.3)
        self.raw.setdefault("weight_decay_type", "linear")
        self.raw.setdefault("final_weight_ratio", 0.0)
        self.raw.setdefault("label_days", 10)
        self.raw.setdefault("max_expr_length", 20)
        self.raw.setdefault("mask_dropout_prob", 1.0)
        self.raw.setdefault("log_freq", 1000)
        self.raw.setdefault("status_filter", ["active", "watch"])
        self.raw.setdefault("max_backtrack_days", 100)
        self.raw.setdefault("max_future_days", 30)
        self.raw.setdefault("use_filled", True)
        self.raw.setdefault("cache_root", "data/cache")
        self.raw.setdefault("seed", 0)
        self.raw.setdefault("cuda", 0)
        
        # 如果没有显式提供domain，尝试从dataset_id推断（保留兼容性）
        if "domain" not in self.raw:
            self.raw["domain"] = self._infer_domain()
            
        # 设置时间戳
        if "created_at" not in self.raw:
            self.raw["created_at"] = datetime.now().isoformat()
    
    def _infer_domain(self) -> str:
        """从dataset_id推断域"""
        dataset_id = self.raw.get("dataset_id", "").lower()
        if "a_share" in dataset_id or "stock" in dataset_id:
            return "A"
        elif "futures" in dataset_id or "fut" in dataset_id:
            return "FUT"
        elif "index" in dataset_id or "idx" in dataset_id:
            return "IDX"
        elif "fx" in dataset_id or "forex" in dataset_id:
            return "FX"
        elif "crypto" in dataset_id:
            return "CRYPTO"
        elif "macro" in dataset_id:
            return "MACRO"
        return "A"


    def validate(self) -> List[str]:
        """验证作业规格"""
        errors = []
        
        # 验证必填字段
        for field in self.REQUIRED_FIELDS:
            if field not in self.raw:
                errors.append(f"Missing required field: {field}")
        
        # 验证日期格式
        date_fields = ["train_start", "train_end", "test_start", "test_end"]
        for field in date_fields:
            if field in self.raw:
                date_str = str(self.raw[field])
                if len(date_str) != 8 or not date_str.isdigit():
                    errors.append(f"Invalid date format for {field}: {date_str}, expected YYYYMMDD")
        
        # 验证数值字段
        numeric_fields = ["n_episodes", "pool_capacity", "entropy_coef", "ssl_weight", "nov_weight"]
        for field in numeric_fields:
            if field in self.raw:
                value = self.raw[field]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Invalid numeric value for {field}: {value}")
        
        # 验证枚举字段
        if "encoder_type" in self.raw:
            encoder_type = self.raw["encoder_type"]
            if encoder_type not in ["gnn", "transformer", "lstm"]:
                errors.append(f"Invalid encoder_type: {encoder_type}")
        
        if "weight_decay_type" in self.raw:
            decay_type = self.raw["weight_decay_type"]
            if decay_type not in ["linear", "exponential"]:
                errors.append(f"Invalid weight_decay_type: {decay_type}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.raw.copy()
    
    # 属性访问
    @property
    def job_id(self) -> str:
        return self.raw.get("job_id", "unknown")
    
    @property
    def name(self) -> str:
        return self.raw.get("name", "unknown")
    
    @property
    def dataset_id(self) -> str:
        return self.raw.get("dataset_id", "unknown")
    
    @property
    def family_id(self) -> str:
        return self.raw.get("family_id", "unknown")
    
    @property
    def segment_name(self) -> str:
        return self.raw.get("segment_name", "default")
    
    @property
    def train_start(self) -> str:
        return self.raw.get("train_start", "20200101")
    
    @property
    def train_end(self) -> str:
        return self.raw.get("train_end", "20201231")
    
    @property
    def test_start(self) -> str:
        return self.raw.get("test_start", "20210101")
    
    @property
    def test_end(self) -> str:
        return self.raw.get("test_end", "20211231")
    
    @property
    def domain(self) -> str:
        return self.raw.get("domain", "A")

    @property
    def status_filter(self) -> List[str]:
        return self.raw.get("status_filter", ["active", "watch"])
    
    @property
    def max_backtrack_days(self) -> int:
        return self.raw.get("max_backtrack_days", 100)
    
    @property
    def max_future_days(self) -> int:
        return self.raw.get("max_future_days", 30)
    
    @property
    def label_days(self) -> int:
        return self.raw.get("label_days", 10)
    
    @property
    def n_episodes(self) -> int:
        return self.raw.get("n_episodes", 10000)
    
    @property
    def pool_capacity(self) -> int:
        return self.raw.get("pool_capacity", 50)
    
    @property
    def encoder_type(self) -> str:
        return self.raw.get("encoder_type", "gnn")
    
    @property
    def entropy_coef(self) -> float:
        return self.raw.get("entropy_coef", 0.01)
    
    @property
    def ssl_weight(self) -> float:
        return self.raw.get("ssl_weight", 1.0)
    
    @property
    def nov_weight(self) -> float:
        return self.raw.get("nov_weight", 0.3)
    
    @property
    def max_expr_length(self) -> int:
        return self.raw.get("max_expr_length", 20)
    
    @property
    def cache_root(self) -> str:
        return self.raw.get("cache_root", "data/cache")
    
    @property
    def output_dir(self) -> str:
        return self.raw.get("output_dir", "output")
    
    @property
    def run_name(self) -> str:
        return self.raw.get("run_name", "default_run")
    
    @property
    def seed(self) -> int:
        return self.raw.get("seed", 0)
    
    @property
    def cuda(self) -> int:
        return self.raw.get("cuda", 0)

    @property
    def log_freq(self) -> int:
        return self.raw.get("log_freq", 1000)

def load_job_spec(spec_path: str) -> MiningJobSpec:
    """加载作业规格"""
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Job spec file not found: {spec_path}")
    
    ext = os.path.splitext(spec_path)[1].lower()
    
    if ext in ['.yaml', '.yml']:
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec_dict = yaml.safe_load(f)
    elif ext == '.json':
        with open(spec_path, 'r', encoding='utf-8') as f:
            spec_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return MiningJobSpec(spec_dict)

class JobSpecValidator:
    """作业规格验证器"""
    
    @staticmethod
    def validate_compatibility(job_spec: MiningJobSpec, dataset_meta: Dict[str, Any], 
                             family_spec: Dict[str, Any]) -> List[str]:
        """验证作业规格与数据集和因子族的兼容性"""
        errors = []
        
        # 验证数据集ID
        if job_spec.dataset_id != dataset_meta.get("dataset_id"):
            errors.append(f"Job dataset_id '{job_spec.dataset_id}' != meta dataset_id '{dataset_meta.get('dataset_id')}'")
        
        # 验证因子族ID
        if job_spec.family_id != family_spec.get("family_id"):
            errors.append(f"Job family_id '{job_spec.family_id}' != spec family_id '{family_spec.get('family_id')}'")
        
        # 验证域兼容性
        meta_domain = dataset_meta.get("domain")
        allowed_domains = family_spec.get("allowed_domains", [])
        if allowed_domains and meta_domain not in allowed_domains:
            errors.append(f"Dataset domain '{meta_domain}' not allowed by family '{job_spec.family_id}': {allowed_domains}")
            
        # 验证频率兼容性
        meta_freq = dataset_meta.get("freq_group") or dataset_meta.get("frequency")
        allowed_freqs = family_spec.get("allowed_freq_groups", [])
        if allowed_freqs and meta_freq not in allowed_freqs:
            errors.append(f"Dataset frequency '{meta_freq}' not allowed by family: {allowed_freqs}")
        
        # 验证日期范围
        if job_spec.train_start < dataset_meta.get("date_range", {}).get("start", "00000000"):
            errors.append(f"Job train_start '{job_spec.train_start}' before dataset start")
        
        if job_spec.test_end > dataset_meta.get("date_range", {}).get("end", "99999999"):
            errors.append(f"Job test_end '{job_spec.test_end}' after dataset end")
            
        # 验证层兼容性
        dataset_layers = set(dataset_meta.get("layers_enabled", []))
        family_layers = set(family_spec.get("enabled_layers", []))
        if family_layers and not family_layers.issubset(dataset_layers):
            errors.append(f"Family requires layers {family_layers - dataset_layers} not enabled in dataset")
            
        # 验证目标价格列
        target_cfg = dataset_meta.get("target", {})
        price_column = target_cfg.get("price_column", "close")
        columns = dataset_meta.get("columns", {})
        if price_column not in columns.values() and price_column not in columns:
            errors.append(f"Target price_column '{price_column}' not found in dataset columns")
        
        return errors
