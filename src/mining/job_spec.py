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
        "cache_root", "output_dir", "run_name", "seed", "cuda"
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
        
        # 设置时间戳
        if "created_at" not in self.raw:
            self.raw["created_at"] = datetime.now().isoformat()
    
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
    
    def to_args(self) -> Dict[str, Any]:
        """转换为命令行参数字典"""
        args = {}
        
        # 映射字段到命令行参数
        field_mapping = {
            "n_episodes": "n_episodes",
            "pool_capacity": "pool_capacity",
            "encoder_type": "encoder_type",
            "entropy_coef": "entropy_coef",
            "ssl_weight": "ssl_weight",
            "nov_weight": "nov_weight",
            "weight_decay_type": "weight_decay_type",
            "final_weight_ratio": "final_weight_ratio",
            "label_days": "label_days",
            "max_expr_length": "max_expr_length",
            "mask_dropout_prob": "mask_dropout_prob",
            "log_freq": "log_freq",
            "status_filter": "status_filter",
            "max_backtrack_days": "max_backtrack_days",
            "max_future_days": "max_future_days",
            "use_filled": "use_filled",
            "cache_root": "cache_root",
            "output_dir": "output_dir",
            "run_name": "run_name",
            "seed": "seed",
            "cuda": "cuda"
        }
        
        for spec_field, arg_field in field_mapping.items():
            if spec_field in self.raw:
                args[arg_field] = self.raw[spec_field]
        
        # 特殊处理
        if "train_start" in self.raw:
            args["train_start"] = self.raw["train_start"]
        if "train_end" in self.raw:
            args["train_end"] = self.raw["train_end"]
        if "test_start" in self.raw:
            args["test_start"] = self.raw["test_start"]
        if "test_end" in self.raw:
            args["test_end"] = self.raw["test_end"]
        
        return args
    
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
    
    # 可选字段属性
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

def load_job_spec(spec_path: str) -> MiningJobSpec:
    """
    加载作业规格
    
    Args:
        spec_path: 规格文件路径
        
    Returns:
        MiningJobSpec实例
    """
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

def save_job_spec(job_spec: MiningJobSpec, output_path: str):
    """
    保存作业规格
    
    Args:
        job_spec: 作业规格
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ext = os.path.splitext(output_path)[1].lower()
    spec_dict = job_spec.to_dict()
    
    if ext in ['.yaml', '.yml']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(spec_dict, f, default_flow_style=False, allow_unicode=True)
    elif ext == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {ext}")
    
    logger.info(f"Saved job spec to {output_path}")

def create_default_job_spec(job_id: str, name: str, dataset_id: str, family_id: str, 
                          segment_name: str = "default") -> MiningJobSpec:
    """
    创建默认的作业规格
    
    Args:
        job_id: 作业ID
        name: 作业名称
        dataset_id: 数据集ID
        family_id: 因子族ID
        segment_name: 区段名称
        
    Returns:
        默认MiningJobSpec实例
    """
    spec_dict = {
        "job_id": job_id,
        "name": name,
        "dataset_id": dataset_id,
        "family_id": family_id,
        "segment_name": segment_name,
        "version": "1.0",
        "description": f"Default job spec for {family_id}",
        "tags": ["default", "auto-generated"],
        "priority": "medium",
        "owner": "system",
        "train_start": "20200101",
        "train_end": "20201231",
        "test_start": "20210101",
        "test_end": "20211231",
        "n_episodes": 10000,
        "pool_capacity": 50,
        "encoder_type": "gnn",
        "entropy_coef": 0.01,
        "ssl_weight": 1.0,
        "nov_weight": 0.3,
        "weight_decay_type": "linear",
        "final_weight_ratio": 0.0,
        "label_days": 10,
        "max_expr_length": 20,
        "mask_dropout_prob": 1.0,
        "log_freq": 1000,
        "status_filter": ["active", "watch"],
        "max_backtrack_days": 100,
        "max_future_days": 30,
        "use_filled": True,
        "cache_root": "data/cache",
        "seed": 0,
        "cuda": 0
    }
    
    return MiningJobSpec(spec_dict)

def list_job_specs(spec_dir: str = "config/jobs") -> List[str]:
    """
    列出可用的作业规格
    
    Args:
        spec_dir: 规格文件目录
        
    Returns:
        可用的作业规格ID列表
    """
    if not os.path.exists(spec_dir):
        return []
    
    job_specs = []
    for filename in os.listdir(spec_dir):
        if filename.endswith(('.yaml', '.yml', '.json')):
            job_id = os.path.splitext(filename)[0]
            job_specs.append(job_id)
    
    return sorted(job_specs)

class JobSpecValidator:
    """作业规格验证器"""
    
    @staticmethod
    def validate_compatibility(job_spec: MiningJobSpec, dataset_meta: Dict[str, Any], 
                             family_spec: Dict[str, Any]) -> List[str]:
        """
        验证作业规格与数据集和因子族的兼容性
        
        Args:
            job_spec: 作业规格
            dataset_meta: 数据集元数据
            family_spec: 因子族规格
            
        Returns:
            兼容性错误列表
        """
        errors = []
        
        # 验证数据集ID
        if job_spec.dataset_id != dataset_meta.get("dataset_id"):
            errors.append(f"Job dataset_id '{job_spec.dataset_id}' != meta dataset_id '{dataset_meta.get('dataset_id')}'")
        
        # 验证因子族ID
        if job_spec.family_id != family_spec.get("family_id"):
            errors.append(f"Job family_id '{job_spec.family_id}' != spec family_id '{family_spec.get('family_id')}'")
        
        # 验证域兼容性 - 使用dataset_meta的domain而不是解析dataset_id
        meta_domain = dataset_meta.get("domain", "unknown")
        # 如果meta中有domain字段，则验证；否则跳过验证
        if meta_domain != "unknown":
            # 从dataset_meta获取domain，而不是解析dataset_id
            job_domain_from_meta = dataset_meta.get("domain", "A")
            if job_domain_from_meta != meta_domain:
                errors.append(f"Job domain from meta '{job_domain_from_meta}' != meta domain '{meta_domain}'")
        
        # 验证日期范围
        if job_spec.train_start < dataset_meta.get("date_range", {}).get("start", "00000000"):
            errors.append(f"Job train_start '{job_spec.train_start}' before dataset start")
        
        if job_spec.test_end > dataset_meta.get("date_range", {}).get("end", "99999999"):
            errors.append(f"Job test_end '{job_spec.test_end}' after dataset end")
        
        return errors

if __name__ == "__main__":
    # 测试功能
    logging.basicConfig(level=logging.INFO)
    
    # 创建默认规格
    default_spec = create_default_job_spec("test_job_001", "Test Job", "cn.a_share.equity.daily.v1", "pv_ts_core")
    print("Default job spec:")
    print(yaml.dump(default_spec.to_dict(), default_flow_style=False))
    
    # 验证规格
    errors = default_spec.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Specification is valid")
    @property
    def domain(self) -> str:
        """获取域 - 从dataset_id中提取"""
        dataset_id = self.raw.get("dataset_id", "")
        # 从dataset_id中提取域，例如 "cn.a_share.equity.daily.v1" -> "A"
        if ".a_share." in dataset_id:
            return "A"
        elif "pv_daily" in dataset_id:
            return "pv_daily"
        elif "moneyflow" in dataset_id:
            return "moneyflow"
        else:
            return "A"  # 默认域

    @property
    def domain(self) -> str:
        """获取域 - 从dataset_id中提取"""
        dataset_id = self.raw.get("dataset_id", "")
        # 从dataset_id中提取域，例如 "cn.a_share.equity.daily.v1" -> "A"
        if ".a_share." in dataset_id:
            return "A"
        elif "pv_daily" in dataset_id:
            return "pv_daily"
        elif "moneyflow" in dataset_id:
            return "moneyflow"
        else:
            return "A"  # 默认域
