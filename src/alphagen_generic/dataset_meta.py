import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class DatasetMeta:
    """数据集元数据描述与验证"""
    
    REQUIRED_FIELDS = [
        "name", "frequency", "domain", "data_dir",
        "files", "columns", "date_range"
    ]
    
    OPTIONAL_FIELDS = [
        "version", "sample_pool", "supported_operators",
        "max_ast_depth", "max_ast_width", "recommended_delta_times",
        "recommended_constants"
    ]
    
    DEFAULT_COLUMN_MAPPING = {
        "date": "trade_date",
        "code": "ts_code",
        "close": "close"
    }
    
    def __init__(self, meta_path: str):
        self.meta_path = meta_path
        self.raw: Dict[str, Any] = self._load()
        self._validate_required_fields()
    
    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Dataset meta not found at {self.meta_path}")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _validate_required_fields(self) -> None:
        missing = [field for field in self.REQUIRED_FIELDS if field not in self.raw]
        if missing:
            raise ValueError(f"Missing required fields in dataset meta: {missing}")
    
    def validate(self) -> List[str]:
        """验证 meta 完整性，返回错误列表"""
        errors = []
        
        # 检查必填字段
        for field in self.REQUIRED_FIELDS:
            if field not in self.raw:
                errors.append(f"Missing required field: {field}")
        
        # 检查 domain 有效性
        if "domain" in self.raw and self.raw["domain"] not in ["A", "B", "C", "E"]:
            errors.append(f"Invalid domain: {self.raw['domain']}, must be A/B/C/E")
        
        # 检查 frequency 有效性
        if "frequency" in self.raw and self.raw["frequency"] not in ["daily", "30min", "5min"]:
            errors.append(f"Invalid frequency: {self.raw['frequency']}, must be daily/30min/5min")
        
        # 检查 files 结构
        if "files" in self.raw:
            if "raw" not in self.raw["files"] and "filled" not in self.raw["files"]:
                errors.append("At least one of 'raw' or 'filled' must be specified in files")
        
        # 检查 columns 结构
        if "columns" in self.raw:
            cols = self.raw["columns"]
            if "date" not in cols and "code" not in cols:
                errors.append("At least one of 'date' or 'code' must be specified in columns")
        
        # 检查 date_range 结构
        if "date_range" in self.raw:
            dr = self.raw["date_range"]
            if "start" not in dr and "end" not in dr:
                errors.append("At least one of 'start' or 'end' must be specified in date_range")
        
        # 检查 recommended_delta_times 格式
        if "recommended_delta_times" in self.raw:
            dt = self.raw["recommended_delta_times"]
            if not isinstance(dt, list) or not all(isinstance(x, int) for x in dt):
                errors.append("recommended_delta_times must be a list of integers")
        
        # 检查 recommended_constants 格式
        if "recommended_constants" in self.raw:
            consts = self.raw["recommended_constants"]
            if not isinstance(consts, list) or not all(isinstance(x, (int, float)) for x in consts):
                errors.append("recommended_constants must be a list of numbers")
        
        return errors
    
    def is_valid(self) -> bool:
        """检查 meta 是否有效"""
        return len(self.validate()) == 0
    
    def to_loader_kwargs(self) -> Dict[str, Any]:
        """转换为 ParquetFeatureLoader.__init__() 的参数 dict"""
        kwargs: Dict[str, Any] = {}
        
        # data_dir
        if "data_dir" in self.raw:
            kwargs["data_dir"] = self.raw["data_dir"]
        
        # feature_file_map_raw / filled
        if "files" in self.raw:
            files = self.raw["files"]
            if "raw" in files:
                kwargs["feature_file_map_raw"] = {
                    self.raw.get("domain", "A"): files["raw"]
                }
            if "filled" in files:
                kwargs["feature_file_map_filled"] = {
                    self.raw.get("domain", "A"): files["filled"]
                }
        
        # columns
        if "columns" in self.raw:
            cols = self.raw["columns"]
            if "date" in cols:
                kwargs["date_column"] = cols["date"]
            if "code" in cols:
                kwargs["code_column"] = cols["code"]
            if "close" in cols:
                kwargs["close_column"] = cols["close"]
        
        # sample_pool
        if "sample_pool" in self.raw:
            kwargs["pool_path"] = self.raw["sample_pool"]
        
        return kwargs
    
    def resolve_parquet_paths(self) -> Dict[str, str]:
        """解析 raw/filled parquet 绝对路径"""
        paths: Dict[str, str] = {}
        data_dir = self.raw.get("data_dir", "")
        
        if "files" in self.raw:
            files = self.raw["files"]
            if "raw" in files:
                raw_path = os.path.join(data_dir, files["raw"])
                paths["raw"] = raw_path
            if "filled" in files:
                filled_path = os.path.join(data_dir, files["filled"])
                paths["filled"] = filled_path
        
        return paths
    
    def get_search_space_hints(self) -> Dict[str, Any]:
        """返回推荐的 operators/delta_times/constants"""
        hints: Dict[str, Any] = {}
        
        if "supported_operators" in self.raw:
            hints["operator_names"] = self.raw["supported_operators"]
        
        if "recommended_delta_times" in self.raw:
            hints["delta_times"] = self.raw["recommended_delta_times"]
        
        if "recommended_constants" in self.raw:
            hints["constants"] = self.raw["recommended_constants"]
        
        if "max_ast_depth" in self.raw:
            hints["max_expr_length"] = self.raw["max_ast_depth"] * 2 + 2
        
        return hints
    
    @property
    def name(self) -> str:
        return self.raw.get("name", "unknown")
    
    @property
    def version(self) -> str:
        return self.raw.get("version", "1.0")
    
    @property
    def frequency(self) -> str:
        return self.raw.get("frequency", "daily")
    
    @property
    def domain(self) -> str:
        return self.raw.get("domain", "A")
    
    @property
    def date_range(self) -> Optional[Tuple[str, str]]:
        dr = self.raw.get("date_range", {})
        start = dr.get("start")
        end = dr.get("end")
        if start and end:
            return (start, end)
        elif start:
            return (start, "99991231")
        elif end:
            return ("00000001", end)
        return None
    
    @property
    def max_ast_depth(self) -> Optional[int]:
        return self.raw.get("max_ast_depth")
    
    @property
    def supported_operators(self) -> Optional[List[str]]:
        return self.raw.get("supported_operators")
    
    @staticmethod
    def create_default(
        name: str,
        domain: str = "A",
        frequency: str = "daily",
        data_dir: str = "data/factor_ready",
        output_path: Optional[str] = None,
    ) -> "DatasetMeta":
        """创建默认 meta 并可选保存到文件"""
        meta_dict = {
            "name": name,
            "version": "1.0",
            "frequency": frequency,
            "domain": domain,
            "data_dir": data_dir,
            "files": {
                "raw": f"feature_{domain.lower()}_price_volume.parquet",
                "filled": f"feature_{domain.lower()}_filled.parquet"
            },
            "columns": {
                "date": "trade_date",
                "code": "ts_code",
                "close": "close"
            },
            "date_range": {},
            "supported_operators": ["Ref", "Mean", "Std", "Delta", "Rank", "Min", "Max"],
            "max_ast_depth": 12,
            "recommended_delta_times": [5, 10, 20, 30, 40, 50, 60],
            "recommended_constants": [0.5, 1.0, 2.0, 3.0, 5.0],
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2, ensure_ascii=False)
        
        return DatasetMeta(output_path if output_path else "<memory>")
