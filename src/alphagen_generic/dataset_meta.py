import json
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetMeta:
    """数据集元数据描述与验证 - 升级版支持多层数据架构"""
    
    REQUIRED_FIELDS = [
        "dataset_id", "name", "frequency", "domain", "data_dir",
        "files", "columns", "date_range"
    ]
    
    OPTIONAL_FIELDS = [
        "version", "region", "market", "asset_type", "freq_group", 
        "timezone", "calendar", "sample_pool", "supported_operators",
        "max_ast_depth", "recommended_delta_times", "recommended_constants",
        "layers_enabled", "operator_whitelist", "target"
    ]
    
    # 配置化枚举，支持扩展
    VALID_DOMAINS = ["A", "B", "C", "E", "FUT", "OPT", "IDX", "FX", "MACRO", "CRYPTO"]
    VALID_FREQUENCIES = ["daily", "30min", "5min", "1min", "weekly", "monthly"]
    VALID_FREQ_GROUPS = ["eod", "intraday", "weekly", "monthly"]
    VALID_REGIONS = ["cn", "us", "hk", "global"]
    VALID_MARKETS = ["a_share", "us_equity", "hk_equity", "futures", "options", "fx", "macro", "index"]
    VALID_ASSET_TYPES = ["equity", "futures", "options", "crypto", "fx", "macro", "index"]
    VALID_LAYERS = ["raw", "filled", "atomic"]
    
    DEFAULT_COLUMN_MAPPING = {
        "date": "trade_date",
        "code": "ts_code", 
        "close": "close"
    }
    
    def __init__(self, meta_source: Union[str, Dict[str, Any]]):
        """
        支持两种构造方式：
        1. meta_path: str - 从JSON/YAML文件加载
        2. meta_source: Dict[str, Any] - 直接从字典构造
        """
        if isinstance(meta_source, dict):
            # 从字典直接构造
            self.meta_path = "<memory>"
            self.raw: Dict[str, Any] = meta_source.copy()
        elif isinstance(meta_source, str):
            # 从文件路径加载
            self.meta_path = meta_source
            self.raw: Dict[str, Any] = self._load()
        else:
            raise ValueError(f"meta_source must be str or dict, got {type(meta_source)}")
            
        self._validate_required_fields()
        self._normalize_meta()
    
    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Dataset meta not found at {self.meta_path}")
        
        ext = os.path.splitext(self.meta_path)[1].lower()
        if ext in ['.yaml', '.yml']:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
    
    def _validate_required_fields(self):
        """验证必填字段"""
        missing = [field for field in self.REQUIRED_FIELDS if field not in self.raw]
        if missing:
            raise ValueError(f"Missing required fields in dataset meta: {missing}")
    
    def _normalize_meta(self) -> None:
        """标准化元数据，填充默认值"""
        # 向后兼容处理
        if "dataset_id" not in self.raw:
            # 从旧格式生成dataset_id
            domain = self.raw.get("domain", "unknown")
            freq = self.raw.get("frequency", "daily")
            self.raw["dataset_id"] = f"{domain}.{freq}.v1"
            logger.warning(f"Missing dataset_id, generated: {self.raw['dataset_id']}")
        
        # 设置默认值
        self.raw.setdefault("version", "1.0")
        self.raw.setdefault("region", "cn")
        self.raw.setdefault("market", "a_share")
        self.raw.setdefault("asset_type", "equity")
        self.raw.setdefault("freq_group", "eod")
        self.raw.setdefault("timezone", "Asia/Shanghai")
        self.raw.setdefault("calendar", "SSE")
        self.raw.setdefault("layers_enabled", ["raw", "filled"])
        self.raw.setdefault("operator_whitelist", self.raw.get("supported_operators", []))
        
        # 验证dataset_id格式
        dataset_id = self.raw.get("dataset_id", "")
        if not isinstance(dataset_id, str) or len(dataset_id.split(".")) < 3:
            logger.warning(f"Dataset_id '{dataset_id}' might not follow the standard naming convention.")
    
    def validate(self) -> List[str]:
        """验证 meta 完整性，返回错误列表"""
        errors = []
        
        # 检查必填字段
        for field in self.REQUIRED_FIELDS:
            if field not in self.raw:
                errors.append(f"Missing required field: {field}")
        
        # 检查 domain 有效性
        if "domain" in self.raw:
            domain = self.raw["domain"]
            if domain not in self.VALID_DOMAINS:
                logger.warning(f"Domain '{domain}' not in standard domains {self.VALID_DOMAINS}")
        
        # 检查 frequency 有效性
        if "frequency" in self.raw:
            freq = self.raw["frequency"]
            if freq not in self.VALID_FREQUENCIES:
                errors.append(f"Invalid frequency: {freq}, must be one of {self.VALID_FREQUENCIES}")
        
        return errors
    
    def is_valid(self) -> bool:
        """检查 meta 是否有效"""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """将meta对象转换为字典"""
        return self.raw.copy()
    
    @property
    def target_config(self) -> Dict[str, Any]:
        """获取目标配置"""
        return self.raw.get("target", {})
        
    @property
    def target_price_column(self) -> str:
        """获取目标价格列名"""
        return self.target_config.get("price_column", self.columns.get("close", "close"))

    @property
    def dataset_id(self) -> str:
        return self.raw.get("dataset_id", "unknown")
    
    @property
    def name(self) -> str:
        return self.raw.get("name", "unknown")
    
    @property
    def frequency(self) -> str:
        return self.raw.get("frequency", "daily")
    
    @property
    def domain(self) -> str:
        return self.raw.get("domain", "A")
    
    @property
    def data_dir(self) -> str:
        return self.raw.get("data_dir", ".")
        
    @property
    def columns(self) -> Dict[str, str]:
        return self.raw.get("columns", {})
        
    @property
    def layers_enabled(self) -> List[str]:
        return self.raw.get("layers_enabled", ["raw", "filled"])
        
    @property
    def freq_group(self) -> str:
        return self.raw.get("freq_group", "eod")

    @property
    def date_range(self) -> Optional[Tuple[str, str]]:
        dr = self.raw.get("date_range", {})
        start = dr.get("start")
        end = dr.get("end")
        if start and end:
            return (str(start), str(end))
        return None

    def to_loader_kwargs(self) -> Dict[str, Any]:
        """转换为 Loader 所需的参数字典"""
        kwargs = {
            "dataset_id": self.dataset_id,
            "domain": self.domain,
            "data_dir": self.data_dir,
            "freq_group": self.freq_group,
            "layers": self.layers_enabled,
            "dataset_meta": self,
        }

        
        # 列名映射
        if self.columns:
            kwargs["date_column"] = self.columns.get("date", "trade_date")
            kwargs["code_column"] = self.columns.get("code", "ts_code")
            kwargs["close_column"] = self.columns.get("close", "close")
            
        # 目标配置
        if self.target_config:
            kwargs["target_config"] = self.target_config
            
        # 样本池
        if "sample_pool" in self.raw:
            kwargs["pool_path"] = self.raw["sample_pool"].get("path")
            
        # 显式文件映射
        if "files" in self.raw:
            files = self.raw["files"]
            if "raw" in files:
                kwargs["feature_file_map_raw"] = files["raw"]
            if "filled" in files:
                kwargs["feature_file_map_filled"] = files["filled"]
            if "atomic" in files:
                kwargs["atomic_path"] = files["atomic"]
                
        return kwargs

    def resolve_parquet_paths(self) -> Dict[str, str]:
        """解析所有可用的 Parquet 文件绝对路径"""
        paths = {}
        data_root = Path(self.data_dir)
        
        if "files" in self.raw:
            for layer, layer_info in self.raw["files"].items():
                if isinstance(layer_info, dict):
                    # 取第一个路径或根据 domain 匹配
                    rel_path = layer_info.get(self.domain) or next(iter(layer_info.values()))
                else:
                    rel_path = layer_info
                
                abs_path = data_root / rel_path
                if abs_path.exists():
                    paths[layer] = str(abs_path)
                    
        return paths
